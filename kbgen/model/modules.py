from typing import Optional
import torch
import torch.nn as nn
from mup import MuReadout, MuSharedReadout
from .positional_encodings import PositionalEncoding
from .transformer import TransformerDecoder, TransformerEncoder
from ..utils import shift_right
from .embeddings import NumericEmbedding


class ModuleDict(nn.ModuleDict):
    def __setitem__(self, key: str, module: nn.Module) -> None:
        super().__setitem__(key.replace(".", "_"), module)

    def __getitem__(self, key: str) -> nn.Module:
        return super().__getitem__(key.replace(".", "_"))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        if affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        if hasattr(self, "weight"):
            return self.weight * x / (norm + self.eps) + self.bias
        return x / (norm + self.eps)


class _ResBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, d_ff: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.linear1 = nn.Linear(self.d_model, 2 * self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.act = nn.GLU()
        # self.layer_norm = RMSNorm(self.d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(self.dropout(x))
        x = self.linear2(x)
        x = self.act(self.dropout(x))
        x = self.layer_norm(x + residual)
        return x


class DecoderModule(nn.Module):
    def __init__(
        self,
        config: dict,
        readout_dim: int,
        embedding: Optional[torch.Tensor] = None,
        numerical: bool = False,
    ) -> None:
        super().__init__()
        self.numerical = numerical
        d_model, d_ff, dropout = (
            config["d_model"],
            config["d_ff_mult"] * config["d_model"],
            config["dropout"],
        )

        self.decoder = nn.Sequential(
            *[
                _ResBlock(d_model, dropout, d_ff)
                for _ in range(config["field_decoder_layers"])
            ]
        )
        if embedding is not None:
            if config["use_mup"]:
                self.readout = MuSharedReadout(embedding, bias=False)
            else:
                self.readout = nn.Linear(d_model, readout_dim, bias=False)
                self.readout.weight = embedding
        else:
            if config["use_mup"]:
                self.readout = MuReadout(d_model, readout_dim)
            else:
                self.readout = nn.Linear(d_model, readout_dim)

    def forward(self, x):
        x = self.decoder(x)
        x = self.readout(x)
        if self.numerical:
            return x.sigmoid() + 1
        return x


class EncoderModule(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        d_model, d_ff, dropout = (
            config["d_model"],
            config["d_ff_mult"] * config["d_model"],
            config["dropout"],
        )

        self.encoder = nn.Sequential(
            *[
                _ResBlock(d_model, dropout, d_ff)
                for _ in range(config["field_encoder_layers"])
            ]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class NumericalEncoder(nn.Module):
    def __init__(self, config, embedding=None) -> None:
        super().__init__()
        if embedding is None:
            embedding = NumericEmbedding(config)
        self.embedding = embedding
        self.encoder = EncoderModule(config)

    def forward(self, x):
        x = self.embedding(x)
        return self.encoder(x)


class CategoricalEncoder(nn.Module):
    def __init__(self, config, embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = EncoderModule(config)
        self.pe = PositionalEncoding(config["d_model"], config["dropout"])

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(self.pe(x)).mean(dim=1)
        return x


class TextModule(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        """Module for text fields, includes both an encoder and a decoder.

        Args:
            config (dict): Configuration dictionary. Must contain the following keys:
                - text_model: "custom" or "t5-small"
                - vocab_size: Size of the vocabulary
                - d_model: Dimension of the model
                - dropout: Dropout rate
                - nhead: Number of attention heads
                - num_layers: Number of layers
                - d_ff_mult: Multiplier for the feedforward dimension (d_ff = d_ff_mult * d_model)
                - text_encoder_layers: Number of layers in the encoder
                - text_decoder_layers: Number of layers in the decoder
                - freeze: Whether to freeze the parameters of the T5 model
                - sparse_embedding: Whether to use sparse embeddings

        Raises:
            NotImplementedError: If text_model is not "custom" or "t5-small".

        Returns:
            TextModule
        """
        super().__init__()
        if config["text_model"] == "custom":
            self.input_embedding = nn.Embedding(
                config["vocab_size"],
                config["d_model"],
                # sparse=config["sparse_embeddings"],
            )
            self.pe = PositionalEncoding(
                config["d_model"], config["dropout"], max_len=256
            )
            self.encoder = TextEncoder(
                config,
                config["text_encoder_layers"],
                self.input_embedding,
                self.pe,
            )
            self.decoder = TextDecoder(
                config,
                num_layers=config["text_decoder_layers"],
                embedding=self.input_embedding,
                pe=self.pe,
            )
        elif config["text_model"] == "t5-small":
            from transformers import T5ForConditionalGeneration  # type: ignore

            self.model = T5ForConditionalGeneration.from_pretrained(  # type: ignore
                "t5-small",
            )

            self.encoder = self.model.encoder  # type: ignore

            class LMHeadDecoder(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.decoder = model.decoder
                    self.lm_head = model.lm_head
                    self.scale = model.model_dim**-0.5

                def forward(self, **kwargs):
                    x = self.decoder(**kwargs).last_hidden_state
                    x = self.lm_head(x * self.scale)
                    return x

            self.decoder = LMHeadDecoder(self.model)  # type: ignore
            self.input_embedding = self.model.get_input_embeddings()  # type: ignore
            if config["freeze"]:
                print("Freezing T5 parameters")
                for param in self.parameters():  # type: ignore
                    param.requires_grad = False
        else:
            raise NotImplementedError

    def _shift_right(self, input_ids, inplace=False):
        return shift_right(input_ids, inplace=inplace)

    def zero_pad(self, pad_token_id):
        self.encoder.embedding.weight.data[pad_token_id] = 0  # type: ignore


class TextEncoder(nn.Module):
    def __init__(self, config, num_layers=None, embedding=None, pe=None):
        super().__init__()
        if num_layers is None:
            num_layers = config["num_layers"]
        self.encoder = TransformerEncoder(
            d_model=config["d_model"],
            dropout=config["dropout"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * 1,
            num_layers=num_layers,
        )
        self.positional_encoding = (
            pe
            if pe is not None
            else PositionalEncoding(
                config["d_model"],
                config["dropout"],
            )
        )
        self.embedding = (
            embedding
            if embedding is not None
            else nn.Embedding(
                config["vocab_size"],
                config["d_model"],
            )
        )

    def forward(
        self,
        src,
        mask=None,
        attention_mask=None,
    ):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.encoder(
            src,
            mask,
            attention_mask,
        )
        return src


class TextDecoder(nn.Module):
    def __init__(self, config, num_layers=None, embedding=None, pe=None, causal=True):
        super().__init__()
        if num_layers is None:
            num_layers = config["num_layers"]
        self.decoder = TransformerDecoder(
            d_model=config["d_model"],
            dropout=config["dropout"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * 4,
            num_layers=num_layers,
        )
        self.emmbedding = embedding
        self.readout = LMHead(config, embedding)

        if pe is None:
            self.positional_encoding = PositionalEncoding(
                config["d_model"],
                config["dropout"],
            )
        else:
            self.positional_encoding = pe
        self.causal = causal

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        memory_mask=None,
        attention_mask=None,
        memory_key_padding_mask=None,
    ):
        if self.emmbedding is not None:
            x = self.emmbedding(input_ids)
        else:
            raise NotImplementedError(
                "Need to implement passing embeddings directly to decoder"
            )
        tgt_mask = self._causal_mask(x.shape[1], x.device) if self.causal else None
        x = self.positional_encoding(x)
        x = self.decoder(
            x,
            encoder_hidden_states,
            tgt_mask,
            memory_mask,
            attention_mask,
            memory_key_padding_mask,
        )
        x = self.readout(x)
        return x

    def _causal_mask(self, size, device=None):
        # TODO does caching help here?
        mask = torch.full((size, size), float("-inf"), device=device)
        mask.triu_(diagonal=1)
        return mask


class LMHead(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        use_mup = config.get("use_mup", False)
        if embedding is not None:
            if use_mup:
                self.linear = MuSharedReadout(embedding.weight, bias=False)
            else:
                self.linear = nn.Linear(config["d_model"], config["vocab_size"])
                self.linear.weight = embedding.weight
        else:
            if use_mup:
                self.linear = MuReadout(config["d_model"], config["vocab_size"])
            else:
                self.linear = nn.Linear(config["d_model"], config["vocab_size"])

    def forward(self, x):
        return self.linear(x)

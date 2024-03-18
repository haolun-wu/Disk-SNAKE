import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from collections import namedtuple

from ..utils.utils import is_missing
from ..utils import TensorDict, random_mask, reduce_by_mask, mean
from ..utils.metrics import Accuracy
from .transformer import TransformerEncoder
from .modules import (
    TextModule,
    DecoderModule,
    CategoricalEncoder,
    NumericalEncoder,
    ModuleDict,
)
from .positional_encodings import RNNPathing
from .embeddings import NumericEmbedding
import warnings

ModelOutputs = namedtuple(
    "ModelOutputs",
    [
        "preds",
        "targets",
        "loss",
        "loss_dict",
        "masked_loss_dict",
        "unmasked_loss_dict",
        "masked_error_dict",
        "unmasked_error_dict",
    ],
)


class KBFormer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        #  Initialize encoders
        num_layers = config["num_layers"]
        nhead = config["nhead"]
        dropout = config["dropout"]
        self.text_model = TextModule(config)

        self.categorical_encoder = CategoricalEncoder(
            config, self.text_model.input_embedding
        )

        numerical_embedding = None
        self.encoder_dict = ModuleDict()
        for field in config["fields"]["numerical"]:
            if config["tie_numerical_embeddings"]:
                if numerical_embedding is None:
                    numerical_embedding = NumericEmbedding(config)
                self.encoder_dict[field] = NumericalEncoder(config, numerical_embedding)
            else:
                self.encoder_dict[field] = NumericalEncoder(config)

        self.mask_embedding = nn.Embedding(len(config["fields"].all_fields), d_model)
        # Initialize Entity-level Attention Models
        self.hierarchy_encoder = RNNPathing(
            config["fields"].all_fields, config["d_model"]
        )
        self.entity_encoder = TransformerEncoder(
            num_layers,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Initialize Field-level decoder Models
        self.decoder_dict = ModuleDict()
        num_decoder = None
        for field in config["fields"]["numerical"]:
            if config["tie_numerical_decoders"] and num_decoder is not None:
                self.decoder_dict[field] = num_decoder
            else:
                self.decoder_dict[field] = (
                    num_decoder := DecoderModule(config, 1, numerical=True)
                )

        for field in config["fields"]["categorical"]:
            num_classes = config["categorical_num_classes"][field]
            self.decoder_dict[field] = DecoderModule(config, num_classes)

        # metrics
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=config["categorical_pad_token_id"], reduction="none"
        )

        self.accuracy = Accuracy(
            fields=config["fields"],
            ignore_idx_cat=config["categorical_pad_token_id"],
            ignore_idx_num=config["numerical_pad_token_id"],
        )

    def encode_properties(
        self,
        tensor_emb_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """
        Returns a tensor containing the encoded representations of the input data.
        The encoded representations are meant to be projections of the input embeddings
        into a common space where the entity-encoder can operate.

        Args:
            tensor_emb_dict (dict): A dictionary containing the tensor embeddings of the input data.
            key_padding_mask (dict): A dictionary containing the key padding masks for the input
            data. This is used to mask out padding tokens in text fields for instance.

        Returns:
            dict: A dictionary containing the encoded representations of the input data.
        """
        all_fields = self.config["fields"].all_fields
        codes = torch.empty(
            (
                tensor_emb_dict.size(),
                len(all_fields),
                self.config["d_model"],
            ),
            device=tensor_emb_dict.device(),
        )
        for idx, field in enumerate(all_fields):
            if field in self.config["fields"]["numerical"]:
                codes[:, idx] = self.encoder_dict[field](tensor_emb_dict[field])
            elif field in self.config["fields"]["categorical"]:
                codes[:, idx] = self.categorical_encoder(tensor_emb_dict[field])
            else:
                if self.config["text_model"] != "custom":
                    kpm = (
                        ~key_padding_mask[field].bool()
                        if key_padding_mask is not None
                        else None
                    )
                else:
                    kpm = (
                        key_padding_mask[field]
                        if key_padding_mask is not None
                        else None
                    )
                # TODO debug attention_mask leading to same result
                output = self.text_model.encoder(
                    tensor_emb_dict[field], attention_mask=kpm
                )
                if self.config["text_model"] == "custom":
                    codes[:, idx] = output[:, 0]
                else:
                    codes[:, idx] = output.last_hidden_state[:, 0]
        return codes

    def decode_properties(
        self,
        entity_embeddings: torch.Tensor,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
    ) -> TensorDict:
        preds = {}
        for idx, field in enumerate(self.config["fields"].all_fields):
            if field in self.config["fields"]["text"]:
                target = input_dict[field]
                # torch transformers expect float masks with
                # 0 for values and -inf for masked values
                # For huggingface we need to convert to
                # True for values and False for masked
                if self.config["text_model"] != "custom":
                    attention_mask = (
                        ~key_padding_mask[field].bool()
                        if key_padding_mask is not None
                        else None
                    )
                else:
                    attention_mask = (
                        key_padding_mask[field]
                        if key_padding_mask is not None
                        else None
                    )
                target = self.text_model._shift_right(target)
                if attention_mask is not None:
                    attention_mask = self.text_model._shift_right(attention_mask)
                pred = self.text_model.decoder(
                    input_ids=target,
                    encoder_hidden_states=entity_embeddings[:, [idx]],
                    attention_mask=attention_mask,
                )
            elif field in self.config["fields"]["numerical"]:
                pred = self.decoder_dict[field](entity_embeddings[:, idx])
            elif field in self.config["fields"]["categorical"]:
                pred = self.decoder_dict[field](entity_embeddings[:, idx])
            else:
                raise ValueError(f"Unknown field {field}. Check config.fields")

            preds[field] = pred
            if preds[field].isnan().any():
                raise ValueError(f"Pred {field} is NaN. It's Debugging time...")
        return TensorDict(preds, fields=self.config["fields"])

    def get_predictions(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[Union[TensorDict, torch.Tensor]] = None,
    ) -> TensorDict:
        # 1. HIERARCHY: generate hierarchy encodings for each proprty
        hierarchy_encodings = self.hierarchy_encoder.get_all_paths()
        # 2. ENCODE: encode each field
        # TODO pass hierarchy encodings to encoder
        codes = self.encode_properties(input_dict, key_padding_mask)

        # 3. MASK: Apply property_mask to codes
        codes, property_mask = self._merge_masks(codes, key_padding_mask, property_mask)

        # 4. ATTEND: Self-attention over all fields
        codes += hierarchy_encodings  # [batch_size, num_fields, d_model]
        # testing attend to all
        out = self.entity_encoder(codes) #, attention_mask=property_mask)
        # 5. DECODE: each field
        preds = self.decode_properties(out, input_dict, key_padding_mask)

        return TensorDict(preds, fields=self.config["fields"])

    # TODO move this to Trainer or Diffusion Class
    def get_metrics(
        self,
        pred_dict: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask: torch.Tensor,
        compute_err: bool = True,
    ) -> Tuple:
        loss, loss_m, loss_u = {}, {}, {}
        # if all elements in a batch are masked the loss will be an empty tensor
        # we can also get a nan?
        for idx, field in enumerate(self.config["fields"].all_fields):
            target = tgt_token_dict[field]
            pred = pred_dict[field]
            p_mask = property_mask[:, idx]
            if field in self.config["fields"]["numerical"]:
                mask = target != self.config["numerical_pad_token_id"]
                pred = pred.squeeze(-1)[mask]  # (bs, 1) -> (bs, )
                target = target[mask]  # (bs, )
                p_mask = p_mask[mask]
                l_ = (pred - target).pow(2)
                # might be empty if batch only has pad_tokens
                lm, lu, l = reduce_by_mask(l_, p_mask)
                # reduce would return empties in this case
                loss_m[field], loss_u[field], loss[field] = (
                    lm.sqrt(),
                    lu.sqrt(),
                    l.sqrt(),
                )
            elif field in self.config["fields"]["categorical"]:
                target = tgt_token_dict[field + "_idx"]
                l_ = self.ce_loss(pred, target)
                loss_m[field], loss_u[field], loss[field] = reduce_by_mask(l_, p_mask)
            elif field in self.config["fields"]["text"]:
                target = target.view(-1)
                pred = pred.view(target.shape[0], -1)
                l_ = self.ce_loss(pred, target)
                loss_m[field], loss_u[field], loss[field] = reduce_by_mask(l_, p_mask)
                if loss[field].isnan().any():
                    raise ValueError(f"Loss {field} is NaN")

        if compute_err:
            # Compute accuracy
            for field in self.config["fields"]["categorical"]:
                tgt_token_dict[field] = tgt_token_dict[field + "_idx"]
                # TODO only use tensors here
            if not isinstance(property_mask, dict):
                # property_mask = property_mask.bool() # does this waste mem
                mask_dict = TensorDict(
                    {
                        field: property_mask[:, idx].bool()
                        for (idx, field) in enumerate(self.config["fields"].all_fields)
                    }
                )
            else:
                mask_dict = property_mask.bool()

            errors_m = {
                k: 1 - v
                for k, v in self.accuracy(pred_dict, tgt_token_dict, mask_dict).items()
            }
            errors_u = {
                k: 1 - v
                for k, v in self.accuracy(pred_dict, tgt_token_dict, ~mask_dict).items()
            }
        else:
            errors_m = errors_u = {}
        return loss, loss_m, loss_u, errors_m, errors_u

    def apply(
        self,
        tgt_dict: TensorDict,
        key_padding_mask: TensorDict,
        eval_mode: bool = False,
    ) -> ModelOutputs:
        property_mask = self._sample_property_mask(
            tgt_dict,
            self.config["mask_rate"][int(eval_mode)],  # select masking rate
            seed=self.config["seed"] if eval_mode else None,  # fix seed for test set
        )
        pred_dict = self(tgt_dict, key_padding_mask, property_mask)
        losses, loss_m, loss_u, errors_m, errors_u = self.get_metrics(
            pred_dict, tgt_dict, property_mask, compute_err=eval_mode
        )

        losses["mean"] = (loss := mean(losses))
        return ModelOutputs(
            preds=pred_dict,
            targets=tgt_dict,
            loss=loss,
            loss_dict=losses,
            masked_loss_dict=loss_m,
            unmasked_loss_dict=loss_u,
            masked_error_dict=errors_m,
            unmasked_error_dict=errors_u,
        )

    def forward(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
    ) -> TensorDict:
        if not self.training:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # torch 2 has buggy warning in transformer decoder
                preds = self.get_predictions(input_dict, key_padding_mask, property_mask)
        else:
            preds = self.get_predictions(input_dict, key_padding_mask, property_mask)
        return preds

    def _merge_masks(
        self, codes: torch.Tensor, attention_mask=None, property_mask=None
    ):
        """Utility function to cleanup the property_mask
        if the property mask is a dictionary converts it to a tensor
        if the property mask is None returns a tensor of zeros to attend
        to everything. If the attention mask (which is the per token padding mask)
        is not None makes sure to ignore the missing properties in the data.

        Args:
            codes (torch.Tensor): [batch_size, num_fields, d_model] tensor of
                encoded properties.
            attention_mask (Any, optional): Token-level mask for padding. Defaults to None.
            property_mask (Any, optional): Mask for the fields (usually random)
                used in masked modeling training. Defaults to None.

        Raises:
            ValueError: If the property_mask is a dictionary and does not contain
                a mask for all fields.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The masked codes and the property_mask
        """
        # convert to tensor
        if isinstance(property_mask, dict):
            fields = self.config["fields"].all_fields
            if not all(field in property_mask for field in fields):
                raise ValueError("property_mask must contain a mask for all fields")
            property_mask = torch.stack(
                [property_mask[field] for field in fields],
                dim=1,
            )
        elif property_mask is None:
            property_mask = torch.zeros_like(codes[:, :, 0], dtype=torch.float32)

        # do not attend to missing fields
        # assumes float masks with 0 for unmask and -inf for masked values
        if attention_mask is not None:
            for idx, field in enumerate(self.config["fields"].all_fields):
                field_type = self.config["fields"].type(field)
                property_mask[:, idx] += is_missing(attention_mask[field], field_type)

        # apply mask embeddings to masked fields in the codes
        if self.config["tie_mask_embeddings"]:
            codes[property_mask.bool()] = self.mask_embedding(
                torch.tensor(0, device=codes.device)
            )
        else:
            codes[property_mask.bool()] = self.mask_embedding(
                torch.nonzero(property_mask)[:, 1]
            )
        return codes, property_mask

    def _sample_property_mask(
        self, attention_mask: TensorDict, mask_rate: float, seed: Optional[int] = None
    ):
        batch_size = len(list(attention_mask.values())[0])
        property_mask = random_mask(
            batch_size,
            len(self.config["fields"].all_fields),
            mask_rate=mask_rate,
            device=next(self.parameters()).device,
            seed=seed,
        )
        return property_mask

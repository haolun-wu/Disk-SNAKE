import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from ..utils import (
    TensorDict,
    random_mask,
    reduce_by_mask,
    mean,
    is_missing,
    ModelOutputs,
    GMMLoss,
    Fields,
)
from ..utils.metrics import Accuracy
from .transformer import TransformerEncoder
from .modules import (
    TextModule,
    DecoderModule,
    MOEDecoderModule,
    CategoricalEncoder,
    NumericalEncoder,
    ModuleDict,
)
from ..data.datasets import CATEGORICAL_PAD_TOKEN_ID, NUMERICAL_PAD_TOKEN_ID
from .positional_encodings import RNNPathing
from .embeddings import NumericEmbedding
import warnings


class KBFormer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        # The code here standardizes with tabddpm
        config["fields"] = Fields(
            numerical=[f"num_{i}" for i in range(config["num_numerical_features"])],
            categorical=[
                f"cat_{i}" for i in range(len(config["num_category_classes"]))
            ],
            text=[],
        )
        config["categorical_num_classes"] = {
            f"cat_{i}": config["num_category_classes"][i]
            for i in range(len(config["num_category_classes"]))
        }
        config["categorical_pad_token_id"] = CATEGORICAL_PAD_TOKEN_ID
        config["numerical_pad_token_id"] = NUMERICAL_PAD_TOKEN_ID

        if config["num_classes"] > 0 and config["is_y_cond"]:
            self.label_emb = nn.Embedding(config["num_classes"], config["d_model"])
        elif config["num_classes"] == 0 and config["is_y_cond"]:
            self.label_emb = nn.Linear(1, config["d_model"])

        #  Initialize encoders
        self.config = config
        d_model = config["d_model"]
        num_layers = config["num_layers"]
        nhead = config["nhead"]
        dropout = config["dropout"]

        numerical_embedding = None
        self.encoder_dict = ModuleDict()
        for field in config["fields"]["numerical"]:
            if config["tie_numerical_embeddings"]:
                if numerical_embedding is None:
                    numerical_embedding = NumericEmbedding(config)
                self.encoder_dict[field] = NumericalEncoder(config, numerical_embedding)
            else:
                self.encoder_dict[field] = NumericalEncoder(config)

        for field in config["fields"]["categorical"]:
            self.encoder_dict[field] = CategoricalEncoder(
                config, vocab_size=config["categorical_num_classes"][field]
            )

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
                    num_decoder := DecoderModule(
                        config, config["num_decoder_mixtures"], numerical=True
                    )
                )

        if config["num_categorical_decoder_experts"] > 0:
            max_num_classes = max(config["categorical_num_classes"].values())
            moe = MOEDecoderModule(config, max_num_classes)
            for field in config["fields"]["categorical"]:
                self.decoder_dict[field] = moe
        elif config["num_categorical_decoder_experts"] == 0:
            # one expert per field
            for field in config["fields"]["categorical"]:
                num_classes = config["categorical_num_classes"][field]
                self.decoder_dict[field] = DecoderModule(config, num_classes)
        else:
            raise ValueError(
                "num_categorical_decoder_experts must be >= 0. Got {}".format(
                    config["num_categorical_decoder_experts"]
                )
            )

        # metrics
        self.ce_loss = nn.CrossEntropyLoss(
            # ignore_index=config["categorical_pad_token_id"],
            reduction="none"
        )
        self.gmm_loss = GMMLoss()

        self.accuracy = Accuracy(
            fields=config["fields"],
            ignore_idx_cat=config["categorical_pad_token_id"],
            ignore_idx_num=config["numerical_pad_token_id"],
        )

    def encode_properties(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """
        Returns a tensor containing the encoded representations of the input data.
        The encoded representations are meant to be projections of the input embeddings
        into a common space where the entity-encoder can operate.

        Args:
            input_dict (dict): A dictionary containing the tensors of the input data.
            key_padding_mask (dict): A dictionary containing the key padding masks for the input
            data. This is used to mask out padding tokens in text fields for instance.

        Returns:
            dict: A dictionary containing the encoded representations of the input data.
        """
        all_fields = self.config["fields"].all_fields
        codes = torch.empty(
            (
                input_dict.size(),
                len(all_fields),
                self.config["d_model"],
            ),
            device=input_dict.device(),
        )
        for idx, field in enumerate(all_fields):
            codes[:, idx] = self.encoder_dict[field](input_dict[field])

        return codes

    def generate_text_logits(self, condition, target, key_padding_mask):
        # torch transformers expect float masks with
        # 0 for values and -inf for masked values
        # For huggingface we need to convert to
        # True for values and False for masked
        if self.config["text_model"] != "custom":
            attention_mask = (
                ~key_padding_mask.bool() if key_padding_mask is not None else None
            )
        else:
            attention_mask = key_padding_mask if key_padding_mask is not None else None
        target = self.text_model._shift_right(target)
        if attention_mask is not None:
            attention_mask = self.text_model._shift_right(attention_mask)
        pred = self.text_model.decoder(
            input_ids=target,
            encoder_hidden_states=condition,
            attention_mask=attention_mask,
        )
        return pred

    def generate_text_autoregressive(self, condition, temp=0.0, max_len=20):
        # make initial input
        batch_size = condition.shape[0]
        current_input = torch.full(
            (batch_size, 1),
            self.config["categorical_pad_token_id"],
            device=condition.device,
        )
        for i in range(max_len):
            pred = self.text_model.decoder(
                input_ids=current_input,
                encoder_hidden_states=condition,
            )
            pred = pred[:, -1, :]
            if temp == 0:
                current_input = torch.cat(
                    (current_input, pred.argmax(-1, keepdim=True)), -1
                )
            else:
                probs = torch.softmax(pred / temp, dim=-1)
                current_input = torch.cat(
                    (current_input, torch.multinomial(probs, 1)), -1
                )

            # if all have EOS, stop
            # if (current_input == self.text_model.tokenizer.eos_token_id).all():
            #     break

        return current_input[:, 1:]

    def _get_probabilistic_params_from_encodings(
        self,
        entity_embeddings: torch.Tensor,
        hierarchy_embeddings: torch.Tensor = None,
    ) -> TensorDict:
        """
        receive the parameters needed to sample from
        numerical: mu, sigma, weights for GMM
        categorical: logits
        text: simply copy the entity embeddings. logits are not enough, because autoregressive generation has to condition on the entity.
        """
        prob_params = {}
        for idx, field in enumerate(self.config["fields"].all_fields):
            if field in self.config["fields"]["text"]:
                params = entity_embeddings[:, [idx]]
            elif field in self.config["fields"]["numerical"]:
                params = self.decoder_dict[field](entity_embeddings[:, idx])
            elif field in self.config["fields"]["categorical"]:
                if self.config["condition_decoders_on_hierarchy"]:
                    decoder_inputs = (
                        entity_embeddings[:, idx],
                        hierarchy_embeddings[:, idx],
                    )
                else:
                    decoder_inputs = (entity_embeddings[:, idx],)
                params = self.decoder_dict[field](*decoder_inputs)
            else:
                raise ValueError(f"Unknown field {field}. Check config.fields")

            prob_params[field] = params
        return TensorDict(prob_params, fields=self.config["fields"])

    def get_all_encodings(
        self,
        input_dict: TensorDict,
        property_mask: Optional[TensorDict] = None,
        target_conditioning: Optional[TensorDict] = None,
    ):
        # 1. HIERARCHY: generate hierarchy encodings for each proprty
        hierarchy_encodings = self.hierarchy_encoder.get_all_paths()
        # shape is [num_fields, d_model]

        # 2. ENCODE: encode each field
        # TODO pass hierarchy encodings to encoder
        codes = self.encode_properties(input_dict)

        # 3. MASK: Apply property_mask to codes
        codes, property_mask = self._merge_and_apply_masks(
            codes, None, property_mask, inplace=True
        )

        # 4. ATTEND: Self-attention over all fields
        codes += hierarchy_encodings  # [batch_size, num_fields, d_model]
        if self.config["is_y_cond"] and target_conditioning is not None:
            if self.config["num_classes"] > 0:
                target_conditioning = target_conditioning.squeeze()
            else:
                target_conditioning = target_conditioning.resize(target_conditioning.size(0), 1).float()
            emb = nn.functional.silu(self.label_emb(target_conditioning))  # batch_size, d_model
            emb = emb.unsqueeze(1).expand(
                -1, len(input_dict), -1
            )  # batch_size, num_fields, d_model
            codes += emb
        return (
            self.entity_encoder(codes, attention_mask=property_mask),
            hierarchy_encodings,
        )

    def sample_with_temp(
        self,
        prob_params,
        temp=0.0,
        teacher_forcing=True,
    ):
        new_samples = TensorDict(fields=self.config["fields"])
        for field in self.config["fields"].all_fields:
            new_samples[field] = self._sample_field_with_temp(
                prob_params[field], temp, field, teacher_forcing=teacher_forcing
            )
        return new_samples

    def _sample_field_with_temp(
        self,
        prob_params,
        temp,
        field,
        target=None,
        key_padding_mask=None,
        teacher_forcing=True,
    ):
        """Sample from a batch of prob_params with temperature."""
        if field in self.config["fields"]["numerical"]:
            return GMMLoss.sample(prob_params, 1, temp)
        elif field in self.config["fields"]["categorical"]:
            if temp == 0:
                return prob_params.argmax(-1)
            else:
                proba = torch.softmax(prob_params / temp, dim=-1)
                return torch.multinomial(proba, 1).view(-1)
        elif field in self.config["fields"]["text"]:
            if not teacher_forcing:
                return self.generate_text_autoregressive(prob_params, temp)
            else:
                assert target is not None
                logits = self.generate_text_logits(
                    prob_params, target, key_padding_mask
                )
                if temp == 0:
                    return logits.argmax(-1)
                else:
                    proba = torch.softmax(logits / temp, dim=-1)
                    shape = proba.shape
                    proba = proba.view(-1, proba.shape[-1])
                    decisions = torch.multinomial(proba, 1)
                    return decisions.view(shape[:-1])

    def get_probabilistic_params(
        self,
        input_dict: TensorDict,
        property_mask: Optional[TensorDict] = None,
        target_conditioning: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        out, hierarchy_encodings = self.get_all_encodings(
            input_dict, property_mask, target_conditioning
        )
        params = self._get_probabilistic_params_from_encodings(out, hierarchy_encodings)
        return TensorDict(params, fields=input_dict.fields)

    def _get_samples(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        target_conditioning: Optional[torch.Tensor] = None,
        temperature: Optional[float] = 0.0,
        teacher_forcing: bool = True,
    ) -> TensorDict:
        params = self.get_probabilistic_params(
            input_dict, property_mask, target_conditioning
        )

        # now get actual predicted samples
        return self.sample_with_temp(
            params,
            temp=temperature,
            teacher_forcing=teacher_forcing,
        )

    # TODO move this to Trainer or Diffusion Class
    def get_loss_from_prob_params(
        self,
        prob_params: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask: torch.Tensor,
    ) -> Tuple:
        loss = {}
        # if all elements in a batch are masked the loss will be an empty tensor
        # we can also get a nan?
        for idx, field in enumerate(self.config["fields"].all_fields):
            target = tgt_token_dict[field]
            prob_param = prob_params[field]
            p_mask = property_mask[:, idx]
            if field in self.config["fields"]["numerical"]:
                token_mask = target != self.config["numerical_pad_token_id"]
                target = target.view(-1)  # Should already be flat
                # prob_param should be shape [batch_size, num_mixtures * 3]
                l_ = self.gmm_loss(prob_param, target)
                loss[field] = reduce_by_mask(l_, p_mask, token_mask)
            elif field in self.config["fields"]["categorical"]:
                target = tgt_token_dict[field]
                token_mask = target != -999.0 # self.config["categorical_pad_token_id"]
                l_ = self.ce_loss(prob_param, target)
                loss[field] = reduce_by_mask(l_, p_mask, token_mask)
        return loss

    def get_metrics_from_prob_params(
        self,
        prob_params: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask: torch.Tensor,
        unscale: bool = False,
        dataset=None,
    ):
        if unscale:
            assert (
                dataset is not None
            ), "Give the dataset to unscale (it has the required info)"

        pred_dict = TensorDict(fields=prob_params.fields)
        for field in prob_params:
            pred_dict[field] = self._sample_field_with_temp(
                prob_params[field],
                temp=0,
                field=field,
                target=tgt_token_dict[field],
                teacher_forcing=True,
            )
        # Compute accuracy
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

        errors = {
            k: v
            for k, v in self.accuracy(
                pred_dict, tgt_token_dict, mask_dict, unscale, dataset
            ).items()
        }

        return errors

    def to_tensor_dict(self, tensor):
        if isinstance(tensor, TensorDict):
            # TODO nan_to_num, this exists only for backward compatibility
            # rename all keys
            tensor_dict = TensorDict(
                {
                    self.config["fields"].all_fields[idx]: tensor[field]
                    for idx, field in enumerate(tensor.fields.all_fields)
                },
                fields=self.config["fields"],
            ) 
            return tensor_dict
        tensor_dict = TensorDict(fields=self.config["fields"])
        for idx, field in enumerate(self.config["fields"].all_fields):
            if field in self.config["fields"]["numerical"]:
                # replace nan with NUMERICAL_PAD_TOKEN_ID
                tensor_dict[field] = tensor[:, idx].nan_to_num(
                    self.config["numerical_pad_token_id"]
                )
            else:
                tensor_dict[field] = tensor[:, idx].long()
        return tensor_dict

    def to_tensor(self, tensor_dict, device=None):
        tensor = torch.empty(
            (tensor_dict.size(), len(self.config["fields"].all_fields)),
            device=device or tensor_dict.device(),
        )
        for idx, field in enumerate(self.config["fields"].all_fields):
            tensor[:, idx] = tensor_dict[field]
        return tensor

    def forward(
        self,
        tgt_dict: torch.Tensor,
        tgt_cond: torch.Tensor=None,  # for conditional generation
        eval_mode=False,
    ):
        # convert to tensor dict from regular tensor
        tgt_dict = self.to_tensor_dict(tgt_dict)

        property_mask = self._sample_property_mask(
            tgt_dict,
            self.config["eval_mask_rate" if eval_mode else "train_mask_rate"],
            seed=self.config["seed"] if eval_mode else None,  # fix seed for test set
        )

        prob_params = self.get_probabilistic_params(
            tgt_dict,
            target_conditioning=tgt_cond,
            property_mask=property_mask,
        )
        losses = self.get_loss_from_prob_params(prob_params, tgt_dict, property_mask)
        losses["mean"] = (loss := mean(losses))
        return ModelOutputs(
            preds=prob_params,
            targets=tgt_dict,
            property_mask=property_mask,
            loss=loss,
            loss_dict=losses,
            error_dict=None,
        )

    # def apply(
    #     self,
    #     tgt_dict: TensorDict,
    #     key_padding_mask: TensorDict,
    #     eval_mode: bool = False,
    #     unscale=False,
    #     dataset=None,
    # ) -> ModelOutputs:
    #     property_mask = self._sample_property_mask(
    #         tgt_dict,
    #         self.config[
    #             "eval_mask_rate" if eval_mode else "train_mask_rate"
    #         ],  # select masking rate
    #         seed=self.config["seed"] if eval_mode else None,  # fix seed for test set
    #     )
    #     output = self.forward(tgt_dict, key_padding_mask, property_mask)

    #     if eval_mode:
    #         output.errors = self.get_metrics_from_prob_params(
    #             output.preds,
    #             tgt_dict,
    #             property_mask,
    #             unscale=unscale,
    #             dataset=dataset,
    #         )

    #     return output

    def sample(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        target_conditioning: Optional[torch.Tensor] = None,
        temperature: Optional[float] = 0.0,
        teacher_forcing: bool = True,
    ) -> TensorDict:
        if not self.training:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # torch 2 has buggy warning in transformer decoder
                preds = self._get_samples(
                    input_dict,
                    key_padding_mask,
                    property_mask,
                    target_conditioning,
                    temperature,
                    teacher_forcing,
                )
        else:
            preds = self._get_samples(
                input_dict,
                key_padding_mask,
                property_mask,
                target_conditioning,
                temperature,
                teacher_forcing,
            )
        return preds

    def _merge_and_apply_masks(
        self, codes: torch.Tensor, padding_mask=None, property_mask=None, inplace=False
    ):
        """Utility function to cleanup the property_mask
        if the property mask is a dictionary converts it to a tensor
        if the property mask is None returns a tensor of zeros to attend
        to everything. If the attention mask (which is the per token padding mask)
        is not None makes sure to ignore the missing properties in the data.

        Args:
            codes (torch.Tensor): [batch_size, num_fields, d_model] tensor of
                encoded properties.
            padding_mask (Any, optional): Token-level mask for padding. Defaults to None.
            property_mask (Any, optional): Mask for the fields (usually random)
                used in masked modeling training. Defaults to None.

        Raises:
            ValueError: If the property_mask is a dictionary and does not contain
                a mask for all fields.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The masked codes and the property_mask
        """
        if not inplace:
            codes = codes.clone()
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
            property_mask = torch.zeros_like(
                codes[:, :, 0], dtype=torch.get_default_dtype()
            )
        else:
            property_mask = property_mask.clone()

        # do not attend to missing fields
        # assumes float masks with 0 for unmask and -inf for masked values
        if padding_mask is not None:
            for idx, field in enumerate(self.config["fields"].all_fields):
                field_type = self.config["fields"].type(field)
                property_mask[:, idx] += is_missing(padding_mask[field], field_type)

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
        self, original_tensors: TensorDict, mask_rate: float, seed: Optional[int] = None
    ):
        batch_size = len(list(original_tensors.values())[0])
        property_mask = random_mask(
            batch_size,
            len(self.config["fields"].all_fields),
            mask_rate=mask_rate,
            device=next(self.parameters()).device,
            seed=seed,
        )
        for field in self.config["never_mask"]:
            try:
                idx = self.config["fields"].all_fields.index(field)
                property_mask[:, idx] = 0
            except ValueError:
                pass
        return property_mask

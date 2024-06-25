import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


from typing import Optional, Tuple, Union

from ..utils import (
    TensorDict,
    random_mask,
    reduce_by_mask,
    mean,
    is_missing,
    ModelOutputs,
    GMMLoss,
    compute_min_ce_loss,
)
from ..utils.metrics import Accuracy, Recall
from .transformer import TransformerEncoder
from .modules import (
    TextModule,
    DecoderModule,
    MOEDecoderModule,
    CategoricalEncoder,
    NumericalEncoder,
    ModuleDict,
)
from .positional_encodings import RNNPathing
from .embeddings import NumericEmbedding
import warnings


class KBFormer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        #  Initialize encoders
        num_layers = config["num_layers"]
        nhead = config["nhead"]
        dropout = config["dropout"]
        if config["fields"]["text"]:
            self.text_model = TextModule(config)

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

        # training, how to choose GT
        self.gt_choose = config["gt_choose"]
        
        # metrics
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=config["categorical_pad_token_id"], reduction="none"
        )
        self.gmm_loss = GMMLoss()

        self.accuracy = Accuracy(
            fields=config["fields"],
            ignore_idx_cat=config["categorical_pad_token_id"],
            ignore_idx_num=config["numerical_pad_token_id"],
        )

        self.recall = Recall(
            fields=config["fields"],
            ignore_idx_cat=config["categorical_pad_token_id"],
            ignore_idx_num=config["numerical_pad_token_id"],
        )

    def encode_properties(
        self,
        input_dict: TensorDict,
        num_mask: int,
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
        fields_num = (
            len(all_fields) - num_mask + 1
        )  # each time, we decode one mask. Equal: num_unmask+1
        codes = torch.empty(
            (
                input_dict.size(),
                fields_num,
                self.config["d_model"],
            ),
            device=input_dict.device(),
        )
        for idx, field in enumerate(all_fields[:-num_mask]):  # Only encode unmasked
            if (
                field
                in self.config["fields"]["numerical"]
                + self.config["fields"]["categorical"]
            ):
                codes[:, idx] = self.encoder_dict[field](input_dict[field])
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
                output = self.text_model.encoder(input_dict[field], padding_mask=kpm)
                if self.config["text_model"] == "custom":
                    codes[:, idx] = output[:, 0]
                else:
                    codes[:, idx] = output.last_hidden_state[:, 0]

        return codes  # shape: (#batch_size, #unmask+1, #d_model)

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
            input_ids=target,  # (batch_size, seq_len)
            encoder_hidden_states=condition,  # (batch_size, 1, d_model) for each field
            attention_mask=attention_mask,
        )
        return pred

    def _get_probabilistic_params_from_encodings(
        self,
        entity_embeddings: torch.Tensor,
        num_mask: int,
        hierarchy_embeddings: torch.Tensor = None,
    ) -> TensorDict:
        """
        receive the parameters needed to sample from
        numerical: mu, sigma, weights for GMM
        categorical: logits
        text: simply copy the entity embeddings. logits are not enough, because autoregressive generation has to condition on the entity.
        """
        all_fields = self.config["fields"].all_fields
        fields_num = (
            len(all_fields) - num_mask + 1
        )  # each time, we decode one mask. Equal: num_unmask+1
        prob_params = {}
        for idx, field in enumerate(self.config["fields"].all_fields[:fields_num]):
            if field in self.config["fields"]["text"]:
                params = entity_embeddings[:, [idx]]  # (batch_size, 1, d_model)
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
            if prob_params[field].isnan().any():
                raise ValueError(f"Pred {field} is NaN. It's Debugging time...")

        return TensorDict(
            prob_params, fields=self.config["fields"]
        )  # no worry about this, surplus fields will be ignored automatically. The length is still num_unmask+1

    def get_all_encodings(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        num_mask: Optional[int] = 1,
        use_path_emb: bool = True,
    ):
        # 1. HIERARCHY: generate hierarchy encodings for each proprty
        all_fields = self.config["fields"].all_fields
        fields_num = (
            len(all_fields) - num_mask + 1
        )  # each time, we decode one mask. Equal: num_unmask+1
        hierarchy_encodings = self.hierarchy_encoder.get_all_paths()[:fields_num]
        # shape is [num_unmask+1, d_model]

        # 2. ENCODE: encode each field
        # TODO pass hierarchy encodings to encoder
        codes = self.encode_properties(input_dict, num_mask, key_padding_mask)

        # 3. MASK: Apply property_mask to codes
        codes, property_mask = self._merge_and_apply_masks(
            codes, key_padding_mask, property_mask, num_mask, inplace=True
        )

        # 4. ATTEND: Self-attention over all fields
        if use_path_emb:
            codes += hierarchy_encodings  # [batch_size, num_unmask+1, d_model]
        else:
            pass

        # testing attend to all
        return (
            self.entity_encoder(
                codes, attention_mask=property_mask
            ),  # property_mask: [batch_size, num_unmask+1]
            hierarchy_encodings,
        )

    def get_probabilistic_params(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        num_mask: Optional[int] = 1,
        use_path_emb: bool = True,
    ) -> TensorDict:
        """
        # Return the sampled tensor for each field
        # If the codes is of shape (batch_size, num_unmask+1, d_model)
        # Then the retuned TensorDict has (num_unmask+1) fields k:v, and each v is of shape (batch_size, 1, d_model)
        """
        out, hierarchy_encodings = self.get_all_encodings(
            input_dict, key_padding_mask, property_mask, num_mask, use_path_emb
        )  # out: (batch_size, num_unmask+1, d_model)
        params = self._get_probabilistic_params_from_encodings(
            out, num_mask, hierarchy_encodings
        )  # length is num_unmask+1
        return TensorDict(params, fields=input_dict.fields)

    def get_loss_from_prob_params(
        self,
        prob_params: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask: torch.Tensor,
        num_mask: int,
    ) -> Tuple:
        # Initialize lists to collect predictions and targets
        batch_size = property_mask.shape[0]
        all_fields = self.config["fields"].all_fields
        fields_num = (
            len(all_fields) - num_mask + 1
        )  # each time, we decode one mask. Equal: num_unmask+1

        # The last field that is masked
        field_masked = all_fields[fields_num - 1]
        prob_param = prob_params[field_masked]

        if self.gt_choose == 'random':
            """Option 1: Randomly select one as ground truth"""
            # Get target and probability parameter for the last field
            target = tgt_token_dict[field_masked]  # Already random, due to permutation during preprocess

            if field_masked in self.config["fields"]["text"]:
                # because prob params for text are not logits, we have to get those first
                token_mask = target != self.config["categorical_pad_token_id"]
                key_padding_mask = (~token_mask).float()
                key_padding_mask[key_padding_mask == 1] = float(
                    "-inf"
                )  # 0 for valid tokens, -inf for masked tokens
                sample = self.generate_text_logits(prob_param, target, key_padding_mask)

                # prob_param: (batch_size, 1, d_model)
                # target: (batch_size, seq_len)
                # sample: (batch_size, seq_len, vocab)
                # p_mask: (batch_size)
                # key_padding_mask: (batch_size, seq_len)
                target = target.view(-1)  # batch * seq
                sample = sample.reshape(target.shape[0], -1)
                loss = self.ce_loss(sample, target)  # The full field is masked
                loss = loss.mean()
        elif self.gt_choose == 'decide':
            """Option 2: choose the highest possible ground truth"""
            # Get the last several masked columns
            masked_fields = all_fields[-num_mask:]

            # Initialize a list to store loss for each possible ground truth
            loss_matrix = []

            for field in masked_fields:
                target = tgt_token_dict[field]
                if field in self.config["fields"]["text"]:
                    # because prob params for text are not logits, we have to get those first
                    token_mask = target != self.config["categorical_pad_token_id"]
                    key_padding_mask = (~token_mask).float()
                    key_padding_mask[key_padding_mask == 1] = float(
                        "-inf"
                    )  # 0 for valid tokens, -inf for masked tokens
                    sample = self.generate_text_logits(prob_param, target, key_padding_mask)

                    # prob_param: (batch_size, 1, d_model)
                    # target: (batch_size, seq_len)
                    # sample: (batch_size, seq_len, vocab)
                    # key_padding_mask: (batch_size, seq_len)
                    target = target.view(-1)  # batch * seq
                    sample = sample.reshape(target.shape[0], -1) # batch * seq, vocab
                    l_ = self.ce_loss(sample, target)  # The full field is masked
                    l_ = l_.view(batch_size, -1)
                    loss_matrix.append(l_.mean(dim=1, keepdim=True))

            # Stack losses to form a matrix of shape (batch_size, num_mask)
            loss_matrix = torch.cat(loss_matrix, dim=1)
            min_loss, _ = loss_matrix.min(dim=1)  # Select the minimum loss for each sample
            loss = min_loss.mean()

        return loss

    def forward(
        self,
        tgt_dict: TensorDict,
        key_padding_mask: TensorDict,
        property_mask: TensorDict,
        num_mask: int,
        use_path_emb: bool = True,
    ):
        prob_params = self.get_probabilistic_params(
            tgt_dict,
            key_padding_mask,
            property_mask,
            num_mask,
            use_path_emb,
        )
        # losses = self.get_loss_from_prob_params(prob_params, tgt_dict, property_mask)
        # losses["mean"] = (loss := mean(losses))
        loss = self.get_loss_from_prob_params(
            prob_params, tgt_dict, property_mask, num_mask
        )
        return (
            ModelOutputs(
                preds=prob_params,
                targets=tgt_dict,
                property_mask=property_mask,
                loss=loss,
                loss_dict=None,
                error_dict=None,
            ),
            prob_params,
        )

    def apply(
        self,
        tgt_dict: TensorDict,
        key_padding_mask: TensorDict,
        eval_mode: bool = False,
        unscale=False,
        dataset=None,
        use_path_emb: bool = True,
    ) -> ModelOutputs:
        property_mask, num_mask = self._sample_property_mask(
            tgt_dict,
            self.config[
                "eval_mask_rate" if eval_mode else "train_mask_rate"
            ],  # select masking rate
            seed=self.config["seed"] if eval_mode else None,  # fix seed for test set
        )
        output, prob_params = self.forward(
            tgt_dict, key_padding_mask, property_mask, num_mask, use_path_emb
        )

        if eval_mode:
            output.error_dict = self.get_metrics_from_prob_params_iterative_update(
                prob_params,
                tgt_dict,
                key_padding_mask,
                property_mask,
                num_mask,
                unscale=unscale,
                dataset=dataset,
            )

        return output

    def _merge_and_apply_masks(
        self,
        codes: torch.Tensor,
        padding_mask=None,
        property_mask=None,
        num_mask=1,
        inplace=False,
    ):
        """Utility function to cleanup the property_mask
        if the property mask is a dictionary converts it to a tensor
        if the property mask is None returns a tensor of zeros to attend
        to everything. If the attention mask (which is the per token padding mask)
        is not None makes sure to ignore the missing properties in the data.

        Args:
            codes (torch.Tensor): [batch_size, num_unmask+1, d_model] tensor of
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

        all_fields = self.config["fields"].all_fields
        fields_num = (
            len(all_fields) - num_mask + 1
        )  # each time, we decode one mask. Equal: num_unmask+1
        # convert to tensor
        if isinstance(property_mask, dict):
            fields = self.config["fields"].all_fields[:fields_num]
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
            property_mask = property_mask[
                :, :fields_num
            ].clone()  # only the last column be -inf

        # do not attend to missing fields
        # assumes float masks with 0 for unmask and -inf for masked values
        if padding_mask is not None:
            for idx, field in enumerate(self.config["fields"].all_fields[:fields_num]):
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
        for field in self.config["never_mask"]:
            try:
                idx = self.config["fields"].all_fields.index(field)
                property_mask[:, idx] = 0
            except ValueError:
                pass
        return property_mask

    """OTHERS for eval"""

    def get_metrics_from_prob_params_iterative_update(
        self,
        prob_params: TensorDict,
        tgt_token_dict: TensorDict,
        key_padding_mask: TensorDict,
        property_mask: torch.Tensor,
        num_mask: int,
        unscale: bool = False,
        dataset=None,
    ):
        # new_tgt_token_dict = deepcopy(tgt_token_dict)
        # pred_dict = deepcopy(tgt_token_dict)

        new_key_padding_mask = deepcopy(key_padding_mask)
        new_property_mask = deepcopy(property_mask)

        if unscale:
            assert (
                dataset is not None
            ), "Give the dataset to unscale (it has the required info)"

        # Focus on those masked fields
        all_fields = self.config["fields"].all_fields
        list_masked_fields = list(all_fields)[-num_mask:]

        assert list(prob_params.keys())[-1] == list_masked_fields[0]

        # Initialize pred_dict on those unmasked fields
        pred_dict = TensorDict(fields=prob_params.fields)
        # last_field_value = list(prob_params.values())[-1] if prob_params else None
        
        # Use the original value for those unmasked fields
        for field in all_fields[:-num_mask]:
            pred_dict[field] = deepcopy(tgt_token_dict[field])

        for field in list_masked_fields:
            pred_cur_field = self._sample_field_with_temp(
                prob_params[field],
                temp=0,
                field=field,
                # target=pred_dict[field],
                teacher_forcing=False,
            )
            # The entire masked column (field) is replaced by the prediction
            pred_dict[field] = pred_cur_field

            # update padding mask and property mask
            new_property_mask[:, -num_mask] = 0  # unmask the last field
            num_mask -= 1  # number of masked fields reduce 1
            if num_mask == 0:
                break
            pad_mask_cur_field = torch.zeros_like(pred_dict[field], dtype=torch.float)
            pad_mask_cur_field[pred_cur_field == 0] = float("-inf")
            new_key_padding_mask[field] = pad_mask_cur_field

            # update prob_params
            prob_params = self.get_probabilistic_params(
                pred_dict,
                new_key_padding_mask,
                new_property_mask,
                num_mask,
                use_path_emb=False,
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

        # errors = {
        #     k: v
        #     for k, v in self.accuracy(
        #         pred_dict, tgt_token_dict, mask_dict, unscale, dataset
        #     ).items()
        # }
        recall_list, correct_list, num_masks_list, pred_seq_list, target_seq_list = (
            self.recall(
                pred_dict,
                tgt_token_dict,
                mask_dict,
                unscale,
                dataset,
                eos_token_id=dataset.tokenizer.eos_token_id,
            )
        )

        return recall_list, correct_list, num_masks_list, pred_seq_list, target_seq_list

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
                # teacher_forcing=True,
                teacher_forcing=False,
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

    def _sample_field_with_temp(
        self,
        prob_params,
        temp,
        field,
        target=None,
        key_padding_mask=None,
        teacher_forcing=False,
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
                assert target is not None, "Target is None with teacher forcing"
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

    def generate_text_autoregressive(self, condition, temp=0.0, max_len=7):
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

        res = current_input[:, 1:]

        return res

    """Only used for GMM"""

    def sample_with_temp(
        self,
        prob_params,
        target_dict=None,
        key_padding_mask=None,
        temp=0.0,
        teacher_forcing=True,
    ):
        new_samples = TensorDict(fields=self.config["fields"])
        for field in self.config["fields"].all_fields:
            if field in self.config["fields"]["text"]:
                new_samples[field] = self._sample_field_with_temp(
                    prob_params[field],
                    temp,
                    field,
                    target_dict[field] if target_dict is not None else None,
                    key_padding_mask[field] if key_padding_mask is not None else None,
                    teacher_forcing,
                )
            else:
                new_samples[field] = self._sample_field_with_temp(
                    prob_params[field], temp, field, teacher_forcing=teacher_forcing
                )
        return new_samples

    def _get_samples(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        temperature: Optional[float] = 0.0,
        teacher_forcing: bool = True,
        use_path_emb: bool = True,
    ) -> TensorDict:
        params = self.get_probabilistic_params(
            input_dict, key_padding_mask, property_mask, use_path_emb
        )

        # now get actual predicted samples
        return self.sample_with_temp(
            params,
            input_dict,
            key_padding_mask,
            temp=temperature,
            teacher_forcing=teacher_forcing,
        )

    def sample(
        self,
        input_dict: TensorDict,
        key_padding_mask: Optional[TensorDict] = None,
        property_mask: Optional[TensorDict] = None,
        temperature: Optional[float] = 0.0,
        teacher_forcing: bool = True,
        use_path_emb: bool = True,
    ) -> TensorDict:
        if not self.training:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # torch 2 has buggy warning in transformer decoder
                preds = self._get_samples(
                    input_dict,
                    key_padding_mask,
                    property_mask,
                    temperature,
                    teacher_forcing,
                    use_path_emb,
                )
        else:
            preds = self._get_samples(
                input_dict,
                key_padding_mask,
                property_mask,
                temperature,
                teacher_forcing,
                use_path_emb,
            )
        return preds

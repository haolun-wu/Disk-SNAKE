import torch
from torch import nn
from ..utils import TensorDict, Fields, ModelOutputs
from typing import Optional
from collections import defaultdict


class Accuracy:
    def __init__(
        self,
        fields: Fields,
        ignore_idx_cat: int = -1,
        ignore_idx_num: Optional[int] = -1000,
    ):
        self.ignore_index = ignore_idx_cat
        self.ignore_index_continuous = ignore_idx_num
        self.fields = fields

    def unscale_for_metrics(
        self, pred_dict: TensorDict, tgt_token_dict: TensorDict, dataset
    ):
        pred_dict = TensorDict(
            dataset.reverse_transform(pred_dict), fields=pred_dict.fields
        )
        tgt_dict = TensorDict(
            dataset.reverse_transform(tgt_token_dict), fields=tgt_token_dict.fields
        )
        return pred_dict, tgt_dict

    def __call__(
        self,
        pred_dict: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask=None,
        unscale=False,
        dataset=None,
    ) -> TensorDict:

        if unscale:
            assert dataset is not None
            pred_dict, tgt_token_dict = self.unscale_for_metrics(
                pred_dict, tgt_token_dict, dataset
            )
        if property_mask is None:
            return self._unmasked_call(pred_dict, tgt_token_dict)
        elif not isinstance(property_mask, dict):
            assert len(property_mask[0]) == len(self.fields.all_fields)
            property_mask = {
                field: property_mask[:, i]
                for i, field in enumerate(self.fields.all_fields)
            }

        masked_preds = TensorDict(fields=pred_dict.fields)
        masked_tgts = TensorDict(fields=pred_dict.fields)
        for field in pred_dict:
            mask = property_mask[field]
            masked_preds[field] = pred_dict[field][mask]
            masked_tgts[field] = tgt_token_dict[field][mask]
        return self._unmasked_call(masked_preds, masked_tgts)

    def _unmasked_call(
        self, pred_dict: TensorDict, tgt_token_dict: TensorDict
    ) -> TensorDict:
        accs = TensorDict(fields=pred_dict.fields)
        with torch.no_grad():
            for field in pred_dict.numerical:
                tgt = tgt_token_dict[field].float()
                # assumes scalar predictions so pred is a 1d tensor
                # New with GMMs
                accs[field] = self.compute_rms(pred_dict[field], tgt)
            for field in pred_dict.categorical:
                # pred must be [batch_size, num_classes]
                pred = pred_dict[field]
                tgt = tgt_token_dict[field].long()
                accs[field] = self.compute_acc(pred, tgt)
            for field in pred_dict.text:
                # pred must be [batch_size, seq_len, num_classes]
                pred = pred_dict[field].flatten(0, 1)
                tgt = tgt_token_dict[field].long().reshape(-1)
                accs[field] = self.compute_acc(pred, tgt)
        print("accs:", accs)
        return accs

    def compute_rms(self, pred: torch.Tensor, tgt: torch.Tensor):
        mask = tgt != self.ignore_index_continuous
        pred, tgt = pred[mask], tgt[mask]
        return ((tgt.flatten() - pred.flatten())).pow(2).mean().sqrt()


    def compute_acc(self, pred: torch.Tensor, tgt: torch.Tensor):
        mask = tgt != self.ignore_index
        pred, tgt = pred[mask], tgt[mask]
        print("mask:", mask.shape)
        print("pred:", pred.shape)
        print("tgt:", tgt.shape)
        return (pred == tgt).float().mean()


def mean(x: dict) -> "torch.Tensor":
    if len(x) == 0:
        return torch.tensor(float("nan"))
    result = torch.tensor(0.0, requires_grad=True)
    total = torch.tensor(0.0)
    for field in x:
        result = result.to(x[field].device)
        if x[field].numel() == 0:
            Warning(f"Empty tensor encountered in {field}")
        elif torch.isnan(x[field]):
            continue
        else:
            result = result + x[field]
            total += 1
    return result / total


class AggregatedMetrics:
    def __init__(self, config):
        self.loss = 0
        self.num_field_samples = defaultdict(
            int
        )  # number of samples masked for each field, depends on property mask (and padding mask)
        self.loss_dict = defaultdict(
            int
        )  # loss on items masked due to property mask
        self.error_dict = defaultdict(
            int
        )  # errors computed uising metrics.py/Accuracy on items masked due to property mask
        self.config = config

    def add_contribution(self, new_outputs: ModelOutputs):
        assert torch.is_floating_point(new_outputs.property_mask), (
            "masking here assumes -inf is masked and 0 is not."
            "there's some ambiguity with bool tensors you have to be careful"
        )
        for idx, field in enumerate(self.config["fields"].all_fields):
            field_mask = new_outputs.property_mask[:, idx].bool()  # True means masked
            tgt = new_outputs.targets[field]
            padding_mask = self._get_padding_mask(tgt, field)  # False means masked

            # need to accomodate the sequence dimension to broadcast correctly
            if (
                field in self.config["fields"]["text"]
                or field in self.config["fields"]["categorical"]
            ):
                field_mask = field_mask.view(-1, 1)

            n_masked_samples = (field_mask & padding_mask).sum()  # might be zero


            if n_masked_samples > 0:
                self.loss_dict[field] = self.weighted_mean(
                    new_outputs.loss_dict[field],
                    self.loss_dict[field],
                    n_masked_samples,
                    self.num_field_samples[field],
                )
                if new_outputs.error_dict is not None:
                    self.error_dict[field] = self.weighted_mean(
                        new_outputs.error_dict[field],
                        self.error_dict[field],
                        n_masked_samples,
                        self.num_field_samples[field],
                    )
                self.num_field_samples[field] += n_masked_samples

    def _get_padding_token(self, field):
        if field in self.config["fields"]["numerical"]:
            return self.config["numerical_pad_token_id"]
        else:
            return self.config["categorical_pad_token_id"]

    def _get_padding_mask(self, tgt, field) -> torch.Tensor:
        # False means masked unlike PyTorch transformers convention which I hate
        return tgt != self._get_padding_token(field)

    def get_output(self):
        loss_dict = self.loss_dict.copy()
        loss_dict["mean"] = mean(self.loss_dict)
        return ModelOutputs(
            preds=None,
            targets=None,
            property_mask=None,
            loss=loss_dict["mean"],
            loss_dict=loss_dict,
            error_dict=self.error_dict
        )

    def weighted_mean(self, new_val, old_val, new_weight, old_weight):
        return (new_val * new_weight + old_val * old_weight) / (new_weight + old_weight)


class GMMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params, x):
        mus, sigmas, weights = GMMLoss._prepare(params)
        return -self.loglikelihood(mus, sigmas, weights, x)

    @staticmethod
    def estimate_mode(params):
        mus, sigmas, weights = GMMLoss._prepare(params)
        return (mus * weights).sum(dim=-1, keepdim=True)

    @staticmethod
    def _prepare(params):
        mus, sigmas, weights = torch.chunk(params, 3, dim=-1)
        sigmas = torch.nn.functional.softplus(sigmas) + 1e-10
        weights = torch.softmax(weights, dim=-1)
        return mus, sigmas, weights

    @staticmethod
    def sample(params, n_samples, temperature=1.0):
        if temperature == 0.0:
            return GMMLoss.estimate_mode(params).repeat(1, n_samples)
        mus, sigmas, weights = GMMLoss._prepare(params)
        batch_size, num_components = mus.shape

        weights = torch.softmax(torch.log(weights) / temperature, dim=-1)
        k = torch.multinomial(weights, n_samples, replacement=True)
        k = k.view(batch_size, n_samples)
        mus = mus.gather(1, k)
        sigmas = sigmas.gather(1, k)
        samples = torch.normal(mus, sigmas * temperature)
        return samples

    def loglikelihood(self, mus, sigmas, weights, x):
        """
        Computes the loglikelihood of x given mus and sigmas.
        Args:
            mus: tensor of shape (batch_size, num_components)
            sigmas: tensor of shape (batch_size, num_components)
            x: tensor of shape (batch_size)
        Returns:
            loglikelihood: tensor of shape (batch_size,)
        """
        _, num_components = mus.shape
        x = x.unsqueeze(1).repeat(
            1, num_components
        )  # x is (batch_size, num_components)
        log_sqrt2pi = 0.9189385175704956
        log_probs = (
            -0.5 * ((x - mus) / sigmas) ** 2 - torch.log(sigmas) - log_sqrt2pi
        )  # normal distribution
        # include weights
        log_probs = log_probs + torch.log(weights)
        # log_probs = log_probs.sum(dim=-1)
        log_probs = torch.logsumexp(log_probs, dim=-1)
        return log_probs

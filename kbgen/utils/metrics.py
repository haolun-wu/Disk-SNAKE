import torch
from ..utils.utils import TensorDict, Fields
from typing import Optional


class Accuracy:
    def __init__(
        self,
        fields: Fields,
        ignore_idx_cat: int = -1,
        ignore_idx_num: Optional[int] = None,
    ):
        self.ignore_index = ignore_idx_cat
        self.ignore_index_continuous = ignore_idx_num
        self.fields = fields

    def __call__(
        self,
        pred_dict: TensorDict,
        tgt_token_dict: TensorDict,
        property_mask=None,
    ) -> TensorDict:
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

    def _unmasked_call(self, pred_dict, tgt_token_dict) -> TensorDict:
        accs = TensorDict(fields=pred_dict.fields)
        with torch.no_grad():
            for field in pred_dict.numerical:
                tgt = tgt_token_dict[field].float()
                # assumes scalar predictions so pred is a 1d tensor
                pred = pred_dict[field].view(len(tgt))
                accs[field] = self.compute_rms(pred, tgt)
            for field in pred_dict.categorical:
                # pred must be [batch_size, num_classes]
                pred = pred_dict[field].argmax(-1)
                tgt = tgt_token_dict[field].long()
                accs[field] = self.compute_acc(pred, tgt)
            for field in pred_dict.text:
                # pred must be [batch_size, seq_len, num_classes]
                pred = pred_dict[field].flatten(0, 1).argmax(-1)
                tgt = tgt_token_dict[field].long().reshape(-1)
                accs[field] = self.compute_acc(pred, tgt)
        return accs

    def compute_rms(self, pred: torch.Tensor, tgt: torch.Tensor):
        mask = tgt != self.ignore_index_continuous
        pred, tgt = pred[mask], tgt[mask]
        norm = tgt.abs()
        return 1 - ((tgt - pred) / norm).pow(2).mean().sqrt()

    def compute_acc(self, pred: torch.Tensor, tgt: torch.Tensor):
        mask = tgt != self.ignore_index
        return (pred[mask] == tgt[mask]).float().mean()


def mean(x: dict, ignore_field="") -> "torch.Tensor":
    if len(x) == 0:
        raise ValueError("Cannot compute mean of empty dict")
    result = torch.tensor(0.0, requires_grad=True)
    total = torch.tensor(0.0)
    for field in x:
        if field == ignore_field:
            continue
        result = result.to(x[field].device)
        if x[field].numel() == 0:
            Warning(f"Empty tensor encountered in {field}")
        elif torch.isnan(x[field]).any():
            raise ValueError(f"NaN encountered in {field}")
        else:
            result = result + x[field]
            total += 1
    return result / total



import torch
import pandas as pd
from typing import Callable, Dict, Literal, Optional, TypeVar
from functools import cached_property
from collections import OrderedDict
from dataclasses import dataclass
import torch.nn.functional as F


def compute_min_ce_loss(prediction, ground_truth, ignore_index=0):
    # prediction: shape (num_masks, seq_len, vocab_size)
    # ground_truth: shape (num_masks, seq_len)

    num_masks, seq_len, vocab_size = prediction.shape

    # Initialize a matrix to store the cross-entropy losses
    ce_loss_matrix = torch.zeros((num_masks, num_masks), device=prediction.device)

    # Compute the cross-entropy loss for all pairs
    for i in range(num_masks):
        for j in range(num_masks):
            # Expand ground truth to have shape (seq_len, vocab_size) for F.cross_entropy
            ground_truth_expanded = ground_truth[j].view(-1)  # (seq_len, vocab_size)
            prediction_expanded = prediction[i].view(-1, vocab_size)  # (seq_len)
            ce_loss_matrix[i, j] = F.cross_entropy(
                prediction_expanded,
                ground_truth_expanded,
                ignore_index=ignore_index,
                reduction="mean",
            )

    # Select the minimum loss for each prediction
    min_ce_loss = torch.min(ce_loss_matrix, dim=1)[0]

    return min_ce_loss


def random_mask(*shape, mask_rate=0.8, device=None, seed=None):
    if mask_rate == 0:
        mask = torch.full(shape, 0.0, device=device)
    elif mask_rate == 1:
        mask = torch.full(shape, -torch.inf, device=device)
    else:
        mask_rate = torch.rand(1) if mask_rate == -1 else mask_rate
        total_elements = shape[-1]
        num_masked_elements = max(1, int(total_elements * mask_rate))
        
        # Create a mask with all elements set to 0.0
        mask = torch.full(shape, 0.0, device=device)
        
        # Set the rightmost elements to -inf to represent the mask
        mask[:, -num_masked_elements:] = -torch.inf
    
    return mask, num_masked_elements


class SortedSet:
    def __init__(self, iterable=None):
        self.items = []
        self.lookup_set = set()
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def add(self, item):
        if item not in self.lookup_set:
            self.items.append(item)
            self.lookup_set.add(item)

    def remove(self, item):
        if item in self.lookup_set:
            self.items.remove(item)
            self.lookup_set.remove(item)

    def __contains__(self, item):
        return item in self.lookup_set

    def __or__(self, other):
        result = SortedSet(self)
        for item in other:
            result.add(item)
        return result

    def __sub__(self, other):
        result = SortedSet()
        for item in self:
            if item not in other:
                result.add(item)
        return result

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"SortedSet({', '.join(map(str, self.items))})"


class TensorDict(dict):
    def __init__(self, base_dict=None, fields: Optional[dict] = None) -> None:
        super().__init__(base_dict if base_dict is not None else {})
        if fields is None:
            fields = {
                "numerical": [],
                "categorical": [],
                "text": [],
            }
        self.fields = fields
        self.numerical = fields["numerical"]
        self.categorical = fields["categorical"]
        self.text = fields["text"]

    @property
    def subset_numerical(self):
        return {field: self[field] for field in self.subset_numerical}

    @property
    def subset_categorical(self):
        return {field: self[field] for field in self.subset_categorical}

    @property
    def subset_text(self):
        return {field: self[field] for field in self.subset_text}

    def to(self, device):
        for field in self:
            self[field] = self[field].to(device)
        return self

    @property
    def iloc(self):
        return iloc(self)

    def __invert__(self):
        return TensorDict(
            {field: ~tensor for field, tensor in self.items()},
            fields=self.fields,
        )

    def to_tensor(self):
        return torch.cat(list(self.values()), dim=-1)

    def float(self):
        return TensorDict(
            {field: tensor.float() for field, tensor in self.items()},
            fields=self.fields,
        )

    def float_(self):
        for field in self:
            self[field] = self[field].float()
        return self

    def bool(self):
        return TensorDict(
            {field: tensor.bool() for field, tensor in self.items()},
            fields=self.fields,
        )

    def bool_(self):
        for field in self:
            self[field] = self[field].bool()
        return self

    def detach(self):
        return TensorDict(
            {field: tensor.detach() for field, tensor in self.items()},
            fields=self.fields,
        )

    def detach_(self):
        for field in self:
            self[field] = self[field].detach()
        return self

    def copy(self):
        return TensorDict(
            {field: tensor.clone() for field, tensor in self.items()},
            fields=self.fields,
        )

    def size(self, idx=None, column=None):
        size = None
        if idx is not None:
            return len(self.values()[idx])
        if column is not None:
            return len(self[column])
        for tensor in self.values():
            if size is None:
                size = len(tensor)
            else:
                assert size == len(tensor), "All tensors must have the same length"
        return size

    def device(self, idx=None, column=None):
        if idx is not None:
            return self.values()[idx].device
        if column is not None:
            return self[column].device
        device = None
        for tensor in self.values():
            if device is None:
                device = tensor.device
            else:
                assert device == tensor.device, "All tensors must be on the same device"
        return device


class iloc:
    def __init__(self, data: TensorDict):
        self.data = data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            key = idx[0]
            indices = idx[1]
            return self.data[key][indices]
        else:
            return TensorDict(
                {key: self.data[key][idx] for key in self.data},
                fields=self.data.fields,
            )

    def __setitem__(self, idx, tensordict):
        if isinstance(idx, tuple):
            key = idx[0]
            indices = idx[1]
            self.data[key][indices] = tensordict[key]
        else:
            for key in self.data:
                self.data[key][idx] = tensordict[key]

    def __call__(self, idx):
        return self[idx]


class Fields(OrderedDict):
    @property
    def all_fields(self):
        all_fields = []
        for field_values in self.values():
            all_fields.extend(field_values)
        return all_fields

    def type(self, field):
        for field_type, fields in self.items():
            if field in fields:
                return field_type
        raise ValueError(f"{field} not found in any field type")


def reduce_by_mask(losses, mask, token_mask=None):
    # careful:
    # this assumes that mask comes as True for things that are masked out (or float -inf)
    # and token_mask has the inverse logic (False for padding tokens)
    assert losses.ndim == 1 and mask.ndim == 1, "Reduce only accepts vectors"
    # p_mask comes in as -inf tensor
    # convert to bool
    if torch.is_floating_point(mask):
        mask = mask.bool()
    if token_mask is None:
        token_mask = torch.ones_like(mask, device=mask.device).bool()

    assert token_mask.ndim == 1, "Token mask should be flat"
    if len(losses) == 0 or not token_mask.any():
        return torch.tensor([], device=mask.device)
    mask = mask.repeat_interleave(len(losses) // len(mask))
    return losses[mask & token_mask].mean()


def shift_right(input_ids, inplace=True):
    decoder_start_token_id = 0

    if not inplace:
        shifted_input_ids = input_ids.clone()
    else:
        shifted_input_ids = input_ids

    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


def get_gpu_status():
    print("-" * 80)
    print(f"Current GPU mem: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    print(f"Max GPU mem: {torch.cuda.max_memory_allocated()/1e6:.2f} MB")
    print("-" * 80 + "\n")


T = TypeVar("T", bound=torch.nn.Module)


def setup_mup_shapes(
    model_callable: Callable[..., T], width_arguments: Dict, config: Dict
):
    base_config = config.copy()
    for key, value in width_arguments.items():
        base_config[key] = value
    base_model = model_callable(base_config)
    for key, value in width_arguments.items():
        base_config[key] = value * 2
    delta_model = model_callable(base_config)

    return base_model, delta_model


def mup_model(
    model_callable: Callable[..., T], width_arguments: Dict, config: Dict
) -> T:
    from mup import set_base_shapes

    base_model, delta_model = setup_mup_shapes(model_callable, width_arguments, config)
    model = model_callable(config)
    return set_base_shapes(model, base_model, delta=delta_model)


def convert_mask_to_float(mask: torch.Tensor) -> torch.Tensor:
    """Convert a mask from bool to float. True elements are converted to -inf and
    False elements are converted to 0."""
    return mask.to(torch.get_default_dtype()).masked_fill_(mask, -torch.inf)


def is_missing(
    padding_mask: torch.Tensor,
    property_type: Literal["categorical", "text", "numerical"],
):
    """Checks if a property is masked for a given batch.
    The padding mask is expected to be a float tensor with 0's for
    valid tokens and - inf's for masked tokens (because this is what
    nn.transformers wants). If the property is categorical or text, we check
    if all but the first token are masked. If the property is numerical,
    we check if the one element in each batch is masked.

    Args:
        padding_mask (torch.Tensor): tensor of shape (batch_size, seq_len)
        property_type (str): Literal["categorical", "text", "numerical"]. The
            type of property to check.
    """
    if property_type == "categorical":
        mask = padding_mask.isinf().view(-1)
    elif property_type == "text":
        # FIXME assumes that the first token is always valid
        # masked out value = eos followed by pads
        mask = padding_mask[:, 1:].all(-1)
    elif property_type == "numerical":
        assert padding_mask.ndim == 1
        mask = padding_mask.isinf()
    else:
        raise ValueError(f"Invalid property type {property_type}")

    return mask.float().masked_fill_(mask, -torch.inf)


def like(src, tgt):
    """make src like tgt"""
    dtype, device, shape = tgt.dtype, tgt.device, tgt.shape
    return src.to(dtype=dtype, device=device).view(*shape)


def trim_padding_(i, p, config):
    # for each key
    for key in config.fields["text"]:
        until = (~p[key].bool()).sum(dim=1).max()
        i[key] = i[key][:, :until].contiguous()
        p[key] = p[key][:, :until].contiguous()
    return i, p


@dataclass
class ModelOutputs:
    preds: TensorDict
    targets: TensorDict
    property_mask: torch.Tensor
    loss: torch.Tensor
    loss_dict: TensorDict
    error_dict: TensorDict

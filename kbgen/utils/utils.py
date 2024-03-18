import torch
import pandas as pd
from typing import Callable, Dict, Literal, Optional, TypeVar
from functools import cached_property
from collections import OrderedDict


def random_mask(*shape, mask_rate=0.8, device=None, seed=None):
    # TODO add dtype support
    # directly uses bool tensor ideally
    if mask_rate == 0:
        mask = torch.full(shape, 0.0, device=device)
    elif mask_rate == 1:
        mask = torch.full(shape, -torch.inf, device=device)
    else:
        mask_rate = torch.rand(1) if mask_rate == -1 else mask_rate
        mask = torch.full(shape, 0.0)
        # set fixed seed for noise
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        mask.masked_fill_(torch.rand(*shape, generator=g) < mask_rate, -torch.inf)
        mask = mask.to(device)
    return mask


def df_tokenize(
    df,
    fields,
    categorical_tokenizer,
    numerical_tokenizer,
):
    """Method to convert a dataframe to a tensor ready for training. Each field
       is tokenized and converted to a tensor of integers.

    Args:
        df (pd.Dataframe): Dataframe containing the data to be converted

    Returns:
        torch.Tensor: Tensor containing the data
    """
    tensor_dict = OrderedDict()
    attention_mask_dict = OrderedDict()
    dtype = torch.get_default_dtype()
    for field in fields.all_fields:
        if field in fields["text"] or field in fields["categorical"]:
            # TODO add dtype support
            # need to change all this to bool
            text = df[field].values.tolist()
            text = categorical_tokenizer(text, padding=True)
            tensor = text["input_ids"]
            am = text["attention_mask"]  # comes in as 1/0 int tensor
            # 1 -> 0, 0 -> -inf
            tensor_dict[field] = (
                tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
            )
            am = am if isinstance(am, torch.Tensor) else torch.tensor(am)
            am = am.bool().logical_not_()
            attention_mask_dict[field] = am.to(dtype).masked_fill_(am, -torch.inf)

        elif field in fields["numerical"]:
            numbers = numerical_tokenizer(df[field].values, dtype=torch.float)
            tensor_dict[field] = numbers
            am = numbers == numerical_tokenizer.pad_token
            attention_mask_dict[field] = am.to(dtype).masked_fill_(am, -torch.inf)
    return TensorDict(tensor_dict, fields=fields), TensorDict(
        attention_mask_dict, fields=fields
    )


def date_to_int_tensor(date):
    if pd.isnull(date) or date.startswith("http"):
        return -1000 * torch.ones(3, dtype=torch.int64)
    year, month, day = date.split("T")[0].split("-")[-3:]
    year = -int(year) if date.startswith("-") else int(year)
    return torch.tensor([int(year), int(month), int(day)], dtype=torch.int64)


def get_key_padding_mask(sequence_dict, pad_token=0):
    # TODO this is broken
    """Returns a mask for the encoder self-attention layer.
    The mask is True (hidden) for all positions corresponding
    to the padding tokens.

    Args:
        sequence_dict (dict): A dictionary of sequences of tokens
        pad_token (int, optional): The padding token. Defaults to 0.

    Returns:
        torch.float32: (batch_size, seq_len) mask with True (0) for all
        non-padding tokens and False (-inf) for all padding tokens
    """
    mask = {
        field: _get_key_padding_for_sequence(sequence, pad_token)
        for field, sequence in sequence_dict.items()
    }
    return TensorDict(mask)


def _get_key_padding_for_sequence(sequence, pad_token=0):
    """Returns a mask for the encoder self-attention layer.
    The mask is True (hidden) for all positions corresponding
    to the padding tokens.

    Args:
        sequence (torch.LongTensor): The sequence of tokens
        pad_token (int, optional): The padding token. Defaults to 0.

    Returns:
        torch.float32: (batch_size, seq_len) mask with True (0) for all
        non-padding tokens and False (-inf) for all padding tokens
    """
    seq = sequence == pad_token
    mask = torch.zeros_like(seq, dtype=torch.get_default_dtype())
    mask[seq] = -torch.inf
    return mask


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

    def __call__(self, idx):
        return self[idx]


class Fields(OrderedDict):
    @cached_property
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


def reduce_by_mask(losses, mask):
    if len(losses) == 0:
        return (torch.tensor([]),) * 3
    # p_mask comes in as -inf tensor
    # convert to bool
    if mask.dtype == torch.float32:
        mask = mask.bool()
    mask = mask.repeat_interleave(len(losses) // len(mask))
    masked = losses[mask].mean()
    unmasked = losses[~mask].mean()
    total = losses.mean()
    return masked, unmasked, total


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
    from mup import make_base_shapes

    base_config = config.copy()
    for key, value in width_arguments.items():
        base_config[key] = value
    base_model = model_callable(base_config)
    for key, value in width_arguments.items():
        base_config[key] = value * 2
    delta_model = model_callable(base_config)

    make_base_shapes(base_model, delta_model, "/tmp/mup_shapes.bsh")


def mup_model(
    model_callable: Callable[..., T], width_arguments: Dict, config: Dict
) -> T:
    from mup import set_base_shapes

    setup_mup_shapes(model_callable, width_arguments, config)
    model = model_callable(config)
    return set_base_shapes(model, "/tmp/mup_shapes.bsh")


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
    if property_type == "categorical" or property_type == "text":
        assert padding_mask.ndim > 1
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

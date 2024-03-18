from typing import Optional, Tuple, Union, MutableSequence
import torch
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from kbgen.config import rootdir
from functools import cached_property
from random import Random
from ..utils import TensorDict, Fields, df_tokenize, schema as schema_utils
from ..utils.tokenizer import (
    CustomTokenizer,
    NumericalTokenizer,
    GPT2Tokenizer,
)


class Dataset:
    def __init__(
        self,
        df,
        fields,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ) -> None:
        self._df = df
        self.fields = fields
        self.df = self._df.loc[:, fields.all_fields]
        self.df = self.transform_numerical()[fields.all_fields]
        self.numerical_pad_token_id = -1000
        self.categorical_pad_token_id = 0
        self.pad_token = "<pad>"
        self.seed = seed
        self.input_dict, self.pad_mask_dict = self.tokenize(
            tokenizer, numerical_tokenizer
        )
        # TODO potential speed up by using cuda tensors
        # now saves memory instead
        indices = range(len(self))
        self.train_idx, self.test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed
        )

    def tokenize(self, tokenizer=None, numerical_tokenizer=None):
        self.df.fillna("", inplace=True)  # previously was "<pad>" TODO
        # TODO: Connect Numerical tokens in categorical fields to numerical embeddings
        if tokenizer is None or tokenizer == "custom":
            self.tokenizer = CustomTokenizer(self.df, self.fields)
        elif tokenizer == "t5-small":
            from transformers import T5TokenizerFast as Tokenizer  # type: ignore

            self.tokenizer = Tokenizer.from_pretrained(tokenizer)
        elif tokenizer == "gpt2":
            self.tokenizer = GPT2Tokenizer()
        else:
            raise NotImplementedError

        self.numerical_tokenizer = NumericalTokenizer(self.numerical_pad_token_id)

        assert (
            self.pad_token == self.tokenizer.pad_token
        ), f"Pad token {self.pad_token} does not match tokenizer {self.tokenizer.pad_token}"
        assert self.categorical_pad_token_id == self.tokenizer.pad_token_id, (
            f"Categorical pad token {self.categorical_pad_token_id} does not "
            f"match tokenizer {self.tokenizer('<pad>')}"
        )
        assert self.numerical_pad_token_id == self.numerical_tokenizer(
            "", return_tensors="list"
        ), (
            f"Numerical pad token {self.numerical_pad_token_id} does not "
            f"match tokenizer {self.numerical_tokenizer('')}"
        )

        input_dict, pad_mask_dict = df_tokenize(
            self.df,
            self.fields,
            self.tokenizer,
            self.numerical_tokenizer,
        )
        for field in self.fields["categorical"]:
            input_dict[field + "_idx"] = self.token_to_label(input_dict[field], field)
        return input_dict, pad_mask_dict

    def transform_numerical(self):
        self._numerical_max = self.df.loc[:, self.fields["numerical"]].max()
        self._numerical_min = self.df.loc[:, self.fields["numerical"]].min()

        self.df.loc[:, self.fields["numerical"]] -= self._numerical_min
        self.df.loc[:, self.fields["numerical"]] /= (
            self._numerical_max - self._numerical_min
        )
        self.df.loc[:, self.fields["numerical"]] += 1
        return self.df

    def reverse_transform_all(self, tensor):
        max_ = self._numerical_max.values
        min_ = self._numerical_min.values
        assert tensor.shape[1] == len(max_)
        return (tensor - 1) * (max_ - min_) + min_

    def reverse_transform(self, dict):
        transformed = {}
        for k in dict.keys():
            if k not in self._numerical_max:
                transformed[k] = dict[k]
                continue
            transformed[k] = self.numerical_decode(k, dict[k])
        return transformed

    def numerical_decode(self, field, tensor):
        max_ = self._numerical_max[field]
        min_ = self._numerical_min[field]
        return (tensor - 1) * (max_ - min_) + min_
        

    def decode(self, seq):
        return self.tokenizer.decode(seq)

    def __len__(self):
        return len(self.df)

    def get_loaders(
        self,
        batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        shuffle: bool = True,
    ):
        if batch_size is None:
            batch_size = (len(self.train_idx), len(self.test_idx))
        if isinstance(batch_size, int):
            batch_size = (batch_size, len(self.test_idx))
        trainloader = DataLoader(
            self.train_idx,
            batch_size=batch_size[0],
            shuffle=shuffle,
            seed=self.seed,
        )
        testloader = DataLoader(
            self.test_idx,
            batch_size=batch_size[1],
            shuffle=False,
            seed=self.seed,
        )
        return trainloader, testloader

    # get item by index
    def __getitem__(self, idx):
        return TensorDict(
            {field: tensor[idx] for field, tensor in self.input_dict.items()},
            fields=self.fields,
        )

    @property
    def train(self):
        return self[self.train_idx]

    @property
    def test(self):
        return self[self.test_idx]

    @cached_property
    def categorical_str_to_id(self):
        # assign pad_token_id to empty string
        categories = {
            field: {"": self.categorical_pad_token_id}
            for field in self.fields["categorical"]
        }
        # BUG the if tokenizer decode is not the same as the original string
        # then we'll have a key error
        for field in categories:
            # give an idx to each category
            # skip masked fields (i.e. empty strings)
            idx = 0
            categories[field].update(
                {
                    value: (idx := idx + 1)
                    for value in self.df[field].unique()
                    if value != ""
                }
            )
        return categories

    @cached_property
    def categorical_id_to_str(self):
        return {
            field: {v: k for k, v in categories.items()}
            for field, categories in self.categorical_str_to_id.items()
        }

    @cached_property
    def categorical_num_classes(self):
        return {
            field: len(categories)
            for field, categories in self.categorical_str_to_id.items()
        }

    def categorical_id_to_token(self, field, id):
        if isinstance(id, torch.Tensor):
            id = id.tolist()
        if not isinstance(id, (list, tuple)):
            id = [id]
        return self.prepare_tokenized_categories[field]["input_ids"][id]
    
    @cached_property
    def prepare_tokenized_categories(self):
        tokenized_categories = {}
        for field, categories in self.categorical_str_to_id.items():
            tokenized_categories[field] = self.tokenizer(
                list(categories.keys()), return_tensors="pt", padding=True
            )
        return tokenized_categories

    def token_to_label(self, token_tensor, field, return_tensor="pt"):
        """
        Given a token tensor and a field, returns the corresponding indices of the categories
        represented by the tokens in the tensor. Example: if the tensor contains the tokens
        [[10, 2]] which corespond to the string "cat", and suppose "cat" is the 5th category in
        the field "animal", then this function will return [5].

        Args:
            token_tensor (torch.Tensor): The tensor containing the tokens to be converted to
                indices.
            field (str): The name of the field whose categories are being converted.
            return_tensor (str, optional): Whether to return the indices as a PyTorch tensor
                ("pt") or a list ("list"). Defaults to "pt".

        Returns:
            torch.Tensor or list: The indices (labels) of the categories represented by
            the okens in the tensor.
        """
        strings = self.tokenizer.batch_decode(token_tensor, skip_special_tokens=True)
        indices = [self.categorical_str_to_id[field][s] for s in strings]
        if return_tensor == "pt":
            return torch.tensor(indices)
        return indices

    def label_to_string(self, labels, field):
        """
        Given a label tensor and a field, returns the corresponding text of the categories
        represented by the labels in the tensor. Example: if the tensor contains the labels
        [5] and suppose "cat" is the 5th category in the field "animal", then this function
        will return ["cat"].

        Args:
            labels (Iterable): The tensor containing the labels to be converted to text.
            field (str): The name of the field whose categories are being converted.

        Returns:
            list: The text of the categories represented by the labels in the tensor.
        """
        strings = [self.categorical_id_to_str[field][i] for i in labels]
        return strings


class GSM(Dataset):
    def __init__(
        self,
        path=None,
        seed=None,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        # initialize arguments
        path = os.path.join(rootdir, "data/gsm") if path is None else path
        seed = 42 if seed is None else seed
        test_size = 0.2 if test_size is None else test_size

        # load data
        # df_path = os.path.join(path, "gsm_processed.csv")
        df_path = os.path.join(path, "gsm_processed_500.csv")
        schema_path = os.path.join(path, "gsm_schema.json")
        for p in [path, df_path, schema_path]:
            if not os.path.exists(p):
                raise ValueError(
                    f"{p} does not exist. To download the dataset"
                    ", please see `exploration_gsm.ipynb`."
                    " Also make sure to set the `rootdir` variable in `config.py`."
                )
        df = pd.read_csv(df_path)
        schema = json.loads(open(schema_path).read())

        # schema is for fields and their basic types
        types_to_nodes = schema_utils.by_types(schema)
        node_ids = schema_utils.get_ids(schema)
        columns = schema_utils.match_fields_to_nodes(df.columns, node_ids).values()
        df.columns = list(columns)  # rename columns to match schema

        self.fields = Fields(
            numerical=types_to_nodes[0],
            # numerical=types_to_nodes[0][:2],
            categorical=types_to_nodes[1],
            # categorical=[],
            text=types_to_nodes[2],
            # text=[],
        )
        super().__init__(
            df,
            self.fields,
            seed,
            test_size,
            tokenizer,
            numerical_tokenizer,
        )
        self.consistency_check()
        self.train_idx, self.test_idx = self.remove_similar(
            self.train_idx, self.test_idx
        )

    def consistency_check(self):
        for field in self.fields["categorical"]:
            proto_tokens = self.prepare_tokenized_categories
            proto_str_to_id = self.categorical_str_to_id
            examples = self.tokenizer.batch_decode(
                proto_tokens[field]["input_ids"], skip_special_tokens=True
            )
            for i in range(len(examples)):
                assert (
                    proto_str_to_id[field][examples[i]] == i
                ), f"Failed categorical index consistency check for {field} {examples[i]} != {i}"

    @staticmethod
    def from_config(config: dict, update: bool = True):
        """Load dataset from config dict. You probably want to import
        a default config from `config.py`. This function will update the config
        dictionary with the dataset's properties.

        Args:
            config (dict): The run config dictionary to use.
            update (bool, optional): Whether to automatically update the input config
                dict with the relevant dataset keys. Defaults to True.

        Returns:
            Dataset: The dataset object.
        """
        dataset = GSM(
            path=config.get("data_path", None),
            tokenizer=config.get("tokenizer", None),
            numerical_tokenizer=config.get("numerical_tokenizer", None),
        )
        if update:
            config.update(
                {
                    "num_fields": len(dataset.df.columns),
                    "vocab_size": len(dataset.tokenizer),
                    "fields": dataset.fields,
                    "categorical_num_classes": dataset.categorical_num_classes,
                    "numerical_pad_token_id": dataset.numerical_pad_token_id,
                    "categorical_pad_token_id": dataset.categorical_pad_token_id,
                }
            )
        return dataset

    def remove_similar(self, train_idx, test_idx):
        train_df = self._df.iloc[train_idx]
        test_df = self._df.iloc[test_idx]
        mask1 = test_df["phone.model"].isin(train_df["phone.model"])
        mask2 = test_df["phone.oem"].isin(train_df["phone.oem"])
        mask = mask1 & mask2
        test_idx = np.array(test_idx)
        new_test_idx = test_idx[~mask]
        dups = test_idx[mask.tolist()]
        new_train_idx = np.concatenate([train_idx, dups])
        return new_train_idx.tolist(), new_test_idx.tolist()


class DataLoader:
    def __init__(
        self, tensor: MutableSequence, batch_size: int, shuffle: bool = False, seed=None
    ):
        self.tensor = tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            Random(self.seed).shuffle(self.tensor)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.tensor):
            raise StopIteration
        batch = self.tensor[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return batch

    def __len__(self):
        div = len(self.tensor) // self.batch_size
        if len(self.tensor) % self.batch_size == 0:
            return div
        else:
            return div + 1

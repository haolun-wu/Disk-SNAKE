from typing import Optional, Tuple, Union, MutableSequence
from collections import OrderedDict
import torch
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from kbgen.config import rootdir
from functools import cached_property
from random import Random
from ..utils import TensorDict, Fields, schema as schema_utils
from ..utils.tokenizer import (
    CustomTokenizer,
    NumericalTokenizer,
    GPT2Tokenizer,
    SimpleTokenizer,
)
from loguru import logger


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
        self.value_to_type = {0: "numerical", 1: "categorical", 2: "text"}
        self._df = df
        self.fields = fields
        self.df = self._df.loc[:, fields.all_fields]
        self.numerical_pad_token_id = -1000
        self.categorical_pad_token_id = 0
        self.df = self.transform_numerical()[fields.all_fields]
        self.df = self.transform_categorical()[fields.all_fields]
        self.pad_token = "<pad>"
        self.seed = seed

        input_dict_path = os.path.join(self.path, f"{tokenizer}_input_dict.pt")
        pad_mask_dict_path = os.path.join(self.path, f"{tokenizer}_pad_mask_dict.pt")
        # if os.path.exists(input_dict_path) and os.path.exists(pad_mask_dict_path):
        #     print("processed data found, loading...")
        #     self.input_dict = torch.load(input_dict_path)
        #     self.pad_mask_dict = torch.load(pad_mask_dict_path)
        #     self.tokenize(tokenizer, numerical_tokenizer, run_tokenizer=False)
        #     print("done")
        # else:
        self.input_dict, self.pad_mask_dict = self.tokenize(
            tokenizer, numerical_tokenizer
        )
        torch.save(self.input_dict, input_dict_path)
        torch.save(self.pad_mask_dict, pad_mask_dict_path)
        # TODO potential speed up by using cuda tensors
        # now saves memory instead
        indices = range(len(self))
        self.train_idx, self.val_idx = train_test_split(
            indices, test_size=test_size, random_state=seed
        )

    def _from_path(
        self,
        path,
        prefix,
    ):
        # initialize arguments
        self.path = path

        # load data
        df_path = os.path.join(path, f"{prefix}.csv")
        schema_path = os.path.join(path, f"{prefix}_schema.json")
        for p in [path, df_path, schema_path]:
            if not os.path.exists(p):
                raise ValueError(
                    f"{p} does not exist. To download the dataset"
                    f", please see `exploration_{prefix}.ipynb`."
                    " Also make sure to set the `rootdir` variable in `config.py`."
                )
        df = pd.read_csv(df_path)
        schema = json.loads(open(schema_path).read())
        self.schema = schema
        types_to_nodes = schema_utils.by_types(schema)

        self.fields = Fields(
            numerical=types_to_nodes[0],
            categorical=types_to_nodes[1],
            text=types_to_nodes[2],
        )
        return df

    @classmethod
    def from_config_(cls, config: dict, update: bool = True):
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
        dataset = cls(
            path=config.get("data_path", None),
            tokenizer=config.get("tokenizer", None),
            numerical_tokenizer=config.get("numerical_tokenizer", None),
        )
        if update:
            logger.info("Updating config with dataset properties.")
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

    def tokenize(self, tokenizer=None, numerical_tokenizer=None, run_tokenizer=True):
        self.df.fillna("", inplace=True)  # previously was "<pad>" TODO
        # convert cat and text columns to strings
        self.df.loc[:, self.fields["text"]] = self.df.loc[
            :, self.fields["text"]
        ].astype(str)

        # TODO: Connect Numerical tokens in categorical fields to numerical embeddings
        if tokenizer == "custom":
            self.tokenizer = CustomTokenizer(self.df, self.fields)
        elif tokenizer == "t5-small":
            from transformers import T5TokenizerFast as Tokenizer  # type: ignore

            self.tokenizer = Tokenizer.from_pretrained(tokenizer)
        elif tokenizer == "gpt2":
            self.tokenizer = GPT2Tokenizer()
        elif tokenizer == "simple":
            self.tokenizer = SimpleTokenizer(" ".join(self.as_strings()))
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

        if run_tokenizer:
            input_dict, pad_mask_dict = self.df_tokenize(
                self.df,
                self.fields,
                self.tokenizer,
                self.numerical_tokenizer,
            )
            return input_dict, pad_mask_dict

    def df_tokenize(
        self,
        df,
        fields,
        text_tokenizer,
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
            if field in fields["text"]:
                # TODO add dtype support
                # need to change all this to bool
                text = df[field].values.tolist()
                text = text_tokenizer(text, padding=True)
                tensor = text["input_ids"]
                am = text["attention_mask"]  # comes in as 1/0 int tensor
                # 1 -> 0, 0 -> -inf
                tensor_dict[field] = (
                    tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
                )
                am = am if isinstance(am, torch.Tensor) else torch.tensor(am)
                am = am.bool().logical_not_()
                am = am.to(dtype).masked_fill_(am, -torch.inf)
                attention_mask_dict[field] = am
            elif field in fields["categorical"]:
                # get set of values and map to integers
                am = df[field] == self.categorical_pad_token_id # get mask of null values (is this right? pad token id is 0)
                am = torch.tensor(am.values, dtype=torch.bool)
                tensor_dict[field] = torch.tensor(df[field].values)
                attention_mask_dict[field] = am.to(dtype).masked_fill_(am, -torch.inf)
            elif field in fields["numerical"]:
                numbers = numerical_tokenizer(df[field].values, dtype=torch.get_default_dtype())
                tensor_dict[field] = numbers
                am = numbers == numerical_tokenizer.pad_token
                attention_mask_dict[field] = am.to(dtype).masked_fill_(am, -torch.inf)
        return TensorDict(tensor_dict, fields=fields), TensorDict(
            attention_mask_dict, fields=fields
        )

    def transform_numerical(self):
        self._numerical_max = self.df.loc[:, self.fields["numerical"]].max()
        self._numerical_min = self.df.loc[:, self.fields["numerical"]].min()

        self.df.loc[:, self.fields["numerical"]] -= self._numerical_min
        self.df.loc[:, self.fields["numerical"]] /= (
            self._numerical_max - self._numerical_min + 1e-10
        )
        return self.df

    def transform_categorical(self):
        for field in self.fields["categorical"]:
            self.df[field] = self.df[field].fillna("").map(self.categorical_str_to_id[field])
        return self.df

    def reverse_transform(self, dict):
        transformed = {}
        for k in dict.keys():
            if k not in self._numerical_max:
                transformed[k] = dict[k].clone()
                continue
            pad_mask = dict[k] == self.numerical_pad_token_id
            transformed[k] = self.numerical_decode(k, dict[k])
            transformed[k].masked_fill_(pad_mask, self.numerical_pad_token_id)
        return transformed

    def numerical_decode(self, field, tensor):
        max_ = self._numerical_max[field]
        min_ = self._numerical_min[field]
        return tensor * (max_ - min_) + min_


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
            batch_size = (len(self.train_idx), len(self.val_idx))
        if isinstance(batch_size, int):
            batch_size = (batch_size, batch_size)
        trainloader = DataLoader(
            self.train_idx,
            batch_size=batch_size[0],
            shuffle=shuffle,
            seed=self.seed,
        )
        testloader = DataLoader(
            self.val_idx,
            batch_size=batch_size[1],
            shuffle=False,
            seed=self.seed,
        )
        return trainloader, testloader


    @property
    def train(self):
        return self[self.train_idx]

    @property
    def test(self):
        return self[self.val_idx]

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
        seed=42,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        # initialize arguments
        path = os.path.join(rootdir, "data/gsm") if path is None else path
        self.path = path
        seed = 42 if seed is None else seed
        test_size = 0.2 if test_size is None else test_size

        # load data
        df_path = os.path.join(path, "gsm_processed.csv")
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
        self.schema = schema

        # schema is for fields and their basic types
        types_to_nodes = schema_utils.by_types(schema)
        node_ids = schema_utils.get_ids(schema)
        columns = schema_utils.match_fields_to_nodes(df.columns, node_ids).values()
        df.columns = list(columns)  # rename columns to match schema

        self.fields = Fields(
            numerical=types_to_nodes[0],
            categorical=types_to_nodes[1],
            text=types_to_nodes[2],
        )
        super().__init__(
            df,
            self.fields,
            seed,
            test_size,
            tokenizer,
            numerical_tokenizer,
        )
        self.train_idx, self.val_idx = self.remove_similar(self.train_idx, self.val_idx)


    def remove_similar(self, train_idx, val_idx):
        train_df = self._df.iloc[train_idx]
        test_df = self._df.iloc[val_idx]
        mask1 = test_df["phone.model"].isin(train_df["phone.model"])
        mask2 = test_df["phone.oem"].isin(train_df["phone.oem"])
        mask = mask1 & mask2
        val_idx = np.array(val_idx)
        new_val_idx = val_idx[~mask]
        dups = val_idx[mask.tolist()]
        new_train_idx = np.concatenate([train_idx, dups])
        return new_train_idx.tolist(), new_val_idx.tolist()


class DataLoader:
    def __init__(
        self,
        *tensor: MutableSequence,
        batch_size: int,
        shuffle: bool = False,
        seed=0,
    ):
        self.tensor = tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            rng = Random(self.seed)
            for t in self.tensor:
                rng.shuffle(t)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.tensor[0]):
            raise StopIteration

        for t in self.tensor:
            assert len(t) == len(self.tensor[0])

        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensor)
        self.i += self.batch_size
        return batch

    def __len__(self):
        div = len(self.tensor[0]) // self.batch_size
        if len(self.tensor) % self.batch_size == 0:
            return div
        else:
            return div + 1


class HomeDepot(Dataset):
    def __init__(
        self,
        path=None,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        path = os.path.join(rootdir, "data/homedepot") if path is None else path
        df = self._from_path(path, "homedepot")
        super().__init__(df, self.fields, seed, test_size, tokenizer, numerical_tokenizer)


class NUCLEAR(Dataset):
    def __init__(
        self,
        path=None,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        path = os.path.join(rootdir, "data/nuclear") if path is None else path
        df = self._from_path(path, "nuclear")
        df["z_cat"] = df["z"]
        df["n_cat"] = df["n"]
        self.fields["numerical"] = ['z', 'n', 'binding_semf', 'radius', 'half_life_sec', 'spin', 'abundance', 'qa', 'qbm', 'qbm_n', 'qec', 'electric_quadrupole', 'volume', 'surface', 'symmetry', 'coulomb']
        self.fields["categorical"] = ["parity", "stability", "z_cat", "n_cat"]
        #self.fields["categorical"] += ["z_cat", "n_cat"]
        super().__init__(df, self.fields, seed, test_size, tokenizer, numerical_tokenizer)


class Gaussians(Dataset):
    def __init__(
        self,
        path=None,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        if path is None:
            self.path = os.path.join(rootdir, "data/gaussians")
        os.makedirs(self.path, exist_ok=True)
        rng = torch.Generator()
        rng.manual_seed(seed)
        n_samples = 10000
        x = torch.randn(n_samples, generator=rng)
        y = torch.randn(n_samples, generator=rng) * 2
        z = x + y
        df = pd.DataFrame({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})
        self.fields = Fields({"numerical": ["x", "y", "z"], "categorical": [], "text": []})
        super().__init__(df, self.fields, seed, test_size, tokenizer, numerical_tokenizer)

class TwoMoons(Dataset):
    def __init__(
        self,
        path=None,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        from sklearn.datasets import make_moons
        if path is None:
            self.path = os.path.join(rootdir, "data/twomoons")
        os.makedirs(self.path, exist_ok=True)

        rng = torch.Generator()
        rng.manual_seed(seed)
        n_samples = 1000
        x, y = make_moons(n_samples=n_samples, noise=0.0, random_state=seed)
        # x = x.round(2)
        df = pd.DataFrame({"x": x[:,0].tolist(), "y": x[:,1].tolist(), "label": y.astype(float).tolist()})
        self.fields = Fields({"numerical": ["x", "y", ], "categorical": ["label"], "text": []})
        super().__init__(df, self.fields, seed, test_size, tokenizer, numerical_tokenizer)

class Iris(Dataset):
    def __init__(
        self,
        path=None,
        seed=0,
        test_size=0.2,
        tokenizer=None,
        numerical_tokenizer=None,
    ):
        from sklearn.datasets import load_iris

        if path is None:
            self.path = os.path.join(rootdir, "data/iris")
        os.makedirs(self.path, exist_ok=True)

        rng = torch.Generator()
        rng.manual_seed(seed)
        x, y = load_iris(return_X_y=True)
        df = pd.DataFrame(
            {
                "slength": x[:, 0].tolist(),
                "swidth": x[:, 1].tolist(),
                "plength": x[:, 2].tolist(),
                "pwidth": x[:, 3].tolist(),
                "label": y.astype(float).tolist(),
            }
        )
        self.fields = Fields(
            {
                "numerical": ["slength", "swidth", "plength", "pwidth"],
                "categorical": ["label"],
                "text": [],
            }
        )
        super().__init__(
            df, self.fields, seed, test_size, tokenizer, numerical_tokenizer
        )


def load_dataset(config):
    if config["dataset"] == "gsm":
        return GSM.from_config_(config)
    elif config["dataset"] == "homedepot":
        return HomeDepot.from_config_(config)
    elif config["dataset"] == "nuclear":
        return NUCLEAR.from_config_(config)
    elif config["dataset"] == "gaussians":
        return Gaussians.from_config_(config)
    elif config["dataset"] == "twomoons":
        return TwoMoons.from_config_(config)
    elif config["dataset"] == "iris":
        return Iris.from_config_(config)

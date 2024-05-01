from ..utils import SortedSet
import torch
from collections.abc import Iterable


class CustomTokenizer:
    def __init__(self, df, fields):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.dictionary = self.train(df, fields)
        self._reverse_tokenizer = list(self.dictionary.keys())

    def __call__(self, string_collection, return_tensors="pt", padding=True):
        sequences = self.encode(
            string_collection, return_tensors=return_tensors, padding=padding
        )
        attention_mask = [
            [token != self.pad_token_id for token in seq] for seq in sequences
        ]
        if return_tensors == "pt":
            sequences = [torch.Tensor(t) for t in sequences]
            attention_mask = [torch.Tensor(t) for t in attention_mask]
            if padding:
                sequences = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=self.pad_token_id,
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    attention_mask,
                    batch_first=True,
                    padding_value=0,
                )
        return_dict = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
        }
        return return_dict

    def encode(self, string_collection, return_tensors="list", padding=True):
        # TODO I don't like multiple return types
        # but following the huggingface API
        # see how to change it later
        def _encode(string_collection):
            if isinstance(string_collection, str):
                tokens = [self.dictionary[word] for word in string_collection.split()]
                tokens += [self.eos_token_id]
                return tokens
            else:
                return [_encode(s) for s in string_collection]

        token_list = _encode(string_collection)
        if padding:
            token_list = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t) for t in token_list],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            if return_tensors == "pt":
                return token_list
            elif return_tensors == "np":
                return token_list.numpy()
        elif return_tensors == "list":
            return token_list
        else:
            raise ValueError(f"return_tensors={return_tensors} not supported")

    def train(self, df, fields):
        # Split the string entries into words
        all_words = SortedSet()
        for field in fields["text"] + fields["categorical"]:
            all_words |= SortedSet(
                word
                for seq in df[field].apply(lambda x: str(x).split()).values
                for word in seq
            )
        # all_words |= SortedSet(
        #     df[fields["categorical"]]
        #     .values.flatten()
        # )
        all_words -= {
            self.pad_token,
            self.sos_token,
            self.eos_token,
        }  # remove special tokens
        dictionary = {
            word: idx
            for idx, word in enumerate(
                [
                    self.pad_token,
                    self.sos_token,
                    self.eos_token,
                ]  # ensure special tokens are first
                + list(all_words)
            )
        }
        return dictionary

    def batch_decode(self, input_ids, attention_mask=None, skip_special_tokens=True):
        return self.decode(input_ids, attention_mask, skip_special_tokens)

    def decode(self, input_ids, attention_mask=None, skip_special_tokens=True):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list):
            if isinstance(input_ids[0], int):
                words = [self.decode(seq) for seq in input_ids]
                return " ".join([word for word in words if word != ""])  # type: ignore
            else:
                return [self.decode(seq) for seq in input_ids]
        string_of_token = self._reverse_tokenizer[input_ids]
        # TODO use attention mask instead
        if skip_special_tokens and string_of_token in [
            "<sos>",
            "<eos>",
            "<pad>",
        ]:
            return ""
        return string_of_token

    def __getitem__(self, key):
        return self.dictionary[key]

    def __len__(self):
        return len(self.dictionary)

    def __contains__(self, key):
        return key in self.dictionary

    def keys(self):
        return self.dictionary.keys()

    @property
    def pad_token_id(self):
        return self.dictionary[self.pad_token]

    @property
    def sos_token_id(self):
        return self.dictionary[self.sos_token]

    @property
    def eos_token_id(self):
        return self.dictionary[self.eos_token]


class NumericalTokenizer:
    def __init__(self, pad_token) -> None:
        self.pad_token = pad_token

    def __call__(self, x, return_tensors="pt", device=None, dtype=None):
        if return_tensors == "pt":
            return torch.tensor(self._encode(x), device=device, dtype=dtype)
        return self._encode(x)

    def _encode(self, x):
        # expects masked elements to be empty string "" not <pad>
        if isinstance(x, str) or isinstance(x, int) or isinstance(x, float):
            return float(x) if not x == "" else self.pad_token
        else:
            return [self._encode(xi) for xi in x]


class SimpleTokenizer:
    def __init__(self, text):
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<->", "<:>", "<|>", "<eos>"]
        self.dictionary = self.train(text)
        self._reverse_tokenizer = list(self.dictionary.keys())
        self.n_vocab = len(self.dictionary)
        self.pad_token_id = self.dictionary["<pad>"]
        self.eos_token_id = self.dictionary["<eos>"]

    def train(self, text):
        # will split by space and turn every word into a token
        all_words = SortedSet(text.split())
        all_words -= SortedSet(self.special_tokens)
        dictionary = {
            word: idx
            for idx, word in enumerate(
                self.special_tokens + list(all_words)  # ensure special tokens are first
            )
        }
        return dictionary

    def __call__(self, string_collection, return_tensors="pt", padding=True):
        sequences = self.encode(
            string_collection, return_tensors=return_tensors, padding=padding
        )
        attention_mask = [
            [token != self.pad_token_id for token in seq] for seq in sequences
        ]
        if return_tensors == "pt":
            sequences = [torch.Tensor(t) for t in sequences]
            attention_mask = [torch.Tensor(t) for t in attention_mask]
            if padding:
                sequences = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=self.pad_token_id,
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    attention_mask,
                    batch_first=True,
                    padding_value=0,
                )
        return_dict = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
        }
        return return_dict

    def encode(
        self, string_collection, return_tensors="list", padding=True, add_eos=True
    ):
        def _encode(string_collection):
            if isinstance(string_collection, str):
                tokens = [self.dictionary[word] for word in string_collection.split()]
                if add_eos:
                    tokens += [self.eos_token_id]
                return tokens
            else:
                return [_encode(s) for s in string_collection]

        token_list = _encode(string_collection)
        if padding:
            token_list = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t) for t in token_list],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
        if return_tensors == "pt":
            return token_list
        elif return_tensors == "np":
            return token_list.numpy()
        elif return_tensors == "list":
            return token_list
        else:
            raise ValueError(f"return_tensors={return_tensors} not supported")

    def decode(self, input_ids, attention_mask=None, skip_special_tokens=True):
        """Decodes a sequence of token ids to a string"""

        def is_special_token(word):
            if skip_special_tokens:
                return word in self.special_tokens or word == ""
            else:
                return False

        words = [self._reverse_tokenizer[id_] for id_ in input_ids]
        return " ".join([word for word in words if not is_special_token(word)])

    def batch_decode(self, input_ids, attention_mask=None, skip_special_tokens=True):
        assert isinstance(input_ids, Iterable), "input_ids must be iterable"
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(input_ids[0], list):
            return [
                self.decode(seq, attention_mask, skip_special_tokens)
                for seq in input_ids
            ]
        elif isinstance(input_ids[0], int):
            return self.decode(input_ids, attention_mask, skip_special_tokens)
        else:
            raise ValueError(f"input_ids type {type(input_ids)} not supported")

    def __getitem__(self, key):
        return self.dictionary[key]

    def __len__(self):
        return len(self.dictionary)

    def __contains__(self, key):
        return key in self.dictionary

    def keys(self):
        return self.dictionary.keys()

from ..utils import SortedSet
import torch
import tiktoken


class GPT2Tokenizer:
    def __init__(self) -> None:
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<pad>"
        base = tiktoken.get_encoding("gpt2")
        base._mergeable_ranks[b"!"] = base.n_vocab
        self.enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                self.pad_token: 0,
            },
        )

        self.pad_token_id = self.enc.encode(
            self.pad_token, allowed_special={self.pad_token}
        )[0]
        self.eos_token_id = self.enc.encode(
            self.eos_token, allowed_special={self.eos_token}
        )[0]

    def encode(self, text: str, allowed_special=set(), disallowed_special="all"):
        return self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.enc.decode(tokens)

    def batch_decode(self, sequence, skip_special_tokens=True) -> list:
        if isinstance(sequence, int):
            sequence = [sequence]
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        if isinstance(sequence[0], int):
            sequence = [sequence]
        if skip_special_tokens:
            stripped = []
            for s in sequence:
                stripped.append(
                    [t for t in s if t not in [self.pad_token_id, self.eos_token_id]]
                )
            sequence = stripped
        return self.enc.decode_batch(sequence)  # type: ignore

    def __call__(self, sequences, return_tensors="pt", padding=True):
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = self.enc.encode_batch(sequences, allowed_special={self.pad_token})
        sequences = [seq + [self.eos_token_id] for seq in sequences]
        attention_mask = [
            [token != self.pad_token_id for token in seq] for seq in sequences
        ]
        if return_tensors == "pt":
            sequences = [torch.tensor(t) for t in sequences]
            attention_mask = [torch.tensor(t) for t in attention_mask]
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

    def __len__(self):
        return self.enc.n_vocab


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
                for seq in df[field].apply(lambda x: x.split()).values
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

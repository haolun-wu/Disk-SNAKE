"""Here we will implement the positional encodings for the KBGEN model.
Unlike the original paper (Vaswani et al. 2017), we will use the positional encodings in a hierarchical tree structure.
The idea is that the positional encoding of a node in the tree is determined by the path
from the root to the node.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod


class Pathing(ABC, nn.Module):
    def __init__(self, schema_ids, d_model, embedding_type="node"):
        """This class implements the positional encodings for the KBGEN model.
        It expects a schma of the entity properties in the dataset in the form of a tree.
        The tree is represented as a dictionary where the keys are the nodes and the values
        are the node ids. The special character "." is used to denote the path from the root
        to a node. The root is the key with id 0. For example, the schema for a person entity
        with 3 attributes Person(name, height, dob) where dob is a date with 3 attributes
        (year, month, day) is represented as follows:

        schema_ids = {
            "person": 0,
            "person.name": 1,
            "person.height": 2,
            "person.dob": 3,
            "person.dob.year": 4,
            "person.dob.month": 5,
            "person.dob.day": 6,
        }


        Args:
            schema_ids (dict): The schema ids of the dataset. Details above.
            d_model (int): The dimension of the embeddings.
            embedding_type (str, optional): Either "node" or "word". "node" means
            that the positional encodings will be based on individual nodes in the
            tree even when they have the same name.
            "word" means that the positional encodings will be based on the sequence
            of words in the path so paths like "person.name" and "person.sibling.name"
            will be made using the same embeddings for "person" and "name".
            How the positional encodings are made from this bag of words is determined
            in the forward method of the subclasses.
            Defaults to "node".

        Raises:
            ValueError: If embedding_type is not "node" or "word".
        """ """"""
        super().__init__()
        if isinstance(schema_ids, list):
            schema_ids = {node: i for i, node in enumerate(schema_ids)}

        if embedding_type == "word":
            self._id_lookup = {id: node for node, id in schema_ids.items()}
            # convert idx to sequence of word ids
            self.words = {}
            self.schema_id_to_word_ids = {}
            for id, node_path in self._id_lookup.items():
                for new_word in node_path.split("."):
                    if new_word not in self.words:
                        self.words[new_word] = len(self.words) + 1
                        self.schema_id_to_word_ids[id] = len(self.words)

            self._schema_id_to_word_ids_seq = lambda id: [
                self.words[word] for word in self._id_lookup[id].split(".")
            ]

            # pad the sequence of word ids
            paths = pad_sequence(
                [
                    torch.tensor(self._schema_id_to_word_ids_seq(id))
                    for id in self._id_lookup
                ],
                batch_first=True,
            )
            self.register_buffer("paths", paths)
            self.register_buffer("blank_x", torch.zeros(1, len(paths), d_model))
        elif embedding_type == "node":
            self.words = schema_ids
            paths = torch.tensor(list(schema_ids.values())).unsqueeze(0)
            self.register_buffer("paths", paths)
            self.register_buffer("blank_x", torch.zeros(1, len(self.words), d_model))
        else:
            raise ValueError("embedding_type must be either 'word' or 'node'")

        self.embeddings = nn.Embedding(len(self.words) + 1, d_model)
        self.scale = 1

    def get_all_paths(self) -> torch.Tensor:
        return self(self.blank_x)

    @abstractmethod
    def forward(self, x):
        pass


class RNNPathing(Pathing):
    def __init__(self, schema, d_model):
        """Positional encodings based on the path from the root to the node.
        The positional encoding of a node is the last hidden state of an RNN
        that encodes the path from the root to the node.

        Usage:
            ```
            rnn_pathing_pe = RNNPathing(schema_ids, d_model)
             schema_ids = {
                 person: 0,
                 person.name: 1,
                 person.height: 2,
                }
            x = torch.tensor(schema_ids.keys())

            pe = rnn_pathing_pe(x) # returns the positional encodings for the nodes in x
            pe.shape # (3, d_model)
            # the RNN processes each node in the path in sequence
            ```

        Args:
            schema (dict): dictionary representing the schema of the dataset.
                See Pathing for more details.
            d_model (int): The dimension of the embeddings.
        """
        super().__init__(schema, d_model, embedding_type="word")
        # simple rnn to encode the paths
        self.rnn = nn.RNN(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.paths is a tensor of shape (num_paths, seq_len)
        out, last = self.rnn(self.embeddings(self.paths))
        # last is a tensor of shape (num_layers, num_paths, d_model)
        # out is a tensor of shape (num_paths, seq_len, d_model)
        # we will use the last hidden state of the rnn as the positional encoding
        return x * self.scale + last[[-1]]


class BagPathing(Pathing):
    def __init__(self, schema, d_model):
        """
        These positional encodings are "bag of words (nodes)" encodings
        constructed by summing the embeddings of the nodes in the path.

        Args:
            schema (dict): dictionary representing the schema of the dataset.
                See Pathing for more details.
            d_model (int): The dimension of the embeddings.
        """
        super().__init__(schema, d_model, embedding_type="word")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.embeddings(self.paths)
        embs[self.paths == 0, :] = 0  # set the root embedding to 0
        # TODO ensure field order is respected
        pos = torch.sum(embs, dim=1)  # sum over the words in the path
        return x * self.scale + pos[: x.shape[1], :]


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model, dropout=0.1, max_len=5000, random=False, trainable=False
    ):
        """Positional encodings from Vaswani et al. 2017."""
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([self.d_model]))
        self.pe = torch.zeros(max_len, self.d_model, dtype=torch.get_default_dtype())

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2)
            * (-torch.log(torch.tensor([10000])) / self.d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.random = random

        self.pe = nn.Parameter(self.pe, requires_grad=trainable)
        self.scale = nn.Parameter(self.scale, requires_grad=trainable)

    def forward(self, x):
        if self.random:
            indices = torch.randint(0, self.pe.shape[0], (x.shape[1],))
            indices = indices.sort()[0]
            x = x * self.scale + self.pe[indices, :]
        else:
            x = x * self.scale + self.pe[: x.shape[1], :]
        return self.dropout(x)  # (batch_size, seq_len, d_model)

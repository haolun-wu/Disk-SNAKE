# %%
"""Here we will implement the positional encodings for the KBGEN model.
Unlike the Vaswani et al original paper, we will use the positional encodings in
a hierarchical tree structure.
The idea is that the positional encoding of a node in the tree is determined by the path
from the root to the node.
"""
import torch
from kbgen.utils import schema as schema_utils
from kbgen.model.positional_encodings import (
    IndepedentPathing,
    RNNPathing,
    BagPathing,
)

# First we will generate the tree structure for the dataset. Or the schema.
# The schema is a dictionary of dictionaries.
# The keys are either the names of the nodes or their type ids.
# Type ids are as follows: 0 for string, 1 for number, 2 for date composite type.
schema = {
    "person": {
        "name": 0,
        "height": 1,
        "dob": {
            2: {"year": 1, "month": 1, "day": 1},
        },
    }
}


# %%
# Example usage
schema_ids = schema_utils.get_ids(schema)
print(schema_ids)
# %%

"""
Now we will generate the positional encodings for the schema.
The module will take the schema as input and generate the positional encodings.
"""
d_model = 3
pe = IndepedentPathing(schema_ids, d_model)
pe_rnn = RNNPathing(schema_ids, d_model)
pe_bag = BagPathing(schema_ids, d_model)
# let's do some tests:
# 1. check that the positional encodings are the same for the same path
# 2. check that the positional encodings are different for different paths

# %%
print(
    "Embeddings IndepedentPathing i.e. each nodes has an indepedent embedding:"
)
for schema_key, schema_id in schema_ids.items():
    print(f"{schema_id}  {str(pe(torch.tensor(schema_id))):<66} {schema_key}")
print("\n")
# %%
print(
    "Embeddings RNNPathing i.e. each nodes has an \
embedding based on the words along the path:"
)
print("Node ID, Word Sequence, RNN embedding, Node Key")
for schema_key, schema_id in schema_ids.items():
    print(
        f"{schema_id}  {pe_rnn.paths[schema_id]}  {pe_rnn(torch.tensor(schema_id))} {schema_key}"
    )
print(
    "Note that the word sequence has its own tokenization.\n\
This could be changed in future iterations where the encoder is the LLM itself.\n"
)
# %%
print(
    "Embeddings BagPathing i.e. each nodes has an embedding based on the \
bag of words along the path:"
)
for schema_key, schema_id in schema_ids.items():
    print(
        f"{schema_id}  {str(pe_bag(torch.tensor(schema_id))):<66} {schema_key}"
    )
print(
    "Currently, this is implemented as a sum over the word embeddings. \
But this could also be changed later.\n"
)
# %%

"""
This file is used to test the implementation of a MuP parameterized model.
We will use the same dataset as in the train.py file.
The goal is to produce the plots from coor_check in:
https://github.com/microsoft/mup/tree/main#checking-correctness-of-parametrization
"""
from kbgen.model import KBFormer
import torch
from kbgen.data.datasets import GSM
from kbgen.utils.cli import parse_args
from kbgen.utils.utils import mup_model

# DATA -----------------------------------------------------
dataset = GSM()
input_dict = dataset.tokenize()

config = parse_args(
    {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 1,
        "epochs": 2,
        "batch_size": 4096,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "dropout": 0.1,
        "train_mask_rate": 0.8,
        "eval_mask_rate": 0,
        "wandb": False,
        "tags": ["test"],
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "num_fields": len(dataset.df.columns),
        "vocab_size": len(dataset.tokenizer),
        "fields": dataset.fields,
        "field_order": dataset.field_order,
    }
)

model = mup_model(KBFormer, {"d_model": config["d_model"]}, config)

torch.manual_seed(config["seed"])

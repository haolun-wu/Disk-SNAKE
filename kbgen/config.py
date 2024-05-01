import os

rootdir = os.environ.get("KBGEN_LOGDIR", None)
if rootdir is None:
    raise ValueError(
        "Please set the KBGEN_LOGDIR environment variable "
        "to a valid path. For example, add the following "
        "line to your .bashrc or .zshrc file: "
        "export KBGEN_LOGDIR=/path/to/logdir"
    )


common_defaults = {
    "dataset": "gsm",  # gsm, nuclear, homedepot
    "d_model": 16,  # Model dimension, must be divisible by nhead
    "d_ff_mult": 2,  # Multiplier for the inner dim in feed forward layer
    "nhead": 4,  # Number of attention heads in Entity encoder + text model if custom
    "num_layers": 2,  # Entity encoder layers for entity Encoder
    "field_encoder_layers": 2,  # number of layers for field encoders
    "field_decoder_layers": 2,  # number of layers for field decoders
    "num_decoder_mixtures": 10,  # number of gaussian mixtures for the decoder for numerical fields
    "num_emb": "dice",  # Type of numerical embedding dice, periodic, binned
    "tie_numerical_embeddings": 0,  # Tie numerical embeddings across fields
    "tie_numerical_decoders": 0,  # Tie numerical decoders across fields
    "num_categorical_decoder_experts": 0,  # Tie categorical embeddings across fields with a MOE model (see modules.py),
    # 0 means individual decoders. n>0 is the number of experts used.
    "condition_decoders_on_hierarchy": 0,
    "tie_mask_embeddings": 0,  # Tie mask embeddings across fields (mask embs are what the entity encoder sees)
    "init_var": 0.02,  # Variance of initialization
    "epochs": 10,  # Number of epochs to train for
    "batch_size": 1024,  # Batch size
    "lr": 1e-2,  # Learning rate
    "weight_decay": 0.0,  # Weight decay
    "dropout": 0.1,  # Dropout
    "train_mask_rate": -1.0,  # Masking rate for properties during train
    "eval_mask_rate": 0.5,  # Masking rate for properties during eval
    "wandb": 0,  # Use wandb for logging (requires wandb login) otherwise nothing is logged
    "tags": ["test_tag"],  # Tags for wandb
    "device": "cpu",  # Device to use
    "seed": 42,  # Random seed used for model initialization and data shuffling
    "rootdir": rootdir,  # Root directory for logging and data
    "ckpt": "",  # Continue training from a checkpoint (must give run name)
    "model": "kbformer",  # Model to use (kbformer, decoder-only)
    "exp_name": "",
    "log_params": 0,  # Log model parameters and gradients to wandb
    "float_precision": "float32",  # float32, float16, float64
    "never_mask": [],  # Properties that are never masked
}

defaults_customLM = common_defaults.copy()
defaults_customLM.update(
    {
        "text_model": "custom",
        "tie_embeddings": 1,  # tie text model embeddings in the readout layer
        "tokenizer": "gpt2",  # tokenizer for the text model (gpt2, t5-small, custom)
        "text_decoder_layers": 2,  # number of layers for text decoders
        "text_encoder_layers": 2,  # number of layers for text encoders
        "encoder_readout": "none",  # use an LM readout layer on the encoder (none, tied, separate)
        "use_mup": 1,  # use mup scaling in the model architecture
    }
)


defaults_hf = common_defaults.copy()
defaults_hf.update(
    {
        "model": "t5",
        "tokenizer": "t5-small",
        "text_model": "t5-small",
        "freeze": False,
        "d_model": 16,
        "all_on_gpu": False,
        "use_mup": False,
    }
)

defaults_text = common_defaults.copy()
defaults_text.update(
    {
        "model": "decoder-only",  # Model to use (kbformer, decoder-only) decoder-only actually uses a transformer encoder
        "tokenizer": "simple",
        "encoder_readout": "separate",  # use an LM readout layer on the encoder (none, tied, separate)
        "use_mup": False,
    }
)

tabddpm_config = {
    "d_model": 16,  # Model dimension, must be divisible by nhead
    "d_ff_mult": 2,  # Multiplier for the inner dim in feed forward layer
    "nhead": 4,  # Number of attention heads in Entity encoder + text model if custom
    "num_layers": 2,  # Entity encoder layers for entity Encoder
    "field_encoder_layers": 2,  # number of layers for field encoders
    "field_decoder_layers": 2,  # number of layers for field decoders
    "num_decoder_mixtures": 10,  # number of gaussian mixtures for the decoder for numerical fields
    "num_emb": "dice",  # Type of numerical embedding dice, periodic, binned
    "tie_numerical_embeddings": 0,  # Tie numerical embeddings across fields
    "tie_numerical_decoders": 0,  # Tie numerical decoders across fields
    "num_categorical_decoder_experts": 0,  # Tie categorical embeddings across fields with a MOE model (see modules.py),
    # 0 means individual decoders. n>0 is the number of experts used.
    "condition_decoders_on_hierarchy": 0,
    "tie_mask_embeddings": 0,  # Tie mask embeddings across fields (mask embs are what the entity encoder sees)
    "epochs": 10,  # Number of epochs to train for
    "batch_size": 1024,  # Batch size
    "lr": 1e-2,  # Learning rate
    "weight_decay": 0.0,  # Weight decay
    "dropout": 0.1,  # Dropout
    "train_mask_rate": -1.0,  # Masking rate for properties during train
    "eval_mask_rate": 0.5,  # Masking rate for properties during eval
    "wandb": 0,  # Use wandb for logging (requires wandb login) otherwise nothing is logged
    "tags": ["test_tag"],  # Tags for wandb
    "device": "cpu",  # Device to use
    "seed": 42,  # Random seed used for model initialization and data shuffling
    "rootdir": rootdir,  # Root directory for logging and data
    "ckpt": "",  # Continue training from a checkpoint (must give run name)
    "model": "kbformer",  # Model to use (kbformer, decoder-only)
    "exp_name": "",
    "log_params": 0,  # Log model parameters and gradients to wandb
    "float_precision": "float32",  # float32, float16, float64
    "never_mask": [],  # Properties that are never masked
    "use_mup": 1,  # use mup scaling in the model architecture
    "tokenizer": "custom",
}

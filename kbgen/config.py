# rootdir = "/work/submit/kitouni/kbgen-logdir"
rootdir = "logdir"
common_defaults = {
    # "d_model": 8,               # Model dimension, must be divisible by nhead
    "d_model": 4,               # Model dimension, must be divisible by nhead
    "d_ff_mult": 2,             # Multiplier for the inner dim in feed forward layer
    "nhead": 2,                 # Number of attention heads in Entity encoder + text model if custom
    # "num_layers": 4,            # Entity encoder layers for entity Encoder
    "num_layers": 2,            # Entity encoder layers for entity Encoder
    "field_encoder_layers": 2,  # number of layers for field encoders
    "field_decoder_layers": 3,  # number of layers for field decoders
    "num_emb": "periodic",      # Type of numerical embedding dice, periodic, binned
    "tie_numerical_embeddings": False,  # Tie numerical embeddings across fields
    "tie_numerical_decoders": False,    # Tie numerical decoders across fields
    "tie_mask_embeddings": True,        # Tie mask embeddings across fields (mask embs are what the entity encoder sees)
    # "epochs": 1000,          # Number of epochs to train for
    "epochs": 10,          # Number of epochs to train for
    "batch_size": 64,        # Batch size
    "lr": 1e-4,              # Learning rate
    "weight_decay": 0,       # Weight decay
    "dropout": 0.0,          # Dropout
    "mask_rate": (-1, 0.5),  # Masking rate for properties, (train_mask_rate, eval_mask_rate)
    "wandb": False,          # Use wandb for logging (requires wandb login) otherwise nothing is logged
    "tags": ["test"],        # Tags for wandb
    "device": "cuda:0",      # Device to use
    "seed": 42,              # Random seed
    "rootdir": rootdir,      # Root directory for logging and data
    "ckpt": "",            # Continue training from a checkpoint (must give run name)
}

defaults_customLM = common_defaults.copy()
defaults_customLM.update(
    {
        "text_model": "custom",
        "tie_embeddings": True, # tie text model embeddings in the readout layer
        "tokenizer": "gpt2",    # tokenizer for the text model (gpt2, t5-small, custom)
        "text_decoder_layers": 4,   # number of layers for text decoders
        "text_encoder_layers": 4,   # number of layers for text encoders
        "use_mup": True,        # use mup scaling in the model architecture
    }
)

defaults_hf = common_defaults.copy()
defaults_hf.update(
    {
        "tokenizer": "t5-small",
        "text_model": "t5-small",
        "freeze": False,
        "d_model": 512,
        "all_on_gpu": False,
        "use_mup": False,
    }
)

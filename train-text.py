# in this notebook we will convert GSM data with schema to json in raw text

# %%
from kbgen.config import defaults_text as config
from kbgen.utils.cli import parse_args
from kbgen.data.datasets import GSM
import torch
from kbgen.data.datasets import DataLoader
import tqdm
from kbgen.model.modules import TextEncoder
from kbgen.utils.log import RunTracker
import math

config = parse_args(config)

# DATA -----------------------------------------------------
device = config["device"] if torch.cuda.is_available() else "cpu"
dataset = GSM.from_config_(config, update=True)
print("Config: ", config)
STRING_COLLECTION = dataset.as_strings()
torch.manual_seed(config["seed"])

# %%
tokenizer = dataset.tokenizer
tokens, padding_mask = tokenizer(STRING_COLLECTION).values()
padding_mask = (padding_mask == 0).float().masked_fill(padding_mask == 0, float("-inf"))

tokens = tokens.to(device)
padding_mask = padding_mask.to(device)

# %%
train_loader = DataLoader(
    dataset.train_idx,
    batch_size=config["batch_size"],
    shuffle=True,
)
val_loader = DataLoader(
    dataset.val_idx,
    batch_size=config["batch_size"],
    shuffle=False,
)

model = TextEncoder(config).to(device)

# %%
ce = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["lr"], max_lr=5*config["lr"], step_size_up=100, cycle_momentum=False)

logger = RunTracker(config).init_run()
pbar = tqdm.trange(config["epochs"]) if not config["wandb"] else range(config["epochs"])

for epoch in pbar:
    for batch in train_loader:
        x = tokens[batch]
        pad_mask = padding_mask[batch]
        # x is (batch_size, seq_len)
        optimizer.zero_grad()
        logits = model.encode(x, padding_mask=pad_mask, shift_right=True, is_causal=True)
        # logits is (batch_size, seq_len, vocab_size)∏
        # transpose to (batch_size, vocab_size, seq_len)
        logits = logits.transpose(1, 2)
        loss = ce(logits, x)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if not config["wandb"]:
            pbar.set_description_str(f"loss: {loss.item():.2e}")
    logger.log({"loss": loss.item()}, "train", epoch)
    for batch in val_loader:
        x = tokens[batch]
        pad_mask = padding_mask[batch]
        logits = model.encode(x, padding_mask=pad_mask, shift_right=True, is_causal=True)
        loss = ce(logits.transpose(1, 2), x)
        if not config["wandb"]:
            pbar.set_postfix_str(f"val_loss: {loss.item():.2e}")
    logger.log({"loss": loss.item()}, "val", epoch)
    if ((epoch) % (math.ceil(config.epochs/5)) == 0) or (epoch == config.epochs - 1 or epoch == 0):
            logger.save_model(model, f"{epoch}.pt")

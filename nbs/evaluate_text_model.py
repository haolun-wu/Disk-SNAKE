
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from kbgen.config import rootdir, defaults_text as config
from kbgen.utils.cli import parse_args
from kbgen.data.datasets import GSM
import torch
from kbgen.data.datasets import DataLoader
from kbgen.utils.tokenizer import GPT2Tokenizer
import tqdm
from kbgen.model.modules import TextEncoder
from kbgen.utils.log import RunTracker

# DATA -----------------------------------------------------
# device = config["device"] if torch.cuda.is_available() else "cpu"
device = "cpu"
dataset = GSM.from_config_(config, update=True)
print("Config: ", config)
STRING_COLLECTION = dataset.as_strings()

tokenizer = dataset.tokenizer
tokens, pad_mask = tokenizer(STRING_COLLECTION).values()
pad_mask = (pad_mask == 0).float().masked_fill(pad_mask == 0, float("-inf"))

stringfy = dataset.stringify

STRING_COLLECTION[0]

# for string in STRING_COLLECTION:
#   for txt in tokenizer(string)['input_ids'].view(-1, 1):
#     element = tokenizer.decode(txt)
#     if ("." in element or ":" in element or "|" in element) and len(element.strip()) > 1:
#       if element != "<|endoftext|>":
#         print("failed:", element.replace(" ", "W"))
#         print(string)

# tracker = RunTracker.from_logdir(rootdir + "/models/09-06-07-34-19train-textdecoder-only_l4_d256")
#tracker = RunTracker.from_logdir("/checkpoint/nolte/kbgen/models/09-07-07-14-31train-textdecoder-only_l4_d256")
tracker = RunTracker.from_logdir("/checkpoint/nolte/kbgen/models/09-07-07-55-09train-textdecoder-only_l4_d256")

model = tracker.load_latest_model()

input_ = dataset.as_strings()[dataset.train_idx[0]]
print([tokenizer.decode(i, skip_special_tokens=False) for i in tokenizer([input_])["input_ids"].view(-1, 1)])
text = tokens[[dataset.train_idx[0]]]
pad_mask_text = pad_mask[[dataset.train_idx[0]]]
print([tokenizer.decode(i, skip_special_tokens=False) for i in model.encode(text, padding_mask=pad_mask_text, is_causal=True, shift_right=True).argmax(-1).view(-1, 1)])
breakpoint()

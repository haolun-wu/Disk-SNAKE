import torch
import sys
import os 
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)

from kbgen.utils.utils import get_gpu_status
from kbgen.model.modules import TextModule

config = {
    "vocab_size": 50000,
    "d_model": 8,
    "dropout": 0.0,
    "nhead": 2,
    "text_encoder_layers": 2,
    "text_decoder_layers": 2,
    "text_model": "custom",
}


torch.set_default_device("cuda")


get_gpu_status()
# Load the model
model = TextModule(config)
total = 0
for name, param in model.named_parameters():
    print(name, param.shape, param.numel())
    if not param.requires_grad:
        print("^ not requires grad")
    total += param.numel() * param.element_size()

print(
    "total number of parameters with grad:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)
print("model memory usage:", total / 1e6, "MB")
model.train()
get_gpu_status()

# Generate some data
print("Generating data...")
batch_size = 4096
max_len = 10
x = torch.randint(
    0, config["vocab_size"]//10, (batch_size, max_len)
)
attention_mask = torch.rand(batch_size, max_len)
attention_mask.masked_fill_(attention_mask < 0.5, -torch.inf)
attention_mask.masked_fill_(attention_mask >= 0.5, 0.0)
total = (
    x.nelement() * x.element_size()
    + attention_mask.nelement() * attention_mask.element_size()
)
print("data memory usage:", total / 1e6, "MB")
get_gpu_status()

# Run the model
print("Running model...")
# enc_output = model.encoder(x, attention_mask=attention_mask)
enc_output = model.encoder(x, attention_mask=None)
print("enc_output", enc_output.shape)
get_gpu_status()
x = model._shift_right(x, inplace=False)
attention_mask = model._shift_right(attention_mask, inplace=False)
# dec_output = model.decoder(x, enc_output[:, :1], attention_mask=attention_mask)
dec_output = model.decoder(x, enc_output[:, :1], attention_mask=None)
print("dec_output", dec_output.shape)
get_gpu_status()
print("Loss...")
loss = torch.nn.functional.cross_entropy(
    dec_output.view(-1, config["vocab_size"]), x.view(-1)
)
get_gpu_status()
print("Backward...")
loss.backward()
get_gpu_status()

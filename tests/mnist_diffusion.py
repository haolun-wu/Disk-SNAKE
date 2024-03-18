# %%
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from kbgen.model.modules import _ResBlock
from kbgen.config import rootdir
from kbgen.diffusion import DiscreteDiffusion


class EncDec(nn.Module):
    def encode(self, x):
        """Encode a batch of inputs to the range [1, 2]"""
        # add one because 0 is reserved for masking
        # normalize to [1, 0]
        x = (x + 1) / (256 + 1) + 1
        return x


net = torch.hub.load(
    "milesial/Pytorch-UNet", "unet_carvana", pretrained=False, scale=0.5
)

class Model(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.readout = nn.Conv2d(2, 256, 1, 1)
        
    def forward(self, x):
        x = self.net(x) # [batch, 3, 32, 32]
        x = self.readout(x) # [batch, 256, 32, 32]
        return x
model = Model(net)
# %%
# parser = ArgumentParser()
# parser.add_argument("--device", default="cuda")
# parser.add_argument("--retrain", action="store_true")
# parser.add_argument("--epochs", type=int, default=0)
# args = parser.parse_args()
args = type("args", (), {"device": "cuda", "retrain": False, "epochs": 0})()
dm = DiscreteDiffusion(model, EncDec(), input_shape=(3, 32, 32))
dm.to(args.device).train()


def preprocess(x):
    x = torch.from_numpy(np.array(x)).to(args.device)
    x = F.pad(x, (2, 2, 2, 2), value=0).unsqueeze(0)
    x = x.repeat(3, 1, 1)
    x[1:] = 0
    return x


train_data = MNIST(
    root=os.path.join(rootdir, "data"),
    train=True,
    download=True,
    transform=transforms.Lambda(lambda x: preprocess(x)),
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# %%
optimizer = optim.Adam(dm.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
mask_rate_schedule = lambda epoch: (epoch / args.epochs) ** 2

file = os.path.join(rootdir, "models/dm.pt")
if not args.retrain and os.path.exists(file):
    try:
        dm.model.load_state_dict(torch.load(file))
        print("Loaded model from models/dm.pt")
    except:
        print("Failed to load model, starting from scratch")
if args.epochs > 0:
    print("Training...")
    for epoch in (pbar := tqdm(range(args.epochs))):
        for x, y in train_loader:
            optimizer.zero_grad()
            rate = mask_rate_schedule(epoch)
            loss = dm.loss(x, rate=rate)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}: {loss.item():.3f}")
        torch.save(dm.model.state_dict(), file)
        print("Saved model to models/dm.pt")

print("Generating samples...")
dm.model.eval()
os.makedirs(os.path.join(rootdir, "images"), exist_ok=True)
for i in tqdm(range(10)):
    samples = dm.generate_sample(n=1, leaps=5, undo=False, temperature=0)
    for j, sample in enumerate(samples):
        plt.subplot(2, 5, j + 1)
        plt.imshow(sample[0].cpu().numpy())
        plt.axis("off")
    plt.tight_layout(h_pad=0, w_pad=0)
    filepath = os.path.join(rootdir, f"images/dm_{i}.png")
    plt.savefig(filepath)
    plt.close()

# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots()
ax.axis("off")
im = ax.imshow(np.zeros((32, 32, 3)), animated=True)


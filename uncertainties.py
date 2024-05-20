# %%
import torch
from kbgen.data.datasets import GSM, NUCLEAR
from kbgen.utils.log import RunTracker
from kbgen.utils import TensorDict, Fields
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from kbgen.Trainer import Trainer

# %%
#run = RunTracker.from_logdir(name="09-23-15-26-17fixed_gaussian-longer_training-save_10_times-puymd_L2td2_te2_d256gsm", force_device="cpu")# <- good model
#run = RunTracker.from_logdir(name="09-23-15-26-08fixed_gaussian-longer_training-save_10_times-gOhAy_L2td2_te2_d256nuclr", force_device="cpu")# <- good model
run = RunTracker.from_logdir(name="09-24-20-40-55nuclr_all_masking-YpZZR_L2td2_te2_d256nuclr", force_device="cpu")# <- good model
print(run.config)


run.config["wandb"] = 0
model = run.load_latest_model().eval()
trainer = Trainer(run.config)
trainer.model.load_state_dict(model.state_dict())
del model
# %%
dataset = trainer.dataset
print(dataset.fields)

# %%

tokens = dataset.input_dict.iloc[dataset.val_idx].to(run.config["device"])
pad_mask = dataset.pad_mask_dict.iloc[dataset.val_idx].to(run.config["device"])

trainer.model.eval()
property_mask = trainer.model._sample_property_mask(tokens, 0.)
property_mask = torch.zeros_like(property_mask)
property_mask[:, dataset.fields.all_fields.index("binding_semf")] = float("-inf")

# make predictions
with torch.no_grad():
    binding_preds = trainer.model.get_probabilistic_params(tokens, pad_mask, property_mask)["binding_semf"]
# %%
results = defaultdict(list)
for idx in tqdm.trange(len(tokens["binding_semf"])):
  # eval likelihood over a range
  truth = tokens["binding_semf"][idx]
  if truth == -1000:
    continue
  range_min = max(0,  - .2)
  range_max = min(1, truth + .2)
  input_ = torch.linspace(range_min, range_max, 200).view(-1, 1)
  loglikelihoods = -torch.cat([trainer.model.gmm_loss(binding_preds[[idx]], i) for i in input_])
  results["truth"].append(truth.item())
  results["LL"].append(loglikelihoods)
  results["input"].append(input_)

# %%
# make a plot: coverage as a function of likelihood range
threshold_range = np.linspace(0.2, .9, 30)
coverage = []
for threshold in threshold_range:
    coverage += [[]]
    for truth, LL, input_ in zip(results["truth"], results["LL"], results["input"]):
        good_idxs = torch.where(LL >  LL.max() * threshold)[0][[0,-1]]
        coverage[-1].append((truth > input_[good_idxs[0]][0]).item() and (truth < input_[good_idxs[-1]][0]).item())
    coverage[-1] = np.mean(coverage[-1])
# %%

plt.plot(threshold_range, coverage)
plt.xlabel("Interval includes >X * max likelihood")
plt.ylabel("Coverage")
plt.show()

# %%
selection = tokens["binding_semf"] != -1000
results["prediction"] = trainer.model.gmm_loss.estimate_mode(binding_preds).squeeze().cpu().numpy()[selection]

# %%
len(results["prediction"]), len(results["truth"])
# %%
for idx in range(3,4):
  input_ = results["input"][idx]
  truth = results["truth"][idx]
  pred = results["prediction"][idx]
  LL = results["LL"][idx]
  plt.figure(figsize=(20, 10))
  plt.plot(input_, LL)
  plt.axvline(truth, alpha=1, color="r")
  plt.axvline(pred, alpha=1, color="purple")
  good_idxs = torch.where(LL >  LL.max() * .5)[0][[0,-1]]
  # shaded region between good_idxs
  plt.axvspan(input_[good_idxs[0]][0], input_[good_idxs[-1]][0], alpha=.5, color="g")
  plt.show()

# %%
import json
with open("results.json", "w") as f:
  json.dump({x: list(y) for x, y in results.items()}, f)

# %%
# %%

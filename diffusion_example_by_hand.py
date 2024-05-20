# %%
import torch
from kbgen.utils.log import RunTracker
from kbgen.utils import TensorDict
import tqdm
from kbgen.Trainer import Trainer
from kbgen.diffusion import HybridDiffusion

# %%
run = RunTracker.from_logdir(
    name="09-23-15-26-17fixed_gaussian-longer_training-save_10_times-puymd_L2td2_te2_d256gsm",
    force_device="cuda",
)
run.config["wandb"] = 0
model = run.load_latest_model().eval()
trainer = Trainer(run.config)
trainer.model.load_state_dict(model.state_dict())
del model

# manually step through diffusion
diffusion = HybridDiffusion(trainer.model, trainer.dataset)

# %%
kpm = trainer.dataset.pad_mask_dict.iloc[trainer.dataset.val_idx]
targets = trainer.dataset.input_dict.iloc[trainer.dataset.val_idx]
# %%
from collections import defaultdict


#evaluate errors when running different masking rates over 5 seeds

errors_m_all = defaultdict(lambda: defaultdict(list))
for masking_rate in tqdm.tqdm(torch.linspace(0, 1, 11)):
    for seed in range(5):
        initial_mask = (
            diffusion.model._sample_property_mask(
                targets,
                mask_rate=masking_rate,
                seed=seed,
            )
            .bool()
            .logical_not()
        )

        diffusion.model.eval()
        out = diffusion.sample(
            n=1, cond=targets, kpm=kpm, mask=initial_mask, leaps=100, temperature=0.0
        )

        mask_dict = TensorDict(
            {
                field: (~initial_mask[:, idx])
                for (idx, field) in enumerate(trainer.dataset.fields.all_fields)
            }
        )

        errors_m = {
            k: v
            for k, v in trainer.model.accuracy(
                out, targets, mask_dict, True, trainer.dataset
            ).items()
        }

        for field in mask_dict.keys():
            errors_m_all[masking_rate.item()][field].append(errors_m[field].item())
# %%

errors_m_all
# %%
import json

with open("errors_all.json", "w") as f:
    json.dump(errors_m_all, f)
# %%

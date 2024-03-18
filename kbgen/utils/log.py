import wandb
from datetime import datetime
import torch
import json
import os
from kbgen.utils.cli import NamespaceDict
from kbgen.utils import Fields
from kbgen.model import KBFormer
from kbgen.utils.utils import mup_model


class RunTracker:
    def __init__(self, config: NamespaceDict) -> None:
        self.config = config
        self.logdir = None
        self.model = None

    def init_run(self):
        run_name = generate_name(self.config)
        self.logdir = os.path.join(self.config.rootdir, f"models/{run_name}")

        if self.config["wandb"]:
            wandb.init(
                project="kb-generator",
                config=self.config,
                tags=self.config["tags"],
                name=run_name,
            )
            artifact = wandb.Artifact("kbgen", type="code")
            artifact.add_file("train.py")
            artifact.add_dir("kbgen", name="kbgen")
            wandb.log_artifact(artifact)
            os.makedirs(self.logdir, exist_ok=True)
            json.dump(self.config, open(f"{self.logdir}/config.json", "w"))
            print(f"Dumped config to {self.logdir}/config.json")
        else:
            print(
                "Not using wandb. RunTracker will not save any models or log metrics."
                " To use wandb, set wandb=True in config.py"
                " or use the --wandb flag in train.py."
            )
        return self

    @classmethod
    def from_logdir(cls, logdir, force_device=None):
        config = NamespaceDict(
            json.load(open(os.path.join(logdir, "config.json"), "r"))
        )
        if force_device is not None:
            config.device = force_device
        config.fields = Fields(
            numerical=config["fields"]["numerical"],
            categorical=config["fields"]["categorical"],
            text=config["fields"]["text"],
        )
        self = cls(config)
        self.logdir = logdir
        if self.config["use_mup"]:
            self.model = mup_model(KBFormer, {"d_model": 8}, self.config)
        else:
            self.model = KBFormer(self.config)
        return self

    def load_model(self, epoch: int) -> KBFormer:
        if self.logdir is None or self.model is None:
            raise ValueError(
                "Cannot load model from logdir. First call init_run() or use from_logdir()."
            )
        model = os.path.join(self.logdir, f"{epoch}.pt")
        if not os.path.exists(model):
            raise ValueError(
                f"Model {model} does not exist. Use run.available_epochs() to see available epochs."
            )
        print(f"Loading model from: {model}")
        self.model.load_state_dict(torch.load(model))
        return self.model

    def available_epochs(self) -> list:
        return [
            int(f.split(".")[0]) for f in os.listdir(self.logdir) if f.endswith(".pt")
        ]

    def load_latest_model(self) -> KBFormer:
        return self.load_model(max(self.available_epochs()))

    def save_model(self, model, file):
        if self.logdir is None:
            raise ValueError(
                "Cannot save model to logdir. First call init_run() or use from_logdir()."
            )
        if not self.config["wandb"]:
            return
        torch.save(model.state_dict(), os.path.join(self.logdir, file))
        print(f"Saved model to {self.logdir}/{file}")
        artifact = wandb.Artifact("kbgen", type="model")
        artifact.add_file(os.path.join(self.logdir, file))

    def log(self, metrics, prefix, step):
        if not self.config["wandb"]:
            return
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_outputs(self, output, step, prefix=None):
        prefix = prefix if prefix is not None else ""
        self.log(output.loss_dict, f"{prefix}/loss", step=step)
        self.log(output.masked_loss_dict, f"{prefix}/masked/loss", step=step)
        self.log(output.unmasked_loss_dict, f"{prefix}/unmasked/loss", step=step)
        self.log(output.masked_error_dict, f"{prefix}/masked/error", step=step)
        self.log(output.unmasked_error_dict, f"{prefix}/unmasked/error", step=step)


def generate_name(config: NamespaceDict) -> str:
    name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    name += "-".join(config.tags)
    if config["text_model"] == "custom":
        name += f"_L{config.num_layers}td{config.text_decoder_layers}"
        name += f"_te{config.text_encoder_layers}"
    else:
        name += f"_{config.text_model}"
    name += f"_d{config.d_model}"
    name += f"_{config.num_emb}"
    if config.tie_numerical_embeddings:
        name += "_tied"
    if config.get("num_target_scalar", False):
        name += "_scalar"
    return name

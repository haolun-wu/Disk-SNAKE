import wandb
from datetime import datetime
import torch
import json
import random
import string
import os
from loguru import logger
from kbgen.config import rootdir
from kbgen.model.model import KBFormer
from kbgen.utils import mup_model
from ..utils.cli import NamespaceDict
from ..utils import Fields, mup_model
from ..model import KBFormer, TextEncoder
from ..config import rootdir
import sys


class RunTracker:
    def __init__(self, config: NamespaceDict) -> None:
        os.environ["WANDB_DATA_DIR"] = os.path.join(rootdir, ".local/wandb_data")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(rootdir, ".local/wandb_cache")
        self.config = config
        self.logdir = None
        self.model = None

    def init_run(self):
        run_name = generate_name(self.config)
        self.logdir = os.path.join(self.config.rootdir, f"models/{run_name}")

        if self.config["wandb"]:
            wandb.init(
                project="DiSK-Set-Efficient-{}-data{}-e{}".format(
                    self.config["dataset"],
                    self.config["num_data"],
                    self.config["epochs"],
                ),
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
            logger.info(f"Dumped config to {self.logdir}/config.json")
        else:
            logger.info(
                "Not using wandb. RunTracker will not save any models or log metrics."
                " To use wandb, set wandb=True in config.py"
                " or set --wandb 1 with train.py."
            )
        return self

    @classmethod
    def from_logdir(cls, logdir=None, name=None, force_device=None):
        if logdir is None and name is None:
            raise ValueError("Either logdir or name must be specified.")
        elif logdir is None:
            logdir = os.path.join(rootdir, "models", name)

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
        self.model = load_model_from_config(self.config)
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
        logger.info(f"Loading model from: {model}")
        self.model.load_state_dict(torch.load(model))
        return self.model.to(self.config["device"])

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
        logger.info(f"Saved model to {self.logdir}/{file}")
        artifact = wandb.Artifact("kbgen", type="model")
        artifact.add_file(os.path.join(self.logdir, file))

    def save_and_exit(self, signal_number, frame, model, file):
        file = file() if callable(file) else file
        logger.info("\nSaving the model before exiting...")
        # pbar get current step
        self.save_model(model, file)
        sys.exit(0)

    def log(self, metrics, prefix, step):
        if not self.config["wandb"]:
            return
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_outputs(self, output, step, prefix=None):
        prefix = prefix if prefix is not None else ""
        self.log(output.loss_dict, f"{prefix}/loss", step=step)
        self.log(output.error_dict, f"{prefix}/metric", step=step)
        if self.config["log_params"]:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.log({name: param.grad.norm()}, step=step, prefix="grad")
                    self.log({name: param.norm()}, step=step, prefix="param")


def random_id():
    return "".join([random.choice(string.ascii_letters) for _ in range(5)])


def generate_name(config: NamespaceDict) -> str:
    name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    name = ""
    name += "-".join(config.tags)
    # add random string to avoid name collisions
    # name += f"-{random_id()}"
    if config["text_model"] == "custom":
        name += f"_L{config.num_layers}td{config.text_decoder_layers}"
        name += f"_te{config.text_encoder_layers}"
    else:
        name += f"_{config.text_model}"
    name += f"_n{config.num_data}"
    name += f"_e{config.epochs}"
    name += f"_d{config.d_model}"
    name += f"_b{config.batch_size}"
    name += f"_lr{config.lr}"
    name += f"_do{config.dropout}"
    name += f"_gt_{config.gt_choose}"
    name += f"_{config.dataset}"
    return name


def load_model_from_config(config: NamespaceDict):
    """Set up model from config. If config["use_mup"] is True, use the mup_model wrapper to scale the model up."""
    model_callable = KBFormer if config["model"] == "kbformer" else TextEncoder
    if config["use_mup"]:
        return mup_model(model_callable, {"d_model": 8}, config)
    else:
        return model_callable(config)


def init_model_from_config_(config: NamespaceDict):
    """Initialize the right model from config. If config["ckpt"] is not empty, load the model from the checkpoint."""
    # config["ckpt"] needs to be a valid run name saved in rootdir/models
    if config["ckpt"] != "":
        logdir = os.path.join(rootdir, "models", config["ckpt"])
        run = RunTracker.from_logdir(logdir)
        logger.info("Overwriting config with loaded config")
        for key in [
            "ckpt",
            "tags",
            "wandb",
            "epochs",
            "lr",
            "weight_decay",
            "train_mask_rate",
            "eval_mask_rate",
        ]:
            run.config.pop(key)
        config.update(run.config)
        logger.info("Loading model...")
        model = run.load_latest_model()
        model.config = config
    else:
        model = load_model_from_config(config)
    return model

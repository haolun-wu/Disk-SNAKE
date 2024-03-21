from kbgen.config import rootdir, defaults_customLM as config
from kbgen.data.datasets import GSM
from kbgen.model import KBFormer
from kbgen.utils.cli import parse_args
from kbgen.utils.log import RunTracker
from kbgen.utils.utils import mup_model
import torch
from torch.optim import AdamW
import tqdm
from mup.optim import MuAdamW
import os


# DATA -----------------------------------------------------
device = config["device"] if torch.cuda.is_available() else "cpu"
dataset = GSM.from_config(config, update=True)


if __name__ == "__main__":
    # parse_args parses the command line arguments and returns a dictionary
    # the defaults for the arguments are given as a dictionary below
    # they are overwritten by the command line arguments of the same name
    config = parse_args(config)

    # Load all data to GPU  (because we can with GSM) ----------------
    input_dict = dataset.input_dict.to(device)
    print("input_dict:", input_dict.keys())
    print("input_dict:", input_dict['phone.weight'])
    pad_mask_dict = dataset.pad_mask_dict.to(device)
    print("pad_mask_dict:", pad_mask_dict.keys())
    print("pad_mask_dict:", pad_mask_dict['phone.weight'])

    trainloader, testloader = dataset.get_loaders(config.batch_size)
    # Init model -----------------------------------------------------
    torch.manual_seed(config.seed)
    if config["ckpt"] != "":
        logdir = os.path.join(rootdir, "models", config["ckpt"])
        run = RunTracker.from_logdir(logdir)
        print("Overwriting config with loaded config")
        for key in ["ckpt", "tags", "wandb", "epochs", "lr", "weight_decay"]:
            run.config.pop(key)
        config.update(run.config)
        print("Loading model...")
        model = run.load_latest_model()
    else:
        model = (
            mup_model(KBFormer, {"d_model": 8}, config)
            if config["use_mup"]
            else KBFormer(config)
        )
    model = model.to(device)
    opt = MuAdamW if config["use_mup"] else AdamW
    optimizer = opt(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["epochs"] // 2 + config["epochs"] % 2, T_mult=1
    )

    # Train -----------------------------------------------------
    logger = RunTracker(config).init_run()
    pbar = tqdm.trange(config.epochs, dynamic_ncols=True)
    step = 0
    for epoch in pbar:
        model.train()
        for batch_idx, batch in enumerate(trainloader):
            # torch.cuda.empty_cache()
            step += 1
            optimizer.zero_grad()
            output = model.apply(input_dict.iloc(batch), pad_mask_dict.iloc(batch))
            output.loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(trainloader))
            pbar.set_description_str(f"Loss: {output.loss.item():^6.3f}")
            logger.log_outputs(output, step, prefix="train")

        with torch.no_grad():
            for batch in testloader:
                model.eval()
                output = model.apply(
                    input_dict.iloc(batch),
                    pad_mask_dict.iloc(batch),
                    eval_mode=True,
                )
                errors_string = [
                    f"{k.split('.')[-1]}{v:^4.1e}"
                    for k, v in output.unmasked_error_dict.items()
                ]
                pbar.set_postfix_str(
                    f"Val L:{output.loss.item():^4.3f}" + "|".join(errors_string)
                )
                logger.log_outputs(output, step, prefix="val")
        # if ((epoch) & (epoch - 1) == 0 and epoch > 100) or (
        if ((epoch) % 50 == 0) or (epoch == config.epochs - 1 or epoch == 0):
            logger.save_model(model, f"{epoch}.pt")

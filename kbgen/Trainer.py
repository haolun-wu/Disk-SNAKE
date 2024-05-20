from kbgen.data.datasets import load_dataset
from kbgen.utils import trim_padding_
from kbgen.utils.log import RunTracker, init_model_from_config_
from kbgen.utils.metrics import AggregatedMetrics
import torch
import tqdm
from mup.optim import MuAdamW
from torch.optim import AdamW
import signal
from functools import partial


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.save_step = max(config.epochs // 10, 1)  # Save five total models
        # set precision -----------------------------------------------------
        torch.set_default_dtype(getattr(torch, config.float_precision))
        # DATA -----------------------------------------------------
        device = config["device"] if torch.cuda.is_available() else "cpu"
        self.dataset = load_dataset(config)

        # Load all data to GPU  (because we can with most of these datasets) ----------------
        self.input_dict = self.dataset.input_dict.to(device)
        self.pad_mask_dict = self.dataset.pad_mask_dict.to(device)

        self.trainloader, self.testloader = self.dataset.get_loaders(config.batch_size)
        # Init model -----------------------------------------------------
        torch.manual_seed(config.seed)
        self.model = init_model_from_config_(config).to(device)
        opt = MuAdamW if config["use_mup"] else AdamW
        self.optimizer = opt(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        n_steps = len(self.trainloader) * config.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, n_steps, eta_min=config.lr / 100
        )

        # Log -----------------------------------------------------
        # Graceful exit
        self.logger = RunTracker(config).init_run()
        self.status = {"epoch": 0}
        # save_and_exit = partial(
        #     self.logger.save_and_exit,
        #     model=self.model,
        #     file=lambda: f"{self.status['epoch']}.pt",
        # )
        # signal.signal(signal.SIGINT, save_and_exit)  # Handler for Ctrl+C (SIGINT)
        # signal.signal(signal.SIGTERM, save_and_exit)  # Handler for SIGTERM (kill)

    def train(self, epochs=None):
        epochs = epochs or self.config["epochs"]
        if self.config.wandb:
            pbar = range(epochs)
        else:
            pbar = tqdm.trange(epochs, dynamic_ncols=True)

        for epoch in pbar:
            # Train -----------------------------------------------------
            self.status["epoch"] = epoch
            self.model.train()
            for (batch,) in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model.apply(
                    *trim_padding_(
                        self.input_dict.iloc(batch),
                        self.pad_mask_dict.iloc(batch),
                        self.config,
                    )
                )
                output.loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if not self.config.wandb:
                pbar.set_description_str(f"Loss: {output.loss.item():^6.3e}")


            # Validation -----------------------------------------------------
            do_log_outputs = epoch % 10 == 0
            do_save_model = epoch % self.save_step == 0 or epoch == epochs - 1

            if do_save_model:
                self.logger.save_model(self.model, f"{epoch}.pt")

            if do_log_outputs:
              self.model.eval()
              for split in ["train", "val"]:
                  loader = self.trainloader if split == "train" else self.testloader

                  aggregator = AggregatedMetrics(self.config)
                  with torch.inference_mode():
                      for (batch,) in loader:
                          output = self.model.apply(
                              *trim_padding_(
                                  self.input_dict.iloc(batch),
                                  self.pad_mask_dict.iloc(batch),
                                  self.config,
                              ),
                              eval_mode=True,
                              unscale=True,
                              dataset=self.dataset,
                          )
                          aggregator.add_contribution(output)
                  output = aggregator.get_output()
                  errors_string = [
                      f"{k.split('.')[-1]}{v:^4.1e}"
                      for k, v in output.error_dict.items()
                  ]
                  if not self.config.wandb:
                      pbar.set_postfix_str(
                          f"Val L:{output.loss.item():^4.1e}" + "|".join(errors_string)
                      )

                  self.logger.log_outputs(output, epoch, prefix=split)

        print(
            "Finished successfully"
        )  # INDICATOR for stdouts to know that a run didn't fail

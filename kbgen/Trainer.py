from kbgen.data.datasets import load_dataset
from kbgen.utils import trim_padding_
from kbgen.utils.log import RunTracker, init_model_from_config_
from kbgen.utils.metrics import AggregatedMetrics
import torch
import tqdm
import wandb
from mup.optim import MuAdamW
from torch.optim import AdamW
import signal
from functools import partial


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.save_step = 100 # save for every 100 epochs
        # set precision -----------------------------------------------------
        torch.set_default_dtype(getattr(torch, config.float_precision))
        # DATA -----------------------------------------------------
        device = config["device"] if torch.cuda.is_available() else "cpu"
        print("device:", device)
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
            print("epoch:", epoch)
            # Train -----------------------------------------------------
            self.status["epoch"] = epoch
            self.model.train()
            batch_train_loss = []
            for (batch,) in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model.apply(
                    *trim_padding_(
                        self.input_dict.iloc(batch),
                        self.pad_mask_dict.iloc(batch),
                        self.config,
                    ),
                    eval_mode=False,
                    unscale=True,
                    dataset=self.dataset,
                    use_path_emb=self.config["use_path_emb"],
                )
                output.loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_train_loss.append(output.loss.item())

            # if not self.config.wandb:
            #     pbar.set_description_str(f"Loss: {output.loss.item():^6.3e}")
            mean_batch_train_loss = sum(batch_train_loss) / len(batch_train_loss)
            print(f"epoch: {epoch}, Training loss: {mean_batch_train_loss:^6.3e}")
            if self.config["wandb"]:
                wandb.log({"training_loss": mean_batch_train_loss}, step=epoch)

            # Validation -----------------------------------------------------
            do_log_outputs = epoch % 10 == 0
            do_save_model = epoch % self.save_step == 0 or epoch == epochs - 1

            if do_save_model:
                self.logger.save_model(self.model, f"{epoch}.pt")

            if do_log_outputs:
                self.model.eval()
                loss_global, metric_global = {}, {}
                for split in ["train", "val"]:
                    loader = self.trainloader if split == "train" else self.testloader
                    (
                        loss_full,
                        recall_list_full,
                        correct_list_full,
                        num_masks_list_full,
                        pred_seq_list_full,
                        target_seq_list_full,
                    ) = (0, [], [], [], [], [])
                    num_batch = 1

                    # aggregator = AggregatedMetrics(self.config)
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
                                use_path_emb=self.config["use_path_emb"],
                            )

                            (
                                recall_list_batch,
                                correct_list_batch,
                                num_masks_list_batch,
                                pred_seq_list_batch,
                                target_seq_list_batch,
                            ) = output.error_dict

                            loss_full += output.loss

                            recall_list_full.extend(recall_list_batch)
                            correct_list_full.extend(correct_list_batch)
                            num_masks_list_full.extend(num_masks_list_batch)
                            pred_seq_list_full.extend(pred_seq_list_batch)
                            target_seq_list_full.extend(target_seq_list_batch)

                            num_batch += 1

                    # Compute global loss and metric for training and val
                    loss_global[split] = (loss_full / num_batch).item()
                    metric_global[split] = sum(correct_list_full) / sum(
                        num_masks_list_full
                    )

                    # log the loss and metric
                    log_output_dict = {}
                    log_output_dict[f"inference on {split}/loss"] = loss_global[split]
                    log_output_dict[f"inference on {split}/metric"] = metric_global[split]

                    for k, v in log_output_dict.items():
                        print(f"{k}: {v}")

                    print("Decode prediction and targte per sample:")
                    for idx in range(len(pred_seq_list_full)):
                        pred = pred_seq_list_full[idx]
                        tgt = target_seq_list_full[idx]
                        pred_seq_list, tgt_seq_list = [], []

                        for pred_seq in pred:
                            if self.dataset.tokenizer.eos_token_id in pred_seq:
                                first_eos_occurence = pred_seq.eq(
                                    self.dataset.tokenizer.eos_token_id
                                ).nonzero(as_tuple=True)[0][0]
                            else:
                                first_eos_occurence = len(pred_seq)

                            text_pred_until_eos = pred_seq[:first_eos_occurence]
                            decoded_pred = self.dataset.tokenizer.decode(
                                text_pred_until_eos
                            )
                            pred_seq_list.append(decoded_pred)

                        for tgt_seq in tgt:
                            first_eos_occurence = tgt_seq.eq(
                                self.dataset.tokenizer.eos_token_id
                            ).nonzero(as_tuple=True)[0][0]
                            decoded_tgt = self.dataset.tokenizer.decode(
                                tgt_seq[:first_eos_occurence]
                            )
                            tgt_seq_list.append(decoded_tgt)

                        print("idx:", idx)
                        print("Prediction seq list:", pred_seq_list)
                        print("Target seq list:", tgt_seq_list)

                        # if self.dataset.tokenizer.eos_token_id in pred:
                        #     first_eos_occurence = pred.eq(self.dataset.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                        # else:
                        #     first_eos_occurence = len(pred)
                        # text_pred_until_eos = pred[:first_eos_occurence]
                        # decoded_pred = self.dataset.tokenizer.decode(text_pred_until_eos)
                        # first_eos_occurence = tgt.eq(self.dataset.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                        # decoded_tgt = self.dataset.tokenizer.decode(tgt[:first_eos_occurence])
                        # #word iou
                        # pred_words = set(decoded_pred.strip().split())
                        # tgt_words = set(decoded_tgt.strip().split())

                    if self.config["wandb"]:
                        wandb.log(log_output_dict, step=epoch)

                    # aggregator.add_contribution(output)
                    # output = aggregator.get_output()
                    # print("output:", output)
                    # errors_string = [
                    #     f"{k.split('.')[-1]}{v:^4.1e}"
                    #     for k, v in output.error_dict.items()
                    # ]
                    # if not self.config.wandb:
                    #     pbar.set_postfix_str(
                    #         f"Val L:{output.loss.item():^4.1e}"
                    #         + "|".join(errors_string)
                    #     )

                    # self.logger.log_outputs(output, epoch, prefix=split)

        print(
            "Finished successfully"
        )  # INDICATOR for stdouts to know that a run didn't fail

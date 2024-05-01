# %%
import torch
from kbgen.utils.log import RunTracker
from kbgen.utils import TensorDict, Fields
import numpy as np
import tqdm
from kbgen.Trainer import Trainer
import tqdm


def eval_run(run_name):
  run = RunTracker.from_logdir(name=run_name, force_device="cuda")
  run.config["wandb"] = 0
  model = run.load_latest_model().eval()
  trainer = Trainer(run.config)
  trainer.model.load_state_dict(model.state_dict())
  # print("numel: ", sum(p.numel() for n,p in trainer.model.named_parameters() if not "text" in n))
  del model
  dataset = trainer.dataset

  if "binding_semf" in dataset.fields["numerical"]:
      dataset._df["binding_semf_unscaled"] = dataset._df["binding_semf"] * (dataset._df["n"] + dataset._df["z"])

  tokens = dataset.input_dict.iloc[dataset.val_idx].to(run.config["device"])
  pad_mask = dataset.pad_mask_dict.iloc[dataset.val_idx].to(run.config["device"])

  trainer.model.eval()
  property_mask = trainer.model._sample_property_mask(tokens, 0.)
  losses = {}
  errs = {}
  for field in tqdm.tqdm(dataset.fields.all_fields):
      property_mask = torch.zeros_like(property_mask)
      property_mask[:, dataset.fields.all_fields.index(field)] = float("-inf")
      with torch.no_grad():
          outputs = trainer.model.get_probabilistic_params(tokens, pad_mask, property_mask)
          err = trainer.model.get_metrics_from_prob_params(outputs, tokens, property_mask, unscale=True, dataset=dataset)
          loss = trainer.model.get_loss_from_prob_params(outputs, tokens, property_mask)

          if field == "binding_semf":
              # do things manually:
              binding_pred = trainer.model._sample_field_with_temp(outputs[field], temp=0, field=field)
              z_pred = trainer.model._sample_field_with_temp(outputs["z"], temp=0, field="z")
              n_pred = trainer.model._sample_field_with_temp(outputs["n"], temp=0, field="n")

              td_fields = Fields(numerical=["binding_semf", "z", "n"], categorical=[], text=[])

              binding_pred = TensorDict({field: binding_pred, "z":z_pred, "n":n_pred}, fields=td_fields)
              mask_dict = TensorDict(
                  {
                      f: property_mask[:, trainer.config.fields.all_fields.index(f)].bool()
                      for f in ["binding_semf", "z", "n"]
                  }, fields=td_fields
              )
              tgt = TensorDict({field: tokens[field], "z": tokens["z"], "n": tokens["n"]}, fields=td_fields)

              preds_unscaled, tgt_unscaled = trainer.model.accuracy.unscale_for_metrics(binding_pred, tgt, dataset)

              preds_unscaled["binding_semf"] = preds_unscaled["binding_semf"] * (tgt_unscaled["n"].view(-1, 1) + tgt_unscaled["z"].view(-1, 1))
              tgt_unscaled["binding_semf"] = tgt_unscaled["binding_semf"] * (tgt_unscaled["n"] + tgt_unscaled["z"])
              tgt_unscaled["binding_semf"][tgt["binding_semf"] == -1000] = -1000
              err = trainer.model.accuracy(
                      preds_unscaled, tgt_unscaled, mask_dict
              )

          if field in dataset.fields["text"]:
              text_pred = trainer.model._sample_field_with_temp(outputs[field], temp=0, field=field, teacher_forcing=False)

              err = []
              for idx in range(len(text_pred)):
                pred = text_pred[idx]
                tgt = tokens[field][idx]
                if dataset.tokenizer.eos_token_id in pred:
                  first_eos_occurence = pred.eq(dataset.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                else:
                  first_eos_occurence = len(pred)
                text_pred_until_eos = pred[:first_eos_occurence]
                decoded_pred = trainer.dataset.tokenizer.decode(text_pred_until_eos)
                first_eos_occurence = tgt.eq(dataset.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                decoded_tgt = trainer.dataset.tokenizer.decode(tgt[:first_eos_occurence])
                #word iou
                pred_words = set(decoded_pred.strip().split())
                tgt_words = set(decoded_tgt.strip().split())
                err.append(len(pred_words & tgt_words) / len(pred_words | tgt_words))

              err = {field: np.mean(err)}


      losses[field] = loss[field].item()
      errs[field] = err[field].item()

  # print("err:")
  # print(*errs.items(), sep="\n")
  return errs

# give a list of run names
run_names = [
"09-28-11-43-43different_seeds_for_good_model_gsm_dataseed42-qcCZz_L2td2_te2_d256gsm",
"09-28-11-40-17different_seeds_for_good_model_gsm_dataseed42-yhDDU_L2td2_te2_d256gsm",
"09-28-11-40-19different_seeds_for_good_model_gsm_dataseed42-ixcYr_L2td2_te2_d256gsm",
"09-28-11-40-17different_seeds_for_good_model_gsm_dataseed42-kTEXC_L2td2_te2_d256gsm",
"09-28-11-40-19different_seeds_for_good_model_gsm_dataseed42-ZhFJF_L2td2_te2_d256gsm",
] #nhead 4, lr 0.001, periodic embs, wd 0.0, dropout 0.1, dataseed 42



#09-28-07-56-03different_seeds_for_good_model_gsm-zdzhu_L2td2_te2_d256gsm
errs = [eval_run(run_name) for run_name in run_names]

means = {}
stds = {}
for field in errs[0].keys():
  means[field] = np.mean([err[field] for err in errs])
  stds[field] = np.std([err[field] for err in errs])

print("means:")
print(*means.items(), sep="\n")
print("\n")
print("stds:")
print(*stds.items(), sep="\n")

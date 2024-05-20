# %%
import torch
from kbgen.utils.log import RunTracker
from kbgen.utils import TensorDict, Fields
import numpy as np
import tqdm
from kbgen.Trainer import Trainer
import tqdm
import os
import pandas as pd
# %%

def get_trainer(run_name):
  run = RunTracker.from_logdir(name=run_name, force_device="cuda")
  run.config["wandb"] = 0
  model = run.load_latest_model().eval()
  trainer = Trainer(run.config)
  trainer.model.load_state_dict(model.state_dict())
  return trainer

# %%
def eval_run(trainer):
  # print("numel: ", sum(p.numel() for n,p in trainer.model.named_parameters() if not "text" in n))
  dataset = trainer.dataset

  if "binding_semf" in dataset.fields["numerical"]:
      dataset._df["binding_semf_unscaled"] = dataset._df["binding_semf"] * (dataset._df["n"] + dataset._df["z"])

  tokens = dataset.input_dict.iloc[dataset.val_idx].to(trainer.config["device"])
  pad_mask = dataset.pad_mask_dict.iloc[dataset.val_idx].to(trainer.config["device"])

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

# %%
def create_df_from(paths):
    df = None
    for path in paths:
      trainer = get_trainer(path)
      errs = eval_run(trainer)
      if df is None:
        df = pd.DataFrame(columns=list(errs.keys()) + list(trainer.config.keys()) + ["path"])
      df.loc[len(df)] = list(errs.values()) + list(trainer.config.values()) + [path]
    return df

# %%
paths = ["11-13-06-04-25dice_periodic_ablation-fixed_dataset_length-fNEdJ_L2td2_te2_d512gsm","11-13-06-04-25dice_periodic_ablation-fixed_dataset_length-aBARF_L2td2_te2_d256gsm","11-13-06-04-25dice_periodic_ablation-fixed_dataset_length-bUyeu_L2td2_te2_d512gsm","11-13-06-04-25dice_periodic_ablation-fixed_dataset_length-RGzfo_L2td2_te2_d512gsm","11-13-06-04-23dice_periodic_ablation-fixed_dataset_length-YvvVl_L2td2_te2_d512gsm","11-13-06-04-22dice_periodic_ablation-fixed_dataset_length-KDLIm_L2td2_te2_d256gsm","11-13-06-04-21dice_periodic_ablation-fixed_dataset_length-lpiGu_L2td2_te2_d512gsm","11-13-06-04-19dice_periodic_ablation-fixed_dataset_length-GJbYL_L2td2_te2_d512gsm","11-13-06-04-18dice_periodic_ablation-fixed_dataset_length-SUDUf_L2td2_te2_d512gsm","11-13-06-04-18dice_periodic_ablation-fixed_dataset_length-WbJwQ_L2td2_te2_d256gsm","11-13-06-04-16dice_periodic_ablation-fixed_dataset_length-NKzzp_L2td2_te2_d512gsm","11-13-06-04-16dice_periodic_ablation-fixed_dataset_length-aDlCd_L2td2_te2_d256gsm","11-13-06-04-14dice_periodic_ablation-fixed_dataset_length-qasMz_L2td2_te2_d512gsm","11-13-06-04-13dice_periodic_ablation-fixed_dataset_length-pacAG_L2td2_te2_d256gsm","11-13-06-04-14dice_periodic_ablation-fixed_dataset_length-UWcSb_L2td2_te2_d256gsm","11-13-06-04-13dice_periodic_ablation-fixed_dataset_length-xDwXe_L2td2_te2_d256gsm","11-13-06-04-13dice_periodic_ablation-fixed_dataset_length-KLNND_L2td2_te2_d256gsm","11-13-06-04-11dice_periodic_ablation-fixed_dataset_length-uAGFh_L2td2_te2_d512gsm","11-13-06-04-05dice_periodic_ablation-fixed_dataset_length-dkaeX_L2td2_te2_d512gsm","11-13-06-04-05dice_periodic_ablation-fixed_dataset_length-XYZPE_L2td2_te2_d256gsm","11-13-06-04-05dice_periodic_ablation-fixed_dataset_length-AJCMd_L2td2_te2_d256gsm","11-13-06-04-05dice_periodic_ablation-fixed_dataset_length-enTcJ_L2td2_te2_d512gsm","11-13-06-04-04dice_periodic_ablation-fixed_dataset_length-rCRfp_L2td2_te2_d512gsm","11-13-06-04-03dice_periodic_ablation-fixed_dataset_length-DWEQz_L2td2_te2_d512gsm","11-13-06-04-03dice_periodic_ablation-fixed_dataset_length-Lquzh_L2td2_te2_d256gsm","11-13-06-04-02dice_periodic_ablation-fixed_dataset_length-ZgVXX_L2td2_te2_d256gsm","11-13-06-04-02dice_periodic_ablation-fixed_dataset_length-dGtKH_L2td2_te2_d512gsm","11-13-06-04-01dice_periodic_ablation-fixed_dataset_length-OObvf_L2td2_te2_d512gsm","11-13-06-03-59dice_periodic_ablation-fixed_dataset_length-BLQkQ_L2td2_te2_d512gsm","11-13-06-03-56dice_periodic_ablation-fixed_dataset_length-BYPbv_L2td2_te2_d512gsm","11-13-06-03-56dice_periodic_ablation-fixed_dataset_length-qCmYe_L2td2_te2_d512gsm","11-13-06-03-51dice_periodic_ablation-fixed_dataset_length-isVed_L2td2_te2_d512gsm","11-13-06-03-51dice_periodic_ablation-fixed_dataset_length-BkMFs_L2td2_te2_d256gsm","11-13-06-03-51dice_periodic_ablation-fixed_dataset_length-xQhug_L2td2_te2_d256gsm","11-13-06-03-51dice_periodic_ablation-fixed_dataset_length-QHshU_L2td2_te2_d512gsm","11-13-06-03-52dice_periodic_ablation-fixed_dataset_length-DWlBu_L2td2_te2_d512gsm","11-13-06-03-52dice_periodic_ablation-fixed_dataset_length-XKJCR_L2td2_te2_d256gsm","11-13-06-03-50dice_periodic_ablation-fixed_dataset_length-fETNl_L2td2_te2_d256gsm","11-13-06-03-51dice_periodic_ablation-fixed_dataset_length-vRDjX_L2td2_te2_d512gsm","11-13-06-03-50dice_periodic_ablation-fixed_dataset_length-fGkLS_L2td2_te2_d256gsm","11-13-06-03-48dice_periodic_ablation-fixed_dataset_length-rrSCY_L2td2_te2_d256gsm","11-13-06-03-46dice_periodic_ablation-fixed_dataset_length-sMVIu_L2td2_te2_d256gsm","11-13-06-03-45dice_periodic_ablation-fixed_dataset_length-FZjNX_L2td2_te2_d256gsm","11-13-06-03-45dice_periodic_ablation-fixed_dataset_length-opPpR_L2td2_te2_d512gsm","11-13-06-03-45dice_periodic_ablation-fixed_dataset_length-xWIdw_L2td2_te2_d256gsm","11-13-06-03-45dice_periodic_ablation-fixed_dataset_length-RnOQa_L2td2_te2_d512gsm","11-13-06-03-45dice_periodic_ablation-fixed_dataset_length-NGDpE_L2td2_te2_d256gsm","11-13-06-03-39dice_periodic_ablation-fixed_dataset_length-TJRNq_L2td2_te2_d256gsm","11-13-06-03-39dice_periodic_ablation-fixed_dataset_length-WnWsu_L2td2_te2_d512gsm","11-13-06-03-35dice_periodic_ablation-fixed_dataset_length-Gdaro_L2td2_te2_d256gsm","11-13-06-03-34dice_periodic_ablation-fixed_dataset_length-FqJeA_L2td2_te2_d512gsm","11-13-06-03-34dice_periodic_ablation-fixed_dataset_length-GdTcQ_L2td2_te2_d512gsm","11-13-06-03-32dice_periodic_ablation-fixed_dataset_length-xEYqo_L2td2_te2_d256gsm","11-13-06-03-33dice_periodic_ablation-fixed_dataset_length-rmmel_L2td2_te2_d256gsm","11-13-06-03-31dice_periodic_ablation-fixed_dataset_length-pDSRc_L2td2_te2_d512gsm","11-13-06-03-31dice_periodic_ablation-fixed_dataset_length-ddgqg_L2td2_te2_d512gsm","11-13-06-03-31dice_periodic_ablation-fixed_dataset_length-SgMbG_L2td2_te2_d512gsm","11-13-06-03-28dice_periodic_ablation-fixed_dataset_length-qEQkF_L2td2_te2_d256gsm","11-13-06-03-27dice_periodic_ablation-fixed_dataset_length-sPevj_L2td2_te2_d256gsm","11-13-06-03-25dice_periodic_ablation-fixed_dataset_length-gSGJv_L2td2_te2_d512gsm","11-13-06-03-26dice_periodic_ablation-fixed_dataset_length-aMtDx_L2td2_te2_d512gsm","11-13-06-03-25dice_periodic_ablation-fixed_dataset_length-iyAVZ_L2td2_te2_d256gsm","11-13-06-03-22dice_periodic_ablation-fixed_dataset_length-KOwbk_L2td2_te2_d256gsm","11-13-06-03-21dice_periodic_ablation-fixed_dataset_length-aulnX_L2td2_te2_d256gsm","11-13-06-03-20dice_periodic_ablation-fixed_dataset_length-uTLHt_L2td2_te2_d512gsm","11-13-06-03-20dice_periodic_ablation-fixed_dataset_length-Dvzfo_L2td2_te2_d256gsm","11-13-06-03-17dice_periodic_ablation-fixed_dataset_length-CwIwl_L2td2_te2_d256gsm","11-13-06-03-17dice_periodic_ablation-fixed_dataset_length-zCcPy_L2td2_te2_d512gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-RKZOg_L2td2_te2_d512gsm","11-13-06-03-17dice_periodic_ablation-fixed_dataset_length-wMEXw_L2td2_te2_d256gsm","11-13-06-03-17dice_periodic_ablation-fixed_dataset_length-LKGCd_L2td2_te2_d512gsm","11-13-06-03-17dice_periodic_ablation-fixed_dataset_length-kDMAG_L2td2_te2_d512gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-LVLrr_L2td2_te2_d512gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-bhgMe_L2td2_te2_d512gsm","11-13-06-03-15dice_periodic_ablation-fixed_dataset_length-HjsqO_L2td2_te2_d256gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-rEbDW_L2td2_te2_d256gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-koLRO_L2td2_te2_d256gsm","11-13-06-03-16dice_periodic_ablation-fixed_dataset_length-rfAvd_L2td2_te2_d256gsm","11-13-06-03-15dice_periodic_ablation-fixed_dataset_length-gDHNx_L2td2_te2_d256gsm","11-13-03-11-59dice_periodic_ablation-fixed_dataset_length-WrMtz_L2td2_te2_d512gsm","11-13-03-11-52dice_periodic_ablation-fixed_dataset_length-lnOZV_L2td2_te2_d512gsm","11-13-03-11-50dice_periodic_ablation-fixed_dataset_length-XlbLb_L2td2_te2_d512gsm","11-13-03-11-50dice_periodic_ablation-fixed_dataset_length-XXaBq_L2td2_te2_d512gsm","11-13-03-11-48dice_periodic_ablation-fixed_dataset_length-kMpXS_L2td2_te2_d256gsm","11-13-03-11-49dice_periodic_ablation-fixed_dataset_length-ANeRp_L2td2_te2_d512gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-PWcLu_L2td2_te2_d256gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-KiSIq_L2td2_te2_d256gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-WuBRu_L2td2_te2_d256gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-NwprZ_L2td2_te2_d256gsm","11-13-03-11-48dice_periodic_ablation-fixed_dataset_length-dgonZ_L2td2_te2_d512gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-hCQak_L2td2_te2_d512gsm","11-13-03-11-47dice_periodic_ablation-fixed_dataset_length-BIlfU_L2td2_te2_d512gsm","11-13-03-11-48dice_periodic_ablation-fixed_dataset_length-tRiNI_L2td2_te2_d512gsm","11-13-03-11-45dice_periodic_ablation-fixed_dataset_length-uQhEu_L2td2_te2_d256gsm","11-13-03-11-46dice_periodic_ablation-fixed_dataset_length-VSuHu_L2td2_te2_d256gsm","11-13-03-11-44dice_periodic_ablation-fixed_dataset_length-herDQ_L2td2_te2_d256gsm","11-13-03-11-44dice_periodic_ablation-fixed_dataset_length-OhaDn_L2td2_te2_d512gsm","11-13-03-11-45dice_periodic_ablation-fixed_dataset_length-OqWkZ_L2td2_te2_d256gsm","11-13-03-11-40dice_periodic_ablation-fixed_dataset_length-tPMNx_L2td2_te2_d256gsm","11-13-03-11-40dice_periodic_ablation-fixed_dataset_length-hydqS_L2td2_te2_d256gsm","11-13-03-11-40dice_periodic_ablation-fixed_dataset_length-EKSdB_L2td2_te2_d512gsm","11-13-03-11-38dice_periodic_ablation-fixed_dataset_length-dDJxf_L2td2_te2_d512gsm","11-13-03-11-38dice_periodic_ablation-fixed_dataset_length-ntgdp_L2td2_te2_d512gsm","11-13-03-11-35dice_periodic_ablation-fixed_dataset_length-kccAW_L2td2_te2_d256gsm","11-13-03-11-35dice_periodic_ablation-fixed_dataset_length-gRsHE_L2td2_te2_d256gsm","11-13-03-11-36dice_periodic_ablation-fixed_dataset_length-zChrw_L2td2_te2_d512gsm","11-13-03-11-36dice_periodic_ablation-fixed_dataset_length-tPJyR_L2td2_te2_d256gsm","11-13-03-11-35dice_periodic_ablation-fixed_dataset_length-CIPeP_L2td2_te2_d512gsm","11-13-03-11-34dice_periodic_ablation-fixed_dataset_length-FCmxI_L2td2_te2_d512gsm","11-13-03-11-29dice_periodic_ablation-fixed_dataset_length-WntLx_L2td2_te2_d512gsm","11-13-03-11-29dice_periodic_ablation-fixed_dataset_length-gPKTH_L2td2_te2_d256gsm","11-13-03-11-29dice_periodic_ablation-fixed_dataset_length-pHpNo_L2td2_te2_d256gsm","11-13-03-11-30dice_periodic_ablation-fixed_dataset_length-wRzaY_L2td2_te2_d256gsm","11-13-03-11-22dice_periodic_ablation-fixed_dataset_length-FDRNB_L2td2_te2_d256gsm","11-13-03-11-22dice_periodic_ablation-fixed_dataset_length-pzFUU_L2td2_te2_d512gsm","11-13-03-11-22dice_periodic_ablation-fixed_dataset_length-fZSUX_L2td2_te2_d256gsm","11-13-03-11-22dice_periodic_ablation-fixed_dataset_length-JlYfU_L2td2_te2_d256gsm","11-13-03-11-22dice_periodic_ablation-fixed_dataset_length-dSAYD_L2td2_te2_d256gsm","11-13-03-11-20dice_periodic_ablation-fixed_dataset_length-kNtcB_L2td2_te2_d512gsm","11-13-03-11-20dice_periodic_ablation-fixed_dataset_length-kkRyI_L2td2_te2_d512gsm","11-13-03-11-18dice_periodic_ablation-fixed_dataset_length-Jcybj_L2td2_te2_d512gsm","11-13-03-11-19dice_periodic_ablation-fixed_dataset_length-KMkJF_L2td2_te2_d512gsm","11-13-03-11-19dice_periodic_ablation-fixed_dataset_length-EWbAq_L2td2_te2_d256gsm","11-13-03-11-16dice_periodic_ablation-fixed_dataset_length-iqeaW_L2td2_te2_d512gsm","11-13-03-11-15dice_periodic_ablation-fixed_dataset_length-EmyGT_L2td2_te2_d256gsm","11-13-03-11-14dice_periodic_ablation-fixed_dataset_length-gzCQj_L2td2_te2_d512gsm","11-13-03-11-14dice_periodic_ablation-fixed_dataset_length-AAZWs_L2td2_te2_d512gsm","11-13-03-11-14dice_periodic_ablation-fixed_dataset_length-LXCnM_L2td2_te2_d256gsm","11-13-03-11-05dice_periodic_ablation-fixed_dataset_length-REomb_L2td2_te2_d256gsm","11-13-03-11-06dice_periodic_ablation-fixed_dataset_length-WMBFO_L2td2_te2_d256gsm","11-13-03-11-06dice_periodic_ablation-fixed_dataset_length-CYttz_L2td2_te2_d256gsm","11-13-03-11-04dice_periodic_ablation-fixed_dataset_length-NoqFT_L2td2_te2_d512gsm","11-13-03-11-04dice_periodic_ablation-fixed_dataset_length-Fzalg_L2td2_te2_d512gsm","11-13-03-11-04dice_periodic_ablation-fixed_dataset_length-onSuL_L2td2_te2_d512gsm","11-13-03-11-03dice_periodic_ablation-fixed_dataset_length-bJBTA_L2td2_te2_d256gsm","11-13-03-11-02dice_periodic_ablation-fixed_dataset_length-DNKjK_L2td2_te2_d512gsm","11-13-03-11-01dice_periodic_ablation-fixed_dataset_length-EHHNa_L2td2_te2_d256gsm","11-13-03-11-02dice_periodic_ablation-fixed_dataset_length-bMBVs_L2td2_te2_d512gsm","11-13-03-11-00dice_periodic_ablation-fixed_dataset_length-dDEOx_L2td2_te2_d512gsm","11-13-03-10-56dice_periodic_ablation-fixed_dataset_length-MWDCH_L2td2_te2_d512gsm","11-13-03-10-55dice_periodic_ablation-fixed_dataset_length-jdNny_L2td2_te2_d512gsm","11-13-03-10-55dice_periodic_ablation-fixed_dataset_length-PrhLm_L2td2_te2_d512gsm","11-13-03-10-54dice_periodic_ablation-fixed_dataset_length-JZvUx_L2td2_te2_d512gsm","11-13-03-10-53dice_periodic_ablation-fixed_dataset_length-UmLfy_L2td2_te2_d256gsm","11-13-03-10-53dice_periodic_ablation-fixed_dataset_length-UfVzj_L2td2_te2_d512gsm","11-13-03-10-51dice_periodic_ablation-fixed_dataset_length-XwTnd_L2td2_te2_d256gsm","11-13-03-10-49dice_periodic_ablation-fixed_dataset_length-qZVSv_L2td2_te2_d256gsm","11-13-03-10-49dice_periodic_ablation-fixed_dataset_length-MgKNg_L2td2_te2_d256gsm","11-13-03-10-49dice_periodic_ablation-fixed_dataset_length-DwsTV_L2td2_te2_d256gsm","11-13-03-10-48dice_periodic_ablation-fixed_dataset_length-cXzji_L2td2_te2_d256gsm","11-13-03-10-48dice_periodic_ablation-fixed_dataset_length-cvnrT_L2td2_te2_d256gsm","11-13-03-10-49dice_periodic_ablation-fixed_dataset_length-xlifY_L2td2_te2_d512gsm","11-13-03-10-49dice_periodic_ablation-fixed_dataset_length-MFZpa_L2td2_te2_d512gsm","11-13-03-10-48dice_periodic_ablation-fixed_dataset_length-KxYLQ_L2td2_te2_d512gsm","11-13-03-10-48dice_periodic_ablation-fixed_dataset_length-ySGJj_L2td2_te2_d256gsm","11-13-03-10-46dice_periodic_ablation-fixed_dataset_length-Dgnyh_L2td2_te2_d256gsm","11-13-03-10-46dice_periodic_ablation-fixed_dataset_length-JHohb_L2td2_te2_d256gsm","11-13-03-10-47dice_periodic_ablation-fixed_dataset_length-DMTpM_L2td2_te2_d512gsm", ]

paths = [

  "11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-rYpcQ_L2td2_te2_d256gsm",
"11-15-03-16-18dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-ofdyv_L2td2_te2_d512gsm",
"11-15-03-15-28dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-VraxR_L2td2_te2_d512gsm",
"11-15-03-15-49dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-zHGcK_L2td2_te2_d512gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-YlJCK_L2td2_te2_d512gsm",
"11-15-03-15-22dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-dfymh_L2td2_te2_d512gsm",
"11-15-03-16-05dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-mxwxS_L2td2_te2_d512gsm",
"11-15-03-16-05dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-oPWnq_L2td2_te2_d512gsm",
"11-15-03-15-22dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-raqxp_L2td2_te2_d512gsm",
"11-15-03-15-22dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-DOMPK_L2td2_te2_d512gsm",
"11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-mifaV_L2td2_te2_d512gsm",
"11-15-03-16-06dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-cnKEj_L2td2_te2_d512gsm",
"11-15-03-16-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-Pggwp_L2td2_te2_d512gsm",
"11-15-03-15-57dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-BvMMT_L2td2_te2_d512gsm",
"11-15-03-15-28dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-txpTg_L2td2_te2_d512gsm",
"11-15-03-15-43dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-gsphj_L2td2_te2_d512gsm",
"11-15-03-15-14dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-YWWxB_L2td2_te2_d512gsm",
"11-15-03-15-13dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-QIINB_L2td2_te2_d512gsm",
"11-15-03-15-08dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-sBblZ_L2td2_te2_d512gsm",
"11-15-03-15-13dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-ceXaC_L2td2_te2_d512gsm",
"11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-gcqPa_L2td2_te2_d512gsm",
"11-15-03-15-33dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-wOujf_L2td2_te2_d512gsm",
"11-15-03-15-09dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-NWgtZ_L2td2_te2_d512gsm",
"11-15-03-15-29dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-ciunf_L2td2_te2_d512gsm",
"11-15-03-15-46dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-guEqn_L2td2_te2_d512gsm",
"11-15-03-15-33dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-iEYbi_L2td2_te2_d512gsm",
"11-15-03-15-47dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-plrdX_L2td2_te2_d512gsm",
"11-15-03-16-17dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-AmiCe_L2td2_te2_d512gsm",
"11-15-03-15-14dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-OEyaC_L2td2_te2_d512gsm",
"11-15-03-16-05dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-qnEEV_L2td2_te2_d512gsm",
"11-15-03-15-08dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-hNaIK_L2td2_te2_d512gsm",
"11-15-03-15-51dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-EEcJz_L2td2_te2_d512gsm",
"11-15-03-15-36dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-oyGoz_L2td2_te2_d512gsm",
"11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-aWNtG_L2td2_te2_d512gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-VJBmn_L2td2_te2_d512gsm",
"11-15-03-15-58dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-WecDj_L2td2_te2_d512gsm",
"11-15-03-15-08dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-xlrCM_L2td2_te2_d512gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-mPHIu_L2td2_te2_d512gsm",
"11-15-03-15-50dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-kQRYA_L2td2_te2_d512gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-hnMAJ_L2td2_te2_d512gsm",
"11-15-03-15-47dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-eYVnC_L2td2_te2_d512gsm",
"11-15-03-15-14dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-riYTs_L2td2_te2_d256gsm",
"11-15-03-15-23dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-czakE_L2td2_te2_d256gsm",
"11-15-03-16-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-Otmut_L2td2_te2_d256gsm",
"11-15-03-15-13dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-HvvlS_L2td2_te2_d256gsm",
"11-15-03-15-22dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-nrzpE_L2td2_te2_d256gsm",
"11-15-03-16-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-kjemJ_L2td2_te2_d256gsm",
"11-15-03-16-17dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-XEsWG_L2td2_te2_d256gsm",
"11-15-03-15-31dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-zxItT_L2td2_te2_d256gsm",
"11-15-03-15-07dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-LzEdz_L2td2_te2_d256gsm",
"11-15-03-15-51dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-cVVQM_L2td2_te2_d256gsm",
"11-15-03-15-39dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-AayOX_L2td2_te2_d256gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-pNwyA_L2td2_te2_d256gsm",
"11-15-03-16-10dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-prqDe_L2td2_te2_d256gsm",
"11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-LvlVV_L2td2_te2_d256gsm",
"11-15-03-15-52dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-SkOmH_L2td2_te2_d256gsm",
"11-15-03-15-36dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-VcAsu_L2td2_te2_d256gsm",
"11-15-03-16-05dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-NzNBT_L2td2_te2_d256gsm",
"11-15-03-15-41dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-nqMhj_L2td2_te2_d256gsm",
"11-15-03-15-14dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-xYoii_L2td2_te2_d256gsm",
"11-15-03-15-28dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-gCMaX_L2td2_te2_d256gsm",
"11-15-03-15-28dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-DOqHu_L2td2_te2_d256gsm",
"11-15-03-16-00dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-zENnL_L2td2_te2_d256gsm",
"11-15-03-16-06dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-HDDDk_L2td2_te2_d256gsm",
"11-15-03-15-07dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-bcRDh_L2td2_te2_d256gsm",
"11-15-03-16-17dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-utZjV_L2td2_te2_d256gsm",
"11-15-03-15-47dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-vOKhI_L2td2_te2_d256gsm",
"11-15-03-14-57dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-rAbAp_L2td2_te2_d256gsm",
"11-15-03-15-45dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-ukOjk_L2td2_te2_d256gsm",
"11-15-03-15-56dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-viSCa_L2td2_te2_d256gsm",
"11-15-03-15-47dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-WjXNE_L2td2_te2_d256gsm",
"11-15-03-15-57dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-AOgQX_L2td2_te2_d256gsm",
"11-15-03-15-15dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-UXdTO_L2td2_te2_d256gsm",
"11-15-03-16-05dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-MTlHd_L2td2_te2_d256gsm",
"11-15-03-15-53dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-ZPzTD_L2td2_te2_d256gsm",
"11-15-03-15-54dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-FgyFR_L2td2_te2_d256gsm",
"11-15-03-15-08dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-FoRrz_L2td2_te2_d256gsm",
"11-15-03-15-40dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-OXDMF_L2td2_te2_d256gsm",
"11-15-03-15-07dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-MUnvV_L2td2_te2_d256gsm",
"11-15-03-16-08dice_periodic_ablation-fixed_dataset_length-tie_numerical_embeddings_ablation-wMFNK_L2td2_te2_d256gsm",
]



# %%
out = create_df_from(paths)
out.to_csv("ablation_results_tied.csv")

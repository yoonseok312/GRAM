from typing import Dict
from typing import List
from typing import Any
from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
import numpy as np
import omegaconf
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from magneto.train.schedulers import get_noam_scheduler
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from gram.knowledge_tracing.components.sbert_regressor import SBERTRegressor
from gram.knowledge_tracing.content_dkt import LightningContentDKT


class EpochwiseAlternatingCKT(LightningContentDKT):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        LightningContentDKT.__init__(self, cfg=cfg)
        self.automatic_optimization = False
        self.training_lm = False
        self.lm = SBERTRegressor(cfg)
        if self.cfg.ckt_dataset_type == "duolingo":
            print("duolingo!!!!!!!!")
            if self.cfg[cfg.ckt_dataset_type].language == "french":
                self.item_count = np.load(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/item_count_french_final.npy",
                    allow_pickle="TRUE",
                ).item()
            else:
                self.item_count = np.load(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/item_count_spanish_final.npy",
                    allow_pickle="TRUE",
                ).item()
            # self.item_count = np.load(f'{self.cfg[self.cfg.ckt_dataset_type].data.root}/item_count_new_french.npy',allow_pickle='TRUE').item()
        elif self.cfg.ckt_dataset_type == "poj":
            with open(
                f"{self.cfg[self.cfg.ckt_dataset_type].data.root}/poj_id_to_count_only_train.pkl",
                "rb",
            ) as handle:
                self.item_count = pickle.load(handle)
        elif self.cfg.ckt_dataset_type == "toeic":
            with open(
                f"{self.cfg[self.cfg.ckt_dataset_type].data.root}/toeic_id_to_count.pkl",
                "rb",
            ) as handle:
                self.item_count = pickle.load(handle)

        item_ids_list = list(self.id_to_text.keys())
        self.non_cold_start_ids = []

    # def on_train_epoch_end(self) -> None:
    #     if self.training_lm:
    #         breakpoint()
    #         self.training_lm = False
    #     else:
    #         breakpoint()
    #         self.training_lm = True

    def on_train_epoch_start(self) -> None:
        print("num_non_csqs", len(self.non_cold_start_ids))
        self.log("num_non_csqs", len(self.non_cold_start_ids))
        for item in self.non_cold_start_ids:
            self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                item
            ] = self.lm([item])
        self.non_cold_start_ids = []

    def validation_epoch_end(self, outputs) -> None:
        """
        single validation epoch for pl.Trainer
        """
        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])
        csqe_labels = np.hstack([o["csqe_labels"] for o in outputs])
        csqe_probs = np.hstack([o["csqe_probs"] for o in outputs])

        # print(loss.shape, labels.shape, probs.shape, preds.shape)

        acc = (preds == labels).sum().item() / len(preds)
        auc = metrics.roc_auc_score(labels, probs)
        try:
            csqe_auc = metrics.roc_auc_score(csqe_labels, csqe_probs)
            # auc_list.append(auc)
        except ValueError:
            csqe_auc = 0
            print("labels", labels)
            print("probs", probs)
            pass
        # auc = metrics.roc_auc_score(labels, probs)
        loss = np.mean(loss)  # .mean().item()

        log = {
            "val_acc": acc,
            "val_loss": loss,
            "val_auc": auc,
            "val_csq_auc": csqe_auc,
        }
        print(log)
        self.log_dict(log)

        return log

    def compute_lm_loss(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = self.cfg[
            self.cfg.experiment_type[self.cfg.current_stage]
        ].weight_count
        assert weight_count in ["linear", "log", "sqrt", "log_add", None]
        # loss = F.mse_loss(logit, label, reduction="sum")
        loss_before_reduction = F.mse_loss(logit, label, reduction="none").sum(-1)
        if weight_count == "linear":
            linear_weight = torch.tensor(
                [
                    self.item_count[int(i.cpu().numpy())]
                    / max(list(self.item_count.values()))
                    for i in x
                ]
            ).type(torch.FloatTensor)
            linear_weighted_loss = torch.dot(
                linear_weight.cuda(), loss_before_reduction
            )
            return linear_weighted_loss
        elif weight_count == "log":
            log_weight = torch.tensor(
                [
                    np.log(self.item_count[int(i.cpu().numpy())])
                    / max(np.log(np.asarray(list(self.item_count.values()))))
                    for i in x
                ]
            ).type(torch.FloatTensor)
            log_weighted_loss = torch.dot(log_weight.cuda(), loss_before_reduction)
            return log_weighted_loss
        elif weight_count == "log_add":
            log_add_weight = torch.tensor(
                [
                    np.log(self.item_count[int(i.cpu().numpy())] + 1)
                    / (max(np.log(np.asarray(list(self.item_count.values())))) + 1)
                    for i in x
                ]
            ).type(torch.FloatTensor)
            log_add_weighted_loss = torch.dot(
                log_add_weight.cuda(), loss_before_reduction
            )
            return log_add_weighted_loss
        elif weight_count == "sqrt":
            sqrt_weight = torch.tensor(
                [
                    np.sqrt(self.item_count[int(i.cpu().numpy())])
                    / max(np.sqrt(np.asarray(list(self.item_count.values()))))
                    for i in x
                ]
            ).type(torch.FloatTensor)
            sqrt_weighted_loss = torch.dot(sqrt_weight.cuda(), loss_before_reduction)
            return sqrt_weighted_loss
        else:
            loss = F.mse_loss(logit, label, reduction="sum")
            return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
        items = batch["item_id"][nonzeros[0], nonzeros[1]]
        items = torch.unique(items)

        items = pd.DataFrame(items.cpu().detach().numpy()).astype("int32")
        self.non_cold_start_ids += items[~items[0].isin(self.non_cold_start_ids)][
            0
        ].tolist()

        kt_opt, lm_opt = self.optimizers()
        kt_sch, lm_sch = self.lr_schedulers()
        # if self.training_lm == False:  # 100 step per epoch
        loss = self.training_step_helper(batch)
        kt_opt.zero_grad()
        self.manual_backward(loss)
        kt_opt.step()
        kt_sch.step()

        self.log("train_kt_loss", loss)
        # else:  # self.steps % 100 == 0: # and self.steps % 10 == 0
        # self.training_lm = True
        if len(self.train_dataloader.dataloader) - 1 == batch_idx:
            embedding_label_list = []
            for id in self.non_cold_start_ids:
                embedding_label_list.append(
                    np.asarray(
                        self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                            id
                        ]
                        .detach()
                        .cpu()
                    )
                )

            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(
                        np.asarray(self.non_cold_start_ids), dtype=torch.float
                    ).cuda(),
                    torch.tensor(
                        np.asarray(embedding_label_list), dtype=torch.float
                    ).cuda(),
                ),
                batch_size=self.cfg.Alternating.regressor_batch_size,
                shuffle=True,
            )

            for ids, labels in tqdm(train_loader):
                # print(data)
                output = self.lm(ids)
                loss = self.compute_lm_loss(labels, output, ids)
                lm_opt.zero_grad()
                self.manual_backward(loss)
                lm_opt.step()
                lm_sch.step()
                self.log("train_lm_loss", loss)
            for item in self.cold_start_ids:
                self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                    item
                ] = self.lm([item])
            self.training_lm = False

    def configure_optimizers(self):
        self.kt_optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr,
        )
        self.kt_scheduler = get_noam_scheduler(
            self.kt_optimizer,
            warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
            only_warmup=False,
            interval="step",
        )
        self.lm_optimizer = torch.optim.Adam(
            self.lm.parameters(),
            lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr,
        )
        self.lm_scheduler = get_noam_scheduler(
            self.lm_optimizer,
            warmup_steps=self.cfg[
                self.cfg.experiment_type[self.cfg.current_stage]
            ].lm_warmup,
            only_warmup=False,
            interval="step",
        )
        return (
            {"optimizer": self.kt_optimizer, "lr_scheduler": self.kt_scheduler},
            {"optimizer": self.lm_optimizer, "lr_scheduler": self.lm_scheduler},
        )

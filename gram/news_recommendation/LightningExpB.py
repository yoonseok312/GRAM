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

from gram.knowledge_tracing.components.input_embedding import AllItemInputEmbedding
from gram.knowledge_tracing.components.SBERT import SBERT, CKTAdditiveAttention
from repoc_common import utils

import wandb

class LightningRegressor(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig, model_sim = None) -> None:
        super().__init__()
        self.save_hyperparameters()  # stores hyperparameters
        self.cfg = cfg
        # self.threshold = 0.5
        self.base_model = SBERTRegressor(cfg, model_sim)
        if self.cfg.ckt_dataset_type == 'duolingo':
            if self.cfg[cfg.ckt_dataset_type].language == 'french':
                self.item_count = np.load(f"{cfg[cfg.ckt_dataset_type].data.root}/item_count_french_final.npy", allow_pickle='TRUE').item()
            else:111
                self.item_count = np.load(f"{cfg[cfg.ckt_dataset_type].data.root}/item_count_spanish_final.npy", allow_pickle='TRUE').item()
        elif self.cfg.ckt_dataset_type == 'poj':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_count_only_train.pkl", "rb") as handle:
                self.item_count = pickle.load(handle)
        elif self.cfg.ckt_dataset_type == 'toeic':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_count.pkl", "rb") as handle:
                self.item_count = pickle.load(handle)
        self.training_lm = False
        self.lr_denominator = 1

    def forward(
        self,
        batch: torch.Tensor,
        # encoded_all: torch.Tensor,
    ) -> torch.Tensor:

        output = self.base_model(batch)
        # print(output.size())
        return output

    def compute_loss(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].weight_count
        assert weight_count in ["linear", "log", "sqrt", "log_add", None]
        if weight_count in ["linear", "log", "log_add", "sqrt"]:

            loss = F.mse_loss(logit, label, reduction="sum")

            linear_weight = torch.tensor([self.item_count[int(i.cpu().numpy())] / max(list(self.item_count.values())) for i in x]).type(torch.FloatTensor)
            log_weight = torch.tensor([np.log(self.item_count[int(i.cpu().numpy())])
                                       / max(np.log(np.asarray(list(self.item_count.values())))) for i in x]).type(torch.FloatTensor)
            log_add_weight = torch.tensor([np.log(self.item_count[int(i.cpu().numpy())]+1)
                                       / (max(np.log(np.asarray(list(self.item_count.values()))))+1) for i in x]).type(
                torch.FloatTensor)
            sqrt_weight = torch.tensor([np.sqrt(self.item_count[int(i.cpu().numpy())])
                                       / max(np.sqrt(np.asarray(list(self.item_count.values())))) for i in x]).type(
                torch.FloatTensor)

            loss_before_reduction = F.mse_loss(logit, label, reduction="none").sum(-1)
            linear_weighted_loss = torch.dot(linear_weight.cuda(),loss_before_reduction)
            # breakpoint()
            log_weighted_loss = torch.dot(log_weight.cuda(),loss_before_reduction)
            log_add_weighted_loss = torch.dot(log_add_weight.cuda(), loss_before_reduction)
            sqrt_weighted_loss = torch.dot(sqrt_weight.cuda(), loss_before_reduction)
        else:
            loss = F.mse_loss(logit, label, reduction="sum")
        if weight_count == "linear":
            return linear_weighted_loss
        elif weight_count == "log":
            return log_weighted_loss
        elif weight_count == "log_add":
            return log_add_weighted_loss
        elif weight_count == "sqrt":
            return sqrt_weighted_loss
        else:
            return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # breakpoint()
        x,y = batch

        logit = self(x)

        loss = self.compute_loss(y, logit, x)
        self.log("train_lm_loss", loss)
        # wandb.log(
        #     {"train_lm_loss": loss}
        # )

        return loss

    def on_train_epoch_start(self) -> None:
        self.training_lm = True

    def on_train_epoch_end(self) -> None:
        self.training_lm = False

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """
        x,y = batch
        logit = self(x)

        loss = self.compute_loss(y,logit,x)
        return {
            "loss": loss.item(),
        }

    def validation_epoch_end(self, outputs):
        """
        single validation epoch for pl.Trainer
        """
        loss = np.array([o["loss"] for o in outputs])
        loss = np.mean(loss)  # .mean().item()

        log = {
            "val_loss": loss,
        }
        print(log)
        self.log_dict(log)
        wandb.log(
            {"val_loss": loss}
        )

        return log

    def configure_optimizers(self) -> Dict[str, Any]:
        # breakpoint()
        if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].alternate_by_epoch:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr/self.lr_denominator)#self.cfg.train.lr)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.cfg.B.lr)

        # self.scheduler = get_noam_scheduler(
        #     self.optimizer,
        #     warmup_steps=4000,
        #     only_warmup=False,
        #     interval="step",
        # )
        return {"optimizer": self.optimizer}

        # return {"optimizer": self.optimizer}
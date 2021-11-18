import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from sklearn import metrics
from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score
import pytorch_lightning as pl
from model_bert import ModelBert
from LightningNRMS import LightningNRMS
from model_bert import NewsEncoder
from repoc_content_kt.news_recommendation.nrms import AlternatingUserEncoder
from repoc_content_kt.news_recommendation.dataset import NewsDataset
from repoc_content_kt.news_recommendation.LightningAlternateNRMSWithLM import LightningAlternatingNRMSWithLM
from magneto.train.schedulers import get_noam_scheduler
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModel, AutoConfig

class LightningExpC(LightningAlternatingNRMSWithLM):
    def __init__(self,
                 cfg,
                 news_index,
                 global_dict,
                 val_news_combined,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0) -> None:
        super().__init__(cfg, news_index, global_dict, val_news_combined)
        self.save_hyperparameters(
            ignore=['news_index', 'global_dict', 'val_news_combined']
        )
        # self.automatic_optimization = True
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx: int, optimizer_idx: int=0) -> torch.Tensor:
        # breakpoint()
        log_ids = batch[0]
        log_mask = batch[1]
        input_ids = batch[2]
        targets = batch[3]
        log_idx = batch[4]
        input_idx = batch[5]
        # breakpoint()
        new_input_idx = torch.transpose(torch.stack(input_idx), 0, 1)
        new_log_idx = torch.transpose(torch.stack(log_idx), 0, 1)

        if self.current_epoch == 0 and batch_idx == 0:
            self.non_cold_start_ids = []
            for item in self.news_index.values():
                if item not in self.cold_start_ids:
                    self.non_cold_start_ids.append(item)
            # FIXME: think about above logic
            # self.non_cold_start_ids = list(self.item_count_news_idx.keys())

            # TODO: make pretrained weight for large dataset
            # print('making dict...')
            # temp_dict = {}
            for item in tqdm(self.non_cold_start_ids):
                # breakpoint()
                # temp_dict[item] = self.lm(torch.tensor(self.global_dict[item]).reshape(-1,72).cuda()).reshape(-1).detach().cpu().numpy()
                self.user_encoder.news_embedding.weight.data[item] = torch.tensor(self.pretrained_weight[item]).cuda()
                self.user_encoder.news_embedding.weight.data[item].requires_grad = True

        user_opt = self.optimizers()
        user_sch = self.lr_schedulers()
        y_hat = self(input_ids, log_ids, new_log_idx, new_input_idx, log_mask, targets)
        loss = self.compute_loss(y_hat, targets)
        user_opt.zero_grad()
        self.manual_backward(loss)
        user_opt.step()
        user_sch.step()
        self.log("train_user_loss", loss)

        return loss

    def configure_optimizers(self):
        self.user_encoder_optimizer = torch.optim.Adam(self.user_encoder.parameters(),lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr)
        self.user_encoder_scheduler = get_noam_scheduler(
            self.user_encoder_optimizer,
            warmup_steps=4000,
            only_warmup=False,
            interval="step",
        )
        return (
            {"optimizer": self.user_encoder_optimizer, "lr_scheduler": self.user_encoder_scheduler},
        )
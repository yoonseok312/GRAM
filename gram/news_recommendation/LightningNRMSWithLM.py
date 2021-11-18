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
import pickle
from tqdm import tqdm
from repoc_content_kt.news_recommendation.dataset import NewsDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig


class LightningNRMSWithLM(pl.LightningModule):
    def __init__(self,
                 cfg,
                 finetuneset,
                 news_index,
                 val_news_combined,
                 news_index_val,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['finetuneset', 'news_index', 'val_news_combined', 'news_index_val']
        )
        self.criterion = nn.CrossEntropyLoss()
        # self.news_index = news_index
        if cfg.dataset_size == 'large':
            with open(f'{cfg.root}/MIND/val_csqs_mind_large_list.pkl', 'rb') as f:
                csqs = pickle.load(f)
        else:
            with open(f'{cfg.root}/MIND_small/test_csqs_mind_small_list.pkl', 'rb') as f:
                csqs = pickle.load(f)
        self.cold_start_ids = []
        self.val_news_index = news_index_val
        for item in tqdm(csqs):
            self.cold_start_ids.append(self.val_news_index[item])
        self.news_scoring = []
        self.news_combined = val_news_combined
        self.cfg = cfg
        config = AutoConfig.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.model,
                                            output_hidden_states=True)
        bert_model = AutoModel.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.model, config=config)
        modules = [bert_model.embeddings,
                   *bert_model.encoder.layer[:4]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        # for name, param in bert_model.named_parameters():
        #     if name not in finetuneset:
        #         param.requires_grad = False
        self.base_model = ModelBert(cfg, bert_model, category_dict_size, domain_dict_size, subcategory_dict_size)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_until_now = 0


    def forward(self,
                input_ids,
                log_ids,
                log_mask,
                targets=None,
                compute_loss=False):
        return self.base_model(input_ids, log_ids, log_mask, targets, False)

    def compute_loss(self, score, targets):
        loss = self.criterion(score, targets)
        return loss

    def on_train_epoch_start(self):
        print("Start recording...")
        self.start.record()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # breakpoint()
        log_ids = batch[0]
        log_mask = batch[1]
        input_ids = batch[2]
        targets = batch[3]

        # print("targets", targets)
        y_hat = self(input_ids, log_ids, log_mask, targets)
        # breakpoint()
        loss = self.compute_loss(y_hat, targets)
        # if loss != loss:
        #     print("LOSS NAN")
        #     breakpoint()
        # breakpoint()
        # self.log("train_loss", loss)

        num_gpus = len(str(self.cfg.gpus).split(','))
        distributed_length = math.ceil(len(self.train_dataloader.dataloader) / num_gpus)
        if distributed_length - 1 == batch_idx:
            print("End recording...")
            self.end.record()
            torch.cuda.synchronize()
            time = self.start.elapsed_time(self.end)
            print("Elapsed time", time)
        return loss

    def csqe_label_and_prob(self, probs, label, id_list, items):
        csqe_items_idx = torch.cat([(items == i).nonzero() for i in id_list])
        csqe_prob = torch.gather(probs, dim=-1, index=csqe_items_idx.squeeze())
        csqe_label = torch.gather(label, dim=-1, index=csqe_items_idx.squeeze())
        # csqe_prob = torch.argmax(torch.softmax(csqe_logit, -1), -1)

        return csqe_label, csqe_prob, len(csqe_prob)

    def on_validation_start(self) -> None:
        self.news_scoring = []
        news_dataset = NewsDataset(self.news_combined)
        news_dataloader = DataLoader(
            news_dataset,
            batch_size=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].batch_size,
            num_workers=self.cfg.num_workers,
        )
        with torch.no_grad():
            for input_ids in tqdm(news_dataloader):
                news_vec = self.base_model.news_encoder(input_ids.cuda())
                news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
                self.news_scoring.extend(news_vec)
        self.news_scoring = np.array(self.news_scoring)

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """

        log_ids = batch[0]
        log_mask = batch[1]
        input_ids = batch[2] # candidate
        labels = batch[3]
        # labels_news_ids = batch[4]
        labels_mask = batch[5]

        user_feature_vecs = self.news_scoring[log_ids.detach().cpu()]
        user_vecs = self.base_model.user_encoder(torch.from_numpy(user_feature_vecs).cuda(), log_mask)
        AUC = []
        CSQ_AUC = []
        MRR = []
        nDCG5 = []
        nDCG10 = []
        all_scores = []
        all_labels = []
        news_ids = []
        csq_errors = 0
        for i in range(user_vecs.shape[0]): # iterate over batch size
            nonzeros = torch.nonzero(labels_mask[i].long(), as_tuple=True)
            labels_i = labels[i][nonzeros[0]]
            input_ids_i = input_ids[i][nonzeros[0]]
            news_vecs = self.news_scoring[input_ids_i.detach().cpu()]
            scores = torch.mm(torch.from_numpy(news_vecs).cuda(), user_vecs[i].unsqueeze(dim=-1)).squeeze()
            # scores = np.dot(news_vecs, user_vecs[i].unsqueeze(dim=-1).detach().cpu().numpy()).squeeze()
            label_np = labels_i.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            all_scores += list(scores)
            all_labels += list(label_np)
            news_ids += input_ids_i.tolist()

            auc = roc_auc_score(label_np, scores)
            mrr = mrr_score(label_np, scores)
            ndcg5 = ndcg_score(label_np, scores, k=5)
            ndcg10 = ndcg_score(label_np, scores, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        return {
            "auc": AUC,
            "mrr": MRR,
            "nDCG5": nDCG5,
            "nDCG10": nDCG10,
            "scores": np.array(all_scores),
            "labels": np.array(all_labels),
            "news_ids": np.array(news_ids),
        }

    def validation_epoch_end(self, outputs):
        """
        single validation epoch for pl.Trainer
        """
        auc = np.hstack([o["auc"] for o in outputs])
        mrr = np.hstack([o["mrr"] for o in outputs])
        nDCG5 = np.hstack([o["nDCG5"] for o in outputs])
        nDCG10 = np.hstack([o["nDCG10"] for o in outputs])
        # csq_auc = np.hstack([o["csq_auc"] for o in outputs])
        # csq_errors = np.hstack([o["csq_errors"] for o in outputs])
        all_scores = np.hstack([o["scores"] for o in outputs])
        all_labels = np.hstack([o["labels"] for o in outputs])
        all_ids = np.hstack([o["news_ids"] for o in outputs])
        print("Calculating CSQ AUC")
        csqe_label, csqe_score, num_csqs = self.csqe_label_and_prob(
            torch.from_numpy(all_scores),
            torch.from_numpy(all_labels),
            self.cold_start_ids,
            torch.from_numpy(all_ids)
        )
        val_csq_auc = roc_auc_score(csqe_label.detach().cpu().numpy(), csqe_score.detach().cpu().numpy())
        # try:
        #     val_csq_auc = roc_auc_score(csqe_label.detach().cpu().numpy(), csqe_score.detach().cpu().numpy())
        # except:
        #     print("Can't calculate CSQ AUC")
        #     val_csq_auc = 0

        log = {
            "val_auc": auc.mean(),
            "val_mrr": mrr.mean(),
            "val_ndcq5": nDCG5.mean(),
            "val_ndcq10": nDCG10.mean(),
            "val_csq_auc": val_csq_auc,
            # "csq_errors": csq_errors.sum() / len(auc),
            "num_csq_interactions": num_csqs,
            "cumulative_training_time": self.time_until_now,
        }
        print(log)
        self.log_dict(log)

        return log

    def on_test_start(self):
        return self.on_validation_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        # breakpoint()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr)#self.cfg.train.lr)
        # self.scheduler = get_noam_scheduler(
        #     self.optimizer,
        #     warmup_steps=4000,
        #     only_warmup=False,
        #     interval="step",
        # )
        return self.optimizer
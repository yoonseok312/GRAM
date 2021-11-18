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
from magneto.train.schedulers import get_noam_scheduler
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModel, AutoConfig



class LightningAlternatingHalfNRMSWithLM(pl.LightningModule):
    def __init__(self,
                 cfg,
                 news_index,
                 global_dict,
                 val_news_combined,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['news_index', 'global_dict', 'val_news_combined']
        )
        self.cfg = cfg
        self.base_model = None
        # self.news_embedding = nn.Embedding(130379+1, 64, padding_idx=0)
        self.user_encoder = AlternatingUserEncoder(cfg, news_index) # this is KT
        self.news_index = news_index
        self.global_dict = global_dict # key: news index의 values, values: 72dim vector (input for news encoder)
        self.automatic_optimization = False

        config = AutoConfig.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.model, output_hidden_states=True)
        if self.cfg.Alternating.use_prev_optimizer:
            config.attention_probs_dropout_prob = 0
            config.hidden_dropout_prob = 0
        bert_model = AutoModel.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.model, config=config)
        self.lm = NewsEncoder(cfg,bert_model,category_dict_size,domain_dict_size,subcategory_dict_size) # this is LM


        # if cfg[cfg.experiment_type[cfg.current_stage]].freeze_layers:
        #     for name, param in bert_model.named_parameters():
        #         if name not in finetuneset:
        #             param.requires_grad = False

        if cfg.dataset_size == 'large':
            with open(f'{cfg.root}/MIND/val_csqs_mind_large_list.pkl', 'rb') as f:
                csqs = pickle.load(f)
        else:
            with open(f'{cfg.root}/MIND_small/test_csqs_mind_small_list.pkl', 'rb') as f:
                csqs = pickle.load(f)
        self.cold_start_ids = []
        for item in csqs:
            self.cold_start_ids.append(self.news_index[item])
        self.non_cold_start_ids = []
        self.criterion = nn.CrossEntropyLoss()
        if cfg.dataset_size == 'large':
            if cfg.Alternating.base_lm.pooling in ['mean', 'init_mean'] and len(self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_attributes) == 3:
                with open(f'{cfg.root}/MIND/MIND_Large_BERT_title_abs_body_400_embeddings_768_mean.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                    self.pretrained_weight = pickle.load(f)
            elif cfg.Alternating.base_lm.pooling in ['mean', 'init_mean']:
                with open(f'{cfg.root}/MIND/MIND_Large_BERT_embeddings_768_mean.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                    self.pretrained_weight = pickle.load(f)
            elif cfg.Alternating.base_lm.pooling == 'att':
                with open(f'{cfg.root}/MIND/MIND_Large_BERT_embeddings_768_additive.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                    self.pretrained_weight = pickle.load(f)
            with open(f'{cfg.root}/MIND/MIND_newsid_to_count_np4_seed0_final.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                item_count = pickle.load(f)
        else:
            with open(f'{cfg.root}/MIND_small/MIND_small_BERT_embeddings_768_mean.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                self.pretrained_weight = pickle.load(f)
            with open(f'{cfg.root}/MIND/MIND_newsid_to_count_np4.pkl', 'rb') as f:#{cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
                item_count = pickle.load(f)
        self.news_combined = val_news_combined
        self.item_count_news_idx = {}
        print("Preparing id to count")
        for news_id in tqdm(item_count.keys()):
            if news_id in self.news_index:
                self.item_count_news_idx[self.news_index[news_id]] = item_count[news_id]
        print("len of idx to count", len(self.item_count_news_idx))
        self.max_count = max(np.sqrt(np.asarray(list(self.item_count_news_idx.values()))))
        print("max count", self.max_count)
        print(len(self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_attributes))
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_until_now = 0
        if len(self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_attributes) == 1:
            self.dim = 72
        else:
            self.dim = 1422


    def forward(self,
                input_ids,
                log_ids,
                log_idx,
                input_idx,
                log_mask,
                targets=None,
                compute_loss=False):
        # breakpoint()
        # input_ids: batch, history, num_words
        # ids_length = input_ids.size(2)
        # input_ids = input_ids.view(-1, ids_length)
        # new_input_idx = torch.transpose(torch.stack(input_idx),0,1)#torch.tensor(list(zip(input_idx[0].tolist(), input_idx[1].tolist())))#.cuda()
        news_vec = self.user_encoder.news_embedding(input_idx)
        news_vec = news_vec.view(
            -1,
            1 + self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].npratio,
            self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_dim
        )

        # batch_size, news_dim
        # log_ids = log_ids.view(-1, ids_length)
        # new_log_idx = torch.transpose(torch.stack(log_idx),0,1)#.cuda()
        log_vec = self.user_encoder.news_embedding(log_idx)
        log_vec = log_vec.view(
            -1,
            self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length,
            self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_dim
        )

        # log_vec, news_vec embedding table에서 bs x 64news
        user_vector = self.user_encoder(log_vec, log_mask)

        # batch_size, 2
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        return score#, new_input_idx, new_log_idx # self.base_model(input_ids, log_ids, log_mask, targets, False)

    def compute_lm_loss(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].weight_count
        assert weight_count in ["sqrt", None]
        if weight_count in ["sqrt"]:

            loss = F.mse_loss(logit, label, reduction="sum")
            sqrt_weight = torch.tensor([np.sqrt(self.item_count_news_idx[int(i.cpu().numpy())])
                                        / self.max_count for i in ids]).type(
                torch.FloatTensor)

            loss_before_reduction = F.mse_loss(logit, label, reduction="none").sum(-1)
            sqrt_weighted_loss = torch.dot(sqrt_weight.cuda(), loss_before_reduction)
        else:
            loss = F.mse_loss(logit, label, reduction="sum") * 0.5
        if weight_count == "sqrt":
            return sqrt_weighted_loss
        else:
            return loss
        return loss

    def compute_loss(self, score, targets):
        loss = self.criterion(score, targets)
        return loss
    # uncomment to log time
    # def on_train_epoch_start(self):
    #     print("Start recording...")
    #     self.start.record()

    # def on_train_epoch_end(self):
    #     print("End recording...")
    #     self.end.record()
    #     torch.cuda.synchronize()
    #     self.time_until_now += self.start.elapsed_time(self.end) / (1000 * 60) # millisecond to minute
    #     self.log("cumul_time_logged", self.time_until_now)

    def on_train_start(self) -> None:
        print("appending csqs")
        self.non_cold_start_ids = []
        for item in self.news_index.values():
            if item not in self.cold_start_ids:
                self.non_cold_start_ids.append(item)
        # FIXME: think about above logic
        # self.non_cold_start_ids = list(self.item_count_news_idx.keys())

        for item in tqdm(self.non_cold_start_ids):
            self.user_encoder.news_embedding.weight.data[item] = torch.tensor(self.pretrained_weight[item]).cuda()
            self.user_encoder.news_embedding.weight.data[item].requires_grad = True

    def on_train_epoch_start(self) -> None:
        print("Updating non cold start questions...")
        for i in range(int(len(self.non_cold_start_ids) / self.cfg.Alternating.num_q_split) + 1):
            if len(self.non_cold_start_ids) % self.cfg.Alternating.num_q_split == 0:
                if i == int(len(self.non_cold_start_ids) / self.cfg.Alternating.num_q_split):
                    continue
            lm_input = torch.stack([torch.from_numpy(self.global_dict[item]) for item in self.non_cold_start_ids[i * self.cfg.Alternating.num_q_split:(i + 1) * self.cfg.Alternating.num_q_split]])
            self.user_encoder.news_embedding.weight.data[self.non_cold_start_ids[i * self.cfg.Alternating.num_q_split:(i + 1) * self.cfg.Alternating.num_q_split]] \
                = self.lm(
                lm_input.reshape(-1, self.dim).cuda(),
            ).clone().detach().requires_grad_(True)

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
        self.unique_ids = []
        self.unique_ids += torch.unique(new_log_idx).tolist()
        self.unique_ids += [item for item in torch.unique(new_input_idx).tolist() if item not in self.unique_ids]
        try:
            self.unique_ids.remove(0)
        except:
            pass

        if self.cfg.Alternating.use_prev_optimizer:
            user_opt, lm_opt = self.optimizers()
            if self.cfg.Alternating.use_lm_noam:
                user_sch, lm_sch = self.lr_schedulers()
            else:
                user_sch = self.lr_schedulers()
        else:
            user_opt, lm_opt, emb_opt = self.optimizers()
            user_sch, lm_sch = self.lr_schedulers()
        y_hat = self(input_ids, log_ids, new_log_idx, new_input_idx, log_mask, targets)

        loss = self.compute_loss(y_hat, targets)
        user_opt.zero_grad()
        if not self.cfg.Alternating.use_prev_optimizer:
            emb_opt.zero_grad()
        self.manual_backward(loss)
        user_opt.step()
        if not self.cfg.Alternating.use_prev_optimizer:
            emb_opt.step()
        user_sch.step()
        self.log("train_user_loss", loss)
        num_gpus = len(str(self.cfg.gpus).split(','))
        distributed_length = math.ceil(len(self.train_dataloader.dataloader) / num_gpus)
        # if batch_idx == 1:
        print(int(len(self.train_dataloader.dataloader)/2))
        if distributed_length - 1 == batch_idx or batch_idx == 10:
            # self.non_cold_start_ids = list(self.item_count_news_idx.keys())
            embedding_label_list = []
            for id in self.unique_ids:
                embedding_label_list.append(
                    np.asarray(
                        self.user_encoder.news_embedding.weight.data[id].detach().cpu()
                    )
                )

            train_loader = DataLoader(
                TensorDataset(torch.tensor(np.asarray(self.unique_ids), dtype=torch.float).cuda(),
                              torch.tensor(np.asarray(embedding_label_list), dtype=torch.float).cuda()),
                batch_size=self.cfg.Alternating.lm_batch_size, shuffle=True)
            print('training LM...', batch_idx)
            for ids, labels in tqdm(train_loader):
                ids_numpy = ids.detach().cpu().numpy()
                temp_ids = []
                for id in ids_numpy:
                    temp_ids.append(self.global_dict[int(id)])
                x_data = torch.tensor(np.asarray(temp_ids)).cuda()
                output = self.lm(x_data)
                loss = self.compute_lm_loss(
                    labels,
                    output,
                    ids
                )
                lm_opt.zero_grad()
                self.manual_backward(loss)
                lm_opt.step()
                if self.cfg.Alternating.use_lm_noam:
                    lm_sch.step()
                # print("lr", lm_opt.state_dict)
                wandb.log(
                    {"train_lm_loss": loss}
                )
                self.log("train_lm_loss", loss)
            # print("End recording...") # uncomment to log time
            # self.end.record()
            # torch.cuda.synchronize()
            # time = self.start.elapsed_time(self.end)
            time = 0
            print("Elapsed time", time)
            self.time_until_now += time / (1000 * 60)  # millisecond to minute
            self.log("cumul_time_logged", self.time_until_now)
            print('Chaning CSQs...')
            if len(self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_attributes) == 1:
                dim = 72
            else:
                dim = 1422
            for item in tqdm(self.cold_start_ids):
                self.user_encoder.news_embedding.weight.data[item] \
                    = self.lm(torch.tensor(self.global_dict[item]).reshape(-1,dim).cuda()).reshape(-1)

        return loss

    # def on_validation_start(self) -> None:
    #     self.news_scoring = []
    #     news_dataset = NewsDataset(self.news_combined)
    #     news_dataloader = DataLoader(
    #         news_dataset,
    #         batch_size=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].batch_size,
    #         num_workers=self.cfg.num_workers,
    #     )
    #     with torch.no_grad():
    #         for input_ids in tqdm(news_dataloader):
    #             news_vec = self.lm(input_ids.cuda())
    #             news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
    #             self.news_scoring.extend(news_vec)
    #     self.news_scoring = np.array(self.news_scoring)

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
        input_id_embedding_idx = batch[6]

        # user_feature_vecs = self.news_scoring[log_ids.detach().cpu()]
        log_vec = self.user_encoder.news_embedding(log_ids)
        log_vec = log_vec.view(
            -1,
            self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length,
            self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_dim
        )
        # user_feature_vecs = self.user_encoder.news_embedding(self.news_index[log_ids])
        user_vecs = self.user_encoder(log_vec, log_mask)

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
            #news_vecs = self.news_scoring[input_ids_i.detach().cpu()]

            news_vecs = self.user_encoder.news_embedding(input_ids_i)
            # breakpoint()
            # scores = np.dot(news_vecs.detach().cpu().numpy(), user_vecs[i].unsqueeze(dim=-1).detach().cpu().numpy()).squeeze()
            scores = torch.mm(news_vecs, user_vecs[i].unsqueeze(dim=-1)).squeeze()
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
            # "csq_auc": CSQ_AUC,
            # "csq_errors": csq_errors,

        }

    def csqe_label_and_prob(self, probs, label, id_list, items):
        csqe_items_idx = torch.cat([(items == i).nonzero() for i in id_list])
        csqe_prob = torch.gather(probs, dim=-1, index=csqe_items_idx.squeeze())
        csqe_label = torch.gather(label, dim=-1, index=csqe_items_idx.squeeze())
        # csqe_prob = torch.argmax(torch.softmax(csqe_logit, -1), -1)

        return csqe_label, csqe_prob, len(csqe_prob)


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
        # print("Calculating CSQ AUC")
        # csqe_label, csqe_score, num_csqs = self.csqe_label_and_prob(
        #     torch.from_numpy(all_scores),
        #     torch.from_numpy(all_labels),
        #     self.cold_start_ids,
        #     torch.from_numpy(all_ids),
        # )
        # val_csq_auc = roc_auc_score(csqe_label.detach().cpu().numpy(), csqe_score.detach().cpu().numpy())


        # if csq_errors.sum() == len(auc):
        #     val_csq_auc = 0
        # else:
        #     val_csq_auc = 0
            # val_csq_auc = csq_auc.mean() # uncomment for csq auc


        log = {
            "val_auc": auc.mean(),
            "val_mrr": mrr.mean(),
            "val_ndcq5": nDCG5.mean(),
            "val_ndcq10": nDCG10.mean(),
            "val_csq_auc": 0,
            # "csq_errors": csq_errors.sum() / len(auc),
            "num_csq_interactions": 0,
            "cumulative_training_time": self.time_until_now,
        }
        print(log)
        self.log_dict(log)

        return log

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


    def configure_optimizers(self):
        if self.cfg.Alternating.use_prev_optimizer:
            self.user_encoder_optimizer = torch.optim.Adam(self.user_encoder.parameters(),lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr)
            self.user_encoder_scheduler = get_noam_scheduler(
                self.user_encoder_optimizer,
                warmup_steps=4000,
                only_warmup=False,
                interval="step",
            )
            self.lm_optimizer = torch.optim.Adam(self.lm.parameters(),
                                                 lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr)
            self.lm_scheduler = get_noam_scheduler(
                self.lm_optimizer,
                warmup_steps=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_warmup,
                only_warmup=False,
                interval="step",
            )
            if self.cfg.Alternating.use_lm_noam:
                return (
                    {"optimizer": self.user_encoder_optimizer, "lr_scheduler": self.user_encoder_scheduler},
                    {"optimizer": self.lm_optimizer, "lr_scheduler": self.lm_scheduler},
                )
            else:
                return (
                    {"optimizer": self.user_encoder_optimizer, "lr_scheduler": self.user_encoder_scheduler},
                    {"optimizer": self.lm_optimizer},
                )
        else:
            my_list = ['news_embedding.weight']
            user_encoder_params = list(filter(lambda kv: kv[0] not in my_list, self.user_encoder.named_parameters()))
            weights = [{'params': pg[1]} for pg in user_encoder_params]
            self.user_encoder_optimizer = torch.optim.Adam(weights, lr=self.cfg[
                self.cfg.experiment_type[self.cfg.current_stage]].lr)
            self.user_encoder_scheduler = get_noam_scheduler(
                self.user_encoder_optimizer,
                warmup_steps=4000,
                only_warmup=False,
                interval="step",
            )
            self.emb_optimizer = torch.optim.SGD(self.user_encoder.news_embedding.parameters(), lr=1)
            self.lm_optimizer = torch.optim.Adam(self.lm.parameters(),
                                                 lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr)
            self.lm_scheduler = get_noam_scheduler(
                self.lm_optimizer,
                warmup_steps=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_warmup,
                only_warmup=False,
                interval="step",
            )
            return (
                {"optimizer": self.user_encoder_optimizer, "lr_scheduler": self.user_encoder_scheduler},
                {"optimizer": self.lm_optimizer, "lr_scheduler": self.lm_scheduler},
                {"optimizer": self.emb_optimizer},
            )


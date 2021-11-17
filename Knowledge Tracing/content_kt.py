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
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

import copy


from repoc_content_kt.models.components.input_embedding import AllItemInputEmbedding
from repoc_content_kt.models.components.SBERT import SBERT
from repoc_common import utils
from repoc_content_kt.models.components.sbert_regressor import LightningRegressor

from magneto.train.schedulers import get_noam_scheduler

from torch.utils.data import DataLoader, TensorDataset
from repoc_content_kt.models.components.sbert_regressor import LightningRegressor


class ContentAllItemGenerator(nn.Module):
    def __init__(self, dim_model, num_items, enc_embed, config):
        super(ContentAllItemGenerator, self).__init__()
        self.cfg = config

        if self.cfg.experiment_type != 'D' or (self.cfg.experiment_type == 'D' and self.cfg.D.use_exp_c_kt_module):
            if self.cfg.param_shared:
                self.question_embedding = enc_embed.embed_feature.shifted_item_id.weight
            else:
                self.generator = nn.Linear(dim_model, num_items)
        if self.cfg.add_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.cfg[self.cfg.ckt_dataset_type].data.max_item_id + 1), requires_grad=True
            )

    def forward(self, x, sbert_embed):
        if self.cfg.experiment_type[self.cfg.current_stage] == 'D' and self.cfg.param_shared:
            output = torch.matmul(x, torch.transpose(sbert_embed, 0, 1))
        elif self.cfg.param_shared:
            output = torch.matmul(x, torch.transpose(self.question_embedding, 0, 1))
        else:
            output = self.generator(x)
        if self.cfg.add_bias:
            output += self.bias
        return output


class ContentAllItemKT(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self, config) -> None:
        super(ContentAllItemKT, self).__init__()
        self.cfg = config
        print("init model")
        self.enc_embed = AllItemInputEmbedding(config, "encoder")
        # self.dec_embed = InputEmbedding(config, "decoder")

        # self.transformer = nn.Transformer(
        #     d_model=self.cfg.train.dim_model,
        #     nhead=self.cfg.train.head_count,
        #     num_encoder_layers=self.cfg.train.encoder_layer_count,
        #     num_decoder_layers=self.cfg.train.decoder_layer_count,
        #     dim_feedforward=self.cfg.train.dim_feedforward,
        #     dropout=self.cfg.train.dropout_rate,
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.train.dim_model,
            nhead=self.cfg.train.head_count,
            dim_feedforward=self.cfg.train.dim_feedforward,
            dropout=self.cfg.train.dropout_rate,
            activation="relu",
        )
        encoder_norm = nn.LayerNorm(self.cfg.train.dim_model)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.train.encoder_layer_count,
            norm=encoder_norm,
        )

        self.generator = ContentAllItemGenerator(
            self.cfg.train.dim_model, self.cfg[self.cfg.ckt_dataset_type].data.max_item_id, self.enc_embed, self.cfg
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # embed_all: torch.Tensor
    ) -> torch.Tensor:
        # enc_input, dec_input = self.enc_embed(batch), self.dec_embed(batch)
        # enc_input, dec_input = map(lambda x: x.transpose(0, 1), (enc_input, dec_input))  # seq_size * batch * dim_model
        enc_input, embed_all = self.enc_embed(batch)
        enc_input = enc_input.transpose(0, 1)

        # 3. Prepare masks for transformer inference
        # [[3], [2], [5]] -> [[ffftt], [ffttt], [fffff]]
        subseq_mask = utils.generate_square_subsequent_mask(self.cfg[self.cfg.ckt_dataset_type].data.max_seq_len).to(
            enc_input.device
        )

        # get transformer output
        # tr_output = self.transformer(
        #     src=enc_input,
        #     tgt=dec_input,
        #     src_mask=subseq_mask,
        #     tgt_mask=subseq_mask,
        #     memory_mask=subseq_mask,
        #     src_key_padding_mask=batch["pad_mask"],
        #     tgt_key_padding_mask=batch["pad_mask"],
        # )
        tr_output = self.transformer(
            src=enc_input,
            mask=subseq_mask,
            src_key_padding_mask=batch["pad_mask"],
        )
        tr_output = torch.transpose(tr_output, 0, 1)  # batch * seq_size * dim_model
        output = self.generator(tr_output, embed_all)
        # output = self.generator(tr_output)
        return output


class LightningContentAllItemKT(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()  # stores hyperparameters
        self.cfg = cfg
        self.threshold = 0.5
        self.base_model = ContentAllItemKT(cfg)
        self.steps = 0
        self.lm_steps = 0
        if cfg.ckt_dataset_type == 'toeic':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/cold_start_ids_part_4_to_7.txt", "rb") as f:
                self.cold_start_ids = pickle.load(f)
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].language == 'spanish':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_test_csqe.pkl", "rb") as f:
                    dic = pickle.load(f)
                    self.cold_start_ids = list(dic.keys())
            else:
                if cfg[cfg.ckt_dataset_type].user_based_split:
                    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe_newsplit_updated.pkl", "rb") as f:
                        dic = pickle.load(f)
                        self.cold_start_ids = list(dic.keys())
                else:
                    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe.pkl", "rb") as f:
                        dic = pickle.load(f)
                        self.cold_start_ids = list(dic.keys())
        elif cfg.ckt_dataset_type == 'poj':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_csqs_new_split.pkl", "rb") as f:
                self.cold_start_ids = pickle.load(f)
        if self.cfg.experiment_type[self.cfg.current_stage] == 'Alternating':
            self.lm = LightningRegressor(cfg)
        if cfg.ckt_dataset_type == 'toeic':
            if cfg[cfg.experiment_type[cfg.current_stage]].with_passage:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text_special_tokens.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
            else:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].language == 'french':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
            elif cfg[cfg.ckt_dataset_type].language == 'spanish':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_all.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'poj':
            print("Truncated text")
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl", "rb") as handle:
                self.id_to_text = pickle.load(handle)

        # dataset = self.train_dataloader()
        # dataset_size = len(dataset)
        # # dataset_size = (
        # #     self.trainer.limit_train_batches
        # #     if self.trainer.limit_train_batches != 0
        # #     else len(dataset)
        # # )
        # num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        #
        # effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        # self.steps_per_epoch = (dataset_size // effective_batch_size) * 1
        # breakpoint()
        self.training_kt_module = True
        self.steps_per_epoch = 1
        # if not self.cfg.init_with_text_embedding:
        #     self.SBERT = SBERT('paraphrase-TinyBERT-L6-v2') # have to uncomment for loading ckpts before refactoring


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # encoded_all: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        output = self.base_model(batch)
        # print(output.size())
        return output

    def csqe_label_and_prob(self, logits, label, id_list, items):
        csqe_items_idx = torch.cat([(items == i).nonzero() for i in id_list])
        # breakpoint()
        csqe_logit = torch.gather(logits, dim=-1, index=csqe_items_idx.squeeze())
        # breakpoint()
        csqe_label = torch.gather(label, dim=-1, index=csqe_items_idx.squeeze())
        # breakpoint()
        csqe_prob = csqe_logit.sigmoid()
        return csqe_label, csqe_prob

    # def on_train_epoch_start(self) -> None:
    #     self.training_kt_module = True
        # dataset = self.train_dataloader()
        # dataset_size = len(dataset)
        # num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        #
        # effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        # self.steps_per_epoch = (dataset_size // effective_batch_size) * 1
        # breakpoint()

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        logit: Dict[str, torch.Tensor],
        is_training: bool = True,
        is_testing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if is_training:
            nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
            items = batch["item_id"][nonzeros[0], nonzeros[1]]
            logits = logit[nonzeros[0], nonzeros[1]]
            logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze()
            prob = logit.sigmoid()
            pred = (prob > self.threshold).long()
            loss = F.binary_cross_entropy_with_logits(logit, label, reduction="mean")
            return loss, label, prob, pred
        else:
            if self.cfg.ckt_dataset_type in ['toeic', 'poj'] or self.cfg[self.cfg.ckt_dataset_type].user_based_split == True:
                nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
                label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
                items = batch["item_id"][nonzeros[0], nonzeros[1]]
                logits = logit[nonzeros[0], nonzeros[1]]
                logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze()
                prob = logit.sigmoid()
                pred = (prob > self.threshold).long()
                loss = F.binary_cross_entropy_with_logits(logit, label, reduction="mean")
                if is_testing:
                    csqe_label, csqe_prob = self.csqe_label_and_prob(
                        logit, label, self.cold_start_ids, items
                    )
                    # non_csq_ids = [id for id in range(0, 19398) if id not in self.cold_start_ids]
                    # non_csqe_label_last, non_csqe_prob_last = self.csqe_label_and_prob(logit, label, non_csq_ids, items)
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
                else:
                    if self.cfg[self.cfg.ckt_dataset_type].run_test_as_val:
                        csqe_label, csqe_prob = self.csqe_label_and_prob(
                            logit, label, self.cold_start_ids, items
                        )
                    else:
                        csqe_label, csqe_prob = torch.LongTensor([1, 0]), torch.LongTensor(
                            [0, 1]
                        )  # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
            else:
                nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
                label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
                items = batch["item_id"][nonzeros[0], nonzeros[1]]
                seq_sizes = batch["sequence_size"]
                last_seq = []
                for idx, seq_size in enumerate(seq_sizes):
                    last_seq += [sum(seq_sizes[idx][0].item() for idx in range(idx + 1)) - 1]
                logits = logit[nonzeros[0], nonzeros[1]]
                logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze()
                last_seq = torch.cuda.LongTensor(last_seq)
                logit_last = torch.gather(logit, dim=-1, index=last_seq)
                label_last = torch.gather(label, dim=-1, index=last_seq)
                label = label_last
                item_last = torch.gather(items, dim=-1, index=last_seq)
                prob = logit_last.sigmoid()
                pred = (prob > self.threshold).long()
                loss = F.binary_cross_entropy_with_logits(logit_last, label_last, reduction="mean")
                if is_testing:
                    # csqe_label, csqe_prob = self.csqe_label_and_prob(
                    #     logit, label, self.cold_start_ids, items
                    # )
                    # # non_csq_ids = [id for id in range(0, 19398) if id not in self.cold_start_ids]
                    # # non_csqe_label_last, non_csqe_prob_last = self.csqe_label_and_prob(logit, label, non_csq_ids, items)

                    csqe_list = self.cold_start_ids
                    csqe_label, csqe_prob = self.csqe_label_and_prob(logit_last, label_last, csqe_list,
                                                                               item_last)
                      # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor([1, 0]), torch.LongTensor(
                        [0, 1])  # to speed up training
                else:
                    if self.cfg[self.cfg.ckt_dataset_type].run_test_as_val:
                        # breakpoint()
                        csqe_label, csqe_prob = self.csqe_label_and_prob(
                            logit_last, label_last, self.cold_start_ids, item_last
                        )
                        # breakpoint()
                    else:
                        csqe_label, csqe_prob = torch.LongTensor([1, 0]), torch.LongTensor(
                            [0, 1]
                        )  # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor([1, 0]), torch.LongTensor(
                        [0, 1])  # to speed up training
            return (
                loss,
                label,
                prob,
                pred,
                csqe_label,
                csqe_prob,
                non_csqe_label_last,
                non_csqe_prob_last,
            )

    def on_train_epoch_end(self) -> None:
        # breakpoint()
        self.training_kt_module = False

    def training_step_helper(self, batch):
        logit = self(batch)
        loss, _, _, _ = self.compute_loss(batch, logit)

        # # manually increase the step size
        # self.scheduler.step_batch()

        self.log("train_loss", loss)
        # wandb.log(
        #     {"train_loss": loss}
        # )

        return loss
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int=0) -> torch.Tensor: # Need to add , optimizer_idx: int for alternating

        # just return loss for now
        # Train KT module
        if self.cfg.experiment_type[self.cfg.current_stage] != "Alternating":
            loss = self.training_step_helper(batch)

            return loss
        elif self.cfg.experiment_type[self.cfg.current_stage] == "Alternating" and self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].alternate_by_epoch:
            if optimizer_idx == 0 and self.lm.training_lm == False:  # 100 step per epoch
                loss = self.training_step_helper(batch)
                return loss
            elif optimizer_idx == 1 and not self.training_kt_module:  # self.steps % 100 == 0: # and self.steps % 10 == 0
                # print("train lm")
                # if self.cfg[self.cfg.experiment_type[
                #     self.cfg.current_stage]].alternate_by_epoch and not self.training_kt_module:
                checkpoint_callback = ModelCheckpoint(
                    dirpath="",
                    filename="",
                    save_top_k=0,
                    every_n_train_steps=0,
                )

                trainer_regressor = pl.Trainer(
                    callbacks=[
                        checkpoint_callback,
                    ],
                    max_epochs=1,
                    accelerator="ddp",
                    gpus=None if self.cfg.gpus is None else str(self.cfg.gpus),
                    logger=pl.loggers.WandbLogger(project="pipeline_B",
                                                  name=self.cfg.exp_name + "_" + self.cfg.current_stage) if self.cfg.B.use_wandb else None,
                    deterministic=True,
                    # val_check_interval=self.cfg.B.val_check_interval,
                )

                item_ids_list = list(self.id_to_text.keys())
                non_cold_start_ids = []
                ### FIXME

                if self.cfg.ckt_dataset_type == 'duolingo':
                    item_count = np.load(f'{self.cfg[self.cfg.ckt_dataset_type].data.root}/item_count_new_french.npy', allow_pickle='TRUE').item()
                elif self.cfg.ckt_dataset_type == 'poj':
                    with open(f"{self.cfg[self.cfg.ckt_dataset_type].data.root}/poj_id_to_count_only_train.pkl",
                              "rb") as handle:
                        item_count = pickle.load(handle)
                elif self.cfg.ckt_dataset_type == 'toeic':
                    with open(f"{self.cfg[self.cfg.ckt_dataset_type].data.root}/toeic_id_to_count.pkl",
                              "rb") as handle:
                        item_count = pickle.load(handle)

                for id in item_ids_list:
                    if id not in self.cold_start_ids and id in item_count:
                        non_cold_start_ids.append(id)

                embedding_label_list = []
                for id in non_cold_start_ids:
                    embedding_label_list.append(
                        np.asarray(
                            self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[id].detach().cpu()))

                train_loader = DataLoader(
                    TensorDataset(torch.tensor(np.asarray(non_cold_start_ids), dtype=torch.float),
                                  torch.tensor(np.asarray(embedding_label_list), dtype=torch.float)),
                    batch_size=self.cfg.Alternating.regressor_batch_size, shuffle=True)
                self.lm.lr_denominator = self.cfg.Alternating.lm_lr_decay_rate * self.current_epoch + 1
                trainer_regressor.fit(self.lm, train_loader)
                self.training_kt_module = True
                for item in non_cold_start_ids:
                    self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item] = self.lm([item])
                for item in self.cold_start_ids:
                    self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item] = self.lm([item])

                return None
        else:
            if optimizer_idx == 0: # 100 step per epoch
                loss = self.training_step_helper(batch)
                return loss
            elif optimizer_idx == 1: # self.steps % 100 == 0: # and self.steps % 10 == 0
                # print("train lm")
                nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
                items = batch["item_id"][nonzeros[0], nonzeros[1]]
                items = torch.unique(items)
                labels = torch.gather(
                    self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data,
                    dim=0,
                    index=items.unsqueeze(-1).repeat(1, 768)
                )
                output = self.lm(items)
                loss = self.lm.compute_loss(labels, output, items)
                # wandb.log(
                #     {"train_lm_loss": loss}
                # )
                if self.cfg.Alternating.change_kt_emb:
                    unique_ids= []
                    id_to_emb = list(zip(items, output))
                    for item in id_to_emb:
                        if item[0] in unique_ids:
                            continue
                        else:
                            unique_ids.append(item[0])
                            self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item[0]] = item[1].clone().detach()
                return loss

    # def on_validation_start(self) -> None:
    #     if self.cfg.experiment_type[self.cfg.current_stage] == 'Alternating' and not self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].alternate_by_epoch:
    #         for qid in self.cold_start_ids:
    #             self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
    #                 qid
    #             ] = self.lm([qid])

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """
        logit = self(batch)
        loss, labels, probs, preds, csqe_labels, csqe_probs, _, _ = self.compute_loss(
            batch, logit, is_training=False
        )
        return {
            "loss": loss.item(),
            "labels": labels.data.cpu().tolist(),
            "probs": probs.data.cpu().tolist(),
            "preds": preds.data.cpu().tolist(),
            "csqe_labels": csqe_labels.data.cpu().tolist(),
            "csqe_probs": csqe_probs.data.cpu().tolist(),
        }

    def validation_epoch_end(self, outputs):
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

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        logit = self(batch)
        # loss, labels, probs, preds, csqe_labels, csqe_preds = self.compute_loss(batch, logit)
        (
            loss,
            labels,
            probs,
            preds,
            csqe_labels,
            csqe_probs,
            non_csqe_labels,
            non_csqe_probs,
        ) = self.compute_loss(batch, logit, is_training=False, is_testing=True)
        return {
            "loss": loss.item(),
            "labels": labels.data.cpu().tolist(),
            "probs": probs.data.cpu().tolist(),
            "preds": preds.data.cpu().tolist(),
            "csqe_labels": csqe_labels.data.cpu().tolist(),
            "csqe_probs": csqe_probs.data.cpu().tolist(),
            "non_csqe_labels": non_csqe_labels.data.cpu().tolist(),
            "non_csqe_probs": non_csqe_probs.data.cpu().tolist(),
        }

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])
        csqe_labels = np.hstack([o["csqe_labels"] for o in outputs])
        csqe_probs = np.hstack([o["csqe_probs"] for o in outputs])
        non_csqe_labels = np.hstack([o["non_csqe_labels"] for o in outputs])
        non_csqe_probs = np.hstack([o["non_csqe_probs"] for o in outputs])

        acc = (preds == labels).sum().item() / len(preds)
        auc = metrics.roc_auc_score(labels, probs)
        loss = loss.mean().item()

        print("num of CSQ interactions", len(csqe_labels))
        # print("num of non CSQ interactions", len(non_csqe_labels))
        print("num of total interactions", len(labels))
        csqe_auc = metrics.roc_auc_score(csqe_labels, csqe_probs)
        non_csqe_auc = metrics.roc_auc_score(non_csqe_labels, non_csqe_probs)

        # try:
        #     csqe_auc = metrics.roc_auc_score(csqe_labels, csqe_preds)
        #     non_csqe_auc = metrics.roc_auc_score(non_csqe_labels, non_csqe_preds)
        #     # auc_list.append(auc)
        # except ValueError:
        #     print("labels", csqe_labels)
        #     print("probs", csqe_preds)
        #     pass

        log = {
            "test_acc": acc,
            "test_loss": loss,
            "test_auc": auc,
            "test_csqe_auc": csqe_auc,
            "test_non_csqe_auc": non_csqe_auc,
        }
        print(log)
        self.log_dict(log)
        # wandb.log({
        #     'test_acc': acc,
        #     'test_loss': loss,
        #     "test_auc": auc,
        #     "test_csqe_auc": csqe_auc,
        #     "test_non_csqe_auc": non_csqe_auc,
        # })

        return log

    def configure_optimizers(self):
        if self.cfg.experiment_type[self.cfg.current_stage] != "Alternating":
            self.kt_optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr)
            self.kt_scheduler = get_noam_scheduler(
                self.kt_optimizer,
                warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
                only_warmup=False,
                interval="step",
            )
            return {"optimizer": self.kt_optimizer, "lr_scheduler": self.kt_scheduler}
            # return [self.kt_optimizer, None], [self.kt_scheduler, None]
            # return [self.kt_optimizer], [self.kt_scheduler]
        else:
            self.kt_optimizer = torch.optim.Adam(self.base_model.parameters(),
                                                 lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr)
            self.kt_scheduler = get_noam_scheduler(
                self.kt_optimizer,
                warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
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
            # return [self.kt_optimizer, self.lm_optimizer], [self.kt_scheduler, self.lm_scheduler]
            return [self.kt_optimizer, self.lm_optimizer]
            # return {
            #     "optimizer": [self.kt_optimizer, self.lm_optimizer],
            #     "lr_scheduler": [self.kt_scheduler, self.lm_scheduler]
            # }

        # return {"optimizer": self.optimizer}

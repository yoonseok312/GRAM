from typing import Dict, Tuple, List, Any

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch_optimizer as advanced_optim
from magneto.train.schedulers import get_noam_scheduler
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import pickle

from repoc_content_kt.models.content_saint_base import ContentSaintBaseModel


def retrieve_from_tensor_with_mask(tensor: torch.Tensor, mask: torch.LongTensor):
    """
    args
        tensor: (B, L) or (B, L, D)
        mask: (B, L)
    returns
        ret: (S,) or (S, D)
    """
    nonzeros = torch.nonzero(mask, as_tuple=True)
    return tensor[nonzeros[0], nonzeros[1]]


class ContentSaintModel(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        super(ContentSaintModel, self).__init__()
        self.save_hyperparameters()  # stores hyperparameters
        self.cfg = cfg
        self.threshold = 0.5
        self.base_model = ContentSaintBaseModel(cfg)
        with open("/root/duolingo/item_id_to_text_test_csqe.pkl", "rb") as f:
            dic = pickle.load(f)
            self.cold_start_ids = list(dic.keys())

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        output = self.base_model(batch)

        return output

    def csqe_label_and_prob(self, logits, label, id_list, items):
        csqe_items_idx = torch.cat([(items == i).nonzero() for i in id_list])
        csqe_logit = torch.gather(logits, dim=-1, index=csqe_items_idx.squeeze())
        csqe_label = torch.gather(label, dim=-1, index=csqe_items_idx.squeeze())
        csqe_prob = csqe_logit.sigmoid()
        return csqe_label, csqe_prob

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        logit: Dict[str, torch.Tensor],
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if is_training:
            nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
            items = batch["item_id"][nonzeros[0], nonzeros[1]]
            logit = logit[nonzeros[0], nonzeros[1]]
            # logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze() # only for all-item KT
            prob = logit.sigmoid()
            pred = (prob > self.threshold).long()
            loss = F.binary_cross_entropy_with_logits(logit, label, reduction="mean")
            # breakpoint()
            return loss, label, prob, pred
        else:
            nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
            items = batch["item_id"][nonzeros[0], nonzeros[1]]
            seq_sizes = batch["sequence_size"]
            last_seq = []
            for idx, seq_size in enumerate(seq_sizes):
                last_seq += [sum(seq_sizes[idx][0].item() for idx in range(idx + 1)) - 1]
            logit = logit[nonzeros[0], nonzeros[1]]
            # logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze()
            last_seq = torch.cuda.LongTensor(last_seq)
            logit_last = torch.gather(logit, dim=-1, index=last_seq)
            label_last = torch.gather(label, dim=-1, index=last_seq)
            item_last = torch.gather(items, dim=-1, index=last_seq)
            prob_last = logit_last.sigmoid()
            pred_last = (prob_last > self.threshold).long()
            loss_last = F.binary_cross_entropy_with_logits(logit_last, label_last, reduction="mean")

            csqe_list = self.cold_start_ids
            # csqe_label_last, csqe_prob_last = self.csqe_label_and_prob(logit_last, label_last, csqe_list, item_last)
            csqe_label_last, csqe_prob_last = torch.LongTensor([1, 0]), torch.LongTensor([0, 1])
            # non_csqe_list = [i for i in range(0, 5001) if i not in csqe_list]
            # non_csqe_label_last, non_csqe_prob_last = self.csqe_label_and_prob(logit_last, label_last, non_csqe_list, item_last)
            non_csqe_label_last, non_csqe_prob_last = torch.LongTensor([1, 0]), torch.LongTensor(
                [0, 1]
            )  # to speed up training
            # breakpoint()
            return (
                loss_last,
                label_last,
                prob_last,
                pred_last,
                csqe_label_last,
                csqe_prob_last,
                non_csqe_label_last,
                non_csqe_prob_last,
            )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:

        # just return loss for now
        logit = self(batch)
        loss, _, _, _ = self.compute_loss(batch, logit)

        # manually increase the step size
        # self.scheduler.step_batch()

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, List]:

        logit = self(batch)
        (
            loss,
            labels,
            probs,
            preds,
            csqe_labels,
            csqe_probs,
            non_csqe_labels,
            non_csqe_probs,
        ) = self.compute_loss(batch, logit, is_training=False)
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

    # def validation_step(
    #     self, batch: Dict[str, torch.Tensor], batch_idx: int
    # ) -> Dict[str, List]:

    #     outputs = {"loss": [], "label": [], "prob": [], "pred": []}
    #     sub_batch_size = len(batch) * 20 # // 10
    #     augmented_batch_size = len(batch["item_id"]) #// 50
    #     for sub_batch_i in range(0, augmented_batch_size, sub_batch_size):
    #         start_idx = sub_batch_i
    #         end_idx = min(
    #             (sub_batch_i + sub_batch_size), augmented_batch_size
    #         )
    #         assert start_idx <= end_idx
    #         cur_sub_batch = {
    #             key: val[start_idx:end_idx] for key, val in batch.items()
    #         }

    #         logit = self(cur_sub_batch)
    #         loss, label, prob, pred = self.compute_loss(cur_sub_batch, logit)
    #         outputs["loss"].append(loss.item())
    #         outputs["label"].append(label)#.data.cpu().tolist()
    #         outputs["prob"].append(prob)#.data.cpu().tolist()
    #         outputs["pred"].append(pred)#.data.cpu().tolist()

    #     # logit = torch.cat(logit, axis=0)
    #     # logit = self(batch)
    #     outputs["label"] = torch.cat(outputs["label"], axis=0).data.cpu().tolist()
    #     outputs["prob"] = torch.cat(outputs["prob"], axis=0).data.cpu().tolist()
    #     outputs["pred"] = torch.cat(outputs["pred"], axis=0).data.cpu().tolist()

    #     # print({k:len(v) for k,v in outputs.items()})

    #     return outputs
    #     # return {
    #     #     "loss": loss.item(),
    #     #     "label": label.data.cpu().tolist(),
    #     #     "prob": prob.data.cpu().tolist(),
    #     #     "pred": pred.data.cpu().tolist(),
    #     # }

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # loss = np.hstack([o["loss"] for o in outputs])
        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])
        csqe_labels = np.hstack([o["csqe_labels"] for o in outputs])
        csqe_probs = np.hstack([o["csqe_probs"] for o in outputs])
        non_csqe_labels = np.hstack([o["non_csqe_labels"] for o in outputs])
        non_csqe_probs = np.hstack([o["non_csqe_probs"] for o in outputs])

        # print(loss.shape, labels.shape, probs.shape, preds.shape)

        acc = (preds == labels).sum().item() / len(preds)
        auc = roc_auc_score(labels, probs)
        if len(csqe_labels) == 0 or len(non_csqe_labels) == 0:
            # breakpoint()
            csqe_auc = 0
            non_csqe_auc = 0
        else:
            # breakpoint()
            csqe_auc = roc_auc_score(csqe_labels, csqe_probs)
            non_csqe_auc = roc_auc_score(non_csqe_labels, non_csqe_probs)
        # try:
        #     csqe_auc = roc_auc_score(csqe_labels, csqe_probs)
        # except ValueError:
        #     csqe_auc = 0
        #     print("labels", labels)
        #     print("probs", probs)
        #     pass
        loss = np.mean(loss)  # .mean().item()

        log = {
            "val_acc": acc,
            "val_loss": loss,
            "val_auc": auc,
            "val_csq_auc": csqe_auc,
            "val_non_csq_auc": non_csqe_auc,
        }
        print(log)
        self.log_dict(log)

        # self.log("val_loss", res["val_loss"])
        return log

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])

        acc = (preds == labels).sum().item() / len(preds)
        auc = roc_auc_score(labels, probs)
        loss = loss.mean().item()

        log = {
            "test_acc": acc,
            "test_loss": loss,
            "test_auc": auc,
        }
        print(log)
        self.log_dict(log)

        return log

    def configure_optimizers(self) -> Dict[str, Any]:

        if self.cfg.train.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters())
        elif self.cfg.train.optimizer == "radam":
            self.optimizer = advanced_optim.RAdam(
                self.parameters(),
                lr=0.001,
                betas=[0.9, 0.999],
                eps=1e-08,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"unknown optimizer: {self.cfg.train.optimizer}")

        self.scheduler = get_noam_scheduler(
            self.optimizer,
            warmup_steps=4000,
            only_warmup=False,
            interval="step",
        )
        # self.scheduler = NoamLR(
        #     optimizer,
        #     self.cfg.train.dim_model,
        #     self.cfg.warmup,
        #     factor=self.cfg.factor,
        # )
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

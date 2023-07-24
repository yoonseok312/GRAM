from typing import Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import wandb


class MLP1(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self) -> None:
        super(MLP1, self).__init__()
        torch.manual_seed(0)
        self.activation = nn.ReLU()
        torch.manual_seed(0)
        self.linear2 = nn.Linear(784, 64)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        hidden_2 = self.linear2(x)

        return hidden_2


class MLP2(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self) -> None:
        super(MLP2, self).__init__()
        self.linear2 = nn.Linear(64, 10)
        self.linear2.weight.data.fill_(0.01)
        self.linear2.bias.data.fill_(0.01)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        output = self.linear2(x)
        return output


class MLP3(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self, total_len) -> None:
        super(MLP3, self).__init__()
        self.embedding = nn.Embedding(total_len, 64, sparse=True)
        self.linear3 = nn.Linear(64, 10)
        self.linear3.weight.data.fill_(0.01)
        self.linear3.bias.data.fill_(0.01)
        # Use hook if you want to check the gradient
        # h = self.linear3.weight.register_hook(
        #     lambda grad: print(
        #         "alt kt MLP3 lienar", torch.sum(grad), torch.sum(self.linear3.weight)
        #     )
        # )

    def forward(
        self,
        x_id,
    ) -> torch.Tensor:
        hidden = self.embedding(x_id)
        output = self.linear3(hidden)
        return output


class LightningMLPD(pl.LightningModule):
    def __init__(self, config):
        super(LightningMLPD, self).__init__()
        self.base_model_lm = MLP1()
        self.base_model_kt = MLP2()
        self.dummy_lm = MLP1()

        self.config = config
        self.comp = []
        self.prev_data = None
        self.prev_id = -1
        self.automatic_optimization = False

    def forward(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.base_model_lm(batch)
        output = self.base_model_kt(hidden)
        return output

    def compute_loss(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        # loss = F.mse_loss(logit, label, reduction="sum")
        loss = F.cross_entropy(logit, label.type(torch.LongTensor))
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch["data"]
        x_id = batch["data_id"]
        y = batch["target"]
        opt_kt, opt_lm = self.optimizers()
        if x_id.tolist()[0] != 0:
            self.comp.append(self.base_model_lm(self.prev_data))
            self.prev_data = x
            self.prev_id = x_id
        else:
            self.comp.append(self.base_model_lm(x))
            self.prev_data = x
            self.prev_id = x_id
        logit = self(x)
        loss = self.compute_loss(y, logit)
        opt_kt.zero_grad()
        opt_lm.zero_grad()
        self.manual_backward(loss)
        opt_kt.step()
        opt_lm.step()
        self.log("train_kt_loss", loss)

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """
        x = batch["data"]
        x_id = batch["data_id"]
        y = batch["target"]
        logit = self(x)
        pred = torch.argmax(logit, -1)
        per = torch.sum(pred == y) / len(y)

        loss = self.compute_loss(y, logit)
        return {
            "loss": loss.item(),
            "correct": per.item(),
        }

    def validation_epoch_end(self, outputs):
        """
        single validation epoch for pl.Trainer
        """
        loss = np.array([o["loss"] for o in outputs])
        loss = np.mean(loss)  # .mean().item()

        pred = np.array([o["correct"] for o in outputs])
        pred = np.mean(pred)  # .mean().item()

        log = {
            "val_loss": loss,
            "val_acc": pred,
        }
        print(log)
        self.log_dict(log)
        wandb.log({"val_loss": loss})
        wandb.log({"val_pred": pred})

        return log

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.config.optimizer == "SGD":
            self.kt_optimizer = torch.optim.SGD(
                self.base_model_kt.parameters(),
                lr=self.config.ktlr,
                momentum=self.config.momentum,
                nesterov=True,
            )
            self.lm_optimizer = torch.optim.SGD(
                self.base_model_lm.parameters(),
                lr=self.config.lmlr,
                momentum=self.config.momentum,
                nesterov=True,
            )
            return (
                {"optimizer": self.kt_optimizer},
                {"optimizer": self.lm_optimizer},
            )
        elif self.config.optimizer == "Adam":
            self.kt_optimizer = torch.optim.Adam(
                self.base_model_kt.parameters(), lr=self.config.ktlr
            )
            self.lm_optimizer = torch.optim.Adam(
                self.base_model_lm.parameters(), lr=self.config.ktlr
            )
            return (
                {"optimizer": self.kt_optimizer},
                {"optimizer": self.lm_optimizer},
            )
        else:
            AssertionError()


class LightningMLP_StepAlternate(LightningMLPD):
    def __init__(self, config, train_dict, test_dict, train_len, test_len):
        super().__init__(config=config)
        self.base_model_lm = MLP1()
        self.config = config
        self.base_model_kt = MLP3(train_len + test_len)
        self.dummy_lm = MLP1()
        self.dummy_kt = MLP3(train_len + test_len)
        self.qid = []
        self.automatic_optimization = False
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.train_len = train_len
        self.test_len = test_len
        self.labels = []
        self.lm_output = []
        self.alternating_interval = config.alternating_interval
        self.mode = "kt"  # change it to enum later
        self.lm_steps = 0
        self.x_stacked = []
        self.x_id_stacked = []
        self.debug = False

    def compute_loss_lm(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(logit, label, reduction="sum") * 0.5
        # breakpoint()
        return loss

    def training_epoch_end(self, output) -> None:
        self.debug = True
        a = 0

    # multistep alternating
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch["data"]
        x_id = batch["data_id"]
        y = batch["target"]
        self.x_stacked.append(x)
        with torch.no_grad():
            lm_output = torch.stack(
                [
                    self.base_model_lm(
                        torch.tensor(self.train_dict[int(item.numpy())]).type(
                            torch.FloatTensor
                        )
                    )
                    for item in x_id
                ]
            )
            self.lm_output.append(lm_output)
            self.base_model_kt.embedding.weight.data[torch.tensor(x_id)] = (
                lm_output.reshape(-1, 64).clone().detach()
            )  # torch.tensor([item]).type(torch.FloatTensor).cuda())

        opt_lm, opt_kt, opt_emb = self.optimizers()
        if self.mode == "kt":
            self.base_model_kt.embedding.requires_grad_(True)
            logit = self.base_model_kt(x_id)
            loss = self.compute_loss(y, logit)
            pred = torch.argmax(logit, -1)
            per = torch.sum(pred == y) / len(y)
            self.log("train acc", per)
            opt_kt.zero_grad()
            opt_emb.zero_grad()
            self.manual_backward(loss)
            opt_kt.step()
            opt_emb.step()

            self.log("train_kt_loss", loss)
            self.labels.append(
                torch.stack(
                    [self.base_model_kt.embedding.weight.data[item] for item in x_id]
                )
            )
            if (self.global_step + 1) % self.alternating_interval == 0:
                self.mode = "lm"
        if self.mode == "lm":
            for i in range(self.alternating_interval):
                labels = self.labels[self.lm_steps]
                lm_output = self.base_model_lm(self.x_stacked[self.lm_steps])

                loss2 = self.compute_loss_lm(labels, lm_output)
                opt_lm.zero_grad()
                self.manual_backward(loss2)
                opt_lm.step()
                self.log("train_lm_loss", loss2)

                self.lm_steps += 1

            self.mode = "kt"
            self.lm_steps = 0
            self.labels = []
            self.lm_output = []
            self.x_stacked = []

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """
        x = batch["data"]
        x_id = batch["data_id"]
        y = batch["target"]
        logit = self.base_model_kt(x_id)
        pred = torch.argmax(logit, -1)
        per = torch.sum(pred == y) / len(y)

        loss = self.compute_loss(y, logit)
        return {
            "loss": loss.item(),
            "correct": per.item(),
        }

    def on_validation_epoch_start(self) -> None:
        for item in self.test_dict.keys():
            self.base_model_kt.embedding.weight.data[
                torch.tensor(item)
            ] = self.base_model_lm(
                torch.tensor(self.test_dict[item]).type(torch.FloatTensor)
            ).reshape(
                -1, 64
            )

    def configure_optimizers(self):
        if self.config.optimizer == "SGD":
            self.optimizer_lm = torch.optim.SGD(
                self.base_model_lm.parameters(),
                lr=self.config.lmlr,
                momentum=self.config.momentum,
                nesterov=True,
            )
            self.optimizer_kt = torch.optim.SGD(
                self.base_model_kt.linear3.parameters(),
                lr=self.config.ktlr,
                momentum=self.config.momentum,
                nesterov=True,
            )
            self.optimizer_emb = torch.optim.SGD(
                self.base_model_kt.embedding.parameters(),
                lr=self.config.ktlr,
                nesterov=False,
            )  # momentum=self.config.momentum,
            # self.optimizer_kt = torch.optim.SGD([
            #
            # ])
        elif self.config.optimizer == "Adam":
            self.optimizer_lm = torch.optim.Adam(
                self.base_model_lm.parameters(), lr=self.config.ktlr
            )  # self.config.lmlr
            self.optimizer_kt = torch.optim.Adam(
                self.base_model_kt.linear3.parameters(), lr=self.config.ktlr
            )
            self.optimizer_emb = torch.optim.SGD(
                self.base_model_kt.embedding.parameters(), lr=1, nesterov=False
            )  # momentum=self.config.momentum, # self.config.ktlr
        elif self.config.optimizer == "Sadam":
            self.optimizer_lm = torch.optim.Adam(
                self.base_model_lm.parameters(), lr=self.config.lmlr
            )
            self.optimizer_kt = torch.optim.Adam(
                self.base_model_kt.linear3.parameters(), lr=self.config.ktlr
            )
            self.optimizer_emb = torch.optim.SparseAdam(
                self.base_model_kt.embedding.parameters(),
                lr=self.config.ktlr,
                betas=(0, 0),
                eps=1,
            )

        return (
            {"optimizer": self.optimizer_lm},
            {"optimizer": self.optimizer_kt},
            {"optimizer": self.optimizer_emb},
        )

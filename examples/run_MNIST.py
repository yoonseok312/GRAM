import os
import time
import glob

from datetime import datetime

import pytorch_lightning as pl
import wandb
import pickle
import torch
import numpy as np

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.datasets import MNIST
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge

from mlp_mnist import LightningMLPD, LightningMLP_StepAlternate
from torch.utils.data import Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

import gc
import random
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf


class MyDataset(Dataset):
    def __init__(self, X, X_id, Y):
        self.X = torch.tensor(X).type(torch.FloatTensor)
        self.X_id = torch.tensor(X_id)  # .type(torch.FloatTensor)
        self.target = torch.tensor(Y).type(torch.FloatTensor)

    def __getitem__(self, index):
        x = self.X[index]
        x_id = self.X_id[index]
        y = self.target[index]

        return {"data": x, "data_id": x_id, "target": y}

    def __len__(self):
        return len(self.X)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/examples"
    cfg_file = "mnist_config.yaml"
    config = OmegaConf.load(os.path.join(root, "configs", cfg_file))
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    mode = config.mode
    print(config)
    if mode == "alt":
        exp_name = f"MNIST_demo_alt_step{config.alternating_interval}_{config.optimizer}_ktlr{config.ktlr}_lmlr{config.lmlr}_m{config.momentum}"
    elif mode == "e2e":
        exp_name = (
            f"MNIST_demo_e2e_{config.optimizer}_lr{config.ktlr}_m{config.momentum}"
        )
    else:
        AssertionError()
    data_size = 50
    batch_size = 64
    epochs = 10

    pl.seed_everything(0, workers=True)
    np.random.seed(0)
    torch.manual_seed(0)

    train = MNIST(root="./data/mnist", train=True, download=True)
    test = MNIST(root="./data/mnist", train=False, download=True)

    X = np.asarray(torch.cat([train.data.float().view(-1, 784) / 255.0]))
    X_id = np.array([i for i in range(len(X))])
    Y = np.asarray(torch.cat([train.targets]))
    train_dict = {k: v for k, v in list(zip(X_id, X))}

    X_test = np.asarray(torch.cat([test.data.float().view(-1, 784) / 255.0]))
    X_id_test = np.array([i for i in range(len(X), len(X) + len(X_test))])
    Y_test = np.asarray(torch.cat([test.targets]))
    test_dict = {k_test: v_test for k_test, v_test in list(zip(X_id_test, X_test))}

    train_length = len(X)
    test_length = len(X_test)

    train_dataset = MyDataset(X, X_id, Y)
    test_dataset = MyDataset(X_test, X_id_test, Y_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)

    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    model_D = LightningMLPD(config)
    model_alternate = LightningMLP_StepAlternate(
        config, train_dict, test_dict, train_length, test_length
    )

    # wandb.init(project='mlp_check', name="Alt " + exp_name, reinit=True)

    if mode == "alt":
        trainer_alternate = pl.Trainer(
            callbacks=[
                EarlyStopping(
                    monitor="val_acc",
                    patience=5,
                    mode="max",
                ),
            ],
            max_epochs=epochs,
            logger=pl.loggers.WandbLogger(project="mlp_check", name="Alt " + exp_name),
            val_check_interval=1.0,
            deterministic=True,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            accelerator="cpu",  # Change to gpu if you want to use gpu
        )

        trainer_alternate.fit(model_alternate, trainloader, testloader)
        trainer_alternate.test(model_alternate, testloader)
    else:
        # wandb.init(project='mlp_check', name="D " + exp_name, reinit=True)

        trainer_D = pl.Trainer(
            callbacks=[
                EarlyStopping(
                    monitor="val_acc",
                    patience=5,
                    mode="max",
                ),
            ],
            max_epochs=epochs,
            logger=pl.loggers.WandbLogger(project="mlp_check", name="D " + exp_name),
            val_check_interval=1.0,
            deterministic=True,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            accelerator="cpu",  # Change to gpu if you want to use gpu
        )

        trainer_D.fit(model_D, trainloader, testloader)
        trainer_D.test(model_D, testloader)


if __name__ == "__main__":
    main()

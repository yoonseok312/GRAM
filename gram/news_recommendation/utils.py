import logging
import os
import sys
import torch
import numpy as np
import argparse
import re
import os
import time
import glob

from datetime import datetime

import pytorch_lightning as pl
import wandb
import pickle
import torch
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import InputExample, losses
import pandas as pd

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from gram.knowledge_tracing.components.SBERT import SBERT
from gram.knowledge_tracing.components.sbert_regressor import LightningRegressor
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

import gc
import random

def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def init_hvd_cuda(enable_hvd=True, enable_gpu=True):
    hvd = None
    if enable_hvd:
        import horovod.torch as hvd

        hvd.init()
        logging.info(
            f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
        )


    hvd_size = hvd.size() if enable_hvd else 1
    hvd_rank = hvd.rank() if enable_hvd else 0
    hvd_local_rank = hvd.local_rank() if enable_hvd else 0

    if enable_gpu:
        torch.cuda.set_device(hvd_local_rank)

    return hvd_size, hvd_rank, hvd_local_rank


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    embedding_matrix = np.random.uniform(size=(len(word_dict) + 1,
                                               word_embedding_dim))
    have_word = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = line[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
    return embedding_matrix, have_word


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    print(os.listdir(directory))
    if len(os.listdir(directory))==0:
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None

def pipeline_B_newsrec(model, news_index, cfg, log_dir):

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename= "SBERT_Regressor_" + cfg.exp_name + "-{epoch}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer_regressor = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg.B.patience),
            checkpoint_callback,
        ],
        max_epochs=cfg.B.max_epochs,
        accelerator="ddp",
        gpus=None if cfg.gpus is None else str(cfg.gpus),
        logger=pl.loggers.WandbLogger(project="pipeline_B", name=cfg.exp_name + "_" + cfg.current_stage) if cfg.B.use_wandb else None,
        deterministic=True,
        val_check_interval=cfg.B.val_check_interval,
    )

    with open(f'{cfg.root}/MIND/MIND_newsid_to_count_np4_seed0_final.pkl', 'rb') as f:  # {cfg.root}/MIND_small/MIND_small_global_id_to_BERT_vec.pkl','rb') as f:
        item_count = pickle.load(f)
    item_ids_list = list(item_count.keys())

    if cfg.dataset_size == 'large':
        with open(f'{cfg.root}/MIND/val_csqs_mind_large_list.pkl', 'rb') as f:
            csqs = pickle.load(f)
    else:
        with open(f'{cfg.root}/MIND_small/test_csqs_mind_small_list.pkl', 'rb') as f:
            csqs = pickle.load(f)
    cold_start_ids = []
    for item in csqs:
        cold_start_ids.append(news_index[item])

    non_cold_start_ids = []
    for id in item_ids_list:
        if id not in cold_start_ids:
            non_cold_start_ids.append(id)

    if cfg.B.use_sim:
        model_regressor = LightningRegressor(cfg, model_sim)
    else:
        # FIXME
        if cfg.B.use_sim:
            print(glob.glob('/tmp/pycharm_project_47/repoc_content_kt/scripts/temp/22200'))
            model_sim = SBERT(glob.glob('/tmp/pycharm_project_47/repoc_content_kt/scripts/temp/22200')[0])

            model_regressor = LightningRegressor(cfg, model_sim)
        else:
            model_regressor = LightningRegressor(cfg)

    embedding_label_list = []
    for id in non_cold_start_ids:
        embedding_label_list.append(np.asarray(model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[id]))
    # breakpoint()
    if cfg.B.monitor == 'train_loss':
        split = int(len(non_cold_start_ids) * 1.0)
        train_loader = DataLoader(TensorDataset(torch.tensor(np.asarray(non_cold_start_ids)[:split], dtype=torch.float),
                                                torch.tensor(np.asarray(embedding_label_list)[:split],
                                                             dtype=torch.float)),
                                  batch_size=cfg.B.regressor_batch_size, shuffle=True)
        trainer_regressor.fit(model_regressor, train_loader)

    else:
        split = int(len(non_cold_start_ids) * 0.9)
        random.seed(0)
        id_emb = list(zip(non_cold_start_ids, embedding_label_list))
        random.shuffle(id_emb)
        non_cold_start_ids, embedding_label_list = zip(*id_emb)
        train_loader = DataLoader(TensorDataset(torch.tensor(np.asarray(non_cold_start_ids)[:split], dtype=torch.float),
                                                torch.tensor(np.asarray(embedding_label_list)[:split],dtype=torch.float)),
                                                batch_size=cfg.B.regressor_batch_size,shuffle=True)

        val_loader = DataLoader(TensorDataset(torch.tensor(np.asarray(non_cold_start_ids)[split:], dtype=torch.float),
                                                torch.tensor(np.asarray(embedding_label_list)[split:],dtype=torch.float)),
                                                batch_size=cfg.B.regressor_batch_size,shuffle=True)
        trainer_regressor.fit(model_regressor, train_loader, val_loader)

    # breakpoint()
    print('log_dir b4: ', log_dir)
    print("glob b4: ", glob.glob(os.path.join(log_dir, "SBERT_Regressor_*.ckpt")))

    # log_dir_aft = "/root/all_item_sbert/*logs/*regression/" + "*" + str(cur_time)
    # # log_dir_aft = log_dir + "/"
    # print('log_dir aft: ', log_dir_aft)
    # print("glob aft: ", glob.glob(os.path.join(log_dir_aft, "*.ckpt")))
    best_val_ckpt = glob.glob(os.path.join(log_dir, "SBERT_Regressor_*.ckpt"))[0]
    # breakpoint()
    best_model = LightningRegressor.load_from_checkpoint(best_val_ckpt, cfg=cfg)
    if cfg.B.change_all_qid_for_test:
        model.base_model.enc_embed.SBERT = best_model.base_model.SBERT_pretrained
        cfg.experiment_type[cfg.current_stage] = 'D'
        del best_model
        gc.collect()
        return model
    else:
        for cid in cold_start_ids:
            model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                cid
            ] = best_model(torch.Tensor([cid])).reshape(-1)

        del best_model
        gc.collect()
        return model

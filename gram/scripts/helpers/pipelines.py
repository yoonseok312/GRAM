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
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge

from sentence_transformers import InputExample, losses
import pandas as pd

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from gram.knowledge_tracing.content_all_item_kt import LightningContentAllItemKT
from gram.knowledge_tracing.content_saint_kt import ContentSaintModel
from gram.knowledge_tracing.content_dkt import LightningContentDKT
from gram.knowledge_tracing.AlternatingCKT import EpochwiseAlternatingCKT
from gram.knowledge_tracing.components.SBERT import SBERT
from gram.knowledge_tracing.components.sbert_regressor import LightningRegressor
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

import gc
import random


def get_model(cfg: DictConfig, load_ckpt: bool = False, ckpt_path: str = None, regressor_ckpt_path: str = None):
    if cfg.model_name == "all_item_kt":
        if load_ckpt:
            model = LightningContentAllItemKT.load_from_checkpoint(ckpt_path, cfg=cfg)
        else:
            model = LightningContentAllItemKT(cfg)
    elif cfg.model_name == "saint":
        if load_ckpt:
            model = ContentSaintModel.load_from_checkpoint(ckpt_path, cfg=cfg)
        else:
            model = ContentSaintModel(cfg)
    elif cfg.model_name == "dkt":
        print("KT CKPT", ckpt_path)
        # breakpoint()
        if load_ckpt:
            if cfg.experiment_type[cfg.current_stage] == 'Alternating' and cfg[cfg.experiment_type[cfg.current_stage]].alternate_by_epoch:
                model = EpochwiseAlternatingCKT(cfg).load_from_checkpoint(ckpt_path, cfg=cfg)
            else:
                model = LightningContentDKT(cfg).load_from_checkpoint(ckpt_path, cfg=cfg)
            if regressor_ckpt_path is not None:
                # breakpoint()
                print("REG CKPT", regressor_ckpt_path)
                finetuned_SBERT = LightningRegressor.load_from_checkpoint(regressor_ckpt_path, cfg=cfg)
                model.base_model.enc_embed.SBERT = finetuned_SBERT.base_model.SBERT_pretrained
                # breakpoint()
        else:
            if cfg.experiment_type[cfg.current_stage] == 'Alternating' and cfg[cfg.experiment_type[cfg.current_stage]].alternate_by_epoch:
                model = EpochwiseAlternatingCKT(cfg)
            else:
                model = LightningContentDKT(cfg)
    else:
        raise NotImplementedError(
            f'Model "{cfg.model_name}" is not supported.'
        )
    return model


def pipeline_A(best_model, cfg: DictConfig):
    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_emb.pkl", "rb") as handle:
        id_to_emb = pickle.load(handle)
    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/cold_start_ids_part_4_to_7.txt", "rb") as f:
        cold_start_ids = pickle.load(f)

    text_embedding_list = []
    kt_embedding_list = []
    cold_text_embedding_list = []
    for item_id in tqdm(id_to_emb.keys()):
        if item_id not in cold_start_ids:
            text_embedding_list.append(torch.Tensor(id_to_emb[item_id]))
            kt_embedding_list.append(best_model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item_id])

    for item_id in cold_start_ids:
        cold_text_embedding_list.append(torch.Tensor(id_to_emb[item_id]))
    kt_embeddings = torch.stack(kt_embedding_list).numpy()
    fine_tuned_text_embeddings = torch.stack(text_embedding_list).numpy()
    cold_start_embeddings = torch.stack(cold_text_embedding_list).numpy()

    if cfg.A.base_regresor == 'Ridge':
        clf_reg = Ridge(alpha=1)
    else:
        clf_reg = Ridge(alpha=1) # FIXME: Add other regressors
    clf_reg.fit(fine_tuned_text_embeddings, kt_embeddings)
    cold_kt_embeddings = clf_reg.predict(cold_start_embeddings)

    count = 0
    for item_id in cold_start_ids:
        best_model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item_id] = torch.tensor(cold_kt_embeddings[count]).reshape(-1)
        count += 1

    return best_model


def pipeline_B(model, cfg, log_dir):

    if cfg.B.regressor_ckpt is not None:
        if cfg.ckt_dataset_type == 'toeic':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/cold_start_ids_part_4_to_7.txt", "rb") as f:
                cold_start_ids = pickle.load(f)
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].user_based_split:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe_newsplit_updated.pkl", "rb") as f:
                    dic = pickle.load(f)
                    cold_start_ids = list(dic.keys())
            else:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe.pkl", "rb") as f:
                    dic = pickle.load(f)
                    cold_start_ids = list(dic.keys())
        elif cfg.ckt_dataset_type == 'poj':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_csqs_new_split.pkl", "rb") as f:
                cold_start_ids = pickle.load(f)
        best_val_ckpt = cfg.B.regressor_ckpt
        best_model = LightningRegressor.load_from_checkpoint(best_val_ckpt, cfg=cfg)

        for cid in cold_start_ids:
            model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                cid
            ] = best_model(torch.Tensor([cid])).reshape(-1)

        del best_model
        gc.collect()
        return model

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename= "SBERT_Regressor_" + cfg.exp_name + "-{epoch}-{val_loss:.3f}",
        save_top_k=cfg.B.max_epochs,
        monitor=cfg.B.monitor,
        mode="min",
    )

    trainer_regressor = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor=cfg.B.monitor, patience=cfg.B.patience),
            checkpoint_callback,
        ],
        max_epochs=cfg.B.max_epochs,
        accelerator="ddp",
        gpus=None if cfg.gpus is None else str(cfg.gpus),
        logger=pl.loggers.WandbLogger(project="pipeline_B", name=cfg.exp_name + "_" + cfg.current_stage) if cfg.B.use_wandb else None,
        deterministic=True,
        val_check_interval=cfg.B.val_check_interval,
    )

    if cfg.ckt_dataset_type == 'toeic':
        with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_item_id_to_emb_BERT_TinyBERT_batch_100.pkl", "rb") as handle:#_TinyBERT_batch_100.pkl", "rb") as handle:
            id_to_emb = pickle.load(handle)
        with open(f"{cfg[cfg.ckt_dataset_type].data.root}/cold_start_ids_part_4_to_7.txt", "rb") as f:
            cold_start_ids = pickle.load(f)
        with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_count.pkl", "rb") as handle:
            item_count = pickle.load(handle)
    elif cfg.ckt_dataset_type == 'duolingo':
        if cfg[cfg.ckt_dataset_type].user_based_split:
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl", "rb") as handle:#_TinyBERT_batch_100.pkl", "rb") as handle:
                id_to_emb = pickle.load(handle)
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe_newsplit_updated.pkl", "rb") as f:
                dic = pickle.load(f)
                cold_start_ids = list(dic.keys())
        else:
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl",
                      "rb") as handle:  # _TinyBERT_batch_100.pkl", "rb") as handle:
                id_to_emb = pickle.load(handle)
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe.pkl", "rb") as f:
                dic = pickle.load(f)
                cold_start_ids = list(dic.keys())
    elif cfg.ckt_dataset_type == 'poj':
        with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_csqs_new_split.pkl", "rb") as f:
            cold_start_ids = pickle.load(f)
        with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_count_only_train.pkl", "rb") as handle:#_TinyBERT_batch_100.pkl", "rb") as handle:
            id_to_emb = pickle.load(handle)
        # with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl",
        #           "rb") as handle:
        #     id_to_emb = pickle.load(handle)
        # with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_french_csq_ids.txt", "rb") as f:
        #     cold_start_ids = pickle.load(f)
        #     print("MAX", max(cold_start_ids))
    item_ids_list = list(item_count.keys())

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

def pipeline_C(model, cfg: DictConfig):
    if cfg.C.load_encoded_embeddings:
        if cfg.ckt_dataset_type == 'toeic':
            if cfg.C.base_lm.pretrained == "paraphrase-TinyBERT-L6-v2":
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_emb_with_passage_TinyBERT_batch_100.pkl", "rb") as handle:
                    id_to_emb = pickle.load(handle)
            elif cfg.C.base_lm.pretrained == "paraphrase-mpnet-base-v2":
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_emb_with_passage_MPNet_batch_100.pkl", "rb") as handle:
                    id_to_emb = pickle.load(handle)
            elif cfg.C.base_lm.pretrained == "nreimers/TinyBERT_L-6_H-768_v2":
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_item_id_to_emb_BERT_TinyBERT_batch_100.pkl", "rb") as handle:
                    id_to_emb = pickle.load(handle)
            else:
                print("ERROR")
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].language == 'french':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_french_pretrained_BERT_embedding.pkl", "rb") as handle:#/duolingo_item_id_to_embedding_TinyBERT_batch_100.pkl", "rb") as handle:
                    id_to_emb = pickle.load(handle)
            elif cfg[cfg.ckt_dataset_type].language == 'spanish':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_spanish_pretrained_BERT_embedding.pkl", "rb") as handle:
                    id_to_emb = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'poj':
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_emb_tinyBERT_with_scraped_truncated.pkl",
                      "rb") as handle:
                id_to_emb = pickle.load(handle)
    else:
        if cfg.C.base_lm == 'SBERT':
            language_model = SBERT(cfg.C.base_lm.pretrained)
        else:
            language_model = SBERT(cfg.C.base_lm.pretrained) # FIXME: Add other LMs
        if cfg.ckt_dataset_type == 'toeic':
            if cfg[cfg.experiment_type[cfg.current_stage]].with_passage:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text_special_tokens.pkl", "rb") as handle:
                    id_to_text = pickle.load(handle)
            else:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text.pkl", "rb") as handle:
                    id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].language == 'french':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl", "rb") as handle:
                    id_to_text = pickle.load(handle)
            elif cfg[cfg.ckt_dataset_type].language == 'spanish':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_all.pkl", "rb") as handle:
                    id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'poj':
            print("Truncated text")
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl",
                      "rb") as handle:
                id_to_text = pickle.load(handle)
        encoded_valid_ids = language_model.encode(
            sentences=[id_to_text[item_id] for item_id in tqdm(id_to_text.keys())],
            batch_size=cfg.C.base_lm.batch_size,
        )

        id_to_emb = {}
        for idx, item in enumerate(id_to_text.keys()):
            id_to_emb[item] = encoded_valid_ids[idx]
    for item_id in tqdm(id_to_emb.keys()):
        model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item_id] = torch.Tensor(id_to_emb[item_id])
    return model

# def pipeline_D(model, cfg: DictConfig, ckpt_path: str = None):
#     if cfg.D.use_exp_c_kt_module:
#         print("model name", cfg.model_name)
#         if cfg.model_name == 'dkt':
#             model = LightningContentDKT.load_from_checkpoint(ckpt_path, cfg=cfg)
#         else:
#             model = LightningContentAllItemKT.load_from_checkpoint(ckpt_path, cfg=cfg)
#
#     if cfg.D.freeze_lm:
#         if cfg.D.base_lm == 'SBERT':
#             language_model = LightningRegressor.load_from_checkpoint(cfg.D.regressor_ckpt, cfg=cfg).cpu()
#             breakpoint()
#         else:
#             language_model = LightningRegressor.load_from_checkpoint(cfg.D.regressor_ckpt, cfg=cfg).cpu()
#             breakpoint()
#         with open(f'{cfg.data.root}/toeic_id_to_text_with_passage.pkl', 'rb') as handle:
#             id_to_text = pickle.load(handle)
#         encoded_valid_ids = language_model.base_model.SBERT_pretrained.encode(
#             sentences=[id_to_text[item_id] for item_id in tqdm(id_to_text.keys())],
#             batch_size=cfg.C.base_lm.batch_size,
#             use_device=False,
#         )
#         del language_model
#         gc.collect()
#
#         id_to_emb = {}
#
#         for idx, item in enumerate(id_to_text.keys()):
#             id_to_emb[item] = encoded_valid_ids[idx]
#
#
#         for item_id in tqdm(id_to_emb.keys()):
#             model.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[item_id] \
#                 = torch.Tensor(id_to_emb[item_id])
#
#         params = model.base_model.enc_embed.embed_feature.shifted_item_id.parameters()
#         for param in params:
#             param.requires_grad = False
#
#     return model

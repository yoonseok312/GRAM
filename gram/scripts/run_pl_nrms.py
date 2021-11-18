import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
from pathlib import Path
from repoc_content_kt.news_recommendation.LightningNRMSWithLM import LightningNRMSWithLM
from repoc_content_kt.news_recommendation.LightningAlternateNRMSWithLM import LightningAlternatingNRMSWithLM
from torch.utils.data import DataLoader
from preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert, load_matrix
# from repoc_content_kt.news_recommendation.deprecated.parameters import parse_args
from repoc_content_kt.news_recommendation.prepare_data import prepare_training_data
from repoc_content_kt.news_recommendation.dataset import InteractionDataset
from LightningAlternateNRMSWithLM import LightningAlternatingNRMSWithLM
import logging
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert, load_matrix
from model_bert import ModelBert
from prepare_data import prepare_training_data
from repoc_content_kt.news_recommendation.dataset import DatasetTrain, InteractionDataset, InteractionDatasetTest
from repoc_content_kt.news_recommendation.LightningNRMS import LightningNRMS
from repoc_content_kt.news_recommendation.LightningExpC import LightningExpC
from transformers import AutoTokenizer, AutoModel, AutoConfig
from repoc_common.utils import save_cfg

import pytorch_lightning as pl

import warnings
import wandb
warnings.filterwarnings('ignore')

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from repoc_content_kt.news_recommendation.LightningAlternatingHalfNRMS import LightningAlternatingHalfNRMSWithLM
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy('file_system')
finetuneset = {
    'encoder.layer.5.attention.self.query.weight',
    'encoder.layer.5.attention.self.query.bias',
    'encoder.layer.5.attention.self.key.weight',
    'encoder.layer.5.attention.self.key.bias',
    'encoder.layer.5.attention.self.value.weight',
    'encoder.layer.5.attention.self.value.bias',
    'encoder.layer.5.attention.output.dense.weight',
    'encoder.layer.5.attention.output.dense.bias',
    'encoder.layer.5.attention.output.LayerNorm.weight',
    'encoder.layer.5.attention.output.LayerNorm.bias',
    'encoder.layer.5.intermediate.dense.weight',
    'encoder.layer.5.intermediate.dense.bias',
    'encoder.layer.5.output.dense.weight',
    'encoder.layer.5.output.dense.bias',
    'encoder.layer.5.output.LayerNorm.weight',
    'encoder.layer.5.output.LayerNorm.bias',
    'encoder.layer.6.attention.self.query.weight',
    'encoder.layer.6.attention.self.query.bias',
    'encoder.layer.6.attention.self.key.weight',
    'encoder.layer.6.attention.self.key.bias',
    'encoder.layer.6.attention.self.value.weight',
    'encoder.layer.6.attention.self.value.bias',
    'encoder.layer.6.attention.output.dense.weight',
    'encoder.layer.6.attention.output.dense.bias',
    'encoder.layer.6.attention.output.LayerNorm.weight',
    'encoder.layer.6.attention.output.LayerNorm.bias',
    'encoder.layer.6.intermediate.dense.weight',
    'encoder.layer.6.intermediate.dense.bias',
    'encoder.layer.6.output.dense.weight',
    'encoder.layer.6.output.dense.bias',
    'encoder.layer.6.output.LayerNorm.weight',
    'encoder.layer.6.output.LayerNorm.bias',
    'pooler.dense.weight',
    'pooler.dense.bias',
    'rel_pos_bias.weight',
    'classifier.weight',
    'classifier.bias'}


def train(cfg, log_dir, train_path, val_path):

    # print("visible device count", torch.cuda.device_count())

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=cfg.exp_name,
        save_top_k=1,
        monitor='val_auc',
        mode='max',
    )

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
            ),
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
        ],
        max_epochs=100,
        accelerator="ddp",
        gpus=str(cfg.gpus), # str(args.gpus), # '6,7',#,1,2,3,4,5,6,7',#'1','2','3','4','5','6','7',
        logger=pl.loggers.WandbLogger(project="news_recommendation",
                                      name=cfg.exp_name),
        val_check_interval=cfg.Alternating.val_check_interval,
        limit_train_batches=1.0,
        deterministic=True,
        num_sanity_val_steps=0,
        # replace_sampler_ddp=True,
    )

    if cfg.experiment_type[cfg.current_stage] in ['D', 'Alternating', 'C']:
        tokenizer = AutoTokenizer.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.model)

        if cfg.dataset_size == 'large':
            news, news_index = read_news_bert(
                os.path.join(f'{train_path}/total_news_val.tsv'),
                cfg,
                tokenizer
            )
        else:
            news, news_index = read_news_bert(
                os.path.join(f'{train_path}/total_news_small.tsv'),
                cfg,
                tokenizer
            )
        news_title, news_title_type, news_title_attmask, \
        news_abstract, news_abstract_type, news_abstract_attmask, \
        news_body, news_body_type, news_body_attmask, global_dict = get_doc_input_bert(
            news, news_index, cfg)

        news_combined = np.concatenate([
            x for x in
            [news_title, news_title_type, news_title_attmask, \
             news_abstract, news_abstract_type, news_abstract_attmask, \
             news_body, news_body_type, news_body_attmask]
            if x is not None], axis=1)
        if cfg.experiment_type[cfg.current_stage] == 'D':
            news_val, news_index_val = read_news_bert(
                os.path.join(f'{val_path}/news.tsv'),
                cfg,
                tokenizer
            )

            news_title_val, news_title_type_val, news_title_attmask_val, \
            news_abstract_val, news_abstract_type_val, news_abstract_attmask_val, \
            news_body_val, news_body_type_val, news_body_attmask_val, global_dict = get_doc_input_bert(
                news_val, news_index_val, cfg)

            news_combined_val = np.concatenate([
                x for x in
                [news_title_val, news_title_type_val, news_title_attmask_val, \
                 news_abstract_val, news_abstract_type_val, news_abstract_attmask_val, \
                 news_body_val, news_body_type_val, news_body_attmask_val]
                if x is not None], axis=1)
        else:
            news_index_val = news_index
            news_combined_val = news_combined
    elif cfg.experiment_type[cfg.current_stage] == 'Vanilla':
        if cfg.dataset_size == 'large':
            news, news_index, word_dict_train = read_news(
                os.path.join(f'{train_path}/news.tsv'),
                cfg,
            )
        else:
            news, news_index, word_dict_train = read_news(
                os.path.join(f'{train_path}/news.tsv'),
                cfg,
            )

        # embedding_matrix, have_word = load_matrix('/MIND/glove.840B.300d.txt',
        #                                                 word_dict_train,
        #                                                 300) # word embedding dim

        news_title = get_doc_input(news, news_index, word_dict_train, cfg)
        news_combined = np.concatenate([x for x in [news_title] if x is not None], axis=1)
        if cfg.dataset_size == 'large':
            news_val, news_index_val = read_news(
                os.path.join(f'{val_path}/news.tsv'),
                cfg,
                mode='test',
            )
        else:
            news_val, news_index_val = read_news(
                os.path.join(f'{val_path}/news.tsv'),
                cfg,
                mode='test',
            )
        news_title_val = get_doc_input(news_val, news_index_val, word_dict_train, cfg)
        news_combined_val = np.concatenate([x for x in [news_title_val] if x is not None], axis=1)

    data_file_path = os.path.join(f'{train_path}/', f'behaviors_np{cfg[cfg.experiment_type[cfg.current_stage]].npratio}_0.tsv')
    data_test_file_path = os.path.join(f'{val_path}/', f'behaviors.tsv')

    dataset = InteractionDataset(data_file_path, news_index, news_combined, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg[cfg.experiment_type[cfg.current_stage]].batch_size,
        num_workers=cfg.num_workers
    ) # num_workers=2, worker_init_fn=worker_init_fn

    print("test batch size", cfg[cfg.experiment_type[cfg.current_stage]].batch_size)
    dataset_test = InteractionDatasetTest(data_test_file_path, news_index_val, news_combined_val, cfg)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg[cfg.experiment_type[cfg.current_stage]].batch_size,
        num_workers=cfg.num_workers,
        # collate_fn=collate_fn,
    )

    if cfg.experiment_type[cfg.current_stage] == 'Alternating':
        if cfg.Alternating.load_ckpt is not None:
            model = LightningAlternatingNRMSWithLM.load_from_checkpoint(
                checkpoint_path=cfg.Alternating.load_ckpt,
                # cfg=cfg,
                # bert_model,
                news_index=news_index,
                global_dict=global_dict,
                val_news_combined=news_combined_val,
            )
        elif cfg.Alternating.alternating_proportion == 0.5:
            model = LightningAlternatingHalfNRMSWithLM(cfg, news_index, global_dict, news_combined_val)
        else:
            model = LightningAlternatingNRMSWithLM(cfg, news_index, global_dict, news_combined_val)
    elif cfg.experiment_type[cfg.current_stage] == 'D':
        if cfg.D.load_ckpt is not None:
            model = LightningNRMSWithLM.load_from_checkpoint(
                checkpoint_path=cfg.D.load_ckpt,
                cfg=cfg,
                finetuneset=finetuneset,
                news_index=news_index,
                val_news_combined=news_combined_val,
                news_index_val=news_index_val,
            )
        else:
            model = LightningNRMSWithLM(cfg, finetuneset, news_index, news_combined_val, news_index_val)
    elif cfg.experiment_type[cfg.current_stage] == 'C':
        model = LightningExpC(cfg, news_index, global_dict, news_combined_val)
    elif cfg.experiment_type[cfg.current_stage] == 'B':
        model = LightningExpC.load_from_checkpoint(cfg, news_index, global_dict, news_combined_val)
    else:
        model = LightningNRMS(cfg, None, news_index, news_combined_val, news_index_val)
    if cfg.mode == 'train':
        trainer.fit(model, dataloader, dataloader_test)
    elif cfg.mode == 'test':
        trainer.test(model, test_dataloaders=dataloader_test)


if __name__ == "__main__":
    root = "/tmp/pycharm_project_295/repoc_content_kt/news_recommendation"
    cfg_file = "newsrec_config.yaml"
    config = OmegaConf.load(os.path.join(root, cfg_file))
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    # pl.seed_everything(config.seed, workers=True)
    pl.seed_everything(config.seed, workers=True)
    # root = "/tmp/pycharm_project_971/research-poc/repoc_content_kt/news_recommendation"
    print("visible device count", torch.cuda.device_count())

    train_path = config[config.dataset_size].train_path
    val_path = config[config.dataset_size].val_path
    if not os.path.isdir(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    # script_name = os.path.splitext(os.path.basename(__file__))[0]  # "run_am"
    _log_dir = os.path.join(config.log_root, f"logs/news_rec")
    if not os.path.isdir(_log_dir):
        os.mkdir(_log_dir)
    log_dir = os.path.join(_log_dir, config.exp_name)
    print(log_dir)
    if config.prepare_dataset:
        total_sample_num = prepare_training_data(f'{train_path}/', 1, config[config.experiment_type[config.current_stage]].npratio, config.seed)
        # _ = prepare_training_data(f'{val_path}/', 1, config[config.experiment_type[config.current_stage]].npratio, config.seed)  # nGPU, npratio, seed
        print("Done preparing")
    script_name = os.path.splitext(os.path.basename(__file__))[0]  # "run_am"
    # comment this for ddp
    # wandb.init(
    #     project="news_recommendation",
    #     name=config.exp_name,
    #     reinit=True,
    #     settings=wandb.Settings(start_method='fork')
    # )
    save_cfg(config, log_dir, 'cfg_'+str(config.exp_name))
    train(config, log_dir, train_path, val_path)


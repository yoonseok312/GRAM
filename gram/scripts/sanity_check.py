import os
import time
import glob

from datetime import datetime

import pytorch_lightning as pl

import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from gram.knowledge_tracing.content_dkt import LightningContentDKT

from repoc_content_kt.datasets.content_toeic_dataset import content_data
from repoc_content_kt.scripts.helpers.pipelines import (
    get_model,
    pipeline_A,
    pipeline_B,
    pipeline_C,
    # pipeline_D,
    # pipeline_C_to_D,
)
from repoc_common.utils import save_cfg
import wandb
import numpy as np

def main(cfg: DictConfig, data_root: str, log_dir: str, ckpt_path: str = None, regressor_ckpt_path: str = None, do_infer: bool = False):
    data_loaders = content_data(data_root, cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="KT_" + cfg.exp_name + "-{epoch}-{val_loss:.3f}-{val_auc:.3f}",
        save_top_k=1,
        monitor=cfg[cfg.ckt_dataset_type].monitor,
        mode=cfg[cfg.ckt_dataset_type].monitor_mode,
    )

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(
                monitor=cfg[cfg.ckt_dataset_type].monitor,
                patience=cfg[cfg.experiment_type[cfg.current_stage]].patience,
                mode=cfg[cfg.ckt_dataset_type].monitor_mode,
            ),
            checkpoint_callback,
        ],
        max_epochs=cfg[cfg.experiment_type[cfg.current_stage]].max_epochs,
        accelerator="ddp",
        gpus=None if cfg.gpus is None else str(cfg.gpus),
        logger=pl.loggers.WandbLogger(project="content_kt", name=cfg.exp_name + "_" + cfg.current_stage) if cfg.use_wandb else None,
        val_check_interval=cfg[cfg.experiment_type[cfg.current_stage]].val_check_interval,
        limit_train_batches=cfg.limit_train_batches,
        deterministic=True,
    )

    model = get_model(cfg, cfg[cfg.experiment_type[cfg.current_stage]].kt_ckpt is not None, ckpt_path, regressor_ckpt_path)

    if do_infer:
        trainer.test(model, test_dataloaders=data_loaders["test"])
        assert cfg[cfg.experiment_type[cfg.current_stage]].kt_ckpt is not None
        return ""
        print("Done testing")
        exit()

    if cfg.experiment_type[cfg.current_stage] in ['A', 'B']:
        assert cfg[cfg.experiment_type[cfg.current_stage]].kt_ckpt is not None
        if cfg.experiment_type.second_stage is None or cfg.experiment_type.third_stage is None:
            exit()

    if cfg.experiment_type[cfg.current_stage] == 'C':
        model = pipeline_C(model, cfg)
        if cfg.C.freeze_item_emb_after_init:
            params = model.base_model.enc_embed.embed_feature.shifted_item_id.parameters()
            for param in params:
                param.requires_grad = False

    print(model)

    # Fitting trainer
    if cfg.experiment_type[cfg.current_stage] == 'A' or cfg.experiment_type[cfg.current_stage] == 'B':
        best_val_ckpt = glob.glob(os.path.join(log_dir, "SBERT_Regressor_*.ckpt"))[0]
        return best_val_ckpt
    if cfg.ckt_dataset_type == 'duolingo':
        trainer.fit(model, data_loaders["train"], data_loaders["test"])
    else:
        trainer.fit(model, data_loaders["train"], data_loaders["val"])
    if cfg.experiment_type[cfg.current_stage] == 'B':
        best_val_ckpt = glob.glob(os.path.join(log_dir, "SBERT_Regressor_*.ckpt"))[0]
    else:
        best_val_ckpt = glob.glob(os.path.join(log_dir, "KT_*.ckpt"))[0]

    best_model = get_model(cfg, load_ckpt=True, ckpt_path=best_val_ckpt)

    if cfg.experiment_type[cfg.current_stage] == 'A':
        assert cfg.done_training_kt_modules == False
        if not cfg.A.exp_zero:
            best_model = pipeline_A(best_model, cfg)
    elif cfg.experiment_type[cfg.current_stage] == 'B':
        best_model = pipeline_B(best_model, cfg, log_dir)
    trainer.test(best_model, test_dataloaders=data_loaders["test"])
    return best_val_ckpt

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    # root = "/tmp/pycharm_project_710/repoc_content_kt"
    root = "/tmp/pycharm_project_47/repoc_content_kt"
    # root = "/root/research-poc/repoc_content_kt"

    script_name = os.path.splitext(os.path.basename(__file__))[0]  # "run_am"
    cfg_file = "config.yaml"
    config = OmegaConf.load(os.path.join(root, "configs", cfg_file))
    config = OmegaConf.merge(config, OmegaConf.from_cli())

    # cur_time = datetime.now().replace(microsecond=0).isoformat()
    if not os.path.isdir(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))

    _log_dir = os.path.join(config.log_root, f"logs/{script_name}")
    if not os.path.isdir(_log_dir):
        os.mkdir(_log_dir)
    log_dir = os.path.join(_log_dir, config.exp_name)
    print(log_dir)

    if config.experiment_type.infer_stage is not None:
        save_cfg(config, log_dir, 'infer_cfg')
        config.current_stage = 'infer_stage'
        wandb.init(project='content_kt', name=config.exp_name + "_" + config.current_stage, reinit=True)
        if config.experiment_type[config.current_stage] == 'D':
            _ = main(
                cfg=config,
                data_root=config[config.ckt_dataset_type].data.root,
                log_dir=log_dir,
                ckpt_path=config.D.kt_ckpt,
                regressor_ckpt_path=config.D.regressor_ckpt,
                do_infer=True,
            )
        else:
            _ = main(
                cfg=config,
                data_root=config[config.ckt_dataset_type].data.root,
                log_dir=log_dir,
                ckpt_path=config[config.experiment_type[config.current_stage]].kt_ckpt,
                do_infer=True,
            )
    elif config.experiment_type.first_stage is not None:
        print("Running first stage...")
        config.current_stage = 'first_stage'
        save_cfg(config, log_dir, f'cfg_{config.current_stage}')
        if config.experiment_type[config.current_stage] == 'B':
            wandb.init(project='pipeline_B', name=config.exp_name + "_" + config.current_stage, reinit=True)
        else:
            wandb.init(project='content_kt', name=config.exp_name + "_" + config.current_stage, reinit=True)
        # breakpoint()
        first_stage_ckpt = main(
            cfg=config,
            data_root=config[config.ckt_dataset_type].data.root,
            log_dir=log_dir,
            ckpt_path=config[config.experiment_type[config.current_stage]].kt_ckpt,
        )
        # model = LightningContentDKT.load_from_checkpoint(first_stage_ckpt, cfg=config)
        # config.experiment_type[config.current_stage] = 'D'
        # model_2 = LightningContentDKT.load_from_checkpoint(first_stage_ckpt, cfg=config)
        # breakpoint()
    if config.experiment_type.second_stage is not None:
        print("Running second stage...")
        config.current_stage = 'second_stage'
        if config[config.experiment_type[config.current_stage]].kt_ckpt is None:
            config[config.experiment_type[config.current_stage]].kt_ckpt = first_stage_ckpt

        if config.experiment_type['first_stage'] is None:
            first_stage_ckpt = config[config.experiment_type[config.current_stage]].kt_ckpt
        save_cfg(config, log_dir, f'cfg_{config.current_stage}')
        if config.experiment_type[config.current_stage] == 'B':
            wandb.init(project='pipeline_B', name=config.exp_name + "_" + config.current_stage, reinit=True)
        else:
            wandb.init(project='content_kt', name=config.exp_name + "_" + config.current_stage, reinit=True)

        # breakpoint()
        second_stage_ckpt = main(
            cfg=config,
            data_root=config[config.ckt_dataset_type].data.root,
            log_dir=log_dir,
            ckpt_path=first_stage_ckpt,
        )
        # breakpoint()
    if config.experiment_type.third_stage is not None:
        print("Running third stage...")
        config.current_stage = 'third_stage'
        assert config.experiment_type[config.current_stage] == 'D' # Other options are not yet supported.
        if config.experiment_type.first_stage is None and config.experiment_type.second_stage is None:
            first_stage_ckpt = config.D.kt_ckpt
            second_stage_ckpt = config.D.regressor_ckpt
        config[config.experiment_type[config.current_stage]].kt_ckpt = first_stage_ckpt
        save_cfg(config, log_dir, f'cfg_{config.current_stage}')
        if config.experiment_type[config.current_stage] == 'B':
            wandb.init(project='pipeline_B', name=config.exp_name + "_" + config.current_stage, reinit=True)
        else:
            wandb.init(project='content_kt', name=config.exp_name + "_" + config.current_stage, reinit=True)
        print("ERROR?", first_stage_ckpt)
        # breakpoint()
        _ = main(
            cfg=config,
            data_root=config[config.ckt_dataset_type].data.root,
            log_dir=log_dir,
            ckpt_path=first_stage_ckpt,
            regressor_ckpt_path=second_stage_ckpt,
        )
    print("Done running.")

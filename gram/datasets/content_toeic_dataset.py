from gram.utils import utils
from typing import (
    Any,
    List,
)

import numpy as np
import omegaconf
import torch
from gram.utils.utils import standard_collate_fn
from blink.data.transforms import (
    ConvertTypes,
    AppendIntegerPosition,
    ShiftRight,
)
from gram.utils.utils import StandardToeicInteractionDataset
from magneto.data.utils.utils_datasets import dataset_split
from gram.utils.utils import (
    get_pre_transform,
    get_standard_collate_fn
)


def content_data(
    data_root: str, cfg: omegaconf.dictconfig.DictConfig
) -> torch.utils.data.DataLoader:
    datasets = content_toeic_dataset(data_root, cfg)
    return get_data_loader(datasets, cfg, is_score=False)


def content_toeic_dataset(
    data_root: str, cfg: omegaconf.dictconfig.DictConfig
) -> torch.utils.data.Dataset:
    assert cfg.ckt_dataset_type in ['toeic', 'duolingo', 'poj']
    if cfg.ckt_dataset_type == 'toeic':
        assert cfg[cfg.ckt_dataset_type].data.max_item_id == 19398
        file_dict = {
            "interactions": "interaction_csqe_part_4_to_7.hdf5",
            "users": "users_part_4_to_7_csqe.hdf5",
            "items": "items.hdf5",
        }
    elif cfg.ckt_dataset_type == 'poj':
        assert cfg[cfg.ckt_dataset_type].data.max_item_id == 4055
        # file_dict = {
        #     "interactions": "poj_interactions_with_scraped.hdf5",
        #     "users": "poj_users_with_scraped.hdf5",
        #     "items": "items.hdf5",
        # }
        file_dict = {
            "interactions": "poj_interactions_new_split.hdf5",
            "users": "poj_users_new_split.hdf5",
            "items": "items.hdf5",
        }
    elif cfg[cfg.ckt_dataset_type].language == 'french':
        assert cfg[cfg.ckt_dataset_type].data.max_item_id == 4331
        for feature in ['part_id', 'timeliness']:
            assert feature not in cfg[cfg.ckt_dataset_type].data.using_features
        if cfg[cfg.ckt_dataset_type].user_based_split:
            file_dict = {
                "interactions": "duolingo_interactions_new_split.hdf5",
                "users": "duolingo_users_new_split.hdf5",
                "items": "items.hdf5",
            }
        else:
            file_dict = {
                "interactions": "duolingo_interactions_longer_test.hdf5",
                "users": "duolingo_users_longer_test.hdf5",
                "items": "items.hdf5",
            }
    elif cfg[cfg.ckt_dataset_type].language == 'spanish':
        assert cfg[cfg.ckt_dataset_type].data.max_item_id == 5001
        for feature in ['part_id', 'timeliness']:
            assert feature not in cfg[cfg.ckt_dataset_type].data.using_features
        file_dict = {
            "interactions": "duolingo_spanish_interactions_longer_test.hdf5",
            "users": "duolingo_spanish_users_longer_test.hdf5",
            "items": "items.hdf5",
        }
    data_path_dict = get_path_dict(
        data_root,
        file_dict=file_dict,
    )
    print("Content KT dataset processing")
    if cfg[cfg.ckt_dataset_type].data.using_features is None:
        using_features = tuple({**cfg[cfg.ckt_dataset_type].embed.enc_feature, **cfg[cfg.ckt_dataset_type].embed.dec_feature})
    else:
        using_features = tuple(cfg[cfg.ckt_dataset_type].data.using_features)

    dset = StandardToeicInteractionDataset(
        data_path_dict,
        using_features,
        min_num_of_interactions=cfg[cfg.ckt_dataset_type].data.min_num_interactions,
    )

    print("split", cfg[cfg.ckt_dataset_type].am.split)

    datasets = dataset_split(dset, split=cfg[cfg.ckt_dataset_type].am.split, random_train_val_split=cfg[cfg.ckt_dataset_type].am.random_split)

    datasets["test"] = StandardToeicInteractionDataset(
        data_path_dict,
        using_features,
        min_num_of_interactions=cfg[cfg.ckt_dataset_type].data.min_num_interactions,
        interactions_object_key="interactions_test",
        users_object_key="users_test",
    )
    # breakpoint()

    return datasets


def get_data_loader(
    datasets: torch.utils.data.Dataset,
    cfg: omegaconf.dictconfig.DictConfig,
    is_score: bool = False,
) -> torch.utils.data.DataLoader:
    assert "train" in datasets

    dataloaders = {}
    for mode in datasets.keys():
        if cfg.train_mode == "pretrain":
            pre_tfm_opt = cfg[cfg.ckt_dataset_type].am.pre_transform_opt
        elif cfg.train_mode == "finetune":
            pre_tfm_opt = cfg.score.pre_transform_opt

        if cfg.model_name in ["saint", "all_item_kt", "dkt"] and mode in ["val", "test"]:
            pre_tfm_opt = "Recent"  # "AllInteractionsTest"

        pre_tfm = get_pre_transform(
            seq_len=cfg[cfg.ckt_dataset_type].data.max_seq_len,
            pad_len=cfg[cfg.ckt_dataset_type].data.max_seq_len,
            pad_value=cfg[cfg.ckt_dataset_type].embed.pad_value,
            option=pre_tfm_opt,
        )

        post_tfm = get_post_transform(cfg[cfg.ckt_dataset_type].data.max_seq_len, cfg[cfg.ckt_dataset_type].embed.start_token)

        def _score_collate_fn(batch):
            return standard_score_collate_fn(
                batch,
                pre_transforms=pre_tfm,
                post_transforms=post_tfm,
                score_tag_keys=[subtype.key for subtype in cfg.data.score_subtype],
            )

        if is_score:
            collate_fn = _score_collate_fn
        else:
            collate_fn = get_standard_collate_fn(
                pre_transforms=pre_tfm,
                post_transforms=post_tfm,
                additional_preprocess_keys=[
                    feature.replace("shifted_", "")
                    for feature in {
                        **cfg[cfg.ckt_dataset_type].embed.dec_feature,
                        **cfg[cfg.ckt_dataset_type].embed.enc_feature,
                    }
                    if feature.endswith("_norm")
                ],
                max_lag_time=86400 * 1000,
                max_elapsed_time=300 * 1000,
            )

        # num_workers = 0 if mode == "train" else cfg.num_workers
        dataloaders[mode] = torch.utils.data.DataLoader(
            datasets[mode],
            batch_size=cfg[cfg.experiment_type[cfg.current_stage]].batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            shuffle=(mode == "train"),
        )

    return dataloaders



def get_post_transform(seq_len: int, start_token=None) -> List[Any]:
    if start_token is None:
        post_transform = []
    else:
        post_transform = [
            ShiftRight(list(start_token.keys()), start_token_dict=start_token),
        ]
    post_transform += [
        AppendIntegerPosition(seq_len),
        ConvertTypes({np.int8: np.int64, np.uint8: np.int64, np.float64: np.float32}),
    ]
    return post_transform

from abc import ABC
from typing import Any, Dict, Iterable, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd
from torch.utils import data

from magneto.data.utils.utils_datasets import (
    get_user_interactions,
    get_user_windows,
    get_interactions_index,
)


class StandardToeicInteractionDataset(data.Dataset, ABC):

    def __init__(
        self,
        data_path_dict: Dict[str, str],
        using_features_from_interaction: Tuple[str] = (
            "user_id",
            "item_id",
            "is_correct",
            "lag_time_in_ms",
            "elapsed_time_in_ms",
        ),
        min_num_of_interactions: Optional[int] = 1,
        interactions_object_key: str = "interactions_train",
        users_object_key: str = "users_train",
        transforms: Optional[Iterable[Any]] = None,
    ):
        assert interactions_object_key in [
            "interactions_train",
            "interactions_test",
        ]
        data.Dataset.__init__(self)

        self.interaction_hf = h5py.File(data_path_dict["interactions"], "r")
        self.user_hf = h5py.File(data_path_dict["users"], "r")
        self.interactions_object_key = interactions_object_key
        self.min_num_of_interactions = min_num_of_interactions
        self.transforms = transforms
        self.using_features_from_interaction = using_features_from_interaction

        self.sample_list = get_user_windows(
            self.user_hf[users_object_key], min_num_of_interactions
        )

        self.index_to_userid = self.get_index_to_userid()
        self.userid_to_indices = self.get_userid_to_indices()

        self.interaction_meta = self.interaction_hf[interactions_object_key][
            using_features_from_interaction
        ]

    def get_index_to_userid(self):
        return {idx: uid for idx, (uid, window) in enumerate(self.sample_list)}

    def get_userid_to_indices(self):
        userids = [uid for uid, window in self.sample_list]
        userid_to_indices = {}
        for idx, uid in enumerate(userids):
            try:
                userid_to_indices[uid].append(idx)
            except KeyError:
                userid_to_indices[uid] = [idx]
        return userid_to_indices

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index: int):
        uid, window = self.sample_list[index]
        user_interactions = get_user_interactions(
            self.interaction_meta,
            window,
        )

        if self.transforms:
            for transform in self.transforms:
                user_interactions = transform(user_interactions)

        return user_interactions


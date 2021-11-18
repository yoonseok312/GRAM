from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from abc import ABC
from typing import Any, Dict, Iterable, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd
from torch.utils import data

import numpy as np
import torch
from deprecate import deprecated

import os

from datetime import datetime
from typing import Tuple

import torch

from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf

def get_path_dict(input_file_root: str, file_dict: Dict[str, str]) -> Dict[str, str]:
    return {key: os.path.join(input_file_root, fname) for key, fname in file_dict.items()}

def get_pre_transform(
    seq_len: int,
    pad_len: int,
    pad_value: Dict[str, int],
    option: str = "Recent",
) -> List[Any]:
    tfm = []
    if option == "RecentRandom":
        tfm += [
            GetOnlyRecent(seq_len * 2),
            GetRandomSample(seq_len, sampling_min_n=seq_len),
        ]
    elif option == "Recent":
        tfm.append(GetOnlyRecent(seq_len))
        tfm.append(PadUptoEnd(pad_len, pad_value_dict=pad_value))
    elif option == "RandomWindow":
        tfm.append(GetRandomWindow(seq_len))
        tfm.append(PadUptoEnd(pad_len, pad_value_dict=pad_value))
    elif option == "Random":
        tfm.append(GetRandomSample(seq_len, sampling_min_n=seq_len))
        tfm.append(PadUptoEnd(pad_len, pad_value_dict=pad_value))
    elif option == "AllInteractionsTest":
        tfm.append(PrepareAllInteractionsTest(seq_len, pad_value))
    else:
        raise ValueError(
            "Pre-transform option should one of followings:"
            " 'Recent', 'Random', 'RecentRandom, or 'RandomWindow'."
        )
    # tfm.append(PadUptoEnd(pad_len, pad_value_dict=pad_value))
    return tfm


def get_post_transform(seq_len: int, start_token=None) -> List[Any]:
    if start_token is None:
        post_transform = []
    else:
        post_transform = [
            ShiftRight(list(start_token.keys()), start_token_dict=start_token),
            ExpandDimension(["shifted_lag_time_norm", "shifted_elapsed_time_norm"]),
        ]
    post_transform += [
        AppendIntegerPosition(seq_len),
        ConvertTypes({np.int8: np.int64, np.uint8: np.int64, np.float64: np.float32}),
    ]
    return post_transform


def get_standard_collate_fn(
    additional_preprocess_keys: Optional[Iterable[str]] = None,
    pre_transforms: Optional[Iterable[Callable]] = None,
    post_transforms: Optional[Iterable[Callable]] = None,
    return_type: Type = Dict[str, torch.Tensor],
    **kwargs,
) -> Dict[str, Any]:
    def _collate(interaction_batch: List[np.ndarray]):
        assert isinstance(interaction_batch, list)
        available_return_type = [
            Dict[str, torch.Tensor],
            Dict[str, np.ndarray],
        ]
        assert return_type in available_return_type

        # do transforms (i.e. pad)
        if pre_transforms:
            # do masking or matching interactions with model.seq_size
            for transform in pre_transforms:
                interaction_batch = [
                    transform(interaction_i) for interaction_i in interaction_batch
                ]

        # convert into dict_of_ndarray
        interaction_batch = convert_to_dictofndarray(interaction_batch)

        if additional_preprocess_keys:
            # all of data should be masked with sequence_size, here
            interaction_batch = preprocess_integrate(
                preprocess_keys=additional_preprocess_keys,
                interaction_meta=interaction_batch,
                **kwargs,
            )

        if post_transforms:
            # other custom post_transforms
            for transform in post_transforms:
                interaction_batch = transform(interaction_batch)

        assert isinstance(interaction_batch, dict)

        # convert to torch tensor from np.array
        if return_type == Dict[str, torch.Tensor]:
            for key, interaction_val in interaction_batch.items():
                interaction_batch[key] = torch.from_numpy(interaction_val)

        # return batch according to return types
        if return_type in [Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
            return interaction_batch
        else:
            raise ValueError(
                "Not supported return_type." + f"{return_type} should be in {available_return_type}"
            )

    return _collate

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def get_config(
    root: str,
    is_test: bool = False,
    model_type: str = "am",
    dataset_type: str = "toeic",
) -> DictConfig:
    if is_test:
        assert model_type in ["am", "compressive", "dpa"]
        assert dataset_type in ["toeic", "sat"]
        base_config_file = f"test_{dataset_type}_{model_type}_cfg.yaml"
    else:
        base_config_file = f"{dataset_type}_{model_type}_cfg.yaml"
    cfg = OmegaConf.load(os.path.join(root, "config", base_config_file))
    if not is_test:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def get_log_dir_path(root: str) -> str:
    cur_time = datetime.now().replace(microsecond=0).isoformat()
    _log_dir = os.path.join(root, "logs")
    if not os.path.isdir(_log_dir):
        os.mkdir(_log_dir)
    log_dir = os.path.join(_log_dir, cur_time)
    return log_dir



def save_cfg(cfg: DictConfig, log_path: str, file_name: str = 'run_cfg') -> None:
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fpath = os.path.join(log_path, f"{file_name}.yaml")
    OmegaConf.save(config=cfg, f=fpath)


def setup_run(
    root: str,
    is_test: bool = False,
    save_log: bool = True,
    dataset_type="toeic",
    model_type="am",
) -> Tuple[DictConfig, str]:
    config = get_config(root, is_test=is_test, dataset_type=dataset_type, model_type=model_type)
    log_path = get_log_dir_path(root)

    if save_log:
        save_cfg(config, log_path)
    return config, log_path


def standard_collate_fn(
    interaction_batch: List[np.ndarray],
    additional_preprocess_keys: Optional[Iterable[str]] = None,
    pre_transforms: Optional[Iterable[Callable]] = None,
    post_transforms: Optional[Iterable[Callable]] = None,
    return_type: Type = Dict[str, torch.Tensor],
    **kwargs,
) -> Dict[str, Any]:
    
    assert isinstance(interaction_batch, list)
    available_return_type = [
        Dict[str, torch.Tensor],
        Dict[str, np.ndarray],
    ]
    assert return_type in available_return_type

    # do transforms (i.e. pad)
    if pre_transforms:
        # do masking or matching interactions with model.seq_size
        for transform in pre_transforms:
            interaction_batch = [
                transform(interaction_i) for interaction_i in interaction_batch
            ]

    # convert into dict_of_ndarray
    interaction_batch = convert_to_dictofndarray(interaction_batch)

    if additional_preprocess_keys:
        # all of data should be masked with sequence_size, here
        interaction_batch = preprocess_integrate(
            preprocess_keys=additional_preprocess_keys,
            interaction_meta=interaction_batch,
            **kwargs,
        )

    if post_transforms:
        # other custom post_transforms
        for transform in post_transforms:
            interaction_batch = transform(interaction_batch)

    assert isinstance(interaction_batch, dict)

    # convert to torch tensor from np.array
    if return_type == Dict[str, torch.Tensor]:
        for key, interaction_val in interaction_batch.items():
            interaction_batch[key] = torch.from_numpy(interaction_val)

    # return batch according to return types
    if return_type in [Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        return interaction_batch
    else:
        raise ValueError(
            "Not supported return_type."
            + f"{return_type} should be in {available_return_type}"
        )

def standard_score_collate_fn(
    batch: List[Dict[str, np.ndarray]],
    additional_preprocess_keys: Optional[Iterable[str]] = None,
    pre_transforms: Optional[Iterable[Callable]] = None,
    post_transforms: Optional[Iterable[Callable]] = None,
    return_type: Type = Dict[str, torch.Tensor],
    interactions_key: str = "interactions",
    score_key: str = "score",
    score_tag_keys: List[str] = ["TOEIC_LC", "TOEIC_RC"],
    **kwargs,
) -> dict:

    available_return_type = [
        Dict[str, torch.Tensor],
        Dict[str, np.ndarray],
    ]
    assert return_type in available_return_type

    interaction_batch = standard_collate_fn(
        [x[interactions_key] for x in batch],
        additional_preprocess_keys=additional_preprocess_keys,
        pre_transforms=pre_transforms,
        post_transforms=post_transforms,
        return_type=return_type,
        **kwargs,
    )

    # `stack` is a function that zips a python array of tensor/ndarray/numbers
    # to one tensor/ndarray, depending on the return type
    if return_type == Dict[str, torch.Tensor]:
        stack = torch.LongTensor  # currently assumes that the values are long
    elif return_type == Dict[str, np.ndarray]:
        stack = np.stack
    else:
        raise ValueError("not supported return_type")

    score_batch = {}
    for tag_key in score_tag_keys:
        key_batch = stack([x[score_key][tag_key] for x in batch])
        score_batch[tag_key] = key_batch

    res_batch = {}
    res_batch["interactions"] = interaction_batch
    res_batch["score"] = score_batch

    if "user_id" in batch[0]:
        user_id_batch = stack([b["user_id"] for b in batch])
        res_batch["user_id"] = user_id_batch
    if "interaction_index" in batch[0]:
        interaction_index_batch = stack(
            [b["interaction_index"] for b in batch]
        )
        res_batch["interaction_index"] = interaction_index_batch

    return res_batch


@deprecated(
    target=standard_score_collate_fn, deprecated_in="0.2.5", remove_in="0.4"
)
def toeic_score_collate_fn(
    batch: List[Dict[str, np.ndarray]],
    additional_preprocess_keys: Optional[Iterable[str]] = None,
    pre_transforms: Optional[Iterable[Any]] = None,
    post_transforms: Optional[Iterable[Any]] = None,
    return_type: Type = Dict[str, torch.Tensor],
    **kwargs,
):
    """for supporting previous version of collate_fn"""
    return standard_score_collate_fn(
        batch,
        additional_preprocess_keys,
        pre_transforms,
        post_transforms,
        return_type,
        score_key="score",
        score_tag_keys=["TOEIC_LC", "TOEIC_RC"],
        **kwargs,
    )

class StandardToeicInteractionDataset(data.Dataset, ABC):
    """Standard pytorch.data.Dataset Class for SantaToeic Data

    [Example Usage]
    input_file_root = f"./sample_data/post_query_results/{date_str}"
    interaction_path = os.path.join(input_file_root, "interactions.hdf5")
    user_path = os.path.join(input_file_root, "users.hdf5")
    item_path = os.path.join(input_file_root, "items.hdf5")

    data_path_dict = {
        "interactions": interaction_path,
        "users": user_path,
        "items": item_path,
    }

    dset = StandardToeicInteractionDataset(
        data_path_dict,
        (
            "user_id",
            "item_id",
            "start_time",
            "elapsed_time_in_ms",
            "timeliness",
            "is_correct",
            "lag_time_in_ms",
        ),
    )

    """

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
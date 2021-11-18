from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type

import numpy as np
import torch
from deprecate import deprecated

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
from typing import Dict

import torch
import torch.nn as nn
import omegaconf
import pickle
from gram.knowledge_tracing.components.SBERT import SBERT, CKTAdditiveAttention
from gram.knowledge_tracing.components.eernn import BidirectionalLSTM, TransformerEncoder
from torchtext.legacy import data
from torchtext.vocab import Vectors
from torchtext.legacy import datasets
from torchtext.data import get_tokenizer
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
import numpy as np
from sentence_transformers import models

def make_embed_feature(
    cfg: omegaconf.dictconfig.DictConfig,
    layer_type: str = "encoder",
) -> nn.ModuleDict:
    """Build embedding layers from feature configuration
    Args:
        cfg: OmegaConfig
    """
    if layer_type == "encoder":
        features = cfg[cfg.ckt_dataset_type].embed.enc_feature
    if layer_type == "decoder":
        features = cfg[cfg.ckt_dataset_type].embed.dec_feature
    pad_values = cfg[cfg.ckt_dataset_type].embed.pad_value

    embed = nn.ModuleDict()
    if "item_id" in features:
        embed["item_id"] = nn.Embedding(
            cfg[cfg.ckt_dataset_type].data.max_item_id + 1,
            features.item_id,
            padding_idx=pad_values.item_id,
        )
    if "part_id" in features:
        embed["part_id"] = nn.Embedding(
            cfg[cfg.ckt_dataset_type].data.max_part_id + 1,
            features.part_id,
            padding_idx=pad_values.part_id,
        )
    if "section" in features:
        embed["section"] = nn.Embedding(
            cfg[cfg.ckt_dataset_type].data.max_section + 1,
            features.section,
            padding_idx=pad_values.section,
        )
    if "is_correct" in features:
        embed["is_correct"] = nn.Embedding(
            3,
            features.is_correct,
            padding_idx=pad_values.is_correct,
        )
    if "timeliness" in features:
        embed["timeliness"] = nn.Embedding(
            3,
            features.timeliness,
            padding_idx=pad_values.timeliness,
        )
    if "elapsed_time_norm" in features:
        embed["elapsed_time_norm"] = nn.Linear(
            1,
            features.elapsed_time_norm,
            bias=False,
        )
    if "lag_time_norm" in features:
        embed["lag_time_norm"] = nn.Linear(1, features.lag_time_norm, bias=False)

    if "shifted_item_id" in features:
        embed["shifted_item_id"] = nn.Embedding(
            cfg[cfg.ckt_dataset_type].data.max_item_id + 1,
            features.shifted_item_id,
            padding_idx=pad_values.item_id,
        )
    if "shifted_part_id" in features:
        embed["shifted_part_id"] = nn.Embedding(
            cfg[cfg.ckt_dataset_type].data.max_part_id + 1,
            features.shifted_part_id,
            padding_idx=pad_values.part_id,
        )
    if "shifted_is_correct" in features:
        embed["shifted_is_correct"] = nn.Embedding(
            3,
            features.shifted_is_correct,
            padding_idx=pad_values.is_correct,
        )
    if "shifted_timeliness" in features:
        embed["shifted_timeliness"] = nn.Embedding(
            3,
            features.shifted_timeliness,
            padding_idx=pad_values.timeliness,
        )
    if "shifted_elapsed_time_norm" in features:
        embed["shifted_elapsed_time_norm"] = nn.Linear(
            1,
            features.shifted_elapsed_time_norm,
            bias=False,
        )
    if "shifted_lag_time_norm" in features:
        embed["shifted_lag_time_norm"] = nn.Linear(1, features.shifted_lag_time_norm, bias=False)

    return embed


class AllItemInputEmbedding(nn.Module):
    """Embeds multiple features into one embedding vector"""

    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
        layer_type: str = "encoder",
    ) -> None:
        super(AllItemInputEmbedding, self).__init__()
        embed_cfg = cfg[cfg.ckt_dataset_type].embed
        self.embed_feature = make_embed_feature(cfg, layer_type)
        self.cfg = cfg
        if cfg.ckt_dataset_type == 'toeic':
            if cfg[cfg.experiment_type[cfg.current_stage]].with_passage:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text_special_tokens.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
            else:
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'duolingo':
            if cfg[cfg.ckt_dataset_type].language == 'french':
                with open(f"{self.cfg[self.cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl",
                          "rb") as handle:
                    self.id_to_text = pickle.load(handle)
            elif cfg[cfg.ckt_dataset_type].language == 'spanish':
                with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_all.pkl", "rb") as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == 'poj':
            print("Truncated text")
            with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl", "rb") as handle:
                self.id_to_text = pickle.load(handle)
        self.layer_type = layer_type
        if layer_type == "encoder":
            embed_features = cfg[cfg.ckt_dataset_type].embed.enc_feature
        if layer_type == "decoder":
            embed_features = cfg[cfg.ckt_dataset_type].embed.dec_feature

        total_embed_feature_dim = sum(embed_features.values())

        if embed_cfg.positional is not None:
            self.positional = nn.Parameter(
                torch.empty((cfg[cfg.ckt_dataset_type].data.max_seq_len, embed_cfg.positional))
            )
            torch.nn.init.xavier_uniform_(self.positional)
            total_embed_feature_dim += embed_cfg.positional
        else:
            self.positional = None

        # self.corr_layer = nn.Linear(cfg[cfg.ckt_dataset_type].enc_feature.shifted_item_id, cfg[cfg.ckt_dataset_type].enc_feature.shifted_item_id)
        # self.incorr_layer = nn.Linear(cfg[cfg.ckt_dataset_type].enc_feature.shifted_item_id, cfg[cfg.ckt_dataset_type].enc_feature.shifted_item_id)
        self.aggregate_embeds = nn.Linear(total_embed_feature_dim, cfg.train.dim_model)


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        # get embedding result for each feature
        embed_results = [
            self.embed_feature[name](tensor)
            for name, tensor in batch.items()
            if name in self.embed_feature
        ]

        if self.cfg.experiment_type[self.cfg.current_stage] == 'D':
            embed_results.pop(0)
            embed_results.insert(0, text_embedding_batch)

        # add positional embedding
        batch_dim = embed_results[0].shape[0]

        if self.positional is not None:
            positional_embed = self.positional.unsqueeze(0).repeat(batch_dim, 1, 1)
            embed_results.append(positional_embed)

        # aggregate embeddings
        all_embeds = torch.cat(embed_results, dim=-1)
        return self.aggregate_embeds(all_embeds)


class SaintInputEmbedding(nn.Module):
    """Embeds multiple features into one embedding vector"""

    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
        layer_type: str = "encoder",
    ) -> None:
        super(SaintInputEmbedding, self).__init__()
        embed_cfg = cfg.embed
        self.embed_feature = make_embed_feature(cfg, layer_type)
        if cfg.experiment_type == 'D':
            self.SBERT = SBERT(cfg[cfg.ckt_dataset_type].D.base_lm.model.pretrained)
        with open(f"{cfg.data.root}/toeic_id_to_text_special_tokens.pkl", "rb") as handle:
            self.id_to_text = pickle.load(handle)
        self.cfg = cfg
        self.layer_type = layer_type
        if layer_type == "encoder":
            embed_features = cfg.embed.enc_feature
        if layer_type == "decoder":
            embed_features = cfg.embed.dec_feature

        total_embed_feature_dim = sum(embed_features.values())
        if embed_cfg.positional is not None:
            self.positional = nn.Parameter(
                torch.empty((cfg[cfg.ckt_dataset_type].data.max_seq_len, embed_cfg.positional))
            )
            torch.nn.init.xavier_uniform_(self.positional)
            total_embed_feature_dim += embed_cfg.positional
        else:
            self.positional = None

        self.aggregate_embeds = nn.Linear(total_embed_feature_dim, cfg.train.dim_model)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # encoded_all: torch.Tensor
    ) -> torch.Tensor:
        # get embedding result for each feature
        if self.cfg.is_saint and self.cfg.finetune_SBERT and "item_id" in batch.keys():
            # assert 'item_id' not in self.cfg.embed.enc_feature
            embed_results = [
                self.embed_feature[name](tensor)
                for name, tensor in batch.items()
                if name in self.embed_feature and not "item_id"
            ]
        else:
            embed_results = [
                self.embed_feature[name](tensor)
                for name, tensor in batch.items()
                if name in self.embed_feature
            ]

        if self.layer_type == "encoder":
            if not self.cfg.init_with_text_embedding and self.cfg.finetune_SBERT:
                item_ids = {item_id.item() for user in batch["item_id"] for item_id in user}
                item_ids_list = [
                    item_id for item_id in self.id_to_text.keys() if item_id in item_ids
                ]
                encoded_valid_ids = self.SBERT.encode(
                    sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                    batch_size=self.cfg.SBERT_batch_size,
                    convert_to_tensor=True,
                )

                text_embedding_batch = []
                for user in batch["item_id"]:
                    text_embedding_user = []
                    for item_id in user:
                        if item_id.item() != self.cfg.embed.pad_value.item_id:
                            item_embedding = encoded_valid_ids[item_ids_list.index(item_id.item())]
                        else:
                            item_embedding = torch.zeros(
                                self.cfg.D.base_lm.text_embedding_dim, device="cuda"
                            )

                        text_embedding_user.append(item_embedding)
                    text_embedding_user = torch.stack(text_embedding_user).to("cuda")
                    text_embedding_batch.append(text_embedding_user)
                text_embedding_batch = torch.stack(text_embedding_batch)

                embed_results.append(text_embedding_batch)
        # torch.autograd.set_detect_anomaly(True)
        batch_dim = embed_results[0].shape[0]
        if self.positional is not None:
            positional_embed = self.positional.unsqueeze(0).repeat(batch_dim, 1, 1)
            embed_results.append(positional_embed)

        all_embeds = torch.cat(embed_results, dim=-1)
        return self.aggregate_embeds(all_embeds)

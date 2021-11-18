from typing import Dict
from typing import List
from typing import Any
from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
import numpy as np
import omegaconf
import torch.nn.functional as F
import pickle

from repoc_content_kt.models.components.input_embedding import AllItemInputEmbedding
from repoc_content_kt.models.components.SBERT import SBERT
from repoc_common import utils

# from magneto.train.schedulers import get_noam_scheduler
from repoc_content_kt.models.content_all_item_kt import ContentAllItemGenerator, LightningContentAllItemKT

class ContentDKT(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self, config) -> None:
        super(ContentDKT, self).__init__()
        self.cfg = config
        print("init model")
        self.enc_embed = AllItemInputEmbedding(config, "encoder")
        # self.dec_embed = InputEmbedding(config, "decoder")

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=self.cfg.train.dim_model,
            hidden_size=self.cfg.train.dim_model,
            num_layers=self.cfg.train.encoder_layer_count,
            # batch_first=True,
            dropout=self.cfg.train.dropout_rate,
        )

        self.generator = ContentAllItemGenerator(
            self.cfg.train.dim_model, self.cfg[self.cfg.ckt_dataset_type].data.max_item_id, self.enc_embed, self.cfg
        )

    def _init_hidden(self, batch_size: int, device: torch.device):
        """
        Get zero-initialized hidden state & cell state vectors
        Args:
            batch_size: batch size
            device: device where input tensor exists now
        Returns:
            hidden_zero or (hidden_zero, cell_zero): zero-initialized tensors
        """

        hidden_zero = torch.zeros(
            self.cfg.train.encoder_layer_count, batch_size, self.cfg.train.dim_model, device=device
        ).requires_grad_()
        cell_zero = torch.zeros(
            self.cfg.train.encoder_layer_count, batch_size, self.cfg.train.dim_model, device=device
        ).requires_grad_()
        return (
            hidden_zero,
            cell_zero,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # embed_all: torch.Tensor
    ) -> torch.Tensor:
        # enc_input, dec_input = self.enc_embed(batch), self.dec_embed(batch)
        # enc_input, dec_input = map(lambda x: x.transpose(0, 1), (enc_input, dec_input))  # seq_size * batch * dim_model
        enc_input, embed_all = self.enc_embed(batch)
        enc_input = enc_input.transpose(0, 1)
        # print(np.sum(self.enc_embed.embed_feature.shifted_item_id.weight.cpu().detach().numpy()))

        device = batch["item_id"].device
        batch_size = batch["item_id"].shape[0]

        init_hidden = self._init_hidden(batch_size, device)
        tr_output, _ = self.lstm(enc_input, init_hidden)
        tr_output = torch.transpose(tr_output, 0, 1)  # batch * seq_size * dim_model
        output = self.generator(tr_output, embed_all)
        # output = self.generator(tr_output)
        return output


class LightningContentDKT(LightningContentAllItemKT):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        super().__init__(cfg=cfg)
        self.save_hyperparameters()  # stores hyperparameters
        self.cfg = cfg
        self.threshold = 0.5
        self.base_model = ContentDKT(cfg)
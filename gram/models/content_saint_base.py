"""
A simple implementation of SAINT. Does not implement training loop
"""
from typing import Dict

import omegaconf
import torch
import torch.nn as nn

from repoc_content_kt.models.components.input_embedding import SaintInputEmbedding


class Generator(nn.Module):
    """
    A submodule of SAINT that converts raw Transformer output to prediction tensor
    3 FC layers with LayerNorms
    """

    def __init__(self, d_model):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(d_model, int(d_model / 2)),
            nn.LayerNorm(int(d_model / 2)),
            nn.ReLU(),
            nn.Linear(int(d_model / 2), int(d_model / 4)),
            nn.LayerNorm(int(d_model / 4)),
            nn.ReLU(),
            nn.Linear(int(d_model / 4), 1),
        )

    def forward(self, val):
        """
        Takes any tensor of shape A x d_model and returns shape A
        """
        return self.generator(val).squeeze(-1)


# used internally
def shift_right(tensor, shift, pad_value=0, dim=-1):
    """
    Args:
        tensor: Tensor to shift
        shift: Distance to shift by
        pad_value: Symbol used to fill the empty spaces left by shifting
        dim: shifting dimension
    Returns:
        A tensor of the same shape as tensor with components shifted
        n units to the right along dimension dim.
    """
    assert 0 <= shift <= tensor.shape[dim]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = shift
    padding = tensor.new_full(pad_shape, pad_value)
    shifted = torch.narrow(tensor, dim, 0, tensor.shape[dim] - shift)
    return torch.cat([padding, shifted], dim=dim)


# def get_pad_mask(tensor: torch.Tensor, pad_value: torch.Tensor):
#     # used for nn.Transformer
#     return tensor == pad_value


class ContentSaintBaseModel(nn.Module):
    """
    Simple SAINT model with no training logic (only inference)
    """

    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        super(ContentSaintBaseModel, self).__init__()
        self.cfg = cfg
        self.enc_embed = SaintInputEmbedding(cfg, "encoder")
        self.dec_embed = SaintInputEmbedding(cfg, "decoder")

        self.transformer = nn.Transformer(
            d_model=self.cfg.train.dim_model,
            nhead=self.cfg.train.head_count,
            num_encoder_layers=self.cfg.train.encoder_layer_count,
            num_decoder_layers=self.cfg.train.decoder_layer_count,
            dim_feedforward=self.cfg.train.dim_feedforward,
            dropout=self.cfg.train.dropout_rate,
        )
        # self.transformer = TorchTransformerBase(
        #     d_model=self.cfg.train.dim_model,
        #     nhead=self.cfg.train.head_count,
        #     num_encoder_layers=self.cfg.train.encoder_layer_count,
        #     num_decoder_layers=self.cfg.train.decoder_layer_count,
        #     dim_feedforward=self.cfg.train.dim_feedforward,
        #     dropout=self.cfg.train.dropout_rate,
        #     layernorm_type="pre",
        # )
        self.generator = Generator(self.cfg.train.dim_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # get transformer input
        enc_input = self.enc_embed(batch)  # batch * seq_size * dim_model
        enc_input = torch.transpose(enc_input, 0, 1)  # seq_size * batch * dim_model

        shifted_keys = set(self.dec_embed.embed_feature.keys())
        # shifted_keys = set(["is_correct", "timeliness"])
        shifted_batch = {
            name: shift_right(
                tensor,
                shift=1,
                pad_value=self.cfg.embed.pad_value[name],
                dim=1,
            )
            for name, tensor in batch.items()
            if name in shifted_keys
        }
        dec_input = self.dec_embed(shifted_batch)  # batch * seq_size * dim_model
        dec_input = torch.transpose(dec_input, 0, 1)  # seq_size * batch * dim_model
        # 3. Prepare masks for transformer inference
        # [[3], [2], [5]] -> [[ffftt], [ffttt], [fffff]]
        subseq_mask = self.transformer.generate_square_subsequent_mask(
            self.cfg[self.cfg.ckt_dataset_type].data.max_seq_len
        ).to(enc_input.device)

        # get transformer output
        tr_output = self.transformer(
            src=enc_input,
            tgt=dec_input,
            src_mask=subseq_mask,
            tgt_mask=subseq_mask,
            memory_mask=subseq_mask,
            src_key_padding_mask=batch["pad_mask"],
            tgt_key_padding_mask=batch["pad_mask"],
        )
        tr_output = torch.transpose(tr_output, 0, 1)  # batch * seq_size * dim_model

        return self.generator(tr_output)

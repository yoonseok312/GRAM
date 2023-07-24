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
import wandb
from gram.knowledge_tracing.components.eernn import (
    BidirectionalLSTM,
    TransformerEncoder,
)
from sentence_transformers import models
from gram.knowledge_tracing.components.SBERT import SBERT, CKTAdditiveAttention
from pytorch_lightning.callbacks import ModelCheckpoint
from torchtext.legacy import data
from torchtext.data import get_tokenizer
from torchtext.legacy.data import TabularDataset
import copy
from tqdm import tqdm
import math
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from gram.knowledge_tracing.components.input_embedding import AllItemInputEmbedding
from gram.knowledge_tracing.components.SBERT import SBERT
from gram.utils import utils
from gram.utils.schedulers import get_noam_scheduler

from torch.utils.data import DataLoader, TensorDataset
from gram.knowledge_tracing.components.sbert_regressor import LightningRegressor


class ContentAllItemGenerator(nn.Module):
    def __init__(self, dim_model, num_items, enc_embed, config):
        super(ContentAllItemGenerator, self).__init__()
        self.cfg = config

        if self.cfg.experiment_type != "D" or (
            self.cfg.experiment_type == "D" and self.cfg.D.use_exp_c_kt_module
        ):
            if self.cfg.param_shared:
                self.question_embedding = enc_embed.embed_feature.shifted_item_id.weight
            else:
                self.generator = nn.Linear(dim_model, num_items)
        if self.cfg.add_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.cfg[self.cfg.ckt_dataset_type].data.max_item_id + 1),
                requires_grad=True,
            )

    def forward(self, x, sbert_embed):
        if (
            self.cfg.experiment_type[self.cfg.current_stage] == "D"
            and self.cfg.param_shared
        ):
            output = torch.matmul(x, torch.transpose(sbert_embed, 0, 1))
        elif self.cfg.param_shared:
            output = torch.matmul(x, torch.transpose(self.question_embedding, 0, 1))
        else:
            output = self.generator(x)
        if self.cfg.add_bias:
            output += self.bias
        return output


class ContentAllItemKT(nn.Module):
    """
    Simple Transformer with SBERT for question embedding
    """

    def __init__(self, config) -> None:
        super(ContentAllItemKT, self).__init__()
        self.cfg = config
        self.enc_embed = AllItemInputEmbedding(config, "encoder")
        # self.dec_embed = InputEmbedding(config, "decoder")

        # self.transformer = nn.Transformer(
        #     d_model=self.cfg.train.dim_model,
        #     nhead=self.cfg.train.head_count,
        #     num_encoder_layers=self.cfg.train.encoder_layer_count,
        #     num_decoder_layers=self.cfg.train.decoder_layer_count,
        #     dim_feedforward=self.cfg.train.dim_feedforward,
        #     dropout=self.cfg.train.dropout_rate,
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.train.dim_model,
            nhead=self.cfg.train.head_count,
            dim_feedforward=self.cfg.train.dim_feedforward,
            dropout=self.cfg.train.dropout_rate,
            activation="relu",
        )
        encoder_norm = nn.LayerNorm(self.cfg.train.dim_model)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.train.encoder_layer_count,
            norm=encoder_norm,
        )

        self.generator = ContentAllItemGenerator(
            self.cfg.train.dim_model,
            self.cfg[self.cfg.ckt_dataset_type].data.max_item_id,
            self.enc_embed,
            self.cfg,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_embedding_batch: torch.Tensor,
        embed_all: torch.Tensor,
    ) -> torch.Tensor:
        # enc_input, dec_input = self.enc_embed(batch), self.dec_embed(batch)
        # enc_input, dec_input = map(lambda x: x.transpose(0, 1), (enc_input, dec_input))  # seq_size * batch * dim_model
        enc_input = self.enc_embed(batch, text_embedding_batch)
        enc_input = enc_input.transpose(0, 1)

        # 3. Prepare masks for transformer inference
        # [[3], [2], [5]] -> [[ffftt], [ffttt], [fffff]]
        subseq_mask = utils.generate_square_subsequent_mask(
            self.cfg[self.cfg.ckt_dataset_type].data.max_seq_len
        ).to(enc_input.device)

        # get transformer output
        # tr_output = self.transformer(
        #     src=enc_input,
        #     tgt=dec_input,
        #     src_mask=subseq_mask,
        #     tgt_mask=subseq_mask,
        #     memory_mask=subseq_mask,
        #     src_key_padding_mask=batch["pad_mask"],
        #     tgt_key_padding_mask=batch["pad_mask"],
        # )
        tr_output = self.transformer(
            src=enc_input,
            mask=subseq_mask,
            src_key_padding_mask=batch["pad_mask"],
        )
        tr_output = torch.transpose(tr_output, 0, 1)  # batch * seq_size * dim_model
        output = self.generator(tr_output, embed_all)
        # output = self.generator(tr_output)
        return output


class LightningContentAllItemKT(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()  # stores hyperparameters
        self.cfg = cfg
        self.dropout_seed = 0
        self.threshold = 0.5
        self.base_model = ContentAllItemKT(cfg)
        self.lm_output = []
        self.alternating_interval = cfg.Alternating.alternating_interval
        self.questions_to_regress = []
        self.mode = "kt"
        self.max_auc = 0
        # self.patience = 0
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_until_now = 0
        self.time_until_best_ckpt = 0
        self.non_cold_start_ids = []
        # config = AutoConfig.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.pretrained,
        #                                     output_hidden_states=True)
        # config.attention_probs_dropout_prob = 0
        # config.hidden_dropout_prob = 0
        # breakpoint()
        # bert_model = AutoModel.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.pretrained, config=config)
        # breakpoint()
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg[cfg.experiment_type[cfg.current_stage]].base_lm.pretrained)
        # self.base_lm = KtTextEncoder(cfg, bert_model, self.tokenizer)
        if "D" in [
            cfg.experiment_type["first_stage"],
            cfg.experiment_type["second_stage"],
            cfg.experiment_type["third_stage"],
            cfg.experiment_type["infer_stage"],
        ] or (
            "Alternating"
            in [
                cfg.experiment_type["first_stage"],
                cfg.experiment_type["second_stage"],
                cfg.experiment_type["third_stage"],
                cfg.experiment_type["infer_stage"],
            ]
            and not cfg.Alternating.alternate_by_epoch
        ):
            assert self.cfg.D.base_lm.text_embedding_dim == self.cfg.train.dim_model
            if cfg.D.base_lm.model == "SBERT":
                if cfg[cfg.experiment_type[cfg.current_stage]].base_lm.use_finetuned:
                    self.SBERT = SBERT(cfg.D.base_lm.pretrained)
                else:
                    word_embedding_model = models.Transformer(
                        cfg.D.base_lm.pretrained,
                        max_seq_length=cfg[
                            cfg.experiment_type[cfg.current_stage]
                        ].base_lm.max_seq_len,
                    )
                    if (
                        self.cfg[
                            self.cfg.experiment_type[self.cfg.current_stage]
                        ].base_lm.pooling
                        != "att"
                    ):
                        pooling_model = models.Pooling(
                            word_embedding_model.get_word_embedding_dimension(),
                            pooling_mode=cfg.D.base_lm.pooling,
                        )
                    else:
                        pooling_model = models.Pooling(
                            word_embedding_model.get_word_embedding_dimension(),
                            pooling_mode="mean",
                        )
                    self.SBERT = SBERT(modules=[word_embedding_model, pooling_model])
                self.SBERT.max_seq_length = cfg[
                    cfg.experiment_type[cfg.current_stage]
                ].base_lm.max_seq_len

                if cfg.ckt_dataset_type == "toeic":
                    word_embedding_model = self.SBERT._first_module()
                    tokens = ["[Q]", "[C]", "[P]", "[MASK]"]
                    word_embedding_model.tokenizer.add_tokens(
                        tokens, special_tokens=True
                    )
                    word_embedding_model.auto_model.resize_token_embeddings(
                        len(word_embedding_model.tokenizer)
                    )
                elif cfg.ckt_dataset_type == "poj":
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl",
                        "rb",
                    ) as handle:
                        self.id_to_text = pickle.load(handle)
                if cfg[cfg.experiment_type[cfg.current_stage]].freeze_layers:
                    auto_model = self.SBERT._first_module().auto_model
                    modules = [
                        auto_model.embeddings,
                        *auto_model.encoder.layer[
                            : cfg[
                                cfg.experiment_type[cfg.current_stage]
                            ].freeze_layer_num
                        ],
                    ]  # Replace 5 by what you want
                    print("Freezing LM layers")
                    for module in modules:
                        for param in module.parameters():
                            param.requires_grad = False
                if (
                    self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].base_lm.pooling
                    == "att"
                ):
                    print("Attention based pooling")
                    self.att_pooling = CKTAdditiveAttention(d_h=768)
                # auto_model = self.SBERT._first_module().auto_model
                # auto_model.add_adapter("poj")
                # auto_model.train_adapter("poj")F
                # for param in auto_model.parameters():
                #     param.requires_grad = False
            elif cfg.D.base_lm.model in ["LSTM", "Transformer"]:
                TEXT = data.Field(
                    sequential=True,
                    use_vocab=True,
                    tokenize=get_tokenizer("basic_english"),
                    lower=True,
                    batch_first=True,
                    # fix_length=20
                )
                if cfg.ckt_dataset_type == "toeic":
                    train_data, _ = TabularDataset.splits(
                        path=".",
                        train=f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_text.csv",
                        test=f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_text.csv",
                        format="csv",
                        fields=[("text", TEXT)],
                        skip_header=True,
                    )
                elif cfg.ckt_dataset_type == "poj":
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl",
                        "rb",
                    ) as handle:
                        self.id_to_text = pickle.load(handle)
                    train_data, _ = TabularDataset.splits(
                        path=".",
                        train=f"{cfg[cfg.ckt_dataset_type].data.root}/poj_text_new_split.csv",
                        test=f"{cfg[cfg.ckt_dataset_type].data.root}/poj_text_new_split.csv",
                        format="csv",
                        fields=[("text", TEXT)],
                        skip_header=True,
                    )
                elif cfg[cfg.ckt_dataset_type].language == "french":
                    train_data, _ = TabularDataset.splits(
                        path=".",
                        train=f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_text.csv",
                        test=f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_text.csv",
                        format="csv",
                        fields=[("text", TEXT)],
                        skip_header=True,
                    )
                elif cfg[cfg.ckt_dataset_type].language == "spanish":
                    train_data, _ = TabularDataset.splits(
                        path=".",
                        train=f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_spanish_text_new_split.csv",
                        test=f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_spanish_text_new_split.csv",
                        format="csv",
                        fields=[("text", TEXT)],
                        skip_header=True,
                    )
                MIN_FREQ = 1
                EMBEDDING_DIM = 50
                # vectors = Vectors(name="eng_w2v")
                TEXT.build_vocab(
                    train_data, min_freq=MIN_FREQ, vectors="glove.6B.50d"
                )  # "glove.6B.50d"
                print("Vocab size : {}".format(len(TEXT.vocab)))
                if cfg.D.base_lm.model == "LSTM":
                    self.content_encoder = BidirectionalLSTM(
                        vocab_size=len(TEXT.vocab),
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=384,
                    )
                elif cfg.D.base_lm.model == "Transformer":
                    print("Transformer")
                    self.content_encoder = TransformerEncoder(
                        vocab_size=len(TEXT.vocab),
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=768,
                        feed_forward_dim=128,
                    )
                if not cfg.D.random_weights and not cfg.is_testing:
                    print("Copying weights")
                    self.content_encoder.embedding.weight.data.copy_(TEXT.vocab.vectors)

                tokenizer = get_tokenizer("basic_english")
                self.qid_to_tokenid = {}
                print("seq len", cfg[cfg.ckt_dataset_type].data.lstm_seq_len)
                for key in self.id_to_text.keys():
                    tokenized_id = []
                    tokenized_text = tokenizer(self.id_to_text[key])
                    for token in tokenized_text:
                        tokenized_id.append(TEXT.vocab.stoi[token])
                    assert len(tokenized_text) == len(tokenized_id)
                    seq_len = cfg[cfg.ckt_dataset_type].data.lstm_seq_len
                    if len(tokenized_id) < seq_len:
                        tokenized_id.extend(
                            [1 for _ in range(seq_len - len(tokenized_id))]
                        )  # 20 for duolingo, 512 for toeic
                    else:
                        tokenized_id = tokenized_id[:seq_len]
                    self.qid_to_tokenid[key] = tokenized_id

        self.temp = []
        # if self.cfg.experiment_type[self.cfg.current_stage] == 'Alternating':
        self.automatic_optimization = False
        # self.steps = 0
        # self.lm_steps = 0
        if cfg.ckt_dataset_type == "toeic":
            with open(
                f"{cfg[cfg.ckt_dataset_type].data.root}/cold_start_ids_part_4_to_7.txt",
                "rb",
            ) as f:
                self.cold_start_ids = pickle.load(f)
        elif cfg.ckt_dataset_type == "duolingo":
            # if cfg[cfg.ckt_dataset_type].language == 'spanish':
            #     with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_test_csqe.pkl", "rb") as f:
            #         dic = pickle.load(f)
            #         self.cold_start_ids = list(dic.keys())
            if cfg[cfg.ckt_dataset_type].language == "spanish":
                if cfg[cfg.ckt_dataset_type].user_based_split:
                    print("spanish user based split!!!!!!!")
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_csqs_newsplit.pkl",
                        "rb",
                    ) as f:
                        self.cold_start_ids = pickle.load(f)
                else:
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_test_csqe.pkl",
                        "rb",
                    ) as f:
                        dic = pickle.load(f)
                        self.cold_start_ids = list(dic.keys())
            else:
                if cfg[cfg.ckt_dataset_type].user_based_split:
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe_newsplit_updated.pkl",
                        "rb",
                    ) as f:
                        dic = pickle.load(f)
                        self.cold_start_ids = list(dic.keys())
                else:
                    with open(
                        f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text_test_csqe.pkl",
                        "rb",
                    ) as f:
                        dic = pickle.load(f)
                        self.cold_start_ids = list(dic.keys())
        elif cfg.ckt_dataset_type == "poj":
            with open(
                f"{cfg[cfg.ckt_dataset_type].data.root}/poj_csqs_new_split.pkl", "rb"
            ) as f:
                self.cold_start_ids = pickle.load(f)
        if self.cfg.experiment_type[self.cfg.current_stage] == "Alternating":
            self.lm = LightningRegressor(cfg)
        if cfg.ckt_dataset_type == "toeic":
            if cfg[cfg.experiment_type[cfg.current_stage]].with_passage:
                with open(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text_special_tokens.pkl",
                    "rb",
                ) as handle:
                    self.id_to_text = pickle.load(handle)
            else:
                with open(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text.pkl", "rb"
                ) as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == "duolingo":
            if cfg[cfg.ckt_dataset_type].language == "french":
                with open(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl",
                    "rb",
                ) as handle:
                    self.id_to_text = pickle.load(handle)
            elif cfg[cfg.ckt_dataset_type].language == "spanish":
                with open(
                    f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_all.pkl",
                    "rb",
                ) as handle:
                    self.id_to_text = pickle.load(handle)
        elif cfg.ckt_dataset_type == "poj":
            print("Truncated text")
            with open(
                f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl",
                "rb",
            ) as handle:
                self.id_to_text = pickle.load(handle)

        self.training_kt_module = True
        self.steps_per_epoch = 1
        if (
            self.cfg.experiment_type[self.cfg.current_stage] == "Alternating"
            or self.cfg.experiment_type[self.cfg.current_stage] == "D"
        ):
            self.automatic_optimization = False
        # breakpoint()

    def lm_forward(
        self,
        batch: Dict[str, torch.Tensor],
        increment_seed=False,
        # encoded_all: torch.Tensor,
    ):
        if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.model in [
            "LSTM",
            "Transformer",
        ]:
            # embed_results.pop(0)
            item_ids = {item_id.item() for user in batch["item_id"] for item_id in user}
            item_ids_list = [
                item_id for item_id in self.id_to_text.keys() if item_id in item_ids
            ]
            lstm_input = []
            for item in item_ids_list:
                # padded = self.qid_to_tokenid[item]
                # padded.extend([1 for _ in range(seq_len - len(padded))])
                lstm_input.append(torch.LongTensor(self.qid_to_tokenid[item]))
            lstm_input = torch.stack(lstm_input).to("cuda")
            encoded_valid_ids = self.content_encoder(lstm_input)

        else:
            # if self.cfg.experiment_type == 'C_to_D':
            # embed_results.pop(0)  # Remove shifted_item_id embedding
            nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            item_ids_list = batch["item_id"][nonzeros[0], nonzeros[1]]
            item_ids_list = torch.unique(item_ids_list, sorted=False)
            item_ids_list = [x.item() for x in item_ids_list]
            if (
                self.cfg[
                    self.cfg.experiment_type[self.cfg.current_stage]
                ].base_lm.pooling
                == "att"
            ):
                torch.cuda.manual_seed(self.dropout_seed)
                encoded_valid_ids = self.SBERT.encode(
                    sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                    batch_size=self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].base_lm.batch_size,
                    convert_to_tensor=True,
                    output_value="token_embeddings",
                    reduce_dim=self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].base_lm.text_embedding_dim
                    != 768,
                )
                encoded_valid_ids = [
                    self.att_pooling(q_emb) for q_emb in encoded_valid_ids
                ]
                encoded_valid_ids = torch.stack(encoded_valid_ids)
            else:
                torch.cuda.manual_seed(self.dropout_seed)
                encoded_valid_ids = self.SBERT.encode(
                    sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                    batch_size=self.cfg.D.base_lm.batch_size,
                    convert_to_tensor=True,
                    reduce_dim=self.cfg.D.base_lm.reduce_dim,
                )
            if increment_seed:
                self.dropout_seed += 1
        return encoded_valid_ids, item_ids_list

    def lm_forward_id(
        self,
        item_ids_list,
        increment_seed=False,
        # filter_unique=True,
        # encoded_all: torch.Tensor,
    ):
        if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.model in [
            "LSTM",
            "Transformer",
        ]:
            lstm_input = []
            for item in item_ids_list:
                lstm_input.append(torch.LongTensor(self.qid_to_tokenid[item]))
            lstm_input = torch.stack(lstm_input).to("cuda")
            encoded_valid_ids = self.content_encoder(lstm_input)

        else:
            if (
                self.cfg[
                    self.cfg.experiment_type[self.cfg.current_stage]
                ].base_lm.pooling
                == "att"
            ):
                torch.cuda.manual_seed(self.dropout_seed)
                encoded_valid_ids = self.SBERT.encode(
                    sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                    batch_size=self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].base_lm.batch_size,
                    convert_to_tensor=True,
                    output_value="token_embeddings",
                    reduce_dim=self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].base_lm.text_embedding_dim
                    != 768,
                )
                encoded_valid_ids = [
                    self.att_pooling(q_emb) for q_emb in encoded_valid_ids
                ]
                encoded_valid_ids = torch.stack(encoded_valid_ids)
            else:
                torch.cuda.manual_seed(self.dropout_seed)
                encoded_valid_ids = self.SBERT.encode(
                    sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                    batch_size=self.cfg.D.base_lm.batch_size,
                    convert_to_tensor=True,
                    reduce_dim=self.cfg.D.base_lm.reduce_dim,
                )
            if increment_seed:
                self.dropout_seed += 1
        return encoded_valid_ids, item_ids_list

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # encoded_all: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.cfg.experiment_type[self.cfg.current_stage] == "D":
            encoded_valid_ids, item_ids_list = self.lm_forward(
                batch, increment_seed=True
            )
            # breakpoint()
            encoded_all = []

            j = 0
            # breakpoint()
            for i in range(0, self.cfg[self.cfg.ckt_dataset_type].data.max_item_id):
                if (j == len(item_ids_list) - 1) or (i < item_ids_list[j]):
                    encoded_all.append(
                        torch.zeros(
                            self.cfg.D.base_lm.text_embedding_dim, device="cuda"
                        )
                    )
                else:
                    # breakpoint()
                    encoded_all.append(encoded_valid_ids[j])
                    j += 1
            encoded_all = torch.stack(encoded_all)
            text_embedding_batch = []
            for user in batch["item_id"]:
                text_embedding_user = []
                for item_id in user:
                    if (
                        item_id.item()
                        != self.cfg[self.cfg.ckt_dataset_type].embed.pad_value.item_id
                    ):
                        item_embedding = encoded_all[item_id.item()]
                    else:
                        item_embedding = torch.zeros(
                            self.cfg.D.base_lm.text_embedding_dim, device="cuda"
                        )

                    text_embedding_user.append(item_embedding)
                text_embedding_user = torch.stack(text_embedding_user).to("cuda")
                text_embedding_batch.append(text_embedding_user)
            text_embedding_batch = torch.stack(text_embedding_batch)

        else:
            text_embedding_batch = torch.zeros(1)
            encoded_all = torch.zeros(1)  # just a placeholder
        output = self.base_model.forward(
            batch=batch,
            text_embedding_batch=text_embedding_batch,
            embed_all=encoded_all,
        )
        return output

    def csqe_label_and_prob(self, logits, label, id_list, items):
        csqe_items_idx = torch.cat([(items == i).nonzero() for i in id_list])
        csqe_logit = torch.gather(logits, dim=-1, index=csqe_items_idx.squeeze())
        csqe_label = torch.gather(label, dim=-1, index=csqe_items_idx.squeeze())
        csqe_prob = csqe_logit.sigmoid()
        return csqe_label, csqe_prob

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        logit: Dict[str, torch.Tensor],
        is_training: bool = True,
        is_testing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if is_training:
            nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
            items = batch["item_id"][nonzeros[0], nonzeros[1]]
            logits = logit[nonzeros[0], nonzeros[1]]
            logit = torch.gather(logits, dim=-1, index=items.unsqueeze(-1)).squeeze()
            prob = logit.sigmoid()
            pred = (prob > self.threshold).long()
            loss = F.binary_cross_entropy_with_logits(logit, label, reduction="mean")
            # breakpoint()
            return loss, label, prob, pred
        else:
            if (
                self.cfg.ckt_dataset_type in ["toeic", "poj"]
                or self.cfg[self.cfg.ckt_dataset_type].user_based_split == True
            ):
                nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
                label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
                items = batch["item_id"][nonzeros[0], nonzeros[1]]
                logits = logit[nonzeros[0], nonzeros[1]]
                logit = torch.gather(
                    logits, dim=-1, index=items.unsqueeze(-1)
                ).squeeze()
                prob = logit.sigmoid()
                pred = (prob > self.threshold).long()
                loss = F.binary_cross_entropy_with_logits(
                    logit, label, reduction="mean"
                )
                if is_testing:
                    csqe_label, csqe_prob = self.csqe_label_and_prob(
                        logit, label, self.cold_start_ids, items
                    )
                    # non_csq_ids = [id for id in range(0, 19398) if id not in self.cold_start_ids]
                    # non_csqe_label_last, non_csqe_prob_last = self.csqe_label_and_prob(logit, label, non_csq_ids, items)
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
                else:
                    if self.cfg[self.cfg.ckt_dataset_type].run_test_as_val:
                        csqe_label, csqe_prob = self.csqe_label_and_prob(
                            logit, label, self.cold_start_ids, items
                        )
                    else:
                        csqe_label, csqe_prob = torch.LongTensor(
                            [1, 0]
                        ), torch.LongTensor(
                            [0, 1]
                        )  # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
            else:
                nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
                label = batch["is_correct"][nonzeros[0], nonzeros[1]].float()
                items = batch["item_id"][nonzeros[0], nonzeros[1]]
                seq_sizes = batch["sequence_size"]
                last_seq = []
                for idx, seq_size in enumerate(seq_sizes):
                    last_seq += [
                        sum(seq_sizes[idx][0].item() for idx in range(idx + 1)) - 1
                    ]
                logits = logit[nonzeros[0], nonzeros[1]]
                logit = torch.gather(
                    logits, dim=-1, index=items.unsqueeze(-1)
                ).squeeze()
                last_seq = torch.cuda.LongTensor(last_seq)
                logit_last = torch.gather(logit, dim=-1, index=last_seq)
                label_last = torch.gather(label, dim=-1, index=last_seq)
                label = label_last
                item_last = torch.gather(items, dim=-1, index=last_seq)
                prob = logit_last.sigmoid()
                pred = (prob > self.threshold).long()
                loss = F.binary_cross_entropy_with_logits(
                    logit_last, label_last, reduction="mean"
                )
                if is_testing:
                    csqe_list = self.cold_start_ids
                    csqe_label, csqe_prob = self.csqe_label_and_prob(
                        logit_last, label_last, csqe_list, item_last
                    )
                    # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
                else:
                    if self.cfg[self.cfg.ckt_dataset_type].run_test_as_val:
                        # breakpoint()
                        csqe_label, csqe_prob = self.csqe_label_and_prob(
                            logit_last, label_last, self.cold_start_ids, item_last
                        )
                        # breakpoint()
                    else:
                        csqe_label, csqe_prob = torch.LongTensor(
                            [1, 0]
                        ), torch.LongTensor(
                            [0, 1]
                        )  # to speed up training
                    non_csqe_label_last, non_csqe_prob_last = torch.LongTensor(
                        [1, 0]
                    ), torch.LongTensor(
                        [0, 1]
                    )  # to speed up training
            return (
                loss,
                label,
                prob,
                pred,
                csqe_label,
                csqe_prob,
                non_csqe_label_last,
                non_csqe_prob_last,
            )

    def on_train_epoch_end(self) -> None:
        # breakpoint()
        self.training_kt_module = False

    def training_step_helper(self, batch):
        # breakpoint()
        logit = self(batch)
        loss, _, _, _ = self.compute_loss(batch, logit)

        # # manually increase the step size
        # self.scheduler.step_batch()

        self.log("train_loss", loss)
        # wandb.log(
        #     {"train_loss": loss}
        # )

        return loss

    def on_train_start(self) -> None:
        if self.cfg.Alternating.alternating_epoch_proportion:
            self.alternating_interval = int(
                (len(self.train_dataloader.dataloader))
                * self.cfg.Alternating.alternating_epoch_proportion
            )

    # multistep support
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:  # Need to add , optimizer_idx: int for alternating
        # self.temp = []
        nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
        item_ids_list = batch["item_id"][nonzeros[0], nonzeros[1]]
        item_ids = torch.unique(item_ids_list, sorted=False)
        items = pd.DataFrame(item_ids.cpu().detach().numpy()).astype("int32")
        self.non_cold_start_ids += items[~items[0].isin(self.non_cold_start_ids)][
            0
        ].tolist()
        item_ids = item_ids.tolist()
        num_gpus = len(str(self.cfg.gpus).split(","))
        distributed_length = math.ceil(len(self.train_dataloader.dataloader) / num_gpus)
        if self.cfg.experiment_type[self.cfg.current_stage] != "Alternating":
            kt_opt = self.optimizers()
            kt_sch = self.lr_schedulers()
            loss = self.training_step_helper(batch)
            kt_opt.zero_grad()
            self.manual_backward(loss)
            kt_opt.step()
            kt_sch.step()
            # self.dropout_seed += 1
        # else, always do stepwise alternating.
        else:
            if self.mode == "kt":
                kt_opt, lm_opt, emb_opt = self.optimizers()
                if self.cfg.Alternating.use_sch:
                    kt_sch, lm_sch = self.lr_schedulers()
                elif self.cfg.Alternating.use_lm_sch:
                    kt_sch, lm_sch = self.lr_schedulers()
                else:
                    kt_sch = self.lr_schedulers()
                # if optimizer_idx == 0: # 100 step per epoch
                # self.lm_output.append(lm_output)

                # for time efficiency
                if self.cfg.Alternating.alternating_epoch_proportion is not None:
                    if batch_idx % self.alternating_interval == 0:
                        for i in range(
                            int(
                                len(self.non_cold_start_ids)
                                / self.cfg[self.cfg.ckt_dataset_type].num_q_split
                            )
                            + 1
                        ):
                            if (
                                len(self.non_cold_start_ids)
                                % self.cfg[self.cfg.ckt_dataset_type].num_q_split
                                == 0
                            ):
                                if i == int(
                                    len(self.non_cold_start_ids)
                                    / self.cfg[self.cfg.ckt_dataset_type].num_q_split
                                ):
                                    continue
                            self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                                torch.tensor(
                                    self.non_cold_start_ids[
                                        i
                                        * self.cfg[
                                            self.cfg.ckt_dataset_type
                                        ].num_q_split : (i + 1)
                                        * self.cfg[
                                            self.cfg.ckt_dataset_type
                                        ].num_q_split
                                    ]
                                )
                            ] = (
                                self.SBERT.encode(
                                    sentences=[
                                        self.id_to_text[item_id]
                                        for item_id in self.non_cold_start_ids[
                                            i
                                            * self.cfg[
                                                self.cfg.ckt_dataset_type
                                            ].num_q_split : (i + 1)
                                            * self.cfg[
                                                self.cfg.ckt_dataset_type
                                            ].num_q_split
                                        ]
                                    ],
                                    batch_size=self.cfg[
                                        self.cfg.experiment_type[self.cfg.current_stage]
                                    ].base_lm.batch_size,
                                    convert_to_tensor=True,
                                )
                                .clone()
                                .detach()
                                .requires_grad_(True)
                            )
                else:
                    # lm_output, item_ids_list = self.lm_forward(batch)
                    change_ids = [
                        item
                        for item in item_ids
                        if item not in self.questions_to_regress
                    ]
                    # change_idx = [item_ids_list.index(item) for item in change_ids]
                    lm_output, item_ids_list = self.lm_forward_id(change_ids)
                    self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                        torch.tensor(item_ids_list).type(torch.cuda.LongTensor)
                    ] = (lm_output.clone().detach().requires_grad_(True))
                    if self.alternating_interval != 1:
                        del lm_output
                    #     if len(self.lm_output) == 0:
                    #         self.lm_output = lm_output
                    #     else:
                    #         self.lm_output = torch.cat([self.lm_output, lm_output], dim=0)
                self.questions_to_regress += [
                    item_id
                    for item_id in item_ids
                    if item_id not in self.questions_to_regress
                ]

                loss = self.training_step_helper(batch)
                kt_opt.zero_grad()
                emb_opt.zero_grad()
                self.manual_backward(loss)
                kt_opt.step()
                emb_opt.step()
                kt_sch.step()
                self.log("train_kt_loss", loss)
                if (
                    (self.global_step + 1) % self.alternating_interval == 0
                    or distributed_length - 1 == batch_idx
                ):
                    self.mode = "lm"
            if self.mode == "lm":
                items_tensor = torch.tensor(self.questions_to_regress).cuda()
                labels = torch.gather(
                    self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data,
                    dim=0,
                    index=items_tensor.unsqueeze(-1).repeat(1, 768),
                )
                # self.alternating_interval = int(len(self.train_dataloader.dataloader) - 1)
                if self.cfg.Alternating.regressor_batch_size is not None:
                    lm_batch_size = self.cfg.Alternating.regressor_batch_size
                else:
                    lm_batch_size = int(
                        len(self.questions_to_regress) / self.alternating_interval
                    )
                    if lm_batch_size == 0:
                        lm_batch_size = 1
                if self.alternating_interval == 1:
                    for i in range(self.cfg.Alternating.lm_epochs):
                        loss_lm = self.compute_lm_loss(
                            labels, lm_output
                        )  # output = tensor, id list
                        lm_opt.zero_grad()
                        self.manual_backward(loss_lm)
                        lm_opt.step()
                        if self.cfg.Alternating.use_sch:
                            lm_sch.step()
                        elif self.cfg.Alternating.use_lm_sch:
                            lm_sch.step()
                        self.log("train_lm_loss", loss_lm)
                        # wandb.log(
                        #     {"train_lm_loss": loss_lm}
                        # )
                else:
                    train_loader = DataLoader(
                        TensorDataset(
                            torch.tensor(
                                np.asarray(self.questions_to_regress), dtype=torch.float
                            ).cuda(),
                            labels,
                        ),
                        # batch_size=self.cfg.Alternating.regressor_batch_size,
                        batch_size=lm_batch_size,
                        shuffle=False,
                    )
                    if not self.cfg.Alternating.lm_single_step:
                        for i in range(self.cfg.Alternating.lm_epochs):
                            for count, (ids, batch_labels) in tqdm(
                                enumerate(train_loader)
                            ):
                                stacked = self.lm_forward_id(ids.tolist())[0]
                                loss_lm = self.compute_lm_loss(
                                    batch_labels, stacked
                                )  # output = tensor, id list
                                lm_opt.zero_grad()
                                self.manual_backward(loss_lm)
                                lm_opt.step()
                                if self.cfg.Alternating.use_sch:
                                    lm_sch.step()
                                elif self.cfg.Alternating.use_lm_sch:
                                    lm_sch.step()
                                self.log("train_lm_loss", loss_lm)
                                # wandb.log(
                                #     {"train_lm_loss": loss_lm}
                                # )
                    else:
                        for i in range(self.cfg.Alternating.lm_epochs):
                            output = []
                            for count, (ids, batch_labels) in tqdm(
                                enumerate(train_loader)
                            ):
                                output.append(self.lm_forward_id(ids.tolist())[0])
                                if len(output) > 1:
                                    stacked_until_last = torch.stack(
                                        output[:-1]
                                    ).reshape(-1, 768)
                                    stacked = torch.cat(
                                        (stacked_until_last, output[-1])
                                    )
                                else:
                                    stacked = torch.stack(output).reshape(-1, 768)
                            # stacked = torch.stack(output)
                            loss_lm = self.compute_lm_loss(
                                labels, stacked
                            )  # output = tensor, id list
                            print("lm loss", loss_lm)
                            lm_opt.zero_grad()
                            self.manual_backward(loss_lm)
                            lm_opt.step()
                            if self.cfg.Alternating.use_sch:
                                lm_sch.step()
                            elif self.cfg.Alternating.use_lm_sch:
                                lm_sch.step()

                            wandb.log({"train_lm_loss": loss_lm})
                self.mode = "kt"
                self.questions_to_regress = []
                self.dropout_seed += 1
                self.lm_output = []

        if distributed_length - 1 == batch_idx:
            self.eval()
            effective_csqs = [
                i for i in self.id_to_text.keys() if i not in self.non_cold_start_ids
            ]
            if self.cfg.experiment_type[self.cfg.current_stage] == "Alternating":
                # for item in self.cold_start_ids:
                self.base_model.enc_embed.embed_feature.shifted_item_id.weight.data[
                    torch.tensor(effective_csqs)
                ] = (
                    self.SBERT.encode(
                        sentences=[
                            self.id_to_text[item_id] for item_id in effective_csqs
                        ],
                        batch_size=self.cfg[
                            self.cfg.experiment_type[self.cfg.current_stage]
                        ].base_lm.batch_size,
                        convert_to_tensor=True,
                    )
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
            self.train()
            self.non_cold_start_ids = []

    def compute_lm_loss(
        self,
        label: torch.Tensor,
        logit: torch.Tensor,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        # breakpoint()
        loss = (
            F.mse_loss(logit, label, reduction=self.cfg.Alternating.lm_loss_reduction)
            * 0.5
        )
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        single validation step for pl.Trainer
        return losses, labels, predictions
        """
        logit = self(batch)
        loss, labels, probs, preds, csqe_labels, csqe_probs, _, _ = self.compute_loss(
            batch, logit, is_training=False
        )
        return {
            "loss": loss.item(),
            "labels": labels.data.cpu().tolist(),
            "probs": probs.data.cpu().tolist(),
            "preds": preds.data.cpu().tolist(),
            "csqe_labels": csqe_labels.data.cpu().tolist(),
            "csqe_probs": csqe_probs.data.cpu().tolist(),
        }

    def validation_epoch_end(self, outputs):
        """
        single validation epoch for pl.Trainer
        """
        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])
        csqe_labels = np.hstack([o["csqe_labels"] for o in outputs])
        csqe_probs = np.hstack([o["csqe_probs"] for o in outputs])

        # print(loss.shape, labels.shape, probs.shape, preds.shape)

        acc = (preds == labels).sum().item() / len(preds)
        auc = metrics.roc_auc_score(labels, probs)
        try:
            csqe_auc = metrics.roc_auc_score(csqe_labels, csqe_probs)
            # auc_list.append(auc)
        except ValueError:
            csqe_auc = 0
            print("labels", labels)
            print("probs", probs)
            pass
        # auc = metrics.roc_auc_score(labels, probs)
        loss = np.mean(loss)  # .mean().item()

        log = {
            "val_acc": acc,
            "val_loss": loss,
            "val_auc": auc,
            "val_csq_auc": csqe_auc,
        }
        print(log)
        self.log_dict(log)

        return log

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        # breakpoint()
        logit = self(batch)
        # loss, labels, probs, preds, csqe_labels, csqe_preds = self.compute_loss(batch, logit)
        (
            loss,
            labels,
            probs,
            preds,
            csqe_labels,
            csqe_probs,
            non_csqe_labels,
            non_csqe_probs,
        ) = self.compute_loss(batch, logit, is_training=False, is_testing=True)
        return {
            "loss": loss.item(),
            "labels": labels.data.cpu().tolist(),
            "probs": probs.data.cpu().tolist(),
            "preds": preds.data.cpu().tolist(),
            "csqe_labels": csqe_labels.data.cpu().tolist(),
            "csqe_probs": csqe_probs.data.cpu().tolist(),
            "non_csqe_labels": non_csqe_labels.data.cpu().tolist(),
            "non_csqe_probs": non_csqe_probs.data.cpu().tolist(),
        }

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        loss = np.array([o["loss"] for o in outputs])
        labels = np.hstack([o["labels"] for o in outputs])
        probs = np.hstack([o["probs"] for o in outputs])
        preds = np.hstack([o["preds"] for o in outputs])
        csqe_labels = np.hstack([o["csqe_labels"] for o in outputs])
        csqe_probs = np.hstack([o["csqe_probs"] for o in outputs])
        non_csqe_labels = np.hstack([o["non_csqe_labels"] for o in outputs])
        non_csqe_probs = np.hstack([o["non_csqe_probs"] for o in outputs])

        acc = (preds == labels).sum().item() / len(preds)
        auc = metrics.roc_auc_score(labels, probs)
        loss = loss.mean().item()

        print("num of CSQ interactions", len(csqe_labels))
        # print("num of non CSQ interactions", len(non_csqe_labels))
        print("num of total interactions", len(labels))
        csq_auc = []
        for i in range(10000):
            np.random.seed(self.cfg.seed)
            bootstrap_idx = np.random.choice(range(len(csqe_labels)), len(csqe_labels))
            fpr, tpr, thresholds = metrics.roc_curve(
                csqe_labels[bootstrap_idx], csqe_probs[bootstrap_idx]
            )
            csq_auc.append(metrics.auc(fpr, tpr))
        csqe_auc = np.mean(csq_auc)
        csqe_auc_std = np.std(csq_auc)
        # csqe_auc = metrics.roc_auc_score(csqe_labels, csqe_probs)
        non_csqe_auc = metrics.roc_auc_score(non_csqe_labels, non_csqe_probs)

        log = {
            "test_acc": acc,
            "test_loss": loss,
            "test_auc": auc,
            "test_csqe_auc": csqe_auc,
            "test_non_csqe_auc": non_csqe_auc,
            "test_csqe_auc_std": csqe_auc_std,
        }
        print(log)
        self.log_dict(log)
        wandb.log(
            {
                "test_acc": acc,
                "test_loss": loss,
                "test_auc": auc,
                "test_csqe_auc": csqe_auc,
                "test_csqe_auc_std": csqe_auc_std,
                "test_non_csqe_auc": non_csqe_auc,
            }
        )

        return log

    def configure_optimizers(self):
        if self.cfg.experiment_type[self.cfg.current_stage] != "Alternating":
            if self.cfg.experiment_type[self.cfg.current_stage] == "D":
                if self.cfg.D.use_sgd:
                    self.kt_optimizer = torch.optim.SGD(
                        [
                            {"params": self.base_model.parameters()},
                            {"params": self.SBERT.parameters(), "lr": self.cfg.D.lmlr},
                        ],
                        lr=self.cfg.D.ktlr,
                        momentum=0.9,
                    )
                else:
                    self.kt_optimizer = torch.optim.Adam(
                        [
                            {"params": self.base_model.parameters()},
                            {"params": self.SBERT.parameters(), "lr": self.cfg.D.lmlr},
                        ],
                        lr=self.cfg.D.ktlr,
                    )
                self.kt_scheduler = get_noam_scheduler(
                    self.kt_optimizer,
                    warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
                    only_warmup=False,
                    interval="step",
                )
                return {
                    "optimizer": self.kt_optimizer,
                    "lr_scheduler": self.kt_scheduler,
                }
            else:
                self.kt_optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr,
                )
                self.kt_scheduler = get_noam_scheduler(
                    self.kt_optimizer,
                    warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
                    only_warmup=False,
                    interval="step",
                )
                return {
                    "optimizer": self.kt_optimizer,
                    "lr_scheduler": self.kt_scheduler,
                }
            # return [self.kt_optimizer, None], [self.kt_scheduler, None]
            # return [self.kt_optimizer], [self.kt_scheduler]
        else:
            if (
                self.cfg.experiment_type.first_stage == "Alternating"
                and self.cfg.Alternating.use_sch
            ):
                self.lm_scheduler = get_noam_scheduler(
                    self.lm_optimizer,
                    warmup_steps=self.cfg[
                        self.cfg.experiment_type[self.cfg.current_stage]
                    ].lm_warmup,
                    only_warmup=False,
                    interval="step",
                )
                return (
                    {"optimizer": self.kt_optimizer, "lr_scheduler": self.kt_scheduler},
                    {"optimizer": self.lm_optimizer, "lr_scheduler": self.lm_scheduler},
                )
            if (
                self.cfg.experiment_type.first_stage == "Alternating"
                and self.cfg.Alternating.use_sgd
            ):
                self.kt_optimizer = torch.optim.SGD(
                    self.base_model.parameters(),
                    lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr,
                    momentum=0.9,
                    nesterov=False,
                )
                self.lm_optimizer = torch.optim.SGD(
                    self.SBERT.parameters(),
                    lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr,
                    momentum=0.9,
                    nesterov=False,
                )
            else:
                my_list = ["enc_embed.embed_feature.shifted_item_id.weight"]
                # params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
                base_kt_params = list(
                    filter(
                        lambda kv: kv[0] not in my_list,
                        self.base_model.named_parameters(),
                    )
                )
                # weights = []
                # for item in base_kt_params:
                #     weights += ['']
                weights = [{"params": pg[1]} for pg in base_kt_params]
                self.kt_optimizer = torch.optim.Adam(
                    weights,
                    lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr,
                    # eps=1e-1,
                    # betas=(1, 1)
                )
                self.emb_optimizer = torch.optim.SGD(
                    self.base_model.enc_embed.embed_feature.shifted_item_id.parameters(),
                    lr=1,
                    # eps=1e-1,
                    # betas=(1, 1)
                )
                # self.emb_optimizer = torch.optim.Adam(self.base_model.enc_embed.embed_feature.shifted_item_id.parameters(),
                #                                      lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lr,
                #                                      # eps=1e-1,
                #                                      # betas=(1, 1)
                #                                      )
                self.lm_optimizer = torch.optim.Adam(
                    self.SBERT.parameters(),
                    lr=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].lm_lr,
                )
            self.kt_scheduler = get_noam_scheduler(
                self.kt_optimizer,
                warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
                only_warmup=False,
                interval="step",
            )
            print("LM optimizer!!!!!")
            self.lm_scheduler = get_noam_scheduler(
                self.lm_optimizer,
                warmup_steps=self.cfg[self.cfg.ckt_dataset_type].warmup,
                only_warmup=False,
                interval="step",
            )
            # return [self.kt_optimizer, self.lm_optimizer], [self.kt_scheduler, self.lm_scheduler]
            print("LM scheduler!!!!")
            if self.cfg.Alternating.use_lm_sch:
                return (
                    {"optimizer": self.kt_optimizer, "lr_scheduler": self.kt_scheduler},
                    {"optimizer": self.lm_optimizer, "lr_scheduler": self.lm_scheduler},
                    {"optimizer": self.emb_optimizer},
                )
            else:
                return (
                    {"optimizer": self.kt_optimizer, "lr_scheduler": self.kt_scheduler},
                    {"optimizer": self.lm_optimizer},
                    {"optimizer": self.emb_optimizer},
                )

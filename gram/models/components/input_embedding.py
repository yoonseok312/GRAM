from typing import Dict

import torch
import torch.nn as nn
import omegaconf
import pickle
from repoc_content_kt.models.components.SBERT import SBERT, CKTAdditiveAttention
from repoc_content_kt.models.components.eernn import BidirectionalLSTM, TransformerEncoder
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
        # self.enc_input_mapper = nn.Linear(768, cfg[cfg.experiment_type[cfg.current_stage]].base_lm.text_embedding_dim)
        if 'D' in [cfg.experiment_type['first_stage'] , cfg.experiment_type['second_stage'], cfg.experiment_type['third_stage'], cfg.experiment_type['infer_stage']]:
            assert self.cfg.D.base_lm.text_embedding_dim == self.cfg.train.dim_model
            if cfg.D.base_lm.model == 'SBERT':
                if cfg[cfg.experiment_type[cfg.current_stage]].base_lm.use_finetuned:
                    self.SBERT = SBERT(cfg.D.base_lm.pretrained)
                else:
                    word_embedding_model = models.Transformer(cfg.D.base_lm.pretrained, max_seq_length=cfg[cfg.experiment_type[cfg.current_stage]].base_lm.max_seq_len)
                    if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.pooling != 'att':
                        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=cfg.D.base_lm.pooling)
                    else:
                        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                                       pooling_mode='mean')
                    self.SBERT = SBERT(modules=[word_embedding_model, pooling_model])
                self.SBERT.max_seq_length = cfg[cfg.experiment_type[cfg.current_stage]].base_lm.max_seq_len
                if cfg.ckt_dataset_type == 'toeic':
                    word_embedding_model = self.SBERT._first_module()
                    print("Adding special tokens")
                    tokens = ["[Q]", "[C]", "[P]", "[MASK]"]
                    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
                    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
                elif cfg.ckt_dataset_type == 'poj':
                    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl", "rb") as handle:
                        self.id_to_text = pickle.load(handle)
                if cfg[cfg.experiment_type[cfg.current_stage]].freeze_layers:
                    auto_model = self.SBERT._first_module().auto_model
                    modules = [auto_model.embeddings, *auto_model.encoder.layer[:cfg[cfg.experiment_type[cfg.current_stage]].freeze_layer_num]]  # Replace 5 by what you want
                    print("Freezing LM layers")
                    for module in modules:
                        for param in module.parameters():
                            param.requires_grad = False
                if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.pooling == 'att':
                    print("Attention based pooling")
                    self.att_pooling = CKTAdditiveAttention(d_h=768)
                # auto_model = self.SBERT._first_module().auto_model
                # auto_model.add_adapter("poj")
                # auto_model.train_adapter("poj")F
                # for param in auto_model.parameters():
                #     param.requires_grad = False
            elif cfg.D.base_lm.model in ['LSTM', 'Transformer']:
                TEXT = data.Field(sequential=True,
                                  use_vocab=True,
                                  tokenize=get_tokenizer("basic_english"),
                                  lower=True,
                                  batch_first=True,
                                  # fix_length=20
                                  )
                if cfg.ckt_dataset_type == 'toeic':
                    train_data, _ = TabularDataset.splits(
                        path='.', train=f'{cfg[cfg.ckt_dataset_type].data.root}/toeic_text.csv',
                        test=f'{cfg[cfg.ckt_dataset_type].data.root}/toeic_text.csv', format='csv',
                        fields=[('text', TEXT)], skip_header=True)
                elif cfg.ckt_dataset_type == 'poj':
                    with open(f"{cfg[cfg.ckt_dataset_type].data.root}/poj_id_to_text_with_scraped_truncated.pkl", "rb") as handle:
                        self.id_to_text = pickle.load(handle)
                    train_data, _ = TabularDataset.splits(
                        path='.', train=f'{cfg[cfg.ckt_dataset_type].data.root}/poj_text_new_split.csv',
                        test=f'{cfg[cfg.ckt_dataset_type].data.root}/poj_text_new_split.csv', format='csv',
                        fields=[('text', TEXT)], skip_header=True)
                elif cfg[cfg.ckt_dataset_type].language == 'french':
                    train_data, _ = TabularDataset.splits(
                        path='.', train=f'{cfg[cfg.ckt_dataset_type].data.root}/duolingo_text.csv', test=f'{cfg[cfg.ckt_dataset_type].data.root}/duolingo_text.csv', format='csv',
                        fields=[('text', TEXT)], skip_header=True)
                elif cfg[cfg.ckt_dataset_type].language == 'spanish':
                    train_data, _ = TabularDataset.splits(
                        path='.', train=f'{cfg[cfg.ckt_dataset_type].data.root}/duolingo_spanish_text.csv',
                        test=f'{cfg[cfg.ckt_dataset_type].data.root}/duolingo_spanish_text.csv', format='csv',
                        fields=[('text', TEXT)], skip_header=True)
                MIN_FREQ = 1
                EMBEDDING_DIM = 50
                # vectors = Vectors(name="eng_w2v")
                TEXT.build_vocab(train_data, min_freq=MIN_FREQ, vectors='glove.6B.50d') # "glove.6B.50d"
                print('Vocab size : {}'.format(len(TEXT.vocab)))
                if cfg.D.base_lm.model == "LSTM":
                    self.content_encoder = BidirectionalLSTM(vocab_size=len(TEXT.vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=384)
                elif cfg.D.base_lm.model == "Transformer":
                    print("Transformer")
                    self.content_encoder = TransformerEncoder(vocab_size=len(TEXT.vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=768,
                                                    feed_forward_dim=128)
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
                        tokenized_id.extend([1 for _ in range(seq_len - len(tokenized_id))]) # 20 for duolingo, 512 for toeic
                    else:
                        tokenized_id = tokenized_id[:seq_len]
                    self.qid_to_tokenid[key] = tokenized_id

                # with open(
                #         f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_qid_to_tokenid.pkl",
                #         "wb") as handle:
                #     pickle.dump(self.qid_to_tokenid, handle)
            # FIXME: Add other LMs as well
        # if cfg.ckt_dataset_type == 'toeic':
        #     with open(f"{cfg[cfg.ckt_dataset_type].data.root}/toeic_id_to_text_with_passage.pkl", "rb") as handle:
        #         self.id_to_text = pickle.load(handle)
        # elif cfg.ckt_dataset_type == 'duolingo':
        #     if cfg[cfg.ckt_dataset_type].language == 'spanish':
        #         with open(f"{cfg[cfg.ckt_dataset_type].data.root}/spanish_item_id_to_text_all.pkl", "rb") as handle:
        #             self.id_to_text = pickle.load(handle)
        #     else:
        #         with open(f"{cfg[cfg.ckt_dataset_type].data.root}/duolingo_item_id_to_text.pkl", "rb") as handle:
        #             self.id_to_text = pickle.load(handle)
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
        # encoded_all: torch.Tensor
    ) -> torch.Tensor:
        # get embedding result for each feature
        embed_results = [
            self.embed_feature[name](tensor)
            for name, tensor in batch.items()
            if name in self.embed_feature
        ]

        if self.cfg.experiment_type[self.cfg.current_stage] == 'D':
            if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.model in ['LSTM', 'Transformer']:
                embed_results.pop(0)
                item_ids = {item_id.item() for user in batch["item_id"] for item_id in user}
                item_ids_list = [item_id for item_id in self.id_to_text.keys() if item_id in item_ids]
                lstm_input = []
                for item in item_ids_list:
                    # padded = self.qid_to_tokenid[item]
                    # padded.extend([1 for _ in range(seq_len - len(padded))])
                    lstm_input.append(torch.LongTensor(self.qid_to_tokenid[item]))
                lstm_input = torch.stack(lstm_input).to("cuda")
                encoded_valid_ids = self.content_encoder(lstm_input)
                # if 0 in item_ids_list:
                #     print(np.sum(encoded_valid_ids[0].cpu().detach().numpy()))
                # breakpoint()
            else:
                # if self.cfg.experiment_type == 'C_to_D':
                embed_results.pop(0)  # Remove shifted_item_id embedding
                item_ids = {item_id.item() for user in batch["item_id"] for item_id in user}
                item_ids_list = [item_id for item_id in self.id_to_text.keys() if item_id in item_ids]
                if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.pooling == 'att':
                    encoded_valid_ids = self.SBERT.encode(
                        sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                        batch_size=self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].base_lm.batch_size,
                        convert_to_tensor=True,
                        output_value='token_embeddings',
                        reduce_dim=self.cfg[
                                       self.cfg.experiment_type[
                                           self.cfg.current_stage]].base_lm.text_embedding_dim != 768,
                    )
                    encoded_valid_ids = [self.att_pooling(q_emb) for q_emb in encoded_valid_ids]
                    encoded_valid_ids = torch.stack(encoded_valid_ids)
                else:
                    encoded_valid_ids = self.SBERT.encode(
                        sentences=[self.id_to_text[item_id] for item_id in item_ids_list],
                        batch_size=self.cfg.D.base_lm.batch_size,
                        convert_to_tensor=True,
                        reduce_dim=self.cfg.D.base_lm.reduce_dim,
                    )

                # encoded_valid_ids = self.enc_input_mapper(encoded_valid_ids)
            encoded_all = []
            for i in range(0, self.cfg[self.cfg.ckt_dataset_type].data.max_item_id):
                if i in item_ids_list:
                    encoded_all.append(encoded_valid_ids[item_ids_list.index(i)])
                else:
                    encoded_all.append(
                        torch.zeros(self.cfg.D.base_lm.text_embedding_dim, device="cuda")
                    )

            # optimization
            #     nonzeros = torch.nonzero(~batch["pad_mask"], as_tuple=True)
            #     item_ids_list = batch["item_id"][nonzeros[0], nonzeros[1]]
            #     item_ids_list = torch.unique(item_ids_list, sorted=True)
            #     # item_ids = {item_id.item() for user in batch["item_id"] for item_id in user}
            #     # item_ids_list = [item.item() for item in items]
            #     item_ids_list = item_ids_list.tolist()
            #     encoded_valid_ids = self.SBERT.encode(
            #         sentences=[self.id_to_text[item] for item in item_ids_list],
            #         batch_size=self.cfg.D.base_lm.batch_size,
            #         convert_to_tensor=True,
            #         reduce_dim=self.cfg.D.base_lm.reduce_dim,
            #     )
            # # encoded_valid_ids = self.enc_input_mapper(encoded_valid_ids)
            # encoded_all = []
            # j = 0
            # # breakpoint()
            # for i in range(0, self.cfg[self.cfg.ckt_dataset_type].data.max_item_id):
            #     if (j == len(item_ids_list) - 1) or (i < item_ids_list[j]):
            #         encoded_all.append(
            #             torch.zeros(self.cfg.D.base_lm.text_embedding_dim, device="cuda")
            #         )
            #     else:
            #         # breakpoint()
            #         encoded_all.append(encoded_valid_ids[j])
            #         j += 1
            #     # if i in item_ids_list:
            #     #     encoded_all.append(encoded_valid_ids[item_ids_list.index(i)])
            #     # else:
            #     #     encoded_all.append(
            #     #         torch.zeros(self.cfg.D.base_lm.text_embedding_dim, device="cuda")
            #     #     )
            encoded_all = torch.stack(encoded_all)
            text_embedding_batch = []
            for user in batch["item_id"]:
                text_embedding_user = []
                for item_id in user:
                    if item_id.item() != self.cfg[self.cfg.ckt_dataset_type].embed.pad_value.item_id:
                        item_embedding = encoded_all[item_id.item()]
                    else:
                        item_embedding = torch.zeros(
                            self.cfg.D.base_lm.text_embedding_dim, device="cuda"
                        )

                    text_embedding_user.append(item_embedding)
                text_embedding_user = torch.stack(text_embedding_user).to("cuda")
                text_embedding_batch.append(text_embedding_user)
            text_embedding_batch = torch.stack(text_embedding_batch)

            embed_results.insert(0, text_embedding_batch)
            # breakpoint()
        else:
            encoded_all = torch.zeros(1) # just a placeholder

        # add positional embedding
        batch_dim = embed_results[0].shape[0]

        if self.positional is not None:
            positional_embed = self.positional.unsqueeze(0).repeat(batch_dim, 1, 1)
            embed_results.append(positional_embed)

        # aggregate embeddings
        all_embeds = torch.cat(embed_results, dim=-1)
        return self.aggregate_embeds(all_embeds), encoded_all


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
        torch.autograd.set_detect_anomaly(True)
        batch_dim = embed_results[0].shape[0]
        if self.positional is not None:
            positional_embed = self.positional.unsqueeze(0).repeat(batch_dim, 1, 1)
            embed_results.append(positional_embed)

        all_embeds = torch.cat(embed_results, dim=-1)
        return self.aggregate_embeds(all_embeds)

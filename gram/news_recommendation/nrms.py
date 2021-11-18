import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from repoc_content_kt.news_recommendation.model_bert import AdditiveAttention, MultiHeadAttention



class NewsEncoder(nn.Module):
    def __init__(self, cfg, embedding_matrix):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = cfg[cfg.experiment_type[cfg.current_stage]].drop_rate
        self.news_dim = cfg[cfg.experiment_type[cfg.current_stage]].news_dim
        self.dim_per_head = cfg[cfg.experiment_type[cfg.current_stage]].news_dim // cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads
        assert cfg[cfg.experiment_type[cfg.current_stage]].news_dim == cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadAttention(
            cfg[cfg.experiment_type[cfg.current_stage]].word_embedding_dim,
            cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.attn = AdditiveAttention(cfg[cfg.experiment_type[cfg.current_stage]].news_dim, cfg[cfg.experiment_type[cfg.current_stage]].news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        word_vecs = F.dropout(self.embedding_matrix(x.long()),
                              p=self.drop_rate,
                              training=self.training)
        multihead_text_vecs = self.multi_head_self_attn(word_vecs, word_vecs, word_vecs, mask)
        multihead_text_vecs = F.dropout(multihead_text_vecs,
                                        p=self.drop_rate,
                                        training=self.training)
        news_vec = self.attn(multihead_text_vecs, mask)
        return news_vec


class UserEncoder(nn.Module):
    def __init__(self, cfg):
        super(UserEncoder, self).__init__()
        self.cfg = cfg
        self.dim_per_head = cfg[cfg.experiment_type[cfg.current_stage]].news_dim // cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads
        assert cfg[cfg.experiment_type[cfg.current_stage]].news_dim == cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadAttention(cfg[cfg.experiment_type[cfg.current_stage]].news_dim, cfg[cfg.experiment_type[cfg.current_stage]].num_attention_heads, self.dim_per_head, self.dim_per_head)
        self.attn = AdditiveAttention(cfg[cfg.experiment_type[cfg.current_stage]].news_dim, cfg[cfg.experiment_type[cfg.current_stage]].user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, cfg[cfg.experiment_type[cfg.current_stage]].news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_mask:
            # if False in np.asarray(news_vecs.cpu() == news_vecs.cpu()):
            #     print("LOSS NAN")
            #     breakpoint()
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask) + 1e-10
            # if False in np.asarray(news_vecs.cpu() == news_vecs.cpu()):
            #     print("LOSS NAN")
            #     breakpoint()
            user_vec = self.attn(news_vecs, log_mask) + 1e-10
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length, -1) + 1e-10
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1)) + 1e-10
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs) + 1e-10
            user_vec = self.attn(news_vecs) + 1e-10
        # if False in np.asarray(user_vec.cpu() == user_vec.cpu()):
        #     print("LOSS NAN")
        #     breakpoint()
        return user_vec

class AlternatingUserEncoder(UserEncoder):
    def __init__(self, cfg, news_index):
        super().__init__(cfg)
        self.news_embedding = nn.Embedding(max(news_index.values()) + 1, self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].news_dim, padding_idx=0).requires_grad_(True)
import logging
from typing import List, Iterable, Union, Optional
import numpy as np
from numpy import ndarray
import torch
from torch import nn, Tensor
from tqdm.autonotebook import trange
from sentence_transformers import SentenceTransformer

from sentence_transformers.util import (
    batch_to_device,
)

logger = logging.getLogger(__name__)
linear = nn.Linear(768, 32)

class CKTAdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg:
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=768):
        super(CKTAdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.mm(x.transpose(0,1), alpha)
        # x = torch.bmm(x.permute(0, 2, 1), alpha) # ideal code
        # x = torch.reshape(x, (bz, -1))  # (bz, 400)
        x = x.squeeze()
        return x

class SBERT(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        SentenceTransformer.__init__(self, model_name_or_path=model_name_or_path, modules=modules, device=device)

    def encode(
        self,
        sentences: Union[str, List[str], List[int]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        reduce_dim: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        use_device: bool = True,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        # self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == "token_embeddings":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        if use_device:
            self.to(device)
            linear.to(device)  # added

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            if use_device:
                features = batch_to_device(features, device)

            # with torch.no_grad():
            out_features = self.forward(features)

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(
                    out_features[output_value], out_features["attention_mask"]
                ):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0 : last_mask_id + 1])
            else:  # Sentence embeddings
                embeddings = out_features[output_value]
                if reduce_dim:
                    if use_device:
                        embeddings = linear(embeddings).to(embeddings.device)  # added
                    else:
                        embeddings = linear(embeddings)
                # embeddings.requires_grad = True # added
                # print("embeddings", embeddings.requires_grad)
                # embeddings = embeddings.detach() #comment detaching to make requires_grad true
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)


        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            if use_device:
                all_embeddings = np.asarray([emb.detach().numpy() for emb in all_embeddings]) # added .detach()
            else:
                all_embeddings = np.asarray([emb.detach().numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        # print(all_embeddings.requires_grad)

        return all_embeddings

import torch
import torch.nn as nn



class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = torch.mean(outputs, 1) # mean pooling
        return outputs

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feed_forward_dim, hidden_dim=768, n_layers=1, dropout=0.1, head_count=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embedding_dim,hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=head_count,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm,
        )

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        embedded = self.linear(embedded)
        # embedded = [sent len, batch size, emb dim]
        outputs = self.transformer(embedded)
        outputs = torch.mean(outputs, 1) # mean pooling
        return outputs


class TopicRNNModel(nn.Module):

    '''
        wcnt: 전체 단어의 갯수
        emb_size: embeding dimension
        topic_size: hidden dim
    '''
    def __init__(self, wcnt, emb_size=384, topic_size=768, num_layers=2):
        super(TopicRNNModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        if num_layers > 1:
            self.emb_size = topic_size
            self.rnn = nn.GRU(emb_size, topic_size, 1,
                              bidirectional=True,
                              dropout=0.1)
            self.output = nn.GRU(topic_size * 2,
                                 topic_size, num_layers - 1,
                                 dropout=0.1)
        else:
            self.emb_size = topic_size // 2 # output 의 shape 는 topic_size 가 될 것임
            self.rnn = nn.GRU(emb_size, topic_size // 2, 1,
                              bidirectional=True)

    def forward(self, input, hidden):
        x = self.embedding(input)
        # print(x.size())
        # exit(0)
        y, h1 = self.rnn(x, hidden[0])
        if self.num_layers > 1:
            y, h2 = self.output(y, hidden[1])
            return y[-1], (h1, h2)
        else:
            y, _ = torch.max(y, 0)
            return y, (h1, None)

    def default_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.emb_size)), \
            Variable(torch.zeros(self.num_layers - 1,
                                 batch_size, self.emb_size)) \
            if self.num_layers > 1 else None

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))

use_cuda = torch.cuda.is_available()

def Variable(*args, **kwargs):
    v = torch.autograd.Variable(*args, **kwargs)
    if use_cuda:
        v = v.cuda()
    return v
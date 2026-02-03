import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)

        self.mask = torch.tril(torch.ones((max_seq_len, max_seq_len)), diagonal=0)

    def forward(self, x):
        seq_len = x.shape[-2]

        K = self.W_k(x)
        Q = self.W_q(x)
        V = self.W_v(x)

        attention_scores = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)
        causal_mask = self.mask[:seq_len, :seq_len]

        attention_scores = attention_scores.masked_fill(causal_mask==0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)

        out = attention_scores @ V
        return out



        
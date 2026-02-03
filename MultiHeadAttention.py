from HeadAttention import HeadAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.layer = nn.Linear(head_size*num_heads, emb_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        heads_out = [head(x) for head in self.heads]
        concat = torch.cat(heads_out, dim=-1)
        out = self.layer(concat)
        out = self.drop(out)
        return out


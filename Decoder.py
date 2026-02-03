from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward 

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.MHA = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.FFN = FeedForward(emb_size, dropout)
        self.norm_1 = nn.LayerNorm(emb_size)
        self.norm_2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        out_1 = self.MHA(x) + x
        out_1 = self.norm_1(out_1)
        out_2 = self.FFN(out_1) + out_1
        out_2 = self.norm_2(out_2)
        return out_2






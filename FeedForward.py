import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout

        self.FFN = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(4*emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.FFN(x)
        return out


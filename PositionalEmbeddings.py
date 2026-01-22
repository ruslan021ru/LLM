class PositionalEmbeddings(nn.Module):
    "Позиционные эмбеддинги"
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embedding_layer = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int) -> torch.Tensor:
        ind = torch.arange(seq_len, dtype=torch.long)
        embeddings = self.embedding_layer(ind)
        return embeddings

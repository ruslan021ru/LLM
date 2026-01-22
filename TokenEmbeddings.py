class TokenEmbeddings(nn.Module):
    "Эмбеддинги токенов"
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(x)
        return embeddings
import torch
import torch.nn as nn
from TransformerEmbedding import TransformerEmbedding
from EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, pad_idx)  # 已实现的嵌入层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        x = self.embedding(x)  # 词嵌入 + 位置编码
        for layer in self.layers:
            x = layer(x, mask)  # 经过N个编码器层
        return x  # 编码器输出，作为解码器的K和V
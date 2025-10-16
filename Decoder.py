import torch
import torch.nn as nn
from TransformerEmbedding import TransformerEmbedding
from DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, pad_idx)  # 嵌入层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, enc_output, self_mask, cross_mask):
        x = self.embedding(x)  # 词嵌入 + 位置编码
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)  # 经过N个解码器层
        return x  # 解码器输出
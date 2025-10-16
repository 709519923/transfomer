import torch
import torch.nn as nn
from TransformerEmbedding import TransformerEmbedding
from MultiHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 编码器-解码器注意力
        self.feed_forward = nn.Sequential(  # 前馈网络
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1（掩码自注意力后）
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2（编码器-解码器注意力后）
        self.norm3 = nn.LayerNorm(d_model)  # 层归一化3（前馈网络后）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩码自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 编码器-解码器注意力 + 残差连接 + 层归一化
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(cross_output))
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x
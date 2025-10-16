import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 多头自注意力
        self.feed_forward = nn.Sequential(  # 前馈网络
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2
        self.dropout1 = nn.Dropout(dropout)  # 残差连接后的dropout
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)  # Q=K=V（自注意力）
        x = self.norm1(x + self.dropout1(attn_output))
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
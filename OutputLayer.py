import torch
import torch.nn as nn


class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)  # 映射到词汇表大小

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model] → 输出 [seq_len, batch_size, vocab_size]
        return self.linear(x)
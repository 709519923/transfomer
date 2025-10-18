import torch
import torch.nn as nn
from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    自定义实现的多头注意力机制 (Multi-Head Attention).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力模块.

        参数:
            d_model (int): 嵌入向量的总维度.
            n_heads (int): 注意力头的数量.
            dropout (float): Dropout的比例.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        # 每个头的维度 d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # 线性变换层，用于将 Q, K, V 投影到 d_model 维度
        # 在 PyTorch 中，通常将这三个投影合并为一个大矩阵，或分开定义三个 nn.Linear。
        # 这里使用三个独立的层，更清晰地表示 Q, K, V 的分离。
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model),  # W_Q
            nn.Linear(d_model, d_model),  # W_K
            nn.Linear(d_model, d_model),  # W_V
            nn.Linear(d_model, d_model)  # W_O (输出的线性变换)
        ])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播.

        参数:
            query, key, value (Tensor): Q, K, V 输入. 形状: [batch_size, seq_len, d_model]
            mask (Tensor, optional): 注意力掩码.

        返回:
            output (Tensor): 多头注意力计算结果. 形状: [batch_size, seq_len, d_model]
        """
        if mask is not None:
            # 对于批次中的所有样本，掩码是相同的，因此扩展维度使其可用于广播
            # mask 形状: [batch_size, 1, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]
            # 我们假设输入 mask 是 [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # 变为 [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 变为 [batch_size, 1, seq_len, seq_len]

        n_batches = query.size(0)

        # 1. 线性投影并分割成 n_heads
        # 原始形状: [batch_size, seq_len, d_model]
        # 线性投影后: [batch_size, seq_len, d_model]
        # view & transpose 后 (分割成头): [batch_size, n_heads, seq_len, d_k]

        def linear_and_split(x, linear):
            """对输入进行线性变换，然后分割成 n_heads"""
            return linear(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)  # -1和heads调换，让head在前面，方便训练

        query, key, value = [
            linear_and_split(x, l)
            for x, l in zip((query, key, value), self.linears[:3])
        ]

        # 2. 计算缩放点积注意力
        # x 形状: [batch_size, n_heads, seq_len, d_k]
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 拼接所有头
        # 恢复形状: [batch_size, seq_len, n_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        # 合并 d_k 和 n_heads: [batch_size, seq_len, d_model]
        x = x.view(n_batches, -1, self.d_model)

        # 4. 最后的线性投影 (W_O)  --DC用于将8个头的特征融合，即不仅仅有拼接，还有W_O的线性缩放
        output = self.linears[-1](x)

        # 返回结果和注意力权重（可选）
        return output, attn

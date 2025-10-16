import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力 (Scaled Dot-Product Attention).

    参数:
        query, key, value (Tensor): Q, K, V 张量. 形状通常为 [batch_size, n_heads, seq_len, d_k]
        mask (Tensor, optional): 掩码张量. 形状通常为 [batch_size, 1, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]
        dropout (nn.Dropout, optional): Dropout 实例.

    返回:
        output (Tensor): 注意力计算结果. 形状为 [batch_size, n_heads, seq_len, d_k]
        attn (Tensor): 注意力权重矩阵. 形状为 [batch_size, n_heads, seq_len, seq_len]
    """
    d_k = query.size(-1)

    # 1. 计算 Q * K^T
    # scores 形状: [batch_size, n_heads, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. 缩放 (Scale)
    scores = scores / math.sqrt(d_k)

    # 3. 应用掩码 (Masking)
    if mask is not None:
        # 使用极小的负数（如-1e9）来填充被掩盖的位置，
        # 经过 softmax 后，这些位置的权重将趋近于 0。
        # mask 应该是一个布尔张量 (True for masked, False for unmasked) 或 float 张量 (1 for masked, 0 for unmasked)
        # 这里假设 mask 值为 0 或 1，其中 0 表示不应该被掩盖，1 表示应该被掩盖

        scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax
    attn = F.softmax(scores, dim=-1)

    # 5. Dropout
    if dropout is not None:
        attn = dropout(attn)

    # 6. 计算 Attention * V
    # output 形状: [batch_size, n_heads, seq_len, d_k]
    output = torch.matmul(attn, value)

    return output, attn

#
# # ------------------------------
# # 测试参数设置
# # ------------------------------
# batch_size = 2  # 批次大小
# n_heads = 8     # 注意力头数
# seq_len = 512   # 序列长度（Q、K、V的序列长度均为512）
# d_k = 512       # 每个头的特征维度（Q、K的最后一维为512）
#
# # ------------------------------
# # 构造输入张量（Q、K、V）
# # ------------------------------
# # Q形状：[batch_size, n_heads, seq_len, d_k] → [2, 8, 512, 512]
# query = torch.randn(batch_size, n_heads, seq_len, d_k)
# # K形状与Q相同：[2, 8, 512, 512]
# key = torch.randn(batch_size, n_heads, seq_len, d_k)
# # V的特征维度通常与K一致（也可为其他值，这里保持d_k=512）
# value = torch.randn(batch_size, n_heads, seq_len, d_k)
#
# # 构造掩码（可选，这里用全1掩码表示无遮挡）
# mask = torch.ones(batch_size, 1, seq_len, seq_len)  # 形状：[2, 1, 512, 512]
#
# # 构造dropout层（可选）
# dropout = torch.nn.Dropout(p=0.1)
#
# # ------------------------------
# # 执行注意力计算
# # ------------------------------
# output, attn = scaled_dot_product_attention(
#     query=query,
#     key=key,
#     value=value,
#     mask=mask,
#     dropout=dropout
# )
#
# # ------------------------------
# # 输出结果形状验证
# # ------------------------------
# print(f"Q形状: {query.shape}")          # 预期：[2, 8, 512, 512]
# print(f"K形状: {key.shape}")            # 预期：[2, 8, 512, 512]
# print(f"V形状: {value.shape}")          # 预期：[2, 8, 512, 512]
# print(f"注意力分数（scores）形状: [2, 8, 512, 512]")  # Q·K^T的结果
# print(f"注意力权重（attn）形状: {attn.shape}")  # 预期：[2, 8, 512, 512]
# print(f"注意力输出（output）形状: {output.shape}")  # 预期：[2, 8, 512, 512]
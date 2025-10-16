import torch


def create_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    创建填充掩码（Padding Mask），用于掩盖序列中的填充token（pad token）

    参数:
        seq: 输入序列，形状 [seq_len, batch_size] 或 [batch_size, seq_len]
        pad_idx: 填充token的索引

    返回:
        mask: 掩码张量，形状 [batch_size, 1, 1, seq_len]（便于注意力计算时广播）
              填充位置为0，有效位置为1
    """
    # 检查输入维度，适应不同格式
    if seq.dim() == 2:
        # 若输入为 [seq_len, batch_size]，转置为 [batch_size, seq_len]
        seq = seq.transpose(0, 1)

    # 生成 [batch_size, 1, 1, seq_len] 的掩码
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(1)
    return mask


def create_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    创建序列掩码（Subsequent Mask），用于解码器的自注意力，防止关注未来的token

    参数:
        seq_len: 序列长度
        device: 设备（CPU/GPU）

    返回:
        mask: 上三角掩码张量，形状 [1, 1, seq_len, seq_len]
              下三角（包括对角线）为1，上三角为0
    """
    # 生成上三角矩阵（对角线以上为1），然后取反得到下三角掩码
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # 转换为 0/1 掩码（0表示需要掩盖，1表示有效）
    mask = ~mask  # 下三角为True（1），上三角为False（0）
    return mask.unsqueeze(0).unsqueeze(0)  # 扩展维度便于广播


def combine_masks(pad_mask: torch.Tensor, subsequent_mask: torch.Tensor) -> torch.Tensor:
    """
    合并填充掩码和序列掩码（用于解码器自注意力）

    参数:
        pad_mask: 填充掩码，形状 [batch_size, 1, 1, seq_len]
        subsequent_mask: 序列掩码，形状 [1, 1, seq_len, seq_len]

    返回:
        combined_mask: 合并后的掩码，形状 [batch_size, 1, seq_len, seq_len]
                       两个掩码中任意一个为0的位置都会被掩盖
    """
    # 取逻辑与（两个掩码都为1的位置才有效）
    return pad_mask & subsequent_mask

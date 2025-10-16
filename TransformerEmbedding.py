import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    实现位置编码模块
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码模块.

        参数:
            d_model (int): 嵌入向量的维度.
            dropout (float): Dropout的比例.
            max_len (int): 序列的最大可能长度.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵 (max_len, d_model)
        # 这个矩阵是固定的，不需要参与梯度下降，所以我们不将其定义为nn.Parameter
        position = torch.arange(max_len).unsqueeze(1)  # shape: [max_len, 1]

        # 计算分母部分，使用log空间可以防止数值溢出
        # div_term shape: [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 创建一个 (max_len, d_model) 的零矩阵来存放位置编码
        pe = torch.zeros(max_len, 1, d_model)

        # 使用广播机制和正弦/余弦函数计算位置编码
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 将pe注册为模型的缓冲区（buffer）。
        # buffer是模型状态的一部分，会被保存和加载，但不会被视为模型参数进行训练。
        # 这对于像位置编码这样固定的、非学习的参数非常有用。
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播.

        参数:
            x (torch.Tensor): 输入的词元嵌入. shape: [seq_len, batch_size, d_model]
                              (注意：很多Transformer实现采用batch-first, 即[batch_size, seq_len, d_model]，
                               这里我们遵循PyTorch官方教程的seq-len-first格式)

        返回:
            torch.Tensor: 添加了位置编码的嵌入向量. shape: [seq_len, batch_size, d_model]
        """
        # 将位置编码pe加到输入的词元嵌入x上
        # self.pe是 [max_len, 1, d_model]，我们只取当前序列长度的部分
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEmbedding(nn.Module):
    """
    完整的Transformer Embedding模块.
    它包含词元嵌入和位置编码。
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1, pad_idx: int = 0):
        """
        初始化.

        参数:
            vocab_size (int): 词汇表的大小.
            d_model (int): 嵌入向量的维度.
            max_len (int): 序列的最大可能长度.
            dropout (float): Dropout的比例.
            pad_idx (int): padding token的索引，该索引处的嵌入向量将被初始化为0且不参与训练。
        """
        super().__init__()
        self.d_model = d_model
        # 词元嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播.

        参数:
            x (torch.Tensor): 输入的词元ID序列. shape: [seq_len, batch_size]

        返回:
            torch.Tensor: 经过嵌入和位置编码后的向量. shape: [seq_len, batch_size, d_model]
        """
        # 1. 获取词元嵌入
        # 输出 shape: [seq_len, batch_size, d_model]
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)

        # 2. 添加位置编码
        # 输出 shape: [seq_len, batch_size, d_model]
        final_emb = self.positional_encoding(token_emb)

        return final_emb

if __name__ == '__main__':
    # --- 参数定义 ---
    VOCAB_SIZE = 10000  # 假设词汇表大小为10000
    D_MODEL = 512  # 模型维度
    MAX_LEN = 100  # 句子的最大长度
    PAD_IDX = 0  # PADDING的ID
    SEQ_LEN = 30  # 当前批次中的序列长度
    BATCH_SIZE = 64  # 批次大小

    # --- 实例化模块 ---
    embedding_layer = TransformerEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX
    )

    # --- 创建一个假的输入数据 ---
    # shape: (SEQ_LEN, BATCH_SIZE)
    # 词元ID在1到VOCAB_SIZE-1之间随机生成
    src_tokens = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE))

    # --- 前向传播 ---
    output_embeddings = embedding_layer(src_tokens)

    # --- 打印结果 ---
    print(f"输入Token序列的形状: {src_tokens.shape}")
    print(f"输出Embedding向量的形状: {output_embeddings.shape}")

    # --- 验证输出形状是否正确 ---
    assert output_embeddings.shape == (SEQ_LEN, BATCH_SIZE, D_MODEL)
    print("\n代码运行成功，输出形状正确！")
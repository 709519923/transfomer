import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Transformer import Transformer
from mask_utils import create_pad_mask, create_subsequent_mask, combine_masks


# ------------------------------
# 1. 配置参数
# ------------------------------
class Config:
    # 数据参数
    src_vocab_size = 50  # 源语言词汇表大小
    tgt_vocab_size = 50  # 目标语言词汇表大小
    max_seq_len = 10  # 最大序列长度
    pad_idx = 0  # 填充符索引
    sos_idx = 1  # 起始符索引
    eos_idx = 2  # 结束符索引

    # 模型参数
    d_model = 64  # 模型维度（小数据集用小维度）
    n_layers = 2  # 编码器/解码器层数
    n_heads = 2  # 注意力头数
    d_ff = 128  # 前馈网络隐藏层维度
    dropout = 0.1  # Dropout比例

    # 训练参数
    batch_size = 32
    epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# ------------------------------
# 2. 构造玩具数据集（源语言→目标语言：数字反转）
# ------------------------------
class ToyDataset(Dataset):
    def __init__(self, num_samples=1000, max_len=10):
        self.num_samples = num_samples
        self.max_len = max_len
        self.pad_idx = config.pad_idx
        self.sos_idx = config.sos_idx
        self.eos_idx = config.eos_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机序列（3-8个数字，范围3-9，避免与特殊符号冲突）
        seq_len = np.random.randint(3, 8)
        src_seq = np.random.randint(3, 10, size=seq_len)  # 源序列：[3-9]的随机数

        # 目标序列：源序列反转（模拟翻译任务）
        tgt_seq = src_seq[::-1].copy()

        # 添加特殊符号（SOS在句首，EOS在句尾）
        src_seq = np.concatenate([src_seq, [self.eos_idx]])
        tgt_seq = np.concatenate([[self.sos_idx], tgt_seq, [self.eos_idx]])

        # 填充到最大长度
        src_pad_len = self.max_len - len(src_seq)
        tgt_pad_len = self.max_len - len(tgt_seq)
        src_seq = np.pad(src_seq, (0, src_pad_len), constant_values=self.pad_idx)
        tgt_seq = np.pad(tgt_seq, (0, tgt_pad_len), constant_values=self.pad_idx)

        return (
            torch.tensor(src_seq, dtype=torch.long),
            torch.tensor(tgt_seq, dtype=torch.long)
        )


# 数据加载器
train_dataset = ToyDataset(num_samples=1000, max_len=config.max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True
)

# ------------------------------
# 3. 初始化模型、损失函数、优化器
# ------------------------------
# 初始化Transformer
model = Transformer(
    src_vocab_size=config.src_vocab_size,
    tgt_vocab_size=config.tgt_vocab_size,
    d_model=config.d_model,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    d_ff=config.d_ff,
    max_len=config.max_seq_len,
    dropout=config.dropout,
    pad_idx=config.pad_idx
).to(config.device)

# 损失函数（忽略填充符）
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
optimizer = optim.Adam(model.parameters(), lr=config.lr)


# ------------------------------
# 4. 训练函数
# ------------------------------
def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    total_loss = 0.0

    for src_seq, tgt_seq in dataloader:
        # 数据移至设备
        # src_seq = src_seq.to(config.device).transpose(0, 1)  # [seq_len, batch_size]  [10, 8]
        # tgt_seq = tgt_seq.to(config.device).transpose(0, 1)  # [seq_len, batch_size]  [9,  8]
        src_seq = src_seq.to(config.device)  # [batch_size, seq_len]  [8， 10]
        tgt_seq = tgt_seq.to(config.device) # [batch_size, seq_len]  [8， 9]

        # 构造输入和标签（解码器输入不含最后一个EOS，标签不含第一个SOS）
        # tgt_input = tgt_seq[:-1, :]  # 解码器输入：[seq_len-1, batch_size]
        # tgt_label = tgt_seq[1:, :]  # 标签：[seq_len-1, batch_size]
        tgt_input = tgt_seq[:, :-1]  # 解码器输入：[batch_size, seq_len-1]
        tgt_label = tgt_seq[:, 1:]  # 标签：[batch_size, seq_len-1]  满足错位 input的index预测值正好是label的index的值
        # 生成掩码
        src_mask = create_pad_mask(src_seq, config.pad_idx).to(config.device)  # 编码器掩码  [batch_size, 1, 1, seq_len]

        tgt_pad_mask = create_pad_mask(tgt_input, config.pad_idx).to(config.device) # [batch_size, 1, 1, seq_len]
        # tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(0), config.device)
        tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(1), config.device) # [1, 1, seq_len, seq_len]
        tgt_self_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)  # 解码器自注意力掩码 # [batch_size, 1, 1, seq_len] & [1, 1, seq_len, seq_len] = [batch_size, 1, 1, seq_len]
        tgt_cross_mask = src_mask  # 解码器-编码器交叉注意力掩码（与编码器掩码相同） [batch_size, 1, 1, seq_len]

        # src_mask  => [8.1.1.10]
        # 前向传播



        optimizer.zero_grad()
        output = model(src_seq, tgt_input, src_mask, tgt_self_mask, tgt_cross_mask) # [batch, seq, d_model]
        # output形状：[seq_len-1, batch_size, tgt_vocab_size] → 转为[batch_size*seq_len-1, vocab_size]
        loss = criterion(
            output.transpose(0, 1).contiguous().view(-1, config.tgt_vocab_size),
            tgt_label.transpose(0, 1).contiguous().view(-1)
        )

        # 反向传播与优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ------------------------------
# 5. 推理函数（贪婪解码）
# ------------------------------
def predict(model, src_seq, config):
    model.eval()
    # src_seq = src_seq.unsqueeze(1).to(config.device)  # [seq_len, 1]（单样本）
    src_seq = src_seq.unsqueeze(0).to(config.device)  # [1, seq_len]（单样本）

    # 生成编码器掩码
    src_mask = create_pad_mask(src_seq, config.pad_idx).to(config.device) #[batch_size, 1, 1, seq_len]
    enc_output = model.encoder(src_seq, src_mask) # [batch, 1, 1, seq]) -> [batch, seq, d_model]

    # 初始化解码器输入（仅含SOS）
    # tgt_seq = torch.tensor([[config.sos_idx]], dtype=torch.long, device=config.device).transpose(0, 1)  # [1, 1]
    tgt_seq = torch.tensor([[config.sos_idx]], dtype=torch.long, device=config.device) # [1, 1] batch-first

    for _ in range(config.max_seq_len - 1):
        # 生成解码器掩码
        tgt_pad_mask = create_pad_mask(tgt_seq, config.pad_idx).to(config.device)
        tgt_subsequent_mask = create_subsequent_mask(tgt_seq.size(0), config.device)
        tgt_self_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)

        # 解码器前向传播
        dec_output = model.decoder(tgt_seq, enc_output, tgt_self_mask, src_mask)
        output = model.output_layer(dec_output[:, -1:, :])  # 取最后一个时间步
        pred_token = torch.argmax(output, dim=-1)  # 贪婪选择

        # 拼接结果
        tgt_seq = torch.cat([tgt_seq, pred_token], dim=1)

        # 若预测到EOS，停止解码
        if pred_token.item() == config.eos_idx:
            break

    return tgt_seq.squeeze(1).cpu().numpy()  # 转为numpy数组


# ------------------------------
# 6. 主流程
# ------------------------------
if __name__ == "__main__":
    # 训练模型
    print(f"使用设备: {config.device}")
    for epoch in range(config.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, config)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {loss:.4f}")

    # 测试推理（随机选3个样本）
    print("\n推理测试:")
    test_samples = [train_dataset[i][0] for i in np.random.choice(100, 3)]  # 取3个源序列
    for src_seq in test_samples:
        # 原始序列（去除填充和EOS）
        src_clean = src_seq[src_seq != config.pad_idx]
        src_clean = src_clean[src_clean != config.eos_idx].numpy()

        # 推理结果（去除SOS、EOS和填充）
        pred = predict(model, src_seq, config)
        pred_clean = pred[(pred != config.sos_idx) & (pred != config.eos_idx) & (pred != config.pad_idx)]

        print(f"源序列: {src_clean}")
        print(f"预测结果: {pred_clean} (预期: {src_clean[::-1]})")
        print("---")



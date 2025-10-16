import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from OutputLayer import OutputLayer



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6,
                 n_heads=8, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        self.output_layer = OutputLayer(d_model, tgt_vocab_size)

    def forward(self, src_seq, tgt_seq, src_mask, tgt_self_mask, tgt_cross_mask):
        enc_output = self.encoder(src_seq, src_mask)  # 编码器输出
        dec_output = self.decoder(tgt_seq, enc_output, tgt_self_mask, tgt_cross_mask)  # 解码器输出
        output = self.output_layer(dec_output)  # 映射到词汇表
        return output
When we learn about a new framework in AI, there are something we should be familiar with.
1. everything in the image of framework
2. shape flow chart along every nn.module and whole module
3. an example of every data(including how to train and how to predict)
By doing the 3 steps above, we can take control of a new design concept and make it as our inner knowledge to use in our new idea whatever in science research or work.
This project will help you get familiar with Transformer thorough the 3 steps. I hope you can enjoy the magic of transformer and get happiness from it.


1. shape flow chart
<img width="572" height="736" alt="image" src="https://github.com/user-attachments/assets/577eb264-4cf4-46c7-9077-f39b4408720b" />
word embedding:

`x = [batch, len] => [batch, len, d_model]`

`position => [batch, 1, d_model]`

`x + postion => [batch, len, d_model]`

MultiHeadAttention：
`x = [batch_size, seq_len, n_heads, d_k] => linear_transform x3 => q,k,v [batch_size, seq_len, n_heads, d_k]  => attention, value * attention(as k,v input of encoder)`

mask:

in multi-head attetion： will be used for self-attention broadcast
`mask = [batch, 1, 1, len] `

`attention = q * kT = [batch, n_head, len, len]`

`mask_score = mask * attention [batch, n_head, len, len]`

`output = attention * value [batch, n_head, len, len] * [batch, n_head, len, d_k] = [batch, n_head, len, d_k]`

in mask-head attention: lower triangular matrix to prevent seeing future token
`triangular_mask = [1, 1, seq_len, seq_len]`

` pad_mask       = [batch, 1, 1, len] `

`combine_masks = triangular_mask & pad_mask = [1, 1, seq_len, seq_len] & [batch, 1, 1, len] = [b, 1, len, len]`

lower triangular matrix makes:

训练时，限制模型获取 “不该看到的信息”

复用训练时的掩码逻辑：

推理时，对于已生成的序列（如[t1, t2, ..., ti]），生成ti+1时，掩码会屏蔽ti+1之后的位置（此时尚未生成，视为 “未来信息”），确保模型仅基于t1~ti进行预测。
意味着[b,1,i, len]表示训练第i+1个token时，应该掩盖的内容


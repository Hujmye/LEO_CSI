import numpy as np
import torch
import torch.nn as nn
d_k = 32 # K(=Q) 维度
d_v = 32 # V 维度


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask, hyper):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        weights = nn.Softmax(dim=-1)(scores)
        # hyper in
        context = (torch.matmul(weights, V)) * hyper.view(batch_size, n_heads_hyper, 8,  d_k)  #1024 16 8 32
        return context, weights

# 定义多头自注意力类
d_embedding = 512  # Embedding 的维度
n_heads_hyper = 16  # Multi-Head Attention 中头的个数
batch_size = 1024
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads_hyper) # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads_hyper) # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads_hyper) # V的线性变换层
        self.linear = nn.Linear(n_heads_hyper * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, Q, K, V, attn_mask, hyper):
        residual = Q
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads_hyper, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads_hyper, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads_hyper, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads_hyper, 1, 1)
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, hyper)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads_hyper * d_v)
        output = self.linear(context)
        output = self.layer_norm(output + residual)
        return output, weights

# 定义逐位置前馈网络类
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=1024):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, inputs, hyper):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = (output + residual)
        output = self.layer_norm(output)
        return output

# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # <PAD>token 的编码值为 0
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask


# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头自注意力层
        self.pos_ffn = PoswiseFeedForwardNet()  # 位置前馈神经网络层
    def forward(self, enc_inputs, enc_self_attn_mask, hyper):
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, hyper)
        enc_outputs = self.pos_ffn(enc_outputs, hyper)
        return enc_outputs, attn_weights

# 定义编码器类
n_layers = 1  # 设置 Encoder 的层数

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))# 编码器层数
    def forward(self, enc_outputs, hyper):
        enc_inputs = enc_outputs[:, :, 0]
        enc_inputs = torch.squeeze(enc_inputs, -1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_weights = []
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask, hyper)
            enc_self_attn_weights.append(enc_self_attn_weight)
        return enc_outputs, enc_self_attn_weights # 返回编码器输出和编码器注意力权重


class Transformer_hyper(nn.Module):
    def __init__(self):
        super(Transformer_hyper, self).__init__()
        self.encoder = Encoder()  # 初始化编码器实例
    def forward(self, enc_inputs, hyper):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, hyper)
        return enc_outputs

import math
from transformer_uplink import *
from transformer_hyper import *
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # (5000) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)




class Transformer_model(nn.Module):
    def __init__(self, M, B, L, L2, Lp, alpha_para, N0, S, batch_size):
        super(Transformer_model, self).__init__()
        self.M = M
        self.B = B
        self.L = L
        self.L2 = L2
        self.Lp = Lp
        self.N0 = N0
        self.S = S
        self.batch_size = batch_size
        self.alpha_para = alpha_para
        self.pe_ul = PositionalEncoding(d_model=512)
        self.pe_dl = PositionalEncoding(d_model=512)
        self.X_tilde_ini = torch.nn.Parameter(torch.sqrt(torch.tensor(1.0 / M)) * torch.randn([M, 2 * L])).cuda()
        self.X_tilde_ini2 = torch.nn.Parameter(torch.sqrt(torch.tensor(1.0)) * torch.randn([1, 2 * L2])).cuda()
        ########################uplink
        self.d1 = nn.Linear(2*M*L2, 512)
        self.transformer_encoder_ul = Transformer()

        ########################donwlink
        self.u1 = nn.Linear(2 * L, 256)
        self.u2 = nn.Linear(256, 256)
        self.u3 = nn.Linear(256, 128)
        self.u4 = nn.Linear(128, B)
        self.dfc1 = nn.Linear(B, 512)
        self.transformer_encoder = Transformer_hyper()
        self.dfc3 = nn.Linear(512, M * 2)


        #######################################
        #######################################

    def forward(self, X1, X2):
        ######################Uplink
        power_normal2 = torch.sqrt(torch.sum(self.X_tilde_ini2[:, 0:self.L2] ** 2 + self.X_tilde_ini2[:, self.L2:2 * self.L2] ** 2, dim=0))
        X_tilde2 = self.X_tilde_ini2 / torch.cat([power_normal2, power_normal2], dim=0)
        X_tilde_complex2 = torch.view_as_complex(X_tilde2.view(1, 1, self.L2, 2)).squeeze(-1)  # 1, 1, L2
        X_tilde_complex2_full = X_tilde_complex2.repeat(self.batch_size, 1, 1)  # 复制数组, (batch_size, 1, L2)
        # X2 (batch_size, S, M) X2[:, 1, :] : (batch_size, M) # X2 应该是时变信道序列？
        # X2[:, 0, :] 表示发送信号，X_tilde_complex2_full 表示信道增益，y2 表示接收到的信号。
        y2 = torch.bmm(X2[:, 0, :].view(self.batch_size, self.M, 1),X_tilde_complex2_full)  # (batch_size, M, 1) (batch_size, 1, L2) = (batch_size, M, L2)
        y22 = y2.view(self.batch_size, 1, self.M * self.L2)
        for s in range(1, self.S):
            temp = torch.bmm(X2[:, s, :].view(self.batch_size, self.M, 1), X_tilde_complex2_full)  # (batch_size, M, L2)
            temp2 = temp.view(self.batch_size, 1, self.M * self.L2)
            y22 = torch.cat([y22, temp2], dim=1)  # 输入hyper的数据 (batch_size, S, M*L2)
        y_real2 = torch.cat([torch.real(y22), torch.imag(y22)], dim=2).cuda()  # (batch_size, S, M*L2*2)
        noise2 = torch.sqrt(torch.tensor(self.N0 / 2)) * torch.randn((self.batch_size, self.S, self.M * self.L2 * 2)).cuda()
        y_noise2 = y_real2 + noise2  # (batch_size, S, 2*M*L2)
        #####################################################################################################################
        encoder_in_ul = self.d1(y_noise2)
        en_ul = self.pe_ul(encoder_in_ul)
        hyper = self.transformer_encoder_ul(en_ul)  # hyper weight

        ######################################################################################################################
        ##########################downlink
        power_normal = torch.sqrt(torch.sum(torch.square(self.X_tilde_ini[:, 0:self.L]) + torch.square(self.X_tilde_ini[:, self.L:2 * self.L]), dim=0))
        X_tilde = self.X_tilde_ini / (torch.cat([power_normal, power_normal], dim=0))
        X_tilde_complex = torch.complex(X_tilde[:, 0:self.L], X_tilde[:, self.L:2*self.L]).view(1, self.M, self.L)
        X_tilde_complex1_full = X_tilde_complex.repeat(self.batch_size, 1, 1)  # (batch_size,M,L1)
        y = torch.matmul(X1, X_tilde_complex1_full)  # (batch_size,S,M) (batch_size,M,L1) =(batch_size *S *L1)
        y_real = torch.cat([torch.real(y), torch.imag(y)], dim=2).cuda()
        noise = (torch.sqrt(torch.tensor(self.N0 / 2)) * torch.randn((self.batch_size, self.S, self.L * 2))).cuda()
        y_noise = y_real + noise  # (batch_size,S,2*L)
        #################################################
        # Process the initial sequence with linear layers
        # 第一部分：处理y_noise
        nu1 = torch.relu(self.u1(y_noise))
        nu2 = torch.relu(self.u2(nu1))
        nu3 = torch.relu(self.u3(nu2))
        q = torch.tanh(self.u4(nu3))
        # 第二部分：处理lstm
        x_t = torch.tanh(self.dfc1(q))  # 1024 8 256
        en_input = self.pe_dl(x_t)
        out = self.transformer_encoder(en_input, hyper)
        hidden_seq = self.dfc3(out) #* hyper
        h_est_complext = torch.complex(hidden_seq[:, :, 0:self.M], hidden_seq[:, :, self.M:2 * self.M])
        return h_est_complext



# 定义自己的损失函数NMSE
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_pred, y_true):
        nmse = torch.mean(torch.square(torch.abs(y_true - y_pred)))
        return nmse


class NMSE_cal(nn.Module):
    def __init__(self):
        super(NMSE_cal, self).__init__()

    def forward(self, y_pred, y_true):
        # 计算误差
        error = y_true - y_pred
        # 计算误差的平方的模
        mse = torch.sum(torch.abs(error) ** 2)
        # 计算真实值的平方的模
        true_energy = torch.sum(torch.abs(y_true) ** 2)
        # 计算NMSE
        nmse = mse / true_energy
        return nmse


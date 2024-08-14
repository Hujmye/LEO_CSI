from hyper_model_transformer import *
from generate_h import *

model = torch.load("testb15rf.path")
model.eval()

np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = 64
B = 10
Lp = 2
L1 = 2
L2 = 8
delta = 0.2
batch_size = 1024
K = 1
input_number = M            #输入特征数
output_number = M * 2
alpha_para_input = 1.0
SNRdb = 10
sigma2 = 1 / (10 ** (SNRdb / 10.0))
N0_input = sigma2
Mx = np.sqrt(M).astype(int)
My = np.sqrt(M).astype(int)
time_length = 8
dl_cor = 0
ul_cor = 0
time_doppler_opt = 1  #参数代表时间-多普勒优化选项
if time_doppler_opt == 1:  # 0.5e-4
    dl_cor = 0.9998
    ul_cor = 0.9998
if time_doppler_opt == 2:  # 0.5e-3
    dl_cor = 0.9829
    ul_cor = 0.9818
if time_doppler_opt == 3:  # 0.8e-3
    dl_cor = 0.9566
    ul_cor = 0.9537
if time_doppler_opt == 4:  # 1e-3
    dl_cor = 0.9326
    ul_cor = 0.9281
if time_doppler_opt == 5:  # 0.2e-2
    dl_cor = 0.7441
    ul_cor = 0.7280
if time_doppler_opt == 6:  # 0.3e-2
    dl_cor = 0.4720
    ul_cor = 0.4422
if time_doppler_opt == 7:  # 0.4e-2
    dl_cor = 0.1698
    ul_cor = 0.1304

loss = NMSE_cal()
avg = 0
nmse = 0

for i in range(100):
    h_dl, h_ul = generate_dl(Mx, My, batch_size=1024, K=1, Lp=Lp, M=64, time_length=time_length, dl_cor=dl_cor,
                             ul_cor=ul_cor, delta=delta)

    h_dl_ture = torch.from_numpy(h_dl).to(device)
    h_dl_ture = h_dl_ture.to(torch.complex64)  # 1024 8 64

    h_ul_ture = torch.from_numpy(h_ul).to(device)  # 1024 8 64
    h_ul_ture = h_ul_ture.to(torch.complex64)

    with torch.no_grad():
        outputs = model(h_dl_ture, h_ul_ture)
        nmse = loss(outputs, h_dl_ture)
        avg = avg + nmse

print(avg.item()/100)
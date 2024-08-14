import torch.optim as optim
from generate_h import *
from hyper_model_transformer import *
np.random.seed(0)
M = 64
B = 15
Lp = 2
L1 = 2
L2 = 2
delta = 0.2
batch_size = 1024
total_epoch = 30000
K = 1
input_number = M
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
time_doppler_opt = 1  # 参数代表时间-多普勒优化选项
print("transformer", "Lp=", Lp, "L1=", L1, "L2=", L2, "B=", B)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer_model(M, B, L1, L2, Lp, alpha_para_input, N0_input, time_length, batch_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) #0.001 0.0001 0.00001
ExpLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 21000], gamma=0.1)  #0.001 0.0001 0.00001
criterion = NMSE_cal()
NMSE_cal = NMSE_cal()
avg_nmse = 0
record_min = 99

for epoch in range(total_epoch):
    h_dl, h_ul = generate_dl(Mx, My, batch_size=1024, K=1, Lp=Lp, M=64, time_length=time_length, dl_cor=dl_cor,ul_cor=ul_cor, delta=delta)

    h_dl_ture = torch.from_numpy(h_dl).to(device)
    h_dl_ture = h_dl_ture.to(torch.complex64)

    h_ul_ture = torch.from_numpy(h_ul).to(device)
    h_ul_ture = h_ul_ture.to(torch.complex64)

    outputs = model(h_dl_ture, h_ul_ture)
    loss = criterion(outputs, h_dl_ture)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ExpLR.step()
    nmse = NMSE_cal(outputs, h_dl_ture)
    avg_nmse = avg_nmse + nmse
    if (epoch + 1) % 10 == 0:
        if record_min > (avg_nmse.item() / 10):
            record_min = (avg_nmse.item() / 10)
        print("Epoch:", '%04d' % (epoch + 1), "train_cost=", "{:.9f}".format(loss.item()), " nmse=",
              avg_nmse.item() / 10, ' min=', record_min)
        avg_nmse = 0

# torch.save(model, "Transformer_B1_LuLd2_Lp2.path")




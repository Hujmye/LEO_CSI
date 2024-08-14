import numpy as np
def generate_dl(Mx, My, batch_size = 1024, K = 1, Lp = 2, M = 64, time_length=8, dl_cor=0.9998,ul_cor=0.9998, delta=0.2):
    h_complex_dl = np.zeros([batch_size, time_length, M], dtype=complex)  # 信道（复数形式）
    h_complex_ul = np.zeros([batch_size, time_length, M], dtype=complex)  # 信道（复数形式）
    gain_dl_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
    gain_dl_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
    gain_dl = gain_dl_real + 1j * gain_dl_img  # 初始化路径增益           #[batch_size, K, Lp]
    gain_ul_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
    gain_ul_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
    gain_ul = gain_ul_real + 1j * gain_ul_img  # 初始化路径增益           #[batch_size, K, Lp]
    KK = 10  # 莱斯因子定为10dB
    theta = (4.0 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # π/18 ~ π/2
    phi = 2 * np.pi * np.random.rand(batch_size, K)  # 0~2π
    for t in range(time_length):
        f_dop_dl = 3000 + 4000 * np.random.rand(batch_size, K)  # 多普勒频移:3000~7000Hz
        f_dop_ul = 3000 + 4000 * np.random.rand(batch_size, K)  # 多普勒频移:3000~7000Hz
        tau_dl = (10.0 * np.random.rand(batch_size, K) + 40.0) * 0.001  # 40ms~50ms
        tau_ul = (10.0 * np.random.rand(batch_size, K) + 40.0) * 0.001  # 40ms~50ms
        h_complex_t_dl = np.zeros([batch_size, M], dtype=complex)  # 信道（复数形式）
        h_complex_t_ul = np.zeros([batch_size, M], dtype=complex)  # 信道（复数形式）
        arv_x = np.zeros([Mx], dtype=complex)  # array response vector_x
        arv_y = np.zeros([My], dtype=complex)  # array response vector_y
        for size in range(batch_size):
            # 计算array response vector
            for mx in range(Mx):
                arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, 0]) * np.sin(phi[size, 0]))
            for my in range(My):
                arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, 0]) * np.sin(phi[size, 0]))
            arv = np.kron(arv_x, arv_y)

            h_los_dl = np.exp(1j * 2 * np.pi * (f_dop_dl[size, 0] - 2 * 1e9 * tau_dl[size, 0])) * arv
            h_nlos_dl = np.zeros([M], dtype=complex)

            h_los_ul = np.exp(1j * 2 * np.pi * (f_dop_ul[size, 0] - (2 * 1e9 + delta * 1e9) * tau_ul[size, 0])) * arv
            h_nlos_ul = np.zeros([M], dtype=complex)

            for p in range(Lp):
                if t == 0:
                    h_nlos_dl = h_nlos_dl + np.sqrt(1 / Lp) * gain_dl[size, 0, p] * np.exp(
                        1j * 2 * np.pi * (f_dop_dl[size, 0] - 2 * 1e9 * tau_dl[size, 0])) * arv
                    h_nlos_last_dl = h_nlos_dl

                    h_nlos_ul = h_nlos_ul + np.sqrt(1 / Lp) * gain_ul[size, 0, p] * np.exp(
                        1j * 2 * np.pi * (f_dop_ul[size, 0] - (2 * 1e9 + delta * 1e9) * tau_ul[size, 0])) * arv
                    h_nlos_last_ul = h_nlos_ul

                else:
                    # downlink 衰落振幅
                    noise_dl = np.sqrt(0.5) * (
                            np.random.standard_normal([M]) + 1j * np.random.standard_normal([M]))
                    h_nlos_dl = h_nlos_last_dl * dl_cor + noise_dl * np.sqrt(1 - np.square(dl_cor))
                    h_nlos_last_dl = h_nlos_dl

                    # ullink 衰落振幅
                    noise_ul = np.sqrt(0.5) * (
                            np.random.standard_normal([M]) + 1j * np.random.standard_normal([M]))
                    h_nlos_ul = h_nlos_last_ul * ul_cor + noise_ul * np.sqrt(1 - np.square(ul_cor))
                    h_nlos_last_ul = h_nlos_ul


            h_complex_t_dl[size, :] = np.sqrt(KK / (KK + 1)) * h_los_dl + np.sqrt(1 / (KK + 1)) * h_nlos_dl  # 计算信道h

            h_complex_t_ul[size, :] = np.sqrt(KK / (KK + 1)) * h_los_ul + np.sqrt(1 / (KK + 1)) * h_nlos_ul  # 计算信道h

        h_complex_dl[:, t, :] = h_complex_t_dl
        h_complex_ul[:, t, :] = h_complex_t_ul

    return h_complex_dl, h_complex_ul








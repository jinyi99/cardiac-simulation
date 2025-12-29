import cupy as cp
import numpy as np


class GPUNivalaRyR:
    """RyR受体类（Nivala模型）"""

    # RyR参数
    KAP = 0.7 * 0.005  # 0.7*0.005
    KAM = 1.0
    KBM = 0.003  # 0.003
    KBMR2C = 1.0 * 0.003  # 使用kbm，不保持详细平衡
    KBP = 0.00075

    def __init__(self, n_cru, n_ryr=100):
        """
        初始化RyR受体

        Args:
            n_cru: CRU数量
            n_ryr: 每个CRU的RyR数量
        """
        self.n_cru = n_cru
        self.n_ryr = n_ryr

        # Markov状态
        self.Nr = cp.zeros(n_cru, dtype=cp.int32)  # 开放状态
        self.Nrc = cp.full(n_cru, int(0.75 * n_ryr), dtype=cp.int32)  # 关闭状态
        self.Nri = cp.zeros(n_cru, dtype=cp.int32)  # 失活状态
        self.Nrr = n_ryr - self.Nr - self.Nrc - self.Nri  # 准备状态

        # 随机数状态
        self.rfire = cp.zeros(n_cru, dtype=cp.float32)

        # 初始化随机数生成器
        self.rng = cp.random.RandomState(seed=42)

        # 初始化第一次随机抽取
        self.rfire = -cp.log(self.rng.uniform(0, 1, n_cru))

    def update_ryr(self, dt, cd, cj):
        """
        更新RyR状态

        Args:
            dt: 时间步长
            cd: dyadic空间钙浓度
            cj: jSR钙浓度
        """
        # 计算速率常数
        alpha = self.KAP * cp.power(cd, 2)
        gamma = self.KBP * cd
        beta = self.KAM
        delta = self.KBM

        # 计算转换速率
        sco = self.Nrc * alpha
        soc = self.Nr * beta
        scr = self.Nrc * gamma
        src = self.Nrr * self.KBMR2C
        sri = self.Nrr * alpha
        sir = self.Nri * beta
        sio = self.Nri * delta
        soi = self.Nr * gamma

        # 计算累积速率
        lam1 = sco + soc
        lam2 = lam1 + scr
        lam3 = lam2 + src
        lam4 = lam3 + sri
        lam5 = lam4 + sir
        lam6 = lam5 + sio
        lamtot = lam6 + soi

        # 更新状态
        rtest = self.rfire - lamtot * dt

        # 找出需要更新的CRU
        update_mask = rtest <= 0
        update_indices = cp.where(update_mask)[0]

        if len(update_indices) > 0:
            # 处理需要更新的CRU
            for idx in update_indices:
                tloc = self.rfire[idx] / lamtot[idx]
                r2 = lamtot[idx] * self.rng.uniform()

                # 根据随机数决定状态转换
                if r2 <= sco[idx]:
                    self.Nrc[idx] -= 1
                    self.Nr[idx] += 1
                elif r2 <= lam1[idx]:
                    self.Nrc[idx] += 1
                    self.Nr[idx] -= 1
                elif r2 <= lam2[idx]:
                    self.Nrc[idx] -= 1
                    self.Nrr[idx] += 1
                elif r2 <= lam3[idx]:
                    self.Nrr[idx] -= 1
                    self.Nrc[idx] += 1
                elif r2 <= lam4[idx]:
                    self.Nrr[idx] -= 1
                    self.Nri[idx] += 1
                elif r2 <= lam5[idx]:
                    self.Nri[idx] -= 1
                    self.Nrr[idx] += 1
                elif r2 <= lam6[idx]:
                    self.Nri[idx] -= 1
                    self.Nr[idx] += 1
                else:
                    self.Nr[idx] -= 1
                    self.Nri[idx] += 1

                # 重新计算速率
                sco_new = self.Nrc[idx] * alpha[idx]
                soc_new = self.Nr[idx] * beta
                scr_new = self.Nrc[idx] * gamma[idx]
                src_new = self.Nrr[idx] * self.KBMR2C
                sri_new = self.Nrr[idx] * alpha[idx]
                sir_new = self.Nri[idx] * beta
                sio_new = self.Nri[idx] * delta
                soi_new = self.Nr[idx] * gamma[idx]

                lam1_new = sco_new + soc_new
                lam2_new = lam1_new + scr_new
                lam3_new = lam2_new + src_new
                lam4_new = lam3_new + sri_new
                lam5_new = lam4_new + sir_new
                lam6_new = lam5_new + sio_new
                lamtot_new = lam6_new + soi_new

                # 更新随机时间
                r1 = self.rng.uniform()
                while r1 <= 0:
                    r1 = self.rng.uniform()

                mlogr1 = -cp.log(r1)
                wt = mlogr1 / lamtot_new
                tloc += wt

                # 处理额外的事件
                while tloc < dt:
                    r2 = lamtot_new * self.rng.uniform()

                    # 根据随机数决定状态转换
                    if r2 <= sco_new:
                        self.Nrc[idx] -= 1
                        self.Nr[idx] += 1
                    elif r2 <= lam1_new:
                        self.Nrc[idx] += 1
                        self.Nr[idx] -= 1
                    elif r2 <= lam2_new:
                        self.Nrc[idx] -= 1
                        self.Nrr[idx] += 1
                    elif r2 <= lam3_new:
                        self.Nrr[idx] -= 1
                        self.Nrc[idx] += 1
                    elif r2 <= lam4_new:
                        self.Nrr[idx] -= 1
                        self.Nri[idx] += 1
                    elif r2 <= lam5_new:
                        self.Nri[idx] -= 1
                        self.Nrr[idx] += 1
                    elif r2 <= lam6_new:
                        self.Nri[idx] -= 1
                        self.Nr[idx] += 1
                    else:
                        self.Nr[idx] -= 1
                        self.Nri[idx] += 1

                    # 重新计算速率
                    sco_new = self.Nrc[idx] * alpha[idx]
                    soc_new = self.Nr[idx] * beta
                    scr_new = self.Nrc[idx] * gamma[idx]
                    src_new = self.Nrr[idx] * self.KBMR2C
                    sri_new = self.Nrr[idx] * alpha[idx]
                    sir_new = self.Nri[idx] * beta
                    sio_new = self.Nri[idx] * delta
                    soi_new = self.Nr[idx] * gamma[idx]

                    lam1_new = sco_new + soc_new
                    lam2_new = lam1_new + scr_new
                    lam3_new = lam2_new + src_new
                    lam4_new = lam3_new + sri_new
                    lam5_new = lam4_new + sir_new
                    lam6_new = lam5_new + sio_new
                    lamtot_new = lam6_new + soi_new

                    # 更新随机时间
                    r1 = self.rng.uniform()
                    while r1 <= 0:
                        r1 = self.rng.uniform()

                    mlogr1 = -cp.log(r1)
                    wt = mlogr1 / lamtot_new
                    tloc += wt

                # 更新rfire
                self.rfire[idx] = mlogr1 + lamtot_new * (-dt + tloc - wt)

        # 更新不需要处理的CRU
        no_update_mask = ~update_mask
        self.rfire[no_update_mask] = rtest[no_update_mask]

    def get_ryr_current(self, cd, cj):
        """计算RyR电流"""
        return self.Nr * 0.000205 * (cj - cd)  # gryr = 0.000205


class GPULCC:
    """L型钙通道类"""

    # LCC参数
    CDBAR = 0.5  # 3.0
    K2 = 0.0001
    TBA = 450.0
    K1P = 0.00413
    RR1 = 0.3
    RR2 = 3.0  # 3.0
    K2P = 0.00224
    S1P = 0.00195
    S2P = S1P * (K2P / K1P) * (RR1 / RR2)

    def __init__(self, n_cru, n_lcc=10):
        """
        初始化L型钙通道

        Args:
            n_cru: CRU数量
            n_lcc: 每个CRU的LCC数量
        """
        self.n_cru = n_cru
        self.n_lcc = n_lcc

        # Markov状态
        self.Nl = cp.zeros(n_cru, dtype=cp.int32)  # 状态L
        self.Nlc = cp.zeros(n_cru, dtype=cp.int32)  # 状态LC
        self.Nlc2 = cp.full(n_cru, n_lcc, dtype=cp.int32)  # 状态LC2
        self.Nli = cp.zeros(n_cru, dtype=cp.int32)  # 状态LI
        self.Nli2 = cp.zeros(n_cru, dtype=cp.int32)  # 状态LI2
        self.Nlb1 = cp.zeros(n_cru, dtype=cp.int32)  # 状态LB1
        self.Nlb2 = cp.zeros(n_cru, dtype=cp.int32)  # 状态LB2

        # 随机数状态
        self.lfire = cp.zeros(n_cru, dtype=cp.float32)

        # 初始化随机数生成器
        self.rng = cp.random.RandomState(seed=43)

        # 初始化第一次随机抽取
        self.lfire = -cp.log(self.rng.uniform(0, 1, n_cru))

    def update_lcc(self, dt, cd, v):
        """
        更新LCC状态

        Args:
            dt: 时间步长
            cd: dyadic空间钙浓度
            v: 膜电位
        """
        # 计算速率常数
        expmv_8 = cp.exp(-v / 8.0)
        alphalcc = 1.0 / (1.0 + expmv_8)
        betalcc = expmv_8 / (1.0 + expmv_8)

        fc = 1.0 / (1.0 + cp.power(self.CDBAR / cd, 3))
        k1 = 0.03 * fc
        k3 = cp.exp(-(v + 40.0) / 3.0) / (3.0 * (1.0 + cp.exp(-(v + 40.0) / 3.0)))

        Rv = 10.0 + 4954.0 * cp.exp(v / 15.6)
        TCa = (78.0329 + 0.1 * cp.power(1.0 + cd / 0.5, 4)) / (1.0 + cp.power(cd / 0.5, 4))

        Ps = 1.0 / (1.0 + cp.exp(-(v + 40.0) / 11.32))
        Pr = 1.0 / (1.0 + cp.exp(-(v + 40.0) / 4))

        tauCa = (Rv - TCa) * Pr + TCa
        tauBa = (Rv - self.TBA) * Pr + self.TBA

        k5 = (1 - Ps) / tauCa
        k6 = fc * Ps / tauCa
        k5p = (1 - Ps) / tauBa
        k6p = Ps / tauBa
        k3p = k3
        k4 = k3 * (k1 / self.K2) * (k5 / k6) / expmv_8
        k4p = k3p * (self.K1P / self.K2P) * (k5p / k6p) / expmv_8

        s1 = 0.02 * fc
        s2 = s1 * (self.K2 / k1) * (self.RR1 / self.RR2)

        # 计算转换速率
        s12 = self.Nli2 * k4
        s13 = s12 + self.Nli2 * k5
        s21 = s13 + self.Nli * k3
        s24 = s21 + self.Nli * self.K2
        s25 = s24 + self.Nli * s2
        s31 = s25 + self.Nlc2 * k6
        s36 = s31 + self.Nlc2 * k6p
        s34 = s36 + self.Nlc2 * alphalcc
        s43 = s34 + self.Nlc * betalcc
        s42 = s43 + self.Nlc * k1
        s45 = s42 + self.Nlc * self.RR1
        s47 = s45 + self.Nlc * self.K1P
        s52 = s47 + self.Nl * s1
        s54 = s52 + self.Nl * self.RR2
        s57 = s54 + self.Nl * self.S1P
        s63 = s57 + self.Nlb2 * k5p
        s67 = s63 + self.Nlb2 * k4p
        s74 = s67 + self.Nlb1 * self.K2P
        s75 = s74 + self.Nlb1 * self.S2P
        s76 = s75 + self.Nlb1 * k3p

        # 更新状态
        rtest = self.lfire - s76 * dt

        # 找出需要更新的CRU
        update_mask = rtest <= 0
        update_indices = cp.where(update_mask)[0]

        if len(update_indices) > 0:
            # 处理需要更新的CRU
            for idx in update_indices:
                tloc = self.lfire[idx] / s76[idx]
                r2 = s76[idx] * self.rng.uniform()

                # 根据随机数决定状态转换
                if r2 <= s12[idx]:
                    self.Nli2[idx] -= 1
                    self.Nli[idx] += 1
                elif r2 <= s13[idx]:
                    self.Nli2[idx] -= 1
                    self.Nlc2[idx] += 1
                elif r2 <= s21[idx]:
                    self.Nli[idx] -= 1
                    self.Nli2[idx] += 1
                elif r2 <= s24[idx]:
                    self.Nli[idx] -= 1
                    self.Nlc[idx] += 1
                elif r2 <= s25[idx]:
                    self.Nli[idx] -= 1
                    self.Nl[idx] += 1
                elif r2 <= s31[idx]:
                    self.Nlc2[idx] -= 1
                    self.Nli2[idx] += 1
                elif r2 <= s36[idx]:
                    self.Nlc2[idx] -= 1
                    self.Nlb2[idx] += 1
                elif r2 <= s34[idx]:
                    self.Nlc2[idx] -= 1
                    self.Nlc[idx] += 1
                elif r2 <= s43[idx]:
                    self.Nlc[idx] -= 1
                    self.Nlc2[idx] += 1
                elif r2 <= s42[idx]:
                    self.Nlc[idx] -= 1
                    self.Nli[idx] += 1
                elif r2 <= s45[idx]:
                    self.Nlc[idx] -= 1
                    self.Nl[idx] += 1
                elif r2 <= s47[idx]:
                    self.Nlc[idx] -= 1
                    self.Nlb1[idx] += 1
                elif r2 <= s52[idx]:
                    self.Nl[idx] -= 1
                    self.Nli[idx] += 1
                elif r2 <= s54[idx]:
                    self.Nl[idx] -= 1
                    self.Nlc[idx] += 1
                elif r2 <= s57[idx]:
                    self.Nl[idx] -= 1
                    self.Nlb1[idx] += 1
                elif r2 <= s63[idx]:
                    self.Nlb2[idx] -= 1
                    self.Nlc2[idx] += 1
                elif r2 <= s67[idx]:
                    self.Nlb2[idx] -= 1
                    self.Nlb1[idx] += 1
                elif r2 <= s74[idx]:
                    self.Nlb1[idx] -= 1
                    self.Nlc[idx] += 1
                elif r2 <= s75[idx]:
                    self.Nlb1[idx] -= 1
                    self.Nl[idx] += 1
                else:
                    self.Nlb1[idx] -= 1
                    self.Nlb2[idx] += 1

                # 重新计算速率
                s12_new = self.Nli2[idx] * k4[idx]
                s13_new = s12_new + self.Nli2[idx] * k5[idx]
                s21_new = s13_new + self.Nli[idx] * k3[idx]
                s24_new = s21_new + self.Nli[idx] * self.K2
                s25_new = s24_new + self.Nli[idx] * s2[idx]
                s31_new = s25_new + self.Nlc2[idx] * k6[idx]
                s36_new = s31_new + self.Nlc2[idx] * k6p[idx]
                s34_new = s36_new + self.Nlc2[idx] * alphalcc[idx]
                s43_new = s34_new + self.Nlc[idx] * betalcc[idx]
                s42_new = s43_new + self.Nlc[idx] * k1[idx]
                s45_new = s42_new + self.Nlc[idx] * self.RR1
                s47_new = s45_new + self.Nlc[idx] * self.K1P
                s52_new = s47_new + self.Nl[idx] * s1[idx]
                s54_new = s52_new + self.Nl[idx] * self.RR2
                s57_new = s54_new + self.Nl[idx] * self.S1P
                s63_new = s57_new + self.Nlb2[idx] * k5p[idx]
                s67_new = s63_new + self.Nlb2[idx] * k4p[idx]
                s74_new = s67_new + self.Nlb1[idx] * self.K2P
                s75_new = s74_new + self.Nlb1[idx] * self.S2P
                s76_new = s75_new + self.Nlb1[idx] * k3p[idx]

                # 更新随机时间
                r1 = self.rng.uniform()
                mlogr1 = -cp.log(r1)
                wt = mlogr1 / s76_new
                tloc += wt

                # 处理额外的事件
                while tloc < dt:
                    r2 = s76_new * self.rng.uniform()

                    # 根据随机数决定状态转换
                    if r2 <= s12_new:
                        self.Nli2[idx] -= 1
                        self.Nli[idx] += 1
                    elif r2 <= s13_new:
                        self.Nli2[idx] -= 1
                        self.Nlc2[idx] += 1
                    elif r2 <= s21_new:
                        self.Nli[idx] -= 1
                        self.Nli2[idx] += 1
                    elif r2 <= s24_new:
                        self.Nli[idx] -= 1
                        self.Nlc[idx] += 1
                    elif r2 <= s25_new:
                        self.Nli[idx] -= 1
                        self.Nl[idx] += 1
                    elif r2 <= s31_new:
                        self.Nlc2[idx] -= 1
                        self.Nli2[idx] += 1
                    elif r2 <= s36_new:
                        self.Nlc2[idx] -= 1
                        self.Nlb2[idx] += 1
                    elif r2 <= s34_new:
                        self.Nlc2[idx] -= 1
                        self.Nlc[idx] += 1
                    elif r2 <= s43_new:
                        self.Nlc[idx] -= 1
                        self.Nlc2[idx] += 1
                    elif r2 <= s42_new:
                        self.Nlc[idx] -= 1
                        self.Nli[idx] += 1
                    elif r2 <= s45_new:
                        self.Nlc[idx] -= 1
                        self.Nl[idx] += 1
                    elif r2 <= s47_new:
                        self.Nlc[idx] -= 1
                        self.Nlb1[idx] += 1
                    elif r2 <= s52_new:
                        self.Nl[idx] -= 1
                        self.Nli[idx] += 1
                    elif r2 <= s54_new:
                        self.Nl[idx] -= 1
                        self.Nlc[idx] += 1
                    elif r2 <= s57_new:
                        self.Nl[idx] -= 1
                        self.Nlb1[idx] += 1
                    elif r2 <= s63_new:
                        self.Nlb2[idx] -= 1
                        self.Nlc2[idx] += 1
                    elif r2 <= s67_new:
                        self.Nlb2[idx] -= 1
                        self.Nlb1[idx] += 1
                    elif r2 <= s74_new:
                        self.Nlb1[idx] -= 1
                        self.Nlc[idx] += 1
                    elif r2 <= s75_new:
                        self.Nlb1[idx] -= 1
                        self.Nl[idx] += 1
                    else:
                        self.Nlb1[idx] -= 1
                        self.Nlb2[idx] += 1

                    # 重新计算速率
                    s12_new = self.Nli2[idx] * k4[idx]
                    s13_new = s12_new + self.Nli2[idx] * k5[idx]
                    s21_new = s13_new + self.Nli[idx] * k3[idx]
                    s24_new = s21_new + self.Nli[idx] * self.K2
                    s25_new = s24_new + self.Nli[idx] * s2[idx]
                    s31_new = s25_new + self.Nlc2[idx] * k6[idx]
                    s36_new = s31_new + self.Nlc2[idx] * k6p[idx]
                    s34_new = s36_new + self.Nlc2[idx] * alphalcc[idx]
                    s43_new = s34_new + self.Nlc[idx] * betalcc[idx]
                    s42_new = s43_new + self.Nlc[idx] * k1[idx]
                    s45_new = s42_new + self.Nlc[idx] * self.RR1
                    s47_new = s45_new + self.Nlc[idx] * self.K1P
                    s52_new = s47_new + self.Nl[idx] * s1[idx]
                    s54_new = s52_new + self.Nl[idx] * self.RR2
                    s57_new = s54_new + self.Nl[idx] * self.S1P
                    s63_new = s57_new + self.Nlb2[idx] * k5p[idx]
                    s67_new = s63_new + self.Nlb2[idx] * k4p[idx]
                    s74_new = s67_new + self.Nlb1[idx] * self.K2P
                    s75_new = s74_new + self.Nlb1[idx] * self.S2P
                    s76_new = s75_new + self.Nlb1[idx] * k3p[idx]

                    # 更新随机时间
                    r1 = self.rng.uniform()
                    mlogr1 = -cp.log(r1)
                    wt = mlogr1 / s76_new
                    tloc += wt

                # 更新lfire
                self.lfire[idx] = mlogr1 + s76_new * (-dt + tloc - wt)

        # 更新不需要处理的CRU
        no_update_mask = ~update_mask
        self.lfire[no_update_mask] = rtest[no_update_mask]

    def get_lcc_current(self, cd, v):
        """计算LCC电流（向量化实现）"""
        phi = v * (96485 / (8314 * 310))
        pca = 0.000913

        mask = cp.abs(phi) > 0.01
        I_ca = cp.where(
            mask,
            self.Nl * 1.0 * pca * 2.0 * phi * (0.341 * 1800.0 - cp.exp(2.0 * phi) * cd) / (cp.exp(2.0 * phi) - 1.0),
            self.Nl * 1.0 * pca * (0.341 * 1800.0 - cp.exp(2.0 * phi) * cd)
        )
        return I_ca


class GPUCRU:
    """钙释放单元类"""

    def __init__(self, n_cru, n_ryr=100, n_lcc=10):
        """
        初始化钙释放单元

        Args:
            n_cru: CRU数量
            n_ryr: 每个CRU的RyR数量
            n_lcc: 每个CRU的LCC数量
        """
        self.n_cru = n_cru

        # 小钙空间
        self.cd = cp.full(n_cru, 0.1, dtype=cp.float32)  # dyadic空间钙浓度
        self.cj = cp.full(n_cru, 1500.0, dtype=cp.float32)  # jSR钙浓度

        # 电流和通量
        self.Ilcc = cp.zeros(n_cru, dtype=cp.float32)

        # RyR受体
        self.RyRs = GPUNivalaRyR(n_cru, n_ryr)

        # LCC通道
        self.LCCs = GPULCC(n_cru, n_lcc)

        # 网络连接索引
        self.networkIdx = cp.zeros(n_cru, dtype=cp.int32)

        # TT disruption相关变量
        self.orphanedCRU = cp.zeros(n_cru, dtype=cp.bool_)
        self.orphanLCCidx = cp.full(n_cru, -1, dtype=cp.int32)

    def initialize_network_idx(self, nx, ny, nz, nx_myo):
        """
        初始化网络连接索引

        Args:
            nx, ny, nz: CRU网格维度
            nx_myo: 每个CRU对应的肌浆体素数量
        """
        # 创建索引数组
        idx = cp.arange(self.n_cru)

        # 计算3D坐标
        x = (idx % nx) * nx_myo + nx_myo // 2
        y = (idx // nx % ny) * nx_myo + nx_myo // 2
        z = (idx // (nx * ny)) * nx_myo + nx_myo // 2

        # 计算线性索引
        self.networkIdx = z * (nx * nx_myo) * (ny * nx_myo) + y * (nx * nx_myo) + x

    def update_flux(self, dt, cm, cs, v, xnai, fix_sr=False):
        """
        更新CRU通量

        Args:
            dt: 时间步长
            cm: 肌浆钙浓度（全体素向量，外部传入）
            cs: 肌浆网钙浓度（全体素向量，外部传入）
            v: 膜电位（可以是标量或长度为 n_cru 的向量）
            xnai: 细胞内钠浓度（目前未显式使用；保留接口）
            fix_sr: 是否固定SR钙浓度（True 则不更新 cj 和 cs）
        """
        # -------------------------------
        # 1) 标量 → 向量 广播（关键修复）
        # -------------------------------
        # 保证 v 是 shape=(n_cru,) 的 CuPy 向量
        if isinstance(v, (float, int, np.floating, np.integer)):
            v = cp.full(self.n_cru, float(v), dtype=cp.float32)
        elif isinstance(v, cp.ndarray):
            if v.shape == ():  # 0-dim
                v = cp.full(self.n_cru, float(v), dtype=cp.float32)
            elif v.size == 1:
                v = cp.full(self.n_cru, float(v.ravel()[0]), dtype=cp.float32)
            else:
                # 若已经是向量则确保 dtype
                v = v.astype(cp.float32, copy=False)
        elif isinstance(v, np.ndarray):
            if v.shape == () or v.size == 1:
                v = cp.full(self.n_cru, float(v.ravel()[0]), dtype=cp.float32)
            else:
                v = cp.asarray(v, dtype=cp.float32)
        else:
            # 兜底：尝试转 float 再广播
            v = cp.full(self.n_cru, float(v), dtype=cp.float32)

        # 若后续你需要用到 xnai 的逐 CRU 版本，可参照 v 的方式广播：
        # if isinstance(xnai, (float, int, np.floating, np.integer)):
        #     xnai_vec = cp.full(self.n_cru, float(xnai), dtype=cp.float32)
        # elif isinstance(xnai, cp.ndarray):
        #     if xnai.shape == () or xnai.size == 1:
        #         xnai_vec = cp.full(self.n_cru, float(xnai.ravel()[0]), dtype=cp.float32)
        #     else:
        #         xnai_vec = xnai.astype(cp.float32, copy=False)
        # elif isinstance(xnai, np.ndarray):
        #     if xnai.shape == () or xnai.size == 1:
        #         xnai_vec = cp.full(self.n_cru, float(xnai.ravel()[0]), dtype=cp.float32)
        #     else:
        #         xnai_vec = cp.asarray(xnai, dtype=cp.float32)
        # else:
        #     xnai_vec = cp.full(self.n_cru, float(xnai), dtype=cp.float32)

        # -------------------------------
        # 2) 更新 LCC 与 RyR（马尔可夫链）
        # -------------------------------
        self.LCCs.update_lcc(dt, self.cd, v)
        self.RyRs.update_ryr(dt, self.cd, self.cj)

        # -------------------------------
        # 3) 计算缓冲系数（沿网络索引采样）
        # -------------------------------
        beta_m = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bcd = 15.0f;
            const float Kcd = 13.0f;
            const float Bsr = 7.0f;
            const float Ksr = 0.3f;
            beta = 1.0f + Bcd * Kcd / ((c+Kcd)*(c+Kcd)) + Bsr * Ksr / ((c+Ksr)*(c+Ksr));
            ''',
            'beta_m_buffering'
        )(cm[self.networkIdx])

        beta_sr = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bnsr = 140.0f;
            const float Knsr = 650.0f;
            beta = 1.0f + Bnsr * Knsr / ((c+Knsr)*(c+Knsr));
            ''',
            'beta_sr_buffering'
        )(cs[self.networkIdx])

        beta_j = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bnsr = 140.0f;
            const float Knsr = 650.0f;
            beta = 1.0f + Bnsr * Knsr / ((c+Knsr)*(c+Knsr));
            ''',
            'beta_j_buffering'
        )(self.cj)

        # -------------------------------
        # 4) 通量与电流
        # -------------------------------
        # RyR 电流（由 RyR 开放数与 jSR-dyad 梯度决定）
        Iryr = self.RyRs.get_ryr_current(self.cd, self.cj)
        # LCC 电流（由通道开放数与跨膜电压/电化学梯度决定）
        self.Ilcc = self.LCCs.get_lcc_current(self.cd, v)

        # 区室体积与导通参数（与你的原始实现一致）
        Vj, Vd = 0.1, 0.00126  # jSR、dyad 体积
        gjsr, gds = 1.0, 0.303 / 0.00126

        # 通过网络索引将全域 myo/SR 场映射到每个 CRU 位置
        cs_at_cru = cs[self.networkIdx]
        cm_at_cru = cm[self.networkIdx]

        # jSR 通量、SR 总体通量（回填至 cs），以及 dyad-肌浆通量（回填至 cm）
        FluxJ = (gjsr * (cs_at_cru - self.cj) - Iryr / Vj) / beta_j
        FluxS = (-gjsr * (cs_at_cru - self.cj) * (Vj / 0.2)) / beta_sr
        FluxM = (gds * (self.cd - cm_at_cru) * (Vd / 5.0)) / beta_m

        # -------------------------------
        # 5) 状态更新（Euler 步进）
        # -------------------------------
        if not fix_sr:
            self.cj += dt * FluxJ
            cs[self.networkIdx] += dt * FluxS

        cm[self.networkIdx] += dt * FluxM

        # -------------------------------
        # 6) orphaned LCCs 处理
        # -------------------------------
        orphaned_mask = self.orphanedCRU & (self.orphanLCCidx >= 0)
        orphaned_indices = cp.where(orphaned_mask)[0]

        if orphaned_indices.size > 0:
            # 将 orphaned LCC 的电流注入到其对应的 myoplasm 体素
            # 注意：orphanLCCidx 这里应为 myo 全局索引，单位/比例与原实现一致
            for idx in orphaned_indices:
                cm[self.orphanLCCidx[idx]] += dt * self.Ilcc[idx] / 5.0  # zetam = 5.0

        # 完全移除的 LCCs，对应电流清零
        removed_mask = self.orphanedCRU & (self.orphanLCCidx < 0)
        self.Ilcc[removed_mask] = 0.0

    def to_numpy(self):
        """将GPU数组转换为NumPy数组"""
        return {
            'cd': cp.asnumpy(self.cd),
            'cj': cp.asnumpy(self.cj),
            'Ilcc': cp.asnumpy(self.Ilcc),
            'networkIdx': cp.asnumpy(self.networkIdx),
            'orphanedCRU': cp.asnumpy(self.orphanedCRU),
            'orphanLCCidx': cp.asnumpy(self.orphanLCCidx),
            'RyR_Nr': cp.asnumpy(self.RyRs.Nr),
            'RyR_Nrc': cp.asnumpy(self.RyRs.Nrc),
            'RyR_Nri': cp.asnumpy(self.RyRs.Nri),
            'RyR_Nrr': cp.asnumpy(self.RyRs.Nrr),
            'LCC_Nl': cp.asnumpy(self.LCCs.Nl),
            'LCC_Nlc': cp.asnumpy(self.LCCs.Nlc),
            'LCC_Nlc2': cp.asnumpy(self.LCCs.Nlc2),
            'LCC_Nli': cp.asnumpy(self.LCCs.Nli),
            'LCC_Nli2': cp.asnumpy(self.LCCs.Nli2),
            'LCC_Nlb1': cp.asnumpy(self.LCCs.Nlb1),
            'LCC_Nlb2': cp.asnumpy(self.LCCs.Nlb2)
        }

    def from_numpy(self, data):
        """从NumPy数组加载数据"""
        self.cd = cp.asarray(data['cd'], dtype=cp.float32)
        self.cj = cp.asarray(data['cj'], dtype=cp.float32)
        self.Ilcc = cp.asarray(data['Ilcc'], dtype=cp.float32)
        self.networkIdx = cp.asarray(data['networkIdx'], dtype=cp.int32)
        self.orphanedCRU = cp.asarray(data['orphanedCRU'], dtype=cp.bool_)
        self.orphanLCCidx = cp.asarray(data['orphanLCCidx'], dtype=cp.int32)

        # 加载RyR状态
        self.RyRs.Nr = cp.asarray(data['RyR_Nr'], dtype=cp.int32)
        self.RyRs.Nrc = cp.asarray(data['RyR_Nrc'], dtype=cp.int32)
        self.RyRs.Nri = cp.asarray(data['RyR_Nri'], dtype=cp.int32)
        self.RyRs.Nrr = cp.asarray(data['RyR_Nrr'], dtype=cp.int32)

        # 加载LCC状态
        self.LCCs.Nl = cp.asarray(data['LCC_Nl'], dtype=cp.int32)
        self.LCCs.Nlc = cp.asarray(data['LCC_Nlc'], dtype=cp.int32)
        self.LCCs.Nlc2 = cp.asarray(data['LCC_Nlc2'], dtype=cp.int32)
        self.LCCs.Nli = cp.asarray(data['LCC_Nli'], dtype=cp.int32)
        self.LCCs.Nli2 = cp.asarray(data['LCC_Nli2'], dtype=cp.int32)
        self.LCCs.Nlb1 = cp.asarray(data['LCC_Nlb1'], dtype=cp.int32)
        self.LCCs.Nlb2 = cp.asarray(data['LCC_Nlb2'], dtype=cp.int32)
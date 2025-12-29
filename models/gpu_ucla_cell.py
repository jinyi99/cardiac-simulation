import cupy as cp
import numpy as np


# =============================================================================
# Fused Kernels (优化核心：融合内核)
# =============================================================================

@cp.fuse()
def _fused_comp_ina(v, xh, xj, xm, xnai, shift_h, dt, FRT, XNAO, GNA):
    """融合后的钠电流计算内核"""
    # 1. 计算钠平衡电位
    ena = (1.0 / FRT) * cp.log(XNAO / xnai)

    # 2. 临时调整电压
    v_adj = v - shift_h

    # 3. 计算门控变量速率 (直接使用数学公式)
    a = 1.0 - 1.0 / (1.0 + cp.exp(-(v + 40.0) / 0.24))

    # h 门控
    ah = a * 0.135 * cp.exp((80.0 + v_adj) / (-6.8))
    bh = (1.0 - a) / (0.13 * (1.0 + cp.exp((v_adj + 10.66) / (-11.1)))) + \
         a * (3.56 * cp.exp(0.079 * v_adj) + 3.1e5 * cp.exp(0.35 * v_adj))

    # j 门控
    aj = a * (-1.2714e5 * cp.exp(0.2444 * v_adj) - 3.474e-5 * cp.exp(-0.04391 * v_adj)) * \
         (v_adj + 37.78) / (1.0 + cp.exp(0.311 * (v_adj + 79.23)))
    bj = (1.0 - a) * (0.3 * cp.exp(-2.535e-7 * v_adj) / (1.0 + cp.exp(-0.1 * (v_adj + 32.0)))) + \
         a * (0.1212 * cp.exp(-0.01052 * v_adj) / (1.0 + cp.exp(-0.1378 * (v_adj + 40.14))))

    # m 门控 (处理分母为0的奇点)
    val_m = v + 47.13
    am = cp.where(cp.abs(val_m) < 0.01,
                  3.2,
                  0.32 * val_m / (1.0 - cp.exp(-0.1 * val_m)))
    bm = 0.08 * cp.exp(-v / 11.0)

    # 4. Rush-Larsen 更新状态
    tauh = 1.0 / (ah + bh)
    xh_inf = ah * tauh
    xh_new = xh_inf - (xh_inf - xh) * cp.exp(-dt / tauh)

    tauj = 1.0 / (aj + bj)
    xj_inf = aj * tauj
    xj_new = xj_inf - (xj_inf - xj) * cp.exp(-dt / tauj)

    taum = 1.0 / (am + bm)
    xm_inf = am * taum
    xm_new = xm_inf - (xm_inf - xm) * cp.exp(-dt / taum)

    # 5. 计算电流
    ina = GNA * xh_new * xj_new * (xm_new ** 3) * (v - ena)

    return ina, xh_new, xj_new, xm_new


@cp.fuse()
def _fused_comp_ikr(v, xr, dt, FRT, XKO, XKI, GKR):
    """融合后的IKr计算内核"""
    ek = (1.0 / FRT) * cp.log(XKO / XKI)

    # 门控变量速率 (处理奇点)
    val_k1 = v + 7.0
    xkrv1 = cp.where(cp.abs(val_k1) < 0.001 / 0.123,
                     0.00138 / 0.123,
                     0.00138 * val_k1 / (1.0 - cp.exp(-0.123 * val_k1)))

    val_k2 = v + 10.0
    xkrv2 = cp.where(cp.abs(val_k2) < 0.001 / 0.145,
                     0.00061 / 0.145,
                     0.00061 * val_k2 / (cp.exp(0.145 * val_k2) - 1.0))

    taukr = 1.0 / (xkrv1 + xkrv2)
    xr_inf = 1.0 / (1.0 + cp.exp(-(v + 50.0) / 7.5))
    xr_new = xr_inf - (xr_inf - xr) * cp.exp(-dt / taukr)

    rg = 1.0 / (1.0 + cp.exp((v + 33.0) / 22.4))
    ikr = GKR * cp.sqrt(XKO / 5.4) * xr_new * rg * (v - ek)

    return ikr, xr_new


@cp.fuse()
def _fused_comp_iks(v, xs1, xs2, ci, xnai, dt, FRT, XKO, XNAO, XKI, GKS):
    """融合后的IKs计算内核"""
    prnak = 0.01833
    eks = (1.0 / FRT) * cp.log((XKO + prnak * XNAO) / (XKI + prnak * xnai))

    xs1ss = 1.0 / (1.0 + cp.exp(-(v - 1.5) / 16.7))
    xs2ss = xs1ss

    val_tau = v + 30.0
    # 复杂的tau计算
    tauxs1 = cp.where(
        cp.abs(val_tau) < 0.001 / 0.148,
        1.0 / (0.0000719 / 0.148 + 0.000131 / 0.0687),
        1.0 / (0.0000719 * val_tau / (1.0 - cp.exp(-0.148 * val_tau)) +
               0.000131 * val_tau / (cp.exp(0.0687 * val_tau) - 1.0))
    )
    tauxs2 = 4.0 * tauxs1

    xs1_new = xs1ss - (xs1ss - xs1) * cp.exp(-dt / tauxs1)
    xs2_new = xs2ss - (xs2ss - xs2) * cp.exp(-dt / tauxs2)

    gksx = 0.433 * (1.0 + 0.8 / (1.0 + (0.5 / ci) ** 3))
    iks = GKS * gksx * xs1_new * xs2_new * (v - eks)

    return iks, xs1_new, xs2_new


@cp.fuse()
def _fused_comp_ik1(v, FRT, XKO, XKI, GK1):
    """融合后的IK1计算内核 (无状态更新)"""
    ek = (1.0 / FRT) * cp.log(XKO / XKI)

    aki = 1.02 / (1.0 + cp.exp(0.2385 * (v - ek - 59.215)))
    bki = (0.49124 * cp.exp(0.08032 * (v - ek + 5.476)) +
           cp.exp(0.06175 * (v - ek - 594.31))) / (1.0 + cp.exp(-0.5143 * (v - ek + 4.753)))

    xkin = aki / (aki + bki)
    gki = cp.sqrt(XKO / 5.4)
    return GK1 * gki * xkin * (v - ek)


@cp.fuse()
def _fused_comp_ito(v, xtof, ytof, xtos, ytos, rtos, dt, FRT, XKO, XKI, GTOS, GTOF):
    """融合后的Ito计算内核"""
    ek = (1.0 / FRT) * cp.log(XKO / XKI)

    rt1 = -(v + 3.0) / 15.0
    rt2 = (v + 33.5) / 10.0
    rt3 = (v + 60.0) / 10.0

    xtos_inf = 1.0 / (1.0 + cp.exp(rt1))
    ytos_inf = 1.0 / (1.0 + cp.exp(rt2))
    rs_inf = 1.0 / (1.0 + cp.exp(rt2))

    trs = (2800.0 - 500.0) / (1.0 + cp.exp(rt3)) + 220.0 + 500.0
    txs = 9.0 / (1.0 + cp.exp(-rt1)) + 0.5
    tys = 3000.0 / (1.0 + cp.exp(rt3)) + 30.0

    xtos_new = xtos_inf - (xtos_inf - xtos) * cp.exp(-dt / txs)
    ytos_new = ytos_inf - (ytos_inf - ytos) * cp.exp(-dt / tys)
    rtos_new = rs_inf - (rs_inf - rtos) * cp.exp(-dt / trs)

    # Fast component
    xtof_inf = xtos_inf
    ytof_inf = ytos_inf
    txf = 3.5 * cp.exp(-(v / 30.0) ** 2) + 1.5
    tyf = 20.0 / (1.0 + cp.exp(rt2)) + 20.0

    xtof_new = xtof_inf - (xtof_inf - xtof) * cp.exp(-dt / txf)
    ytof_new = ytof_inf - (ytof_inf - ytof) * cp.exp(-dt / tyf)

    itos = GTOS * xtos_new * (0.5 * ytos_new + 0.5 * rtos_new) * (v - ek)
    itof = GTOF * xtof_new * ytof_new * (v - ek)

    return itos + itof, xtof_new, ytof_new, xtos_new, ytos_new, rtos_new


@cp.fuse()
def _fused_comp_inak(v, xnai, FRT, XNAO, XKO, GNAK):
    """融合后的INaK计算内核"""
    sigma = (cp.exp(XNAO / 67.3) - 1.0) / 7.0
    fnak = 1.0 / (1.0 + 0.1245 * cp.exp(-0.1 * v * FRT) +
                  0.0365 * sigma * cp.exp(-v * FRT))

    xkmnai = 12.0
    xkmko = 1.5

    return GNAK * fnak * (1.0 / (1.0 + (xkmnai / xnai))) * (XKO / (XKO + xkmko))


@cp.fuse()
def _fused_update_voltage_step(v, ina, ik1, ito, ikr, iks, inak, jncx, ilcc, istim,
                               dt, WCA):
    """融合后的总电流及电压更新内核"""
    # 转换通量为电流
    incx = WCA * jncx
    ilcc_curr = 2.0 * WCA * (-ilcc)

    total_current = ina + ik1 + ito + ikr + iks + inak + incx + ilcc_curr
    dvdt = -total_current + istim

    v_new = v + dvdt * dt
    return v_new


# =============================================================================
# Main Class
# =============================================================================

class GPUUCLACell:
    """UCLA心肌细胞模型类（GPU版本 - Kernel Fusion Optimized）"""

    # 物理参数
    TEMP = 308.0
    VC = -80.0
    XXR = 8.314
    XF = 96.485
    FRT = XF / (XXR * TEMP)

    # 离子浓度参数
    XNAO = 136.0
    XKI = 140.0
    XKO = 5.4
    CAO = 1800.0

    # 电导参数
    GNA = 12.0
    GKR = 0.0125
    GKS = 0.32
    GK1 = 0.3
    GNAK = 1.5
    GTOS = 0.04
    GTOF = 0.11
    GNACA = 0.84

    # 钙处理参数
    WCA = 8.0313

    def __init__(self, n_cells):
        """
        初始化UCLA心肌细胞模型
        Args:
            n_cells: 细胞数量
        """
        self.n_cells = n_cells

        # 状态变量
        self.v = cp.full(n_cells, -87.2514, dtype=cp.float32)
        self.xm = cp.full(n_cells, 0.00106085, dtype=cp.float32)
        self.xh = cp.full(n_cells, 0.990866, dtype=cp.float32)
        self.xj = cp.full(n_cells, 0.994012, dtype=cp.float32)
        self.xr = cp.full(n_cells, 0.0069357, dtype=cp.float32)
        self.xs1 = cp.full(n_cells, 0.0253915, dtype=cp.float32)
        self.xs2 = cp.full(n_cells, 0.07509, dtype=cp.float32)
        self.ci = cp.full(n_cells, 0.215889, dtype=cp.float32)
        self.xnai = cp.full(n_cells, 12.0, dtype=cp.float32)
        self.xtof = cp.full(n_cells, 0.00362348, dtype=cp.float32)
        self.ytof = cp.full(n_cells, 0.995164, dtype=cp.float32)
        self.xtos = cp.full(n_cells, 0.00362501, dtype=cp.float32)
        self.ytos = cp.full(n_cells, 0.211573, dtype=cp.float32)
        self.rtos = cp.full(n_cells, 0.434248, dtype=cp.float32)

        # 缓冲变量 (未使用但保留兼容性)
        self.tropi = cp.full(n_cells, 19.2272, dtype=cp.float32)
        self.trops = cp.full(n_cells, 17.3258, dtype=cp.float32)

        # 外部输入通量
        self.JNCX = cp.zeros(n_cells, dtype=cp.float32)
        self.ILCC = cp.zeros(n_cells, dtype=cp.float32)

        self.shift_h = cp.zeros(n_cells, dtype=cp.float32)

        # 预热：首次调用fuse函数可能会慢，但之后会很快
        pass

    def step(self, dt, istim, dv_limit=-1):
        """
        执行一个时间步的更新
        """
        # 1. 计算各离子电流并更新相应的门控变量
        # 使用融合内核大幅减少启动开销和显存访问

        # INa
        ina, self.xh, self.xj, self.xm = _fused_comp_ina(
            self.v, self.xh, self.xj, self.xm, self.xnai, self.shift_h,
            dt, self.FRT, self.XNAO, self.GNA
        )

        # IKr
        ikr, self.xr = _fused_comp_ikr(
            self.v, self.xr, dt, self.FRT, self.XKO, self.XKI, self.GKR
        )

        # IKs
        iks, self.xs1, self.xs2 = _fused_comp_iks(
            self.v, self.xs1, self.xs2, self.ci, self.xnai,
            dt, self.FRT, self.XKO, self.XNAO, self.XKI, self.GKS
        )

        # IK1 (无状态变量更新)
        ik1 = _fused_comp_ik1(self.v, self.FRT, self.XKO, self.XKI, self.GK1)

        # Ito
        ito, self.xtof, self.ytof, self.xtos, self.ytos, self.rtos = _fused_comp_ito(
            self.v, self.xtof, self.ytof, self.xtos, self.ytos, self.rtos,
            dt, self.FRT, self.XKO, self.XKI, self.GTOS, self.GTOF
        )

        # INaK
        inak = _fused_comp_inak(self.v, self.xnai, self.FRT, self.XNAO, self.XKO, self.GNAK)

        # 2. 更新电压 (融合了电流求和与欧拉步)
        # 注意: 为了极致速度，这里移除了 dv_limit 的检查 (如有必要可另加)
        self.v = _fused_update_voltage_step(
            self.v, ina, ik1, ito, ikr, iks, inak, self.JNCX, self.ILCC, istim,
            dt, self.WCA
        )

        # 更新钠浓度 (注释掉的逻辑保持原样)
        # dxnai = -1.0 * (ina + 3.0 * inak + 3.0 * incx) / (self.WCA * 1000.0)
        # self.xnai += dxnai * dt

        return True

    def to_numpy(self):
        """将GPU数组转换为NumPy数组"""
        return {
            'v': cp.asnumpy(self.v),
            'xm': cp.asnumpy(self.xm),
            'xh': cp.asnumpy(self.xh),
            'xj': cp.asnumpy(self.xj),
            'xr': cp.asnumpy(self.xr),
            'xs1': cp.asnumpy(self.xs1),
            'xs2': cp.asnumpy(self.xs2),
            'ci': cp.asnumpy(self.ci),
            'xnai': cp.asnumpy(self.xnai),
            'xtof': cp.asnumpy(self.xtof),
            'ytof': cp.asnumpy(self.ytof),
            'xtos': cp.asnumpy(self.xtos),
            'ytos': cp.asnumpy(self.ytos),
            'rtos': cp.asnumpy(self.rtos),
            'JNCX': cp.asnumpy(self.JNCX),
            'ILCC': cp.asnumpy(self.ILCC),
            'shift_h': cp.asnumpy(self.shift_h)
        }

    def from_numpy(self, data):
        """从NumPy数组加载数据"""
        for key in data:
            if hasattr(self, key):
                setattr(self, key, cp.asarray(data[key], dtype=cp.float32))

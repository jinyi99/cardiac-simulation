import cupy as cp
import numpy as np


class GPUUCLACell:
    """UCLA心肌细胞模型类（GPU版本）"""

    # 物理参数
    TEMP = 308.0  # 温度 (K)
    VC = -80.0  # 初始电压 (mV)
    XXR = 8.314  # 通用气体常数 (J/mol·K)
    XF = 96.485  # 法拉第常数 (C/mmol)
    FRT = XF / (XXR * TEMP)  # F/RT

    # 离子浓度参数
    XNAO = 136.0  # 细胞外钠浓度 (mM)
    XKI = 140.0  # 细胞内钾浓度 (mM)
    XKO = 5.4  # 细胞外钾浓度 (mM)
    CAO = 1800.0  # 细胞外钙浓度 (μM)

    # 电导参数
    GNA = 12.0  # 钠电导
    GKR = 0.0125  # 快速延迟整流钾电导
    GKS = 0.32  # 慢速延迟整流钾电导
    GK1 = 0.3  # 内向整流钾电导
    GNAK = 1.5  # Na-K泵电导
    GTOS = 0.04  # 慢速瞬时外向钾电导
    GTOF = 0.11  # 快速瞬时外向钾电导
    GNACA = 0.84  # Na-Ca交换电导

    # 其他参数
    TAUS = 4.0  # 扩散时间常数
    TAUA = 100.0  # NSR-JSR弛豫时间常数
    AV = 11.3  # SR释放-负载关系参数
    CSTAR = 90.0  # SR释放-负载关系参数
    KMNSCA = 0.45  # 非特异性钙激活电流参数
    GINSCA = 0.0  # 非特异性钙激活电流电导

    # 钙处理参数 (从空间模型获取)
    WCA = 8.0313  # 转换因子 (μM/ms 到 μA/μF)

    def __init__(self, n_cells):
        """
        初始化UCLA心肌细胞模型

        Args:
            n_cells: 细胞数量
        """
        self.n_cells = n_cells

        # 状态变量 (使用CuPy GPU数组)
        self.v = cp.full(n_cells, -87.2514, dtype=cp.float32)  # 电压 (mV)
        self.xm = cp.full(n_cells, 0.00106085, dtype=cp.float32)  # 钠m门控变量
        self.xh = cp.full(n_cells, 0.990866, dtype=cp.float32)  # 钠h门控变量
        self.xj = cp.full(n_cells, 0.994012, dtype=cp.float32)  # 钠j门控变量
        self.xr = cp.full(n_cells, 0.0069357, dtype=cp.float32)  # IKr门控变量
        self.xs1 = cp.full(n_cells, 0.0253915, dtype=cp.float32)  # IKs门控变量1
        self.xs2 = cp.full(n_cells, 0.07509, dtype=cp.float32)  # IKs门控变量2
        self.ci = cp.full(n_cells, 0.215889, dtype=cp.float32)  # 肌浆钙浓度 (μM)
        self.xnai = cp.full(n_cells, 12.0, dtype=cp.float32)  # 细胞内钠浓度 (mM)
        self.xtof = cp.full(n_cells, 0.00362348, dtype=cp.float32)  # Ito快激活
        self.ytof = cp.full(n_cells, 0.995164, dtype=cp.float32)  # Ito快失活
        self.xtos = cp.full(n_cells, 0.00362501, dtype=cp.float32)  # Ito慢激活
        self.ytos = cp.full(n_cells, 0.211573, dtype=cp.float32)  # Ito慢失活
        self.rtos = cp.full(n_cells, 0.434248, dtype=cp.float32)  # Ito慢恢复
        self.tropi = cp.full(n_cells, 19.2272, dtype=cp.float32)  # 肌浆钙蛋白缓冲
        self.trops = cp.full(n_cells, 17.3258, dtype=cp.float32)  # 膜下钙蛋白缓冲

        # 从空间模型获取的电流
        self.JNCX = cp.zeros(n_cells, dtype=cp.float32)  # NCX通量 (μM/ms)
        self.ILCC = cp.zeros(n_cells, dtype=cp.float32)  # LCC通量 (μM/ms)

        # h门控变量偏移
        self.shift_h = cp.zeros(n_cells, dtype=cp.float32)

        # 定义元素级内核用于常用计算
        self.exp_kernel = cp.ElementwiseKernel(
            'float32 x',
            'float32 y',
            'y = exp(x);',
            'exp_kernel'
        )

        self.log_kernel = cp.ElementwiseKernel(
            'float32 x',
            'float32 y',
            'y = log(x);',
            'log_kernel'
        )

    def step(self, dt, istim, dv_limit=-1):
        """
        执行一个时间步的更新

        Args:
            dt: 时间步长 (ms)
            istim: 刺激电流 (μA/μF)
            dv_limit: 电压变化限制 (mV)

        Returns:
            success: 是否成功更新
        """
        # 计算离子电流
        ina = self.comp_ina(dt)
        ik1 = self.comp_ik1()
        ito = self.comp_ito(dt)
        ikr = self.comp_ikr(dt)
        iks = self.comp_iks(dt)
        inak = self.comp_inak()

        # 从空间模型转换通量为电流
        incx = self.WCA * self.JNCX  # NCX电流 (μA/μF)
        ilcc = 2.0 * self.WCA * (-self.ILCC)  # LCC电流 (μA/μF)

        # 计算总电流和电压变化
        total_current = ina + ik1 + ito + ikr + iks + inak + incx + ilcc
        dvdt = -total_current + istim
        dv = dvdt * dt

        # 检查电压变化限制
        if dv_limit > 0:
            over_limit = cp.abs(dv) > dv_limit
            if cp.any(over_limit):
                return False  # 电压变化超过限制

        # 更新电压
        self.v += dv

        # 更新钠浓度 (简化版本，假设钠浓度基本不变)
        # dxnai = -1.0 * (ina + 3.0 * inak + 3.0 * incx) / (self.WCA * 1000.0)
        # self.xnai += dxnai * dt

        return True

    def comp_ina(self, dt):
        """计算钠电流"""
        # 计算钠平衡电位
        ena = (1.0 / self.FRT) * cp.log(self.XNAO / self.xnai)

        # 计算门控变量速率
        a = 1.0 - 1.0 / (1.0 + cp.exp(-(self.v + 40.0) / 0.24))

        # 临时调整电压以考虑h门控偏移
        v_adj = self.v - self.shift_h

        # h门控变量速率
        ah = a * 0.135 * cp.exp((80.0 + v_adj) / (-6.8))
        bh = (1.0 - a) / (0.13 * (1 + cp.exp((v_adj + 10.66) / (-11.1)))) + \
             a * (3.56 * cp.exp(0.079 * v_adj) + 3.1e5 * cp.exp(0.35 * v_adj))

        # j门控变量速率
        aj = a * (-1.2714e5 * cp.exp(0.2444 * v_adj) - 3.474e-5 * cp.exp(-0.04391 * v_adj)) * \
             (v_adj + 37.78) / (1.0 + cp.exp(0.311 * (v_adj + 79.23)))
        bj = (1.0 - a) * (0.3 * cp.exp(-2.535e-7 * v_adj) / (1 + cp.exp(-0.1 * (v_adj + 32)))) + \
             a * (0.1212 * cp.exp(-0.01052 * v_adj) / (1 + cp.exp(-0.1378 * (v_adj + 40.14))))

        # m门控变量速率
        am = cp.where(
            cp.abs(self.v + 47.13) < 0.01,
            cp.full(self.n_cells, 3.2, dtype=cp.float32),
            0.32 * (self.v + 47.13) / (1.0 - cp.exp(-0.1 * (self.v + 47.13)))
        )
        bm = 0.08 * cp.exp(-self.v / 11.0)

        # 使用Rush-Larsen方法更新门控变量
        tauh = 1.0 / (ah + bh)
        xh_inf = ah * tauh
        self.xh = xh_inf - (xh_inf - self.xh) * cp.exp(-dt / tauh)

        tauj = 1.0 / (aj + bj)
        xj_inf = aj * tauj
        self.xj = xj_inf - (xj_inf - self.xj) * cp.exp(-dt / tauj)

        taum = 1.0 / (am + bm)
        xm_inf = am * taum
        self.xm = xm_inf - (xm_inf - self.xm) * cp.exp(-dt / taum)

        # 计算钠电流
        return self.GNA * self.xh * self.xj * self.xm ** 3 * (self.v - ena)

    def comp_ikr(self, dt):
        """计算快速延迟整流钾电流"""
        # 计算钾平衡电位
        ek = (1.0 / self.FRT) * cp.log(self.XKO / self.XKI)

        # 计算门控变量速率
        xkrv1 = cp.where(
            cp.abs(self.v + 7.0) < 0.001 / 0.123,
            cp.full(self.n_cells, 0.00138 / 0.123, dtype=cp.float32),
            0.00138 * (self.v + 7.0) / (1.0 - cp.exp(-0.123 * (self.v + 7.0)))
        )

        xkrv2 = cp.where(
            cp.abs(self.v + 10.0) < 0.001 / 0.145,
            cp.full(self.n_cells, 0.00061 / 0.145, dtype=cp.float32),
            0.00061 * (self.v + 10.0) / (cp.exp(0.145 * (self.v + 10.0)) - 1.0)
        )

        # 使用Rush-Larsen方法更新门控变量
        taukr = 1.0 / (xkrv1 + xkrv2)
        xr_inf = 1.0 / (1.0 + cp.exp(-(self.v + 50.0) / 7.5))
        self.xr = xr_inf - (xr_inf - self.xr) * cp.exp(-dt / taukr)

        # 计算IKr电流
        rg = 1.0 / (1.0 + cp.exp((self.v + 33.0) / 22.4))
        return self.GKR * cp.sqrt(self.XKO / 5.4) * self.xr * rg * (self.v - ek)

    def comp_iks(self, dt):
        """计算慢速延迟整流钾电流"""
        # 计算钾平衡电位 (考虑钠渗透性)
        prnak = 0.01833
        eks = (1.0 / self.FRT) * cp.log((self.XKO + prnak * self.XNAO) / (self.XKI + prnak * self.xnai))

        # 计算门控变量速率
        xs1ss = 1.0 / (1.0 + cp.exp(-(self.v - 1.5) / 16.7))
        xs2ss = xs1ss

        # 计算时间常数
        tauxs1 = cp.where(
            cp.abs(self.v + 30.0) < 0.001 / 0.148,
            cp.full(self.n_cells, 1.0 / (0.0000719 / 0.148 + 0.000131 / 0.0687), dtype=cp.float32),
            1.0 / (0.0000719 * (self.v + 30.0) / (1.0 - cp.exp(-0.148 * (self.v + 30.0))) +
                   0.000131 * (self.v + 30.0) / (cp.exp(0.0687 * (self.v + 30.0)) - 1.0))
        )

        tauxs2 = 4.0 * tauxs1

        # 使用Rush-Larsen方法更新门控变量
        self.xs1 = xs1ss - (xs1ss - self.xs1) * cp.exp(-dt / tauxs1)
        self.xs2 = xs2ss - (xs2ss - self.xs2) * cp.exp(-dt / tauxs2)

        # 计算钙依赖性电导缩放
        gksx = 0.433 * (1.0 + 0.8 / (1.0 + (0.5 / self.ci) ** 3))

        # 计算IKs电流
        return self.GKS * gksx * self.xs1 * self.xs2 * (self.v - eks)

    def comp_ik1(self):
        """计算内向整流钾电流"""
        # 计算钾平衡电位
        ek = (1.0 / self.FRT) * cp.log(self.XKO / self.XKI)

        # 计算门控变量
        aki = 1.02 / (1.0 + cp.exp(0.2385 * (self.v - ek - 59.215)))
        bki = (0.49124 * cp.exp(0.08032 * (self.v - ek + 5.476)) +
               cp.exp(0.06175 * (self.v - ek - 594.31))) / (1.0 + cp.exp(-0.5143 * (self.v - ek + 4.753)))

        xkin = aki / (aki + bki)

        # 计算IK1电流
        gki = cp.sqrt(self.XKO / 5.4)  # 电导缩放
        return self.GK1 * gki * xkin * (self.v - ek)

    def comp_ito(self, dt):
        """计算瞬时外向钾电流"""
        # 计算钾平衡电位
        ek = (1.0 / self.FRT) * cp.log(self.XKO / self.XKI)

        # 计算Ito慢速成分
        rt1 = -(self.v + 3.0) / 15.0
        rt2 = (self.v + 33.5) / 10.0
        rt3 = (self.v + 60.0) / 10.0

        xtos_inf = 1.0 / (1.0 + cp.exp(rt1))
        ytos_inf = 1.0 / (1.0 + cp.exp(rt2))
        rs_inf = 1.0 / (1.0 + cp.exp(rt2))

        trs = (2800.0 - 500.0) / (1.0 + cp.exp(rt3)) + 220.0 + 500.0
        txs = 9.0 / (1.0 + cp.exp(-rt1)) + 0.5
        tys = 3000.0 / (1.0 + cp.exp(rt3)) + 30.0

        # 使用Rush-Larsen方法更新门控变量
        self.xtos = xtos_inf - (xtos_inf - self.xtos) * cp.exp(-dt / txs)
        self.ytos = ytos_inf - (ytos_inf - self.ytos) * cp.exp(-dt / tys)
        self.rtos = rs_inf - (rs_inf - self.rtos) * cp.exp(-dt / trs)

        # 计算Ito快速成分
        xtof_inf = xtos_inf
        ytof_inf = ytos_inf

        txf = 3.5 * cp.exp(-(self.v / 30.0) ** 2) + 1.5
        tyf = 20.0 / (1.0 + cp.exp(rt2)) + 20.0

        # 使用Rush-Larsen方法更新门控变量
        self.xtof = xtof_inf - (xtof_inf - self.xtof) * cp.exp(-dt / txf)
        self.ytof = ytof_inf - (ytof_inf - self.ytof) * cp.exp(-dt / tyf)

        # 计算Ito电流
        itos = self.GTOS * self.xtos * (0.5 * self.ytos + 0.5 * self.rtos) * (self.v - ek)
        itof = self.GTOF * self.xtof * self.ytof * (self.v - ek)

        return itos + itof

    def comp_inak(self):
        """计算Na-K泵电流"""
        # 计算sigma因子
        sigma = (cp.exp(self.XNAO / 67.3) - 1.0) / 7.0

        # 计算电压依赖性因子
        fnak = 1.0 / (1.0 + 0.1245 * cp.exp(-0.1 * self.v * self.FRT) +
                      0.0365 * sigma * cp.exp(-self.v * self.FRT))

        # 计算INaK电流
        xkmnai = 12.0  # 细胞内钠半最大激活常数
        xkmko = 1.5  # 细胞外钾半最大激活常数

        return self.GNAK * fnak * (1.0 / (1.0 + (xkmnai / self.xnai))) * \
            (self.XKO / (self.XKO + xkmko))

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
            'tropi': cp.asnumpy(self.tropi),
            'trops': cp.asnumpy(self.trops),
            'JNCX': cp.asnumpy(self.JNCX),
            'ILCC': cp.asnumpy(self.ILCC),
            'shift_h': cp.asnumpy(self.shift_h)
        }

    def from_numpy(self, data):
        """从NumPy数组加载数据"""
        self.v = cp.asarray(data['v'], dtype=cp.float32)
        self.xm = cp.asarray(data['xm'], dtype=cp.float32)
        self.xh = cp.asarray(data['xh'], dtype=cp.float32)
        self.xj = cp.asarray(data['xj'], dtype=cp.float32)
        self.xr = cp.asarray(data['xr'], dtype=cp.float32)
        self.xs1 = cp.asarray(data['xs1'], dtype=cp.float32)
        self.xs2 = cp.asarray(data['xs2'], dtype=cp.float32)
        self.ci = cp.asarray(data['ci'], dtype=cp.float32)
        self.xnai = cp.asarray(data['xnai'], dtype=cp.float32)
        self.xtof = cp.asarray(data['xtof'], dtype=cp.float32)
        self.ytof = cp.asarray(data['ytof'], dtype=cp.float32)
        self.xtos = cp.asarray(data['xtos'], dtype=cp.float32)
        self.ytos = cp.asarray(data['ytos'], dtype=cp.float32)
        self.rtos = cp.asarray(data['rtos'], dtype=cp.float32)
        self.tropi = cp.asarray(data['tropi'], dtype=cp.float32)
        self.trops = cp.asarray(data['trops'], dtype=cp.float32)
        self.JNCX = cp.asarray(data['JNCX'], dtype=cp.float32)
        self.ILCC = cp.asarray(data['ILCC'], dtype=cp.float32)
        self.shift_h = cp.asarray(data['shift_h'], dtype=cp.float32)
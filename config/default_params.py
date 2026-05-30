"""
默认参数配置文件
包含心肌细胞电生理-钙循环模型的所有默认参数
"""

import cupy as cp
import numpy as np


class DefaultParams:
    """默认参数类"""

    # 时间参数
    DT = 0.01  # 时间步长 (ms)
    OUTPUT_DT = 1.0  # 输出间隔 (ms)
    SIMULATION_DURATION = 600.0  # 默认模拟时长 (ms)

    # 空间参数
    NX = 10  # X方向CRU数量
    NY = 20  # Y方向CRU数量
    NZ = 10  # Z方向CRU数量
    NX_MYO = 5  # 每个CRU对应的肌浆体素数量
    DX = 0.2  # 空间步长 (μm)

    # 扩散参数
    D_M = 0.3  # 肌浆扩散系数 (μm²/ms)
    D_SR = 0.06  # 肌浆网扩散系数 (μm²/ms)

    # 物理常数
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

    # 钙处理参数
    WCA = 8.0313  # 转换因子 (μM/ms 到 μA/μF)
    VUP = 0.32  # SERCA泵最大速率
    KUP = 1.0  # SERCA泵半最大激活常数
    GLEAK = 0.000004  # SR泄漏电导
    GBG = 0.00003  # 背景钙电导

    # CRU参数
    N_RYR = 100  # 每个CRU的RyR数量
    N_LCC = 10  # 每个CRU的LCC数量
    G_RYR = 0.000205  # RyR电导
    G_LCC = 1.0  # LCC电导

    # NCX参数
    V2 = 0.8  # NCX最大速率
    KMCA = 0.11  # 钙半最大激活常数
    KMNA = 12.3  # 钠半最大激活常数
    KMCO = 1.3  # 细胞外钙半最大激活常数
    KMNO = 87.5  # 细胞外钠半最大激活常数

    # 体积参数
    VJ = 0.1  # jSR体积
    VD = 0.00126  # dyadic空间体积
    ZETAS = 0.2  # SR体积分数
    ZETAM = 5.0  # 肌浆体积分数

    # 电导参数
    GDS = 0.303 / 0.00126  # dyadic-肌浆电导
    GJSR = 1.0  # SR-jSR电导

    # LCC参数
    CDBAR = 0.5  # LCC钙依赖性参数
    K2 = 0.0001  # LCC速率常数
    TBA = 450.0  # LCC钡时间常数
    K1P = 0.00413  # LCC速率常数
    RR1 = 0.3  # LCC速率常数
    RR2 = 6.0  # LCC速率常数    3.0
    K2P = 0.00224  # LCC速率常数
    S1P = 0.00195  # LCC速率常数
    S2P = S1P * (K2P / K1P) * (RR1 / RR2)  # LCC速率常数

    # RyR参数
    KAP = 0.7 * 0.005  # RyR速率常数
    KAM = 1.0  # RyR速率常数
    KBM = 0.003  # RyR速率常数
    KBMR2C = 1.0 * KBM  # RyR速率常数
    KBP = 0.00075  # RyR速率常数

    # 刺激参数
    STIM_AMPLITUDE = 40.0  # 刺激幅度 (μA/μF)
    STIM_DURATION = 2.0  # 刺激持续时间 (ms)
    STIM_START = 10.0  # 刺激开始时间 (ms)

    # 初始条件
    INITIAL_V = -87.2514  # 初始电压 (mV)
    INITIAL_CM = 0.1  # 初始肌浆钙浓度 (μM)
    INITIAL_CS = 1500.0  # 初始肌浆网钙浓度 (μM)
    INITIAL_CD = 0.1  # 初始dyadic空间钙浓度 (μM)
    INITIAL_CJ = 1500.0  # 初始jSR钙浓度 (μM)
    INITIAL_XNAI = 12.0  # 初始细胞内钠浓度 (mM)

    # 随机数种子
    RNG_SEED = 1111

    # 输出配置
    SAVE_RESULTS = True
    SAVE_INTERVAL = 100  # 保存间隔 (ms)
    PLOT_RESULTS = True
    SAVE_PLOTS = True

    @classmethod
    def get_all_params(cls):
        """获取所有参数字典"""
        params = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                params[attr] = getattr(cls, attr)
        return params

    @classmethod
    def update_params(cls, param_dict):
        """
        更新参数

        Args:
            param_dict: 参数字典
        """
        for key, value in param_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"警告: 参数 {key} 不存在，已忽略")

    @classmethod
    def get_cell_params(cls):
        """获取细胞参数"""
        return {
            'nx': cls.NX,
            'ny': cls.NY,
            'nz': cls.NZ,
            'dt': cls.DT,
            'D_m': cls.D_M,
            'D_sr': cls.D_SR,
            'dx': cls.DX
        }

    @classmethod
    def get_ion_channel_params(cls):
        """获取离子通道参数"""
        return {
            'g_na': cls.GNA,
            'g_kr': cls.GKR,
            'g_ks': cls.GKS,
            'g_k1': cls.GK1,
            'g_nak': cls.GNAK,
            'g_tos': cls.GTOS,
            'g_tof': cls.GTOF,
            'g_naca': cls.GNACA
        }

    @classmethod
    def get_calcium_params(cls):
        """获取钙处理参数"""
        return {
            'vup': cls.VUP,
            'kup': cls.KUP,
            'gleak': cls.GLEAK,
            'gbg': cls.GBG,
            'v2': cls.V2,
            'kmca': cls.KMCA,
            'kmna': cls.KMNA,
            'kmco': cls.KMCO,
            'kmno': cls.KMNO
        }

    @classmethod
    def get_cru_params(cls):
        """获取CRU参数"""
        return {
            'n_ryr': cls.N_RYR,
            'n_lcc': cls.N_LCC,
            'g_ryr': cls.G_RYR,
            'g_lcc': cls.G_LCC,
            'vj': cls.VJ,
            'vd': cls.VD,
            'zetas': cls.ZETAS,
            'zetam': cls.ZETAM,
            'gds': cls.GDS,
            'gjsr': cls.GJSR
        }

    @classmethod
    def get_stim_params(cls):
        """获取刺激参数"""
        return {
            'stim_amplitude': cls.STIM_AMPLITUDE,
            'stim_duration': cls.STIM_DURATION,
            'stim_start': cls.STIM_START
        }

    @classmethod
    def get_initial_conditions(cls):
        """获取初始条件"""
        return {
            'initial_v': cls.INITIAL_V,
            'initial_cm': cls.INITIAL_CM,
            'initial_cs': cls.INITIAL_CS,
            'initial_cd': cls.INITIAL_CD,
            'initial_cj': cls.INITIAL_CJ,
            'initial_xnai': cls.INITIAL_XNAI
        }
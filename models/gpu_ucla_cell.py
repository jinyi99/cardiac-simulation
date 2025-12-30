import cupy as cp

# =============================================================================
# Single Fused Kernel for Whole Cell Model (Complete Physics)
# =============================================================================
# 将所有离子电流计算和状态更新合并到一个 C++ 内核中，极大减少 GPU 启动开销

ucla_kernel_source = r'''
extern "C" __global__ void update_ucla_cell_kernel(
    float* v, float* xm, float* xh, float* xj, 
    float* xr, float* xs1, float* xs2, 
    float* xtof, float* ytof, float* xtos, float* ytos, float* rtos,
    float* xnai, float* ci,
    const float* Jncx_ptr, const float* Ilcc_ptr,
    const float istim, const float dt,
    const float FRT, const float XNAO, const float XKI, const float XKO,
    const float GNA, const float GKR, const float GKS, const float GK1, 
    const float GNAK, const float GTOS, const float GTOF, const float WCA
) {
    // 单细胞模型，通常只有 idx=0 有效
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0) return; 

    // 读取当前状态
    float _v = v[idx];
    float _xnai = xnai[idx];
    float _ci = ci[idx];

    // =================================================================
    // 1. INa (Sodium Current)
    // =================================================================
    float ena = (1.0f / FRT) * logf(XNAO / _xnai);

    // m-gate
    float val_m = _v + 47.13f;
    float am = (abs(val_m) < 0.01f) ? 3.2f : (0.32f * val_m / (1.0f - expf(-0.1f * val_m)));
    float bm = 0.08f * expf(-_v / 11.0f);

    // h-gate
    float ah, bh;
    if (_v >= -40.0f) {
        ah = 0.0f;
        bh = 1.0f / (0.13f * (1.0f + expf((_v + 10.66f) / -11.1f)));
    } else {
        ah = 0.135f * expf((80.0f + _v) / -6.8f);
        bh = 3.56f * expf(0.079f * _v) + 3.1e5f * expf(0.35f * _v);
    }

    // j-gate
    float aj, bj;
    if (_v >= -40.0f) {
        aj = 0.0f;
        bj = 0.3f * expf(-2.535e-7f * _v) / (1.0f + expf(-0.1f * (_v + 32.0f)));
    } else {
        aj = (-1.2714e5f * expf(0.2444f * _v) - 3.474e-5f * expf(-0.04391f * _v)) * (_v + 37.78f) / (1.0f + expf(0.311f * (_v + 79.23f)));
        bj = 0.1212f * expf(-0.01052f * _v) / (1.0f + expf(-0.1378f * (_v + 40.14f)));
    }

    // Rush-Larsen Update for INa gates
    float taum = 1.0f / (am + bm); 
    float minf = am * taum; 
    float xm_new = minf - (minf - xm[idx]) * expf(-dt / taum);

    float tauh = 1.0f / (ah + bh); 
    float hinf = ah * tauh; 
    float xh_new = hinf - (hinf - xh[idx]) * expf(-dt / tauh);

    float tauj = 1.0f / (aj + bj); 
    float jinf = aj * tauj; 
    float xj_new = jinf - (jinf - xj[idx]) * expf(-dt / tauj);

    float ina = GNA * xh_new * xj_new * xm_new * xm_new * xm_new * (_v - ena);

    // =================================================================
    // 2. IKr (Rapid Delayed Rectifier Potassium Current)
    // =================================================================
    float ek = (1.0f / FRT) * logf(XKO / XKI);

    float val_k1 = _v + 7.0f;
    float xkrv1 = (abs(val_k1) < 0.001f) ? (0.00138f / 0.123f) : (0.00138f * val_k1 / (1.0f - expf(-0.123f * val_k1)));

    float val_k2 = _v + 10.0f;
    float xkrv2 = (abs(val_k2) < 0.001f) ? (0.00061f / 0.145f) : (0.00061f * val_k2 / (expf(0.145f * val_k2) - 1.0f));

    float taukr = 1.0f / (xkrv1 + xkrv2);
    float xrinf = 1.0f / (1.0f + expf(-(_v + 50.0f) / 7.5f));
    float xr_new = xrinf - (xrinf - xr[idx]) * expf(-dt / taukr);

    float rg = 1.0f / (1.0f + expf((_v + 33.0f) / 22.4f));
    float ikr = GKR * sqrtf(XKO / 5.4f) * xr_new * rg * (_v - ek);

    // =================================================================
    // 3. IKs (Slow Delayed Rectifier Potassium Current)
    // =================================================================
    float prnak = 0.01833f;
    float eks = (1.0f / FRT) * logf((XKO + prnak * XNAO) / (XKI + prnak * _xnai));

    float xs1ss = 1.0f / (1.0f + expf(-(_v - 1.5f) / 16.7f));
    float val_tau = _v + 30.0f;
    float tauxs1;
    if (abs(val_tau) < 0.001f) {
        tauxs1 = 1.0f / (0.0000719f / 0.148f + 0.000131f / 0.0687f);
    } else {
        tauxs1 = 1.0f / (0.0000719f * val_tau / (1.0f - expf(-0.148f * val_tau)) + 
                         0.000131f * val_tau / (expf(0.0687f * val_tau) - 1.0f));
    }
    float tauxs2 = 4.0f * tauxs1;

    float xs1_new = xs1ss - (xs1ss - xs1[idx]) * expf(-dt / tauxs1);
    float xs2_new = xs1ss - (xs1ss - xs2[idx]) * expf(-dt / tauxs2); 

    float gksx = 0.433f * (1.0f + 0.8f / (1.0f + powf(0.5f / _ci, 3.0f)));
    float iks = GKS * gksx * xs1_new * xs2_new * (_v - eks);

    // =================================================================
    // 4. IK1 (Inward Rectifier Potassium Current)
    // =================================================================
    float aki = 1.02f / (1.0f + expf(0.2385f * (_v - ek - 59.215f)));
    float bki = (0.49124f * expf(0.08032f * (_v - ek + 5.476f)) + 
                 expf(0.06175f * (_v - ek - 594.31f))) / 
                (1.0f + expf(-0.5143f * (_v - ek + 4.753f)));
    float xkin = aki / (aki + bki);
    float ik1 = GK1 * sqrtf(XKO / 5.4f) * xkin * (_v - ek);

    // =================================================================
    // 5. INaK (Sodium-Potassium Pump Current)
    // =================================================================
    float sigma = (expf(XNAO / 67.3f) - 1.0f) / 7.0f;
    float fnak = 1.0f / (1.0f + 0.1245f * expf(-0.1f * _v * FRT) + 0.0365f * sigma * expf(-_v * FRT));
    float inak = GNAK * fnak * (1.0f / (1.0f + powf(12.0f / _xnai, 1.0f))) * (XKO / (XKO + 1.5f));

    // =================================================================
    // 6. Ito (Transient Outward Potassium Current)
    // =================================================================
    float rt1 = -(_v + 3.0f) / 15.0f;
    float rt2 = (_v + 33.5f) / 10.0f;
    float rt3 = (_v + 60.0f) / 10.0f;

    float xtos_inf = 1.0f / (1.0f + expf(rt1));
    float ytos_inf = 1.0f / (1.0f + expf(rt2));
    float rs_inf = 1.0f / (1.0f + expf(rt2)); 

    float trs = (2800.0f - 500.0f) / (1.0f + expf(rt3)) + 720.0f;
    float txs = 9.0f / (1.0f + expf(-rt1)) + 0.5f;
    float tys = 3000.0f / (1.0f + expf(rt3)) + 30.0f;

    float xtos_new = xtos_inf - (xtos_inf - xtos[idx]) * expf(-dt / txs);
    float ytos_new = ytos_inf - (ytos_inf - ytos[idx]) * expf(-dt / tys);
    float rtos_new = rs_inf - (rs_inf - rtos[idx]) * expf(-dt / trs);

    // Fast component (Ito,f)
    float xtof_inf = xtos_inf;
    float ytof_inf = ytos_inf;
    float txf = 3.5f * expf(-powf(_v / 30.0f, 2.0f)) + 1.5f;
    float tyf = 20.0f / (1.0f + expf(rt2)) + 20.0f;

    float xtof_new = xtof_inf - (xtof_inf - xtof[idx]) * expf(-dt / txf);
    float ytof_new = ytof_inf - (ytof_inf - ytof[idx]) * expf(-dt / tyf);

    float itos = GTOS * xtos_new * (0.5f * ytos_new + 0.5f * rtos_new) * (_v - ek);
    float itof = GTOF * xtof_new * ytof_new * (_v - ek);
    float ito = itos + itof;

    // =================================================================
    // 7. Total Current & Voltage Update
    // =================================================================
    float jncx = Jncx_ptr[0]; 
    float ilcc = Ilcc_ptr[0];
    float incx = WCA * jncx;
    float ilcc_curr = 2.0f * WCA * (-ilcc);

    float itotal = ina + ikr + iks + ik1 + inak + incx + ilcc_curr + ito;

    // Euler update for voltage
    v[idx] = _v + dt * (-itotal + istim);

    // =================================================================
    // 8. Write back states
    // =================================================================
    xm[idx] = xm_new; xh[idx] = xh_new; xj[idx] = xj_new;
    xr[idx] = xr_new; xs1[idx] = xs1_new; xs2[idx] = xs2_new;
    xtof[idx] = xtof_new; ytof[idx] = ytof_new;
    xtos[idx] = xtos_new; ytos[idx] = ytos_new; rtos[idx] = rtos_new;
}
'''

# 编译内核
ucla_kernel = cp.RawKernel(ucla_kernel_source, 'update_ucla_cell_kernel')


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

    def step(self, dt, istim):
        """
        执行一个时间步的更新
        """
        # --- 关键修正：确保标量是 Python float 类型，避免 CuPy Implicit conversion 错误 ---
        val_istim = float(istim) if not isinstance(istim, cp.ndarray) else float(istim.item())
        val_dt = float(dt) if not isinstance(dt, cp.ndarray) else float(dt.item())

        # 调用单一融合内核
        ucla_kernel((1,), (1,), (
            self.v, self.xm, self.xh, self.xj,
            self.xr, self.xs1, self.xs2,
            self.xtof, self.ytof, self.xtos, self.ytos, self.rtos,
            self.xnai, self.ci,
            self.JNCX, self.ILCC,
            cp.float32(val_istim), cp.float32(val_dt),
            cp.float32(self.FRT), cp.float32(self.XNAO), cp.float32(self.XKI), cp.float32(self.XKO),
            cp.float32(self.GNA), cp.float32(self.GKR), cp.float32(self.GKS), cp.float32(self.GK1),
            cp.float32(self.GNAK), cp.float32(self.GTOS), cp.float32(self.GTOF), cp.float32(self.WCA)
        ))

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
            'ILCC': cp.asnumpy(self.ILCC)
        }

    def from_numpy(self, data):
        """从NumPy数组加载数据"""
        for key in data:
            if hasattr(self, key):
                setattr(self, key, cp.asarray(data[key], dtype=cp.float32))

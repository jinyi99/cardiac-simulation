import cupy as cp
import numpy as np

# =============================================================================
# CUDA Kernels for Stochastic Gating & Flux Calculation
# =============================================================================

# 随机数生成器辅助函数 (XORWOW)
rng_device_func = r'''
__device__ float next_rand(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)x / 4294967296.0f;
}
'''

# --- 1. RyR Update Kernel (Stochastic) ---
ryr_kernel_source = rng_device_func + r'''
extern "C" __global__ void update_ryr_kernel(
    int* Nr, int* Nrc, int* Nri, int* Nrr,
    float* rfire,
    const float* cd, const float* cj,
    const float KAP, const float KAM, const float KBP, const float KBM, const float KBMR2C,
    const float dt, const int n_cru, const unsigned int seed_base
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cru) return;

    unsigned int rng_state = seed_base + idx;
    next_rand(&rng_state);

    // Load State
    int nr = Nr[idx];
    int nrc = Nrc[idx];
    int nri = Nri[idx];
    int nrr = Nrr[idx];
    float current_rfire = rfire[idx];
    float _cd = cd[idx];

    // Compute Rates locally (Vectorized inside kernel)
    float a = KAP * _cd * _cd;
    float b = KAM;
    float g = KBP * _cd;
    float d = KBM;
    float k_src = KBMR2C;

    float t_elapsed = 0.0f;
    while (t_elapsed < dt) {
        float sco = nrc * a;
        float soc = nr * b;
        float scr = nrc * g;
        float src = nrr * k_src;
        float sri = nrr * a;
        float sir = nri * b;
        float sio = nri * d;
        float soi = nr * g;

        float lamtot = sco + soc + scr + src + sri + sir + sio + soi;

        if (lamtot <= 1e-6f) {
            t_elapsed = dt;
            break;
        }

        float dt_step = dt - t_elapsed;
        if (current_rfire > lamtot * dt_step) {
            current_rfire -= lamtot * dt_step;
            t_elapsed = dt;
        } else {
            t_elapsed += current_rfire / lamtot;
            float r2 = lamtot * next_rand(&rng_state);

            // Branching logic for state transitions
            float acc = sco;
            if (r2 <= acc) { nrc--; nr++; }
            else {
                acc += soc;
                if (r2 <= acc) { nrc++; nr--; }
                else {
                    acc += scr;
                    if (r2 <= acc) { nrc--; nrr++; }
                    else {
                        acc += src;
                        if (r2 <= acc) { nrr--; nrc++; }
                        else {
                            acc += sri;
                            if (r2 <= acc) { nrr--; nri++; }
                            else {
                                acc += sir;
                                if (r2 <= acc) { nri--; nrr++; }
                                else {
                                    acc += sio;
                                    if (r2 <= acc) { nri--; nr++; }
                                    else { nr--; nri++; }
                                }
                            }
                        }
                    }
                }
            }

            float u = next_rand(&rng_state);
            while(u <= 0.0f || u >= 1.0f) u = next_rand(&rng_state);
            current_rfire = -logf(u);
        }
    }

    Nr[idx] = nr;
    Nrc[idx] = nrc;
    Nri[idx] = nri;
    Nrr[idx] = nrr;
    rfire[idx] = current_rfire;
}
'''

# --- 2. LCC Update Kernel (Stochastic) ---
lcc_kernel_source = rng_device_func + r'''
extern "C" __global__ void update_lcc_kernel(
    int* Nli2, int* Nli, int* Nlc2, int* Nlc, int* Nl, int* Nlb2, int* Nlb1,
    float* lfire,
    const float* cd, const float v_val,
    const float dt, const int n_cru, const unsigned int seed_base
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cru) return;

    unsigned int rng_state = seed_base + idx;
    next_rand(&rng_state);

    // Load State
    int n1 = Nli2[idx]; int n2 = Nli[idx]; int n3 = Nlc2[idx]; int n4 = Nlc[idx];
    int n5 = Nl[idx]; int n6 = Nlb2[idx]; int n7 = Nlb1[idx];
    float current_lfire = lfire[idx];
    float _cd = cd[idx];

    // --- Compute Parameters Locally (Avoid global memory reads for arrays) ---
    const float CDBAR = 0.5f;
    const float K2 = 0.0001f;
    const float TBA = 450.0f;
    const float K1P = 0.00413f;
    const float RR1 = 0.3f;
    const float RR2 = 3.0f;
    const float K2P = 0.00224f;
    const float S1P = 0.00195f;
    const float S2P = 0.00195f * (0.00224f / 0.00413f) * (0.3f / 3.0f);

    float expmv_8 = expf(-v_val / 8.0f);
    float alphalcc = 1.0f / (1.0f + expmv_8);
    float betalcc = expmv_8 / (1.0f + expmv_8);

    float fc = 1.0f / (1.0f + powf(CDBAR / _cd, 3.0f));
    float k1 = 0.03f * fc;
    float k3 = expf(-(v_val + 40.0f) / 3.0f) / (3.0f * (1.0f + expf(-(v_val + 40.0f) / 3.0f)));

    float Rv = 10.0f + 4954.0f * expf(v_val / 15.6f);
    float TCa = (78.0329f + 0.1f * powf(1.0f + _cd / 0.5f, 4.0f)) / (1.0f + powf(_cd / 0.5f, 4.0f));

    float Ps = 1.0f / (1.0f + expf(-(v_val + 40.0f) / 11.32f));
    float Pr = 1.0f / (1.0f + expf(-(v_val + 40.0f) / 4.0f));

    float tauCa = (Rv - TCa) * Pr + TCa;
    float tauBa = (Rv - TBA) * Pr + TBA;

    float k5 = (1.0f - Ps) / tauCa;
    float k6 = fc * Ps / tauCa;
    float k5p = (1.0f - Ps) / tauBa;
    float k6p = Ps / tauBa;

    float k4 = k3 * (k1 / K2) * (k5 / k6) / expmv_8;
    float k4p = k3 * (K1P / K2P) * (k5p / k6p) / expmv_8; // k3p = k3

    float s1 = 0.02f * fc;
    float s2 = s1 * (K2 / k1) * (RR1 / RR2);

    float t_elapsed = 0.0f;
    while (t_elapsed < dt) {
        float lamtot = n1*(k4 + k5) + n2*(k3 + K2 + s2) + n3*(k6 + k6p + alphalcc) +
                       n4*(betalcc + k1 + RR1 + K1P) + n5*(s1 + RR2 + S1P) +
                       n6*(k5p + k4p) + n7*(K2P + S2P + k3);

        if (lamtot <= 1e-6f) { t_elapsed = dt; break; }

        float dt_step = dt - t_elapsed;
        if (current_lfire > lamtot * dt_step) {
            current_lfire -= lamtot * dt_step;
            t_elapsed = dt;
        } else {
            t_elapsed += current_lfire / lamtot;
            float r2 = lamtot * next_rand(&rng_state);

            // Simplified selection logic (mimics original 20-case branch)
            float acc = n1*k4; if(r2<=acc){n1--;n2++;}
            else{acc+=n1*k5; if(r2<=acc){n1--;n3++;}
            else{acc+=n2*k3; if(r2<=acc){n2--;n1++;}
            else{acc+=n2*K2; if(r2<=acc){n2--;n4++;}
            else{acc+=n2*s2; if(r2<=acc){n2--;n5++;}
            else{acc+=n3*k6; if(r2<=acc){n3--;n1++;}
            else{acc+=n3*k6p;if(r2<=acc){n3--;n6++;}
            else{acc+=n3*alphalcc;if(r2<=acc){n3--;n4++;}
            else{acc+=n4*betalcc;if(r2<=acc){n4--;n3++;}
            else{acc+=n4*k1; if(r2<=acc){n4--;n2++;}
            else{acc+=n4*RR1;if(r2<=acc){n4--;n5++;}
            else{acc+=n4*K1P;if(r2<=acc){n4--;n7++;}
            else{acc+=n5*s1; if(r2<=acc){n5--;n2++;}
            else{acc+=n5*RR2;if(r2<=acc){n5--;n4++;}
            else{acc+=n5*S1P;if(r2<=acc){n5--;n7++;}
            else{acc+=n6*k5p;if(r2<=acc){n6--;n3++;}
            else{acc+=n6*k4p;if(r2<=acc){n6--;n7++;}
            else{acc+=n7*K2P;if(r2<=acc){n7--;n4++;}
            else{acc+=n7*S2P;if(r2<=acc){n7--;n5++;}
            else{n7--;n6++;}}}}}}}}}}}}}}}}}}}

            float u = next_rand(&rng_state);
            while(u <= 0.0f || u >= 1.0f) u = next_rand(&rng_state);
            current_lfire = -logf(u);
        }
    }

    Nli2[idx] = n1; Nli[idx] = n2; Nlc2[idx] = n3; Nlc[idx] = n4;
    Nl[idx] = n5; Nlb2[idx] = n6; Nlb1[idx] = n7;
    lfire[idx] = current_lfire;
}
'''

# --- 3. Flux Calculation Kernel (Vectorized & Indexed) ---
flux_kernel_source = r'''
extern "C" __global__ void cru_flux_kernel(
    float* cd, float* cj, float* Ilcc_out,
    float* cm, float* cs, 
    const int* networkIdx, 
    const int* Nr, const int* Nl,
    const float v_val, const float dt, 
    const bool fix_sr, const int n_cru
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cru) return;

    // 1. 读取索引和本地状态
    int map_idx = networkIdx[idx]; 
    float _cd = cd[idx];
    float _cj = cj[idx];
    int _nr = Nr[idx];
    int _nl = Nl[idx];

    // 2. 读取 Myo/SR 浓度
    float _cm_myo = cm[map_idx];
    float _cs_myo = cs[map_idx];

    // 3. 计算电流 (RyR & LCC)
    float Iryr = _nr * 0.000205f * (_cj - _cd);

    float phi = v_val * (96485.0f / (8314.0f * 310.0f));
    if (abs(phi) < 1e-5f) phi = 1e-5f;
    float denom = expf(2.0f * phi) - 1.0f;
    float Ilcc = _nl * 0.000913f * 2.0f * phi * (0.341f * 1800.0f - expf(2.0f * phi) * _cd) / denom;

    Ilcc_out[idx] = Ilcc; 

    // 4. 计算通量
    float Vj = 0.1f;
    float Vd = 0.00126f;
    float gjsr = 1.0f;
    float gds = 0.303f / 0.00126f;

    // Buffering factors
    float beta_m = 1.0f + 15.0f * 13.0f / ((_cm_myo + 13.0f) * (_cm_myo + 13.0f)) +
                          7.0f * 0.3f / ((_cm_myo + 0.3f) * (_cm_myo + 0.3f));
    float beta_sr = 1.0f + 140.0f * 650.0f / ((_cs_myo + 650.0f) * (_cs_myo + 650.0f));
    float beta_j = 1.0f + 140.0f * 650.0f / ((_cj + 650.0f) * (_cj + 650.0f));

    float FluxJ = (gjsr * (_cs_myo - _cj) - Iryr / Vj) / beta_j;
    float FluxS = (-gjsr * (_cs_myo - _cj) * (Vj / 0.2f)) / beta_sr;
    float FluxM = (gds * (_cd - _cm_myo) * (Vd / 5.0f)) / beta_m;

    // 5. 更新状态
    if (!fix_sr) {
        cj[idx] = _cj + dt * FluxJ;
        atomicAdd(&cs[map_idx], dt * FluxS); 
    }

    cd[idx] = _cd - dt * FluxM; 
    atomicAdd(&cm[map_idx], dt * FluxM);
}
'''

# 编译内核
ryr_kernel = cp.RawKernel(ryr_kernel_source, 'update_ryr_kernel')
lcc_kernel = cp.RawKernel(lcc_kernel_source, 'update_lcc_kernel')
flux_kernel = cp.RawKernel(flux_kernel_source, 'cru_flux_kernel')


class GPUNivalaRyR:
    # 参数定义
    KAP, KAM, KBM, KBMR2C, KBP = 0.7 * 0.005, 1.0, 0.003, 1.0 * 0.003, 0.00075

    def __init__(self, n_cru, n_ryr=100):
        self.n_cru = n_cru
        self.Nr = cp.zeros(n_cru, dtype=cp.int32)
        self.Nrc = cp.full(n_cru, int(0.75 * n_ryr), dtype=cp.int32)
        self.Nri = cp.zeros(n_cru, dtype=cp.int32)
        self.Nrr = n_ryr - self.Nr - self.Nrc - self.Nri
        self.rng = cp.random.RandomState(seed=42)
        self.rfire = -cp.log(self.rng.uniform(0, 1, n_cru)).astype(cp.float32)

    def update_ryr(self, dt, cd, cj):
        grid = (self.n_cru + 255) // 256
        seed = int(cp.random.randint(0, 2 ** 31 - 1))
        ryr_kernel((grid,), (256,), (
            self.Nr, self.Nrc, self.Nri, self.Nrr, self.rfire,
            cd, cj,
            cp.float32(self.KAP), cp.float32(self.KAM), cp.float32(self.KBP),
            cp.float32(self.KBM), cp.float32(self.KBMR2C),
            cp.float32(dt), cp.int32(self.n_cru), cp.uint32(seed)
        ))


class GPULCC:
    def __init__(self, n_cru, n_lcc=10):
        self.n_cru = n_cru
        self.Nl = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlc = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlc2 = cp.full(n_cru, n_lcc, dtype=cp.int32)
        self.Nli = cp.zeros(n_cru, dtype=cp.int32)
        self.Nli2 = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlb1 = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlb2 = cp.zeros(n_cru, dtype=cp.int32)
        self.rng = cp.random.RandomState(seed=43)
        self.lfire = -cp.log(self.rng.uniform(0, 1, n_cru)).astype(cp.float32)

    def update_lcc(self, dt, cd, v_val):
        grid = (self.n_cru + 255) // 256
        seed = int(cp.random.randint(0, 2 ** 31 - 1))
        lcc_kernel((grid,), (256,), (
            self.Nli2, self.Nli, self.Nlc2, self.Nlc, self.Nl, self.Nlb2, self.Nlb1,
            self.lfire, cd, cp.float32(v_val),
            cp.float32(dt), cp.int32(self.n_cru), cp.uint32(seed)
        ))


class GPUCRU:
    """钙释放单元类 - Optimized with Flux Kernel"""

    def __init__(self, n_cru, n_ryr=100, n_lcc=10):
        self.n_cru = n_cru
        self.cd = cp.full(n_cru, 0.1, dtype=cp.float32)
        self.cj = cp.full(n_cru, 1500.0, dtype=cp.float32)
        self.Ilcc = cp.zeros(n_cru, dtype=cp.float32)
        self.RyRs = GPUNivalaRyR(n_cru, n_ryr)
        self.LCCs = GPULCC(n_cru, n_lcc)
        self.networkIdx = cp.zeros(n_cru, dtype=cp.int32)
        self.orphanedCRU = cp.zeros(n_cru, dtype=cp.bool_)
        self.orphanLCCidx = cp.full(n_cru, -1, dtype=cp.int32)

    def initialize_network_idx(self, nx, ny, nz, nx_myo):
        idx = cp.arange(self.n_cru)
        x = (idx % nx) * nx_myo + nx_myo // 2
        y = (idx // nx % ny) * nx_myo + nx_myo // 2
        z = (idx // (nx * ny)) * nx_myo + nx_myo // 2
        self.networkIdx = (z * (nx * nx_myo) * (ny * nx_myo) + y * (nx * nx_myo) + x).astype(cp.int32)

    def update_flux(self, dt, cm, cs, v, xnai, fix_sr=False):
        # 1. 健壮的电压值提取 (修复 IndexError)
        # v 可能是标量、0维数组或1维数组
        if isinstance(v, cp.ndarray):
            if v.ndim == 0:
                v_val = float(v)
            elif v.size == 1:
                v_val = float(v.flatten()[0])
            else:
                v_val = float(v[0])  # 默认取第一个
        else:
            v_val = float(v)

        # 2. 并行更新 LCC 和 RyR 状态
        self.LCCs.update_lcc(dt, self.cd, v_val)
        self.RyRs.update_ryr(dt, self.cd, self.cj)

        # 3. 计算通量并更新 (CruFluxKernel)
        grid = (self.n_cru + 255) // 256
        flux_kernel(
            (grid,), (256,),
            (
                self.cd, self.cj, self.Ilcc,
                cm, cs,
                self.networkIdx,
                self.RyRs.Nr, self.LCCs.Nl,
                cp.float32(v_val), cp.float32(dt),
                cp.bool_(fix_sr), cp.int32(self.n_cru)
            )
        )

    def to_numpy(self):
        return {
            'cd': cp.asnumpy(self.cd),
            'cj': cp.asnumpy(self.cj),
            'Ilcc': cp.asnumpy(self.Ilcc)
        }

    def from_numpy(self, data):
        self.cd = cp.asarray(data['cd'], dtype=cp.float32)
        self.cj = cp.asarray(data['cj'], dtype=cp.float32)
        self.Ilcc = cp.asarray(data['Ilcc'], dtype=cp.float32)

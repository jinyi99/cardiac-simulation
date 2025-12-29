import cupy as cp
import numpy as np

# =============================================================================
# CUDA Kernels for Stochastic Gating
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

# RyR 更新内核
ryr_kernel_source = rng_device_func + r'''
extern "C" __global__ void update_ryr_kernel(
    int* Nr, int* Nrc, int* Nri, int* Nrr,
    float* rfire,
    const float* alpha, const float* beta, const float* gamma, const float* delta,
    const float KBMR2C,
    const float dt, const int n_cru, const unsigned int seed_base
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cru) return;

    // 初始化随机数状态
    unsigned int rng_state = seed_base + idx;
    // 预热 RNG
    next_rand(&rng_state);

    // 加载状态
    int nr = Nr[idx];
    int nrc = Nrc[idx];
    int nri = Nri[idx];
    int nrr = Nrr[idx];
    float current_rfire = rfire[idx];

    // 加载速率常数
    float a = alpha[idx];
    float b = beta[idx];
    float g = gamma[idx];
    float d = delta[idx];
    float k_src = KBMR2C;

    float t_elapsed = 0.0f;

    while (t_elapsed < dt) {
        // 计算各转换速率 (Propensities)
        float sco = nrc * a;
        float soc = nr * b;
        float scr = nrc * g;
        float src = nrr * k_src;
        float sri = nrr * a;
        float sir = nri * b;
        float sio = nri * d;
        float soi = nr * g;

        // 计算累积速率
        float lam1 = sco + soc;
        float lam2 = lam1 + scr;
        float lam3 = lam2 + src;
        float lam4 = lam3 + sri;
        float lam5 = lam4 + sir;
        float lam6 = lam5 + sio;
        float lamtot = lam6 + soi;

        if (lamtot <= 1e-6f) {
            t_elapsed = dt; // 无事件发生
            break;
        }

        // 检查是否发生事件
        float dt_step = dt - t_elapsed;
        if (current_rfire > lamtot * dt_step) {
            // 本步内无更多事件
            current_rfire -= lamtot * dt_step;
            t_elapsed = dt;
        } else {
            // 发生事件
            // 更新时间
            t_elapsed += current_rfire / lamtot;
            
            // 确定哪个反应发生
            float r2 = lamtot * next_rand(&rng_state);

            if (r2 <= sco) {
                nrc--; nr++;
            } else if (r2 <= lam1) {
                nrc++; nr--;
            } else if (r2 <= lam2) {
                nrc--; nrr++;
            } else if (r2 <= lam3) {
                nrr--; nrc++;
            } else if (r2 <= lam4) {
                nrr--; nri++;
            } else if (r2 <= lam5) {
                nri--; nrr++;
            } else if (r2 <= lam6) {
                nri--; nr++;
            } else {
                nr--; nri++;
            }

            // 重置 rfire (指数分布: -log(u))
            float u = next_rand(&rng_state);
            while(u <= 0.0f || u >= 1.0f) u = next_rand(&rng_state);
            current_rfire = -logf(u);
        }
    }

    // 写回状态
    Nr[idx] = nr;
    Nrc[idx] = nrc;
    Nri[idx] = nri;
    Nrr[idx] = nrr;
    rfire[idx] = current_rfire;
}
'''

# LCC 更新内核
lcc_kernel_source = rng_device_func + r'''
extern "C" __global__ void update_lcc_kernel(
    int* Nli2, int* Nli, int* Nlc2, int* Nlc, int* Nl, int* Nlb2, int* Nlb1,
    float* lfire,
    const float* k1, const float* k2, const float* k3, const float* k4,
    const float* k5, const float* k6, const float* k1p, const float* k2p,
    const float* k3p, const float* k4p, const float* k5p, const float* k6p,
    const float* s1, const float* s2, const float* alphalcc, const float* betalcc,
    const float RR1, const float RR2, const float S1P, const float S2P,
    const float dt, const int n_cru, const unsigned int seed_base
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cru) return;

    unsigned int rng_state = seed_base + idx;
    next_rand(&rng_state);

    // 加载状态 (Nli2:1, Nli:2, Nlc2:3, Nlc:4, Nl:5, Nlb2:6, Nlb1:7)
    int n1 = Nli2[idx];
    int n2 = Nli[idx];
    int n3 = Nlc2[idx];
    int n4 = Nlc[idx];
    int n5 = Nl[idx];
    int n6 = Nlb2[idx];
    int n7 = Nlb1[idx];
    
    float current_lfire = lfire[idx];

    // 加载参数
    float _k1 = k1[idx]; float _k2 = k2[idx]; float _k3 = k3[idx]; float _k4 = k4[idx];
    float _k5 = k5[idx]; float _k6 = k6[idx]; 
    float _k1p = k1p[idx]; float _k2p = k2p[idx]; float _k3p = k3p[idx]; float _k4p = k4p[idx];
    float _k5p = k5p[idx]; float _k6p = k6p[idx];
    float _s1 = s1[idx]; float _s2 = s2[idx];
    float _alpha = alphalcc[idx]; float _beta = betalcc[idx];
    
    float t_elapsed = 0.0f;

    while (t_elapsed < dt) {
        // 计算转换速率
        float s12 = n1 * _k4;
        float s13 = n1 * _k5;
        float s21 = n2 * _k3;
        float s24 = n2 * _k2;
        float s25 = n2 * _s2;
        float s31 = n3 * _k6;
        float s36 = n3 * _k6p;
        float s34 = n3 * _alpha;
        float s43 = n4 * _beta;
        float s42 = n4 * _k1;
        float s45 = n4 * RR1;
        float s47 = n4 * _k1p;
        float s52 = n5 * _s1;
        float s54 = n5 * RR2;
        float s57 = n5 * S1P;
        float s63 = n6 * _k5p;
        float s67 = n6 * _k4p;
        float s74 = n7 * _k2p;
        float s75 = n7 * S2P;
        float s76 = n7 * _k3p;

        // 累积速率
        float rate_sum = s12;
        float r_s13 = rate_sum + s13; rate_sum = r_s13;
        float r_s21 = rate_sum + s21; rate_sum = r_s21;
        float r_s24 = rate_sum + s24; rate_sum = r_s24;
        float r_s25 = rate_sum + s25; rate_sum = r_s25;
        float r_s31 = rate_sum + s31; rate_sum = r_s31;
        float r_s36 = rate_sum + s36; rate_sum = r_s36;
        float r_s34 = rate_sum + s34; rate_sum = r_s34;
        float r_s43 = rate_sum + s43; rate_sum = r_s43;
        float r_s42 = rate_sum + s42; rate_sum = r_s42;
        float r_s45 = rate_sum + s45; rate_sum = r_s45;
        float r_s47 = rate_sum + s47; rate_sum = r_s47;
        float r_s52 = rate_sum + s52; rate_sum = r_s52;
        float r_s54 = rate_sum + s54; rate_sum = r_s54;
        float r_s57 = rate_sum + s57; rate_sum = r_s57;
        float r_s63 = rate_sum + s63; rate_sum = r_s63;
        float r_s67 = rate_sum + s67; rate_sum = r_s67;
        float r_s74 = rate_sum + s74; rate_sum = r_s74;
        float r_s75 = rate_sum + s75; rate_sum = r_s75;
        float r_s76 = rate_sum + s76; rate_sum = r_s76;

        float lamtot = rate_sum;

        if (lamtot <= 1e-6f) {
            t_elapsed = dt;
            break;
        }

        float dt_step = dt - t_elapsed;
        if (current_lfire > lamtot * dt_step) {
            current_lfire -= lamtot * dt_step;
            t_elapsed = dt;
        } else {
            t_elapsed += current_lfire / lamtot;
            float r2 = lamtot * next_rand(&rng_state);

            // 状态更新逻辑 (20个分支)
            if (r2 <= s12) { n1--; n2++; }
            else if (r2 <= r_s13) { n1--; n3++; }
            else if (r2 <= r_s21) { n2--; n1++; }
            else if (r2 <= r_s24) { n2--; n4++; }
            else if (r2 <= r_s25) { n2--; n5++; }
            else if (r2 <= r_s31) { n3--; n1++; }
            else if (r2 <= r_s36) { n3--; n6++; }
            else if (r2 <= r_s34) { n3--; n4++; }
            else if (r2 <= r_s43) { n4--; n3++; }
            else if (r2 <= r_s42) { n4--; n2++; }
            else if (r2 <= r_s45) { n4--; n5++; }
            else if (r2 <= r_s47) { n4--; n7++; }
            else if (r2 <= r_s52) { n5--; n2++; }
            else if (r2 <= r_s54) { n5--; n4++; }
            else if (r2 <= r_s57) { n5--; n7++; }
            else if (r2 <= r_s63) { n6--; n3++; }
            else if (r2 <= r_s67) { n6--; n7++; }
            else if (r2 <= r_s74) { n7--; n4++; }
            else if (r2 <= r_s75) { n7--; n5++; }
            else { n7--; n6++; }

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

# 编译内核
ryr_kernel = cp.RawKernel(ryr_kernel_source, 'update_ryr_kernel')
lcc_kernel = cp.RawKernel(lcc_kernel_source, 'update_lcc_kernel')


class GPUNivalaRyR:
    """RyR受体类（Nivala模型） - Optimized with RawKernel"""

    # RyR参数
    KAP = 0.7 * 0.005
    KAM = 1.0
    KBM = 0.003
    KBMR2C = 1.0 * 0.003
    KBP = 0.00075

    def __init__(self, n_cru, n_ryr=100):
        self.n_cru = n_cru
        self.n_ryr = n_ryr

        # Markov状态 (使用int32以配合atomic操作潜力，虽然这里单线程处理)
        self.Nr = cp.zeros(n_cru, dtype=cp.int32)
        self.Nrc = cp.full(n_cru, int(0.75 * n_ryr), dtype=cp.int32)
        self.Nri = cp.zeros(n_cru, dtype=cp.int32)
        self.Nrr = n_ryr - self.Nr - self.Nrc - self.Nri

        self.rng = cp.random.RandomState(seed=42)
        # 初始随机发射时间
        self.rfire = -cp.log(self.rng.uniform(0, 1, n_cru)).astype(cp.float32)

    def update_ryr(self, dt, cd, cj):
        """使用CUDA Kernel并行更新RyR状态"""
        # 1. 预计算依赖于浓度的速率常数 (向量化计算，极快)
        alpha = (self.KAP * cp.power(cd, 2)).astype(cp.float32)
        gamma = (self.KBP * cd).astype(cp.float32)
        beta = cp.full(self.n_cru, self.KAM, dtype=cp.float32)
        delta = cp.full(self.n_cru, self.KBM, dtype=cp.float32)

        # 2. 调用 CUDA Kernel
        threads_per_block = 256
        blocks_per_grid = (self.n_cru + threads_per_block - 1) // threads_per_block
        
        # 生成一个随机种子基数，确保每步随机性不同
        seed_base = int(cp.random.randint(0, 2**31 - 1))

        ryr_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                self.Nr, self.Nrc, self.Nri, self.Nrr,
                self.rfire,
                alpha, beta, gamma, delta,
                cp.float32(self.KBMR2C),
                cp.float32(dt), cp.int32(self.n_cru), cp.uint32(seed_base)
            )
        )

    def get_ryr_current(self, cd, cj):
        return self.Nr * 0.000205 * (cj - cd)


class GPULCC:
    """L型钙通道类 - Optimized with RawKernel"""

    # LCC参数
    CDBAR = 0.5
    K2 = 0.0001
    TBA = 450.0
    K1P = 0.00413
    RR1 = 0.3
    RR2 = 3.0
    K2P = 0.00224
    S1P = 0.00195
    S2P = S1P * (K2P / K1P) * (RR1 / RR2)

    def __init__(self, n_cru, n_lcc=10):
        self.n_cru = n_cru
        self.n_lcc = n_lcc

        # Markov状态 (Nli2:1, Nli:2, Nlc2:3, Nlc:4, Nl:5, Nlb2:6, Nlb1:7)
        self.Nl = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlc = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlc2 = cp.full(n_cru, n_lcc, dtype=cp.int32)
        self.Nli = cp.zeros(n_cru, dtype=cp.int32)
        self.Nli2 = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlb1 = cp.zeros(n_cru, dtype=cp.int32)
        self.Nlb2 = cp.zeros(n_cru, dtype=cp.int32)

        self.rng = cp.random.RandomState(seed=43)
        self.lfire = -cp.log(self.rng.uniform(0, 1, n_cru)).astype(cp.float32)

    def update_lcc(self, dt, cd, v):
        """使用CUDA Kernel并行更新LCC状态"""
        # 确保输入是float32
        v = v.astype(cp.float32)
        cd = cd.astype(cp.float32)

        # 1. 预计算速率常数 (向量化)
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

        # 准备常量数组
        k2_arr = cp.full(self.n_cru, self.K2, dtype=cp.float32)
        k1p_arr = cp.full(self.n_cru, self.K1P, dtype=cp.float32)
        k2p_arr = cp.full(self.n_cru, self.K2P, dtype=cp.float32)

        # 2. 调用 CUDA Kernel
        threads_per_block = 256
        blocks_per_grid = (self.n_cru + threads_per_block - 1) // threads_per_block
        seed_base = int(cp.random.randint(0, 2**31 - 1))

        lcc_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                self.Nli2, self.Nli, self.Nlc2, self.Nlc, self.Nl, self.Nlb2, self.Nlb1,
                self.lfire,
                k1.astype(cp.float32), k2_arr, k3.astype(cp.float32), k4.astype(cp.float32),
                k5.astype(cp.float32), k6.astype(cp.float32),
                k1p_arr, k2p_arr, k3p.astype(cp.float32), k4p.astype(cp.float32),
                k5p.astype(cp.float32), k6p.astype(cp.float32),
                s1.astype(cp.float32), s2.astype(cp.float32),
                alphalcc.astype(cp.float32), betalcc.astype(cp.float32),
                cp.float32(self.RR1), cp.float32(self.RR2), 
                cp.float32(self.S1P), cp.float32(self.S2P),
                cp.float32(dt), cp.int32(self.n_cru), cp.uint32(seed_base)
            )
        )

    def get_lcc_current(self, cd, v):
        phi = v * (96485.0 / (8314.0 * 310.0))
        pca = 0.000913
        
        # 避免除零
        phi = cp.where(cp.abs(phi) < 1e-5, 1e-5, phi)
        
        denom = cp.exp(2.0 * phi) - 1.0
        # 如果 denom 太小，也处理一下，不过上面 phi 处理过应该还好
        
        I_ca = self.Nl * pca * 2.0 * phi * (0.341 * 1800.0 - cp.exp(2.0 * phi) * cd) / denom
        return I_ca


class GPUCRU:
    """钙释放单元类"""

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
        self.networkIdx = z * (nx * nx_myo) * (ny * nx_myo) + y * (nx * nx_myo) + x

    def update_flux(self, dt, cm, cs, v, xnai, fix_sr=False):
        # 1. 广播与维度检查 (确保 v 是 shape=(n_cru,) 的向量)
        if isinstance(v, (float, int)):
            v = cp.full(self.n_cru, float(v), dtype=cp.float32)
        else:
            v = cp.asarray(v, dtype=cp.float32)
            if v.ndim == 0 or v.size == 1:
                v = cp.full(self.n_cru, float(v), dtype=cp.float32)
            else:
                v = v.reshape(-1) # 确保展平

        # 2. 并行更新 LCC 和 RyR (现在这里调用的是 fast kernels)
        self.LCCs.update_lcc(dt, self.cd, v)
        self.RyRs.update_ryr(dt, self.cd, self.cj)

        # 3. 计算缓冲 (Vectorized)
        beta_m = (1.0 + 15.0 * 13.0 / ((cm[self.networkIdx] + 13.0) ** 2) +
                  7.0 * 0.3 / ((cm[self.networkIdx] + 0.3) ** 2))
        
        beta_sr = (1.0 + 140.0 * 650.0 / ((cs[self.networkIdx] + 650.0) ** 2))
        beta_j = (1.0 + 140.0 * 650.0 / ((self.cj + 650.0) ** 2))

        # 4. 计算通量
        Iryr = self.RyRs.get_ryr_current(self.cd, self.cj)
        self.Ilcc = self.LCCs.get_lcc_current(self.cd, v)

        Vj, Vd = 0.1, 0.00126
        gjsr, gds = 1.0, 0.303 / 0.00126

        cs_at_cru = cs[self.networkIdx]
        cm_at_cru = cm[self.networkIdx]

        FluxJ = (gjsr * (cs_at_cru - self.cj) - Iryr / Vj) / beta_j
        FluxS = (-gjsr * (cs_at_cru - self.cj) * (Vj / 0.2)) / beta_sr
        FluxM = (gds * (self.cd - cm_at_cru) * (Vd / 5.0)) / beta_m

        # 5. 更新状态
        if not fix_sr:
            self.cj += dt * FluxJ
            # 注意: 这里可能有race condition如果多个CRU对应同一个myo voxel
            # 但在常规设置下 CRU 是一一对应的或稀疏的。
            # 如果确实密集，应使用 cp.scatter_add (atomicAdd)
            # 但这里为了性能先保持原样，通常 networkIdx 是唯一的。
            cs[self.networkIdx] += dt * FluxS

        cm[self.networkIdx] += dt * FluxM

        # 6. 处理 Orphaned LCCs
        orphaned_mask = self.orphanedCRU & (self.orphanLCCidx >= 0)
        if cp.any(orphaned_mask):
            idx = cp.where(orphaned_mask)[0]
            # 这里使用了 orphanLCCidx 作为目标索引
            # 注意：如果多个 LCC 注入同一个 voxel，这里需要 atomic add
            # cp.add.at(cm, self.orphanLCCidx[idx], dt * self.Ilcc[idx] / 5.0)
            # 简单起见：
            targets = self.orphanLCCidx[idx]
            values = dt * self.Ilcc[idx] / 5.0
            cp.scatter_add(cm, targets, values)

        removed_mask = self.orphanedCRU & (self.orphanLCCidx < 0)
        self.Ilcc[removed_mask] = 0.0

    def to_numpy(self):
        return {
            'cd': cp.asnumpy(self.cd),
            'cj': cp.asnumpy(self.cj),
            'Ilcc': cp.asnumpy(self.Ilcc),
            'networkIdx': cp.asnumpy(self.networkIdx),
            'RyR_Nr': cp.asnumpy(self.RyRs.Nr),
            'RyR_Nrc': cp.asnumpy(self.RyRs.Nrc),
            'LCC_Nl': cp.asnumpy(self.LCCs.Nl),
            'LCC_Nlc': cp.asnumpy(self.LCCs.Nlc)
        }

    def from_numpy(self, data):
        self.cd = cp.asarray(data['cd'], dtype=cp.float32)
        self.cj = cp.asarray(data['cj'], dtype=cp.float32)
        self.Ilcc = cp.asarray(data['Ilcc'], dtype=cp.float32)
        self.networkIdx = cp.asarray(data['networkIdx'], dtype=cp.int32)
        self.RyRs.Nr = cp.asarray(data['RyR_Nr'], dtype=cp.int32)
        self.RyRs.Nrc = cp.asarray(data['RyR_Nrc'], dtype=cp.int32)
        self.LCCs.Nl = cp.asarray(data['LCC_Nl'], dtype=cp.int32)
        self.LCCs.Nlc = cp.asarray(data['LCC_Nlc'], dtype=cp.int32)

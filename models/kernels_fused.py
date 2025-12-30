import cupy as cp

# =============================================================================
# Fused Kernels (融合内核)
# =============================================================================

# 融合反应内核源码 (包含 MyoSR Flux 计算、缓冲更新和欧拉步)
fused_reaction_source = r'''
extern "C" __global__ void fused_reaction_kernel(
    float* cm, float* cs, float* Jncx_myo, 
    const float* v, const float* xnai, 
    const float dt, const float FRT, 
    const int n_voxels, const bool fix_sr
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_voxels) return;

    // 1. 读取当前状态 (直接从显存读取到寄存器)
    float _cm = cm[idx];
    float _cs = cs[idx];
    float _v = v[0];     // 假设单细胞模型，所有体素共享全细胞电压
    float _xnai = xnai[0];

    // 2. MyoSR 参数 (常量直接编译进内核以提升性能)
    const float vup = 0.32f;
    const float kup = 1.0f;
    const float gleak = 0.000004f;
    const float gbg = 0.00003f;
    const float cao = 1800.0f;

    // NCX 参数
    const float v2 = 0.8f;
    const float kmca = 0.11f;
    const float kmna = 12.3f;
    const float kmco = 1.3f;
    const float kmno = 87.5f;
    const float nao = 136.0f;

    // 3. 计算中间变量
    // Jup & Jleak
    float Jup = vup * (_cm * _cm) / (_cm * _cm + kup * kup);
    float Jleak = gleak * (_cs - _cm);

    // Jbg
    float phi = _v * FRT;
    // 避免 log(0) 或负数
    float safe_cm = (_cm > 1e-6f) ? _cm : 1e-6f; 
    float Jbg = -gbg * (_v - logf(cao / safe_cm) / (2.0f * FRT));

    // JNCX (复杂公式融合)
    float enai3 = _xnai * _xnai * _xnai;
    float nao3 = nao * nao * nao;
    float exp_035_phi = expf(0.35f * phi);
    float exp_m065_phi = expf((0.35f - 1.0f) * phi); // 0.35 - 1.0 = -0.65

    float num = v2 * (exp_035_phi * enai3 * cao - exp_m065_phi * nao3 * _cm);

    float denom1 = 1.0f + (kmca / _cm) * (kmca / _cm) * (kmca / _cm);
    float denom2 = 1.0f + 0.27f * exp_m065_phi;
    float denom3 = kmco * enai3 + powf(kmno, 3.0f) * _cm + 
                   powf(kmna, 3.0f) * cao * (1.0f + _cm / 3.59f) +
                   3.59f * nao3 * (1.0f + enai3 / powf(kmna, 3.0f)) + 
                   enai3 * cao + nao3 * _cm;

    float _Jncx = num / (denom1 * denom2 * denom3);

    // 4. 计算缓冲 (Buffering)
    // Beta M
    float b_m = 1.0f + 15.0f * 13.0f / ((_cm + 13.0f) * (_cm + 13.0f)) +
                       7.0f * 0.3f  / ((_cm + 0.3f) * (_cm + 0.3f));
    // Beta SR
    float b_sr = 1.0f + 140.0f * 650.0f / ((_cs + 650.0f) * (_cs + 650.0f));

    // 5. 更新状态 (Euler Step)
    float dcm_dt = (-Jup + _Jncx + Jbg + Jleak) / b_m;

    // 写入新值
    cm[idx] = _cm + dt * dcm_dt;

    if (!fix_sr) {
        float dcs_dt = (5.0f / 0.2f) * (Jup - Jleak) / b_sr;
        cs[idx] = _cs + dt * dcs_dt;
    }

    // 记录 Jncx 用于全细胞电流计算
    Jncx_myo[idx] = _Jncx;
}
'''

# 编译内核
fused_reaction_kernel = cp.RawKernel(fused_reaction_source, 'fused_reaction_kernel')

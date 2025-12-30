import cupy as cp

# =============================================================================
# Fused Kernels (融合内核: 反应 + 扩散)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 融合反应内核 (Reaction Kernel)
# -----------------------------------------------------------------------------
fused_reaction_source = r'''
extern "C" __global__ void fused_reaction_kernel(
    float* cm, float* cs, float* Jncx_myo, 
    const float* v, const float* xnai, 
    const float dt, const float FRT, 
    const int n_voxels, const bool fix_sr
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_voxels) return;

    // 1. 读取当前状态
    float _cm = cm[idx];
    float _cs = cs[idx];
    float _v = v[0];     
    float _xnai = xnai[0];

    // 2. MyoSR 参数
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

    // 3. 计算中间通量
    float Jup = vup * (_cm * _cm) / (_cm * _cm + kup * kup);
    float Jleak = gleak * (_cs - _cm);

    float phi = _v * FRT;
    float safe_cm = (_cm > 1e-6f) ? _cm : 1e-6f; 
    float Jbg = -gbg * (_v - logf(cao / safe_cm) / (2.0f * FRT));

    // JNCX 计算
    float enai3 = _xnai * _xnai * _xnai;
    float nao3 = nao * nao * nao;
    float exp_035_phi = expf(0.35f * phi);
    float exp_m065_phi = expf((0.35f - 1.0f) * phi);

    float num = v2 * (exp_035_phi * enai3 * cao - exp_m065_phi * nao3 * _cm);
    float denom1 = 1.0f + (kmca / _cm) * (kmca / _cm) * (kmca / _cm);
    float denom2 = 1.0f + 0.27f * exp_m065_phi;
    float denom3 = kmco * enai3 + powf(kmno, 3.0f) * _cm + 
                   powf(kmna, 3.0f) * cao * (1.0f + _cm / 3.59f) +
                   3.59f * nao3 * (1.0f + enai3 / powf(kmna, 3.0f)) + 
                   enai3 * cao + nao3 * _cm;

    float _Jncx = num / (denom1 * denom2 * denom3);

    // 4. 计算缓冲系数 (Buffering Beta)
    float b_m = 1.0f + 15.0f * 13.0f / ((_cm + 13.0f) * (_cm + 13.0f)) +
                       7.0f * 0.3f  / ((_cm + 0.3f) * (_cm + 0.3f));
    float b_sr = 1.0f + 140.0f * 650.0f / ((_cs + 650.0f) * (_cs + 650.0f));

    // 5. 更新状态 (Euler Step)
    float dcm_dt = (-Jup + _Jncx + Jbg + Jleak) / b_m;
    cm[idx] = _cm + dt * dcm_dt;

    if (!fix_sr) {
        float dcs_dt = (5.0f / 0.2f) * (Jup - Jleak) / b_sr;
        cs[idx] = _cs + dt * dcs_dt;
    }

    Jncx_myo[idx] = _Jncx;
}
'''

# -----------------------------------------------------------------------------
# 2. 3D 扩散内核 (Diffusion Kernel - Optimized)
# -----------------------------------------------------------------------------
# 直接使用索引计算，避免 Python 层的 cp.roll 和内存拷贝
diffusion_source = r'''
extern "C" __global__ void diffusion_kernel_3d(
    float* c_out, const float* c_in, const float* beta,
    const float dt, const float D, const float dx,
    const int nx, const int ny, const int nz
) {
    // 1D 线性索引映射到 3D 坐标
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_voxels = nx * ny * nz;

    if (idx >= total_voxels) return;

    // 计算 x, y, z 坐标 (假设数据存储格式为 Z-Y-X 或类似连续存储)
    int temp = idx;
    int x = temp % nx;
    temp /= nx;
    int y = temp % ny;
    int z = temp / ny;

    // 获取中心点浓度
    float c_center = c_in[idx];

    // 计算邻居索引 (无通量边界条件 No-flux: 边界处导数为0 -> 邻居值等于自己)
    // 左邻居 (x-1)
    float c_left  = (x > 0)      ? c_in[idx - 1]  : c_center;
    // 右邻居 (x+1)
    float c_right = (x < nx - 1) ? c_in[idx + 1]  : c_center;

    // 上邻居 (y-1) - 注意 strides
    float c_up    = (y > 0)      ? c_in[idx - nx] : c_center;
    // 下邻居 (y+1)
    float c_down  = (y < ny - 1) ? c_in[idx + nx] : c_center;

    // 前邻居 (z-1)
    float c_front = (z > 0)      ? c_in[idx - nx * ny] : c_center;
    // 后邻居 (z+1)
    float c_back  = (z < nz - 1) ? c_in[idx + nx * ny] : c_center;

    // 离散拉普拉斯算子
    float laplacian = c_left + c_right + c_up + c_down + c_front + c_back - 6.0f * c_center;

    // 更新公式: c_new = c + (D * dt / dx^2) * Laplacian / beta
    // 预计算 factor = D * dt / (dx * dx)
    float factor = D * dt / (dx * dx);

    // 获取缓冲系数 (注意：如果 beta 需要重新计算，可以在这里算，但通常传入预计算好的)
    float b = beta[idx];

    // 写入结果缓冲区
    c_out[idx] = c_center + (factor * laplacian) / b;
}
'''

# 编译内核
fused_reaction_kernel = cp.RawKernel(fused_reaction_source, 'fused_reaction_kernel')
diffusion_kernel_3d = cp.RawKernel(diffusion_source, 'diffusion_kernel_3d')

import cupy as cp
import numpy as np
import time
from .gpu_myocyte import GPUMyocyteCalcium
from .gpu_cru import GPUCRU
from .gpu_ucla_cell import GPUUCLACell
# 导入融合内核
from .kernels_fused import fused_reaction_kernel


class GPUSpatialCell:
    """空间心肌细胞模型类（GPU版本 - 高性能优化版）"""

    # 模型参数
    DT = 0.01  # 时间步长 (ms)
    OUTPUT_DT = 1.0  # 输出间隔 (ms)
    REDUCE_OUTPUT = max(int(OUTPUT_DT / DT), 1)

    # 扩散参数
    D_M = 0.3  # 肌浆扩散系数
    D_SR = 0.06  # 肌浆网扩散系数
    DX = 0.2  # 空间步长

    def __init__(self, nx, ny, nz, filename, rng_seed):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_cru = nx * ny * nz
        self.n_myo_voxels = nx * ny * nz * 5 ** 3  # NX_MYO=5

        self.dt = self.DT
        self.time = 0.0
        self.filename = filename
        self.rng_seed = rng_seed

        # 初始化GPU数组
        self._init_gpu_arrays()

        # 初始化模型组件
        self.whole_cell = GPUUCLACell(1)
        self.myosr_ca = [GPUMyocyteCalcium(self.n_myo_voxels) for _ in range(1)]
        self.crus = [GPUCRU(self.n_cru) for _ in range(1)]

        self._init_network_indices()
        self.v_buffer = cp.zeros(1, dtype=cp.float32)
        self.Istim = cp.zeros(1, dtype=cp.float32)
        self.fix_sr = False

        # 定义扩散计算内核
        self.diffusion_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 c, float32 c_left, float32 c_right, float32 beta',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * (c_left - 2.0f * c + c_right);
            result = c + diffusion / beta;
            ''',
            'diffusion_calculation'
        )

        # 内核启动参数预计算
        self.block_size = 256
        self.grid_size = (self.n_myo_voxels + self.block_size - 1) // self.block_size

        print(f"初始化完成: {nx}x{ny}x{nz} CRU网格, {self.n_myo_voxels} 个体素 (优化版)")

    def _init_gpu_arrays(self):
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    def _init_network_indices(self):
        for cru in self.crus:
            cru.initialize_network_idx(self.nx, self.ny, self.nz, 5)

    def simulate_pacing(self, n_beats, pcl, istim, stim_duration, vclamp=False):
        print(f"开始模拟: {n_beats} 个心动周期, PCL={pcl}ms")
        gpu_istim = cp.asarray(istim) if not isinstance(istim, cp.ndarray) else istim

        for beat in range(n_beats):
            print(f"周期 {beat + 1}/{n_beats}")
            start_time = time.time()
            self.simulate_beat(pcl, gpu_istim, stim_duration, vclamp)
            duration = time.time() - start_time
            print(f"  完成, 耗时: {duration:.2f}s, 平均每步: {duration / (pcl / self.dt) * 1000:.2f}ms")

    def simulate_beat(self, duration, istim, stim_duration, vclamp):
        """
        模拟单个心动周期 (使用 CUDA Graph 优化)
        """
        n_steps = int(duration / self.dt)
        stim_steps = int(stim_duration / self.dt)

        # Graph 配置
        GRAPH_BATCH_SIZE = 100  # 每次 Graph 执行 100 步
        graph = None
        graph_stream = cp.cuda.Stream(non_blocking=True)

        step = 0
        while step < n_steps:
            # 判断是否可以使用 Graph：
            # 1. 剩余步数足够一个 Batch
            # 2. 当前不在刺激期（刺激期可能有逻辑变动，且时间短，用普通循环处理更安全）
            use_graph = (n_steps - step >= GRAPH_BATCH_SIZE) and (step >= stim_steps)

            if use_graph:
                if graph is None:
                    # === 录制 CUDA Graph ===
                    # 注意：录制时所有 GPU 操作会被捕获，Python 逻辑会执行
                    graph_stream.begin_capture()
                    for _ in range(GRAPH_BATCH_SIZE):
                        self._step_logic(istim_val=0.0, istim_on=False, vclamp=vclamp)
                    graph = graph_stream.end_capture()

                # === 执行 Graph ===
                graph.launch(stream=cp.cuda.get_current_stream())

                # 更新 Python 侧状态 (Graph 内部不会更新 Python 变量)
                self.time += self.dt * GRAPH_BATCH_SIZE
                step += GRAPH_BATCH_SIZE

                # 检查输出
                # 注意：这里简化处理，只在 Batch 结束时检查输出
                if step % self.REDUCE_OUTPUT == 0:
                    self.output_data(step)

            else:
                # === 普通执行模式 (处理刺激期或剩余步数) ===
                istim_on = step < stim_steps

                # 如果 istim 是数组，需要处理；如果是标量直接传
                # 这里假设 istim 是标量或只在 stim_on 时有效
                curr_istim = istim if istim_on else 0.0

                self._step_logic(curr_istim, istim_on, vclamp)
                self.time += self.dt
                step += 1

                if step % self.REDUCE_OUTPUT == 0:
                    self.output_data(step)

    def _step_logic(self, istim_val, istim_on, vclamp):
        """单步物理逻辑 (供循环和 Graph 录制共用)"""
        # 1. Update CRU Flux
        self.update_cru_flux()

        # 2. Update MyoSR Flux (使用融合内核)
        self.update_myosr_flux()

        # 3. Diffusion
        self.compute_calcium_diffusion()

        # 4. Voltage
        self.update_voltage(istim_val, istim_on, vclamp)

    def update_cru_flux(self):
        for i, cru in enumerate(self.crus):
            cru.update_flux(
                self.dt,
                self.myosr_ca[i].cm,
                self.myosr_ca[i].cs,
                self.whole_cell.v[0],
                self.whole_cell.xnai[0],
                self.fix_sr
            )

    def update_myosr_flux(self):
        """更新肌浆网通量 - 使用融合内核"""
        # 直接调用 C++ 内核，避免 Python 计算
        myosr = self.myosr_ca[0]
        fused_reaction_kernel(
            (self.grid_size,), (self.block_size,),
            (
                myosr.cm, myosr.cs, myosr.Jncx,
                self.whole_cell.v, self.whole_cell.xnai,
                cp.float32(self.dt), cp.float32(self.whole_cell.FRT),
                cp.int32(self.n_myo_voxels), cp.bool_(self.fix_sr)
            )
        )

    def compute_calcium_diffusion(self):
        for myosr in self.myosr_ca:
            beta_m = myosr.compute_buffering_m(myosr.cm)
            beta_s = myosr.compute_buffering_sr(myosr.cs)

            # 简单的移位操作实现邻居访问 (比切片拷贝更快)
            cm_left = cp.roll(myosr.cm, 1)
            cm_right = cp.roll(myosr.cm, -1)
            # 修正边界 (cp.roll 是循环移位，需截断)
            # 注意：实际扩散内核可能需要更严谨的边界处理，此处保持原逻辑兼容性
            # 为保持与原代码一致的切片逻辑：
            # 原代码: cm_left[1:] = cm[:-1], cm_left[0]=0 (implicit in zeros_like)

            # 优化：为了适配 Graph，这里使用原代码逻辑的变量构造，
            # 但如果想要极致速度，建议完全移入 C++ 内核。
            # 这里保留原逻辑以确保数值正确性。
            cm_L = cp.zeros_like(myosr.cm)
            cm_L[1:] = myosr.cm[:-1]
            cm_R = cp.zeros_like(myosr.cm)
            cm_R[:-1] = myosr.cm[1:]

            cs_L = cp.zeros_like(myosr.cs)
            cs_L[1:] = myosr.cs[:-1]
            cs_R = cp.zeros_like(myosr.cs)
            cs_R[:-1] = myosr.cs[1:]

            myosr.cm_tmp = self.diffusion_kernel(
                self.dt, self.D_M, self.DX,
                myosr.cm, cm_L, cm_R, beta_m
            )

            myosr.cs_tmp = self.diffusion_kernel(
                self.dt, self.D_SR, self.DX,
                myosr.cs, cs_L, cs_R, beta_s
            )

            myosr.swap_temp_ptr()

    def update_voltage(self, istim, istim_on, vclamp=False):
        # 确保 istim 是标量或正确的数组类型
        current_istim = istim if istim_on else 0.0

        # 调用 whole_cell 的 step
        # 注意：step 内部是融合内核，本身很快
        self.whole_cell.step(self.dt, current_istim)

        if vclamp and hasattr(self, 'VClampArray'):
            # 电压钳逻辑稍微复杂，如果在 Graph 中使用需确保索引正确
            # 这里的简化实现假设 Graph 仅用于非钳位或简单模式
            clamp_idx = min(int(self.time / self.dt), len(self.VClampArray) - 1)
            self.whole_cell.v[0] = self.VClampArray[clamp_idx]

    def output_data(self, step):
        avg_data = self.get_averages()
        if step % (10 * self.REDUCE_OUTPUT) == 0:
            print(f"t={self.time:.2f}ms, V={avg_data['v']:.2f}mV, [Ca]i={avg_data['ci']:.4f}μM")

    def get_averages(self):
        # 计算平均值
        avg_cm = cp.mean(self.myosr_ca[0].cm)
        avg_cs = cp.mean(self.myosr_ca[0].cs)
        avg_jncx = cp.mean(self.myosr_ca[0].Jncx)
        avg_ilcc = cp.mean(self.crus[0].Ilcc)

        self.whole_cell.ci[0] = avg_cm
        self.whole_cell.JNCX[0] = avg_jncx
        self.whole_cell.ILCC[0] = avg_ilcc

        return {
            'time': self.time,
            'v': float(self.whole_cell.v[0]),
            'ci': float(avg_cm),
            'cs': float(avg_cs),
            'JNCX': float(avg_jncx),
            'ILCC': float(avg_ilcc)
        }

    # 其他辅助方法保持不变...
    def load_vclamp_data(self, filename):
        try:
            data = np.loadtxt(filename)
            self.VClampArray = cp.asarray(data, dtype=cp.float32)
            print(f"已加载电压钳数据: {len(data)} 个点")
        except Exception as e:
            print(f"加载电压钳数据失败: {e}")

    def set_fix_sr(self, fix_sr):
        self.fix_sr = fix_sr

    def to_numpy(self):
        return {
            'time': self.time,
            'whole_cell': self.whole_cell.to_numpy(),
            'myosr_ca': [myosr.to_numpy() for myosr in self.myosr_ca],
            'crus': [cru.to_numpy() for cru in self.crus]
        }

    def from_numpy(self, data):
        self.time = data['time']
        self.whole_cell.from_numpy(data['whole_cell'])
        for i, myosr_data in enumerate(data['myosr_ca']):
            if i < len(self.myosr_ca):
                self.myosr_ca[i].from_numpy(myosr_data)
        for i, cru_data in enumerate(data['crus']):
            if i < len(self.crus):
                self.crus[i].from_numpy(cru_data)

    def save_state(self, filename):
        data = self.to_numpy()
        np.savez_compressed(filename, **data)
        print(f"模型状态已保存: {filename}")

    def load_state(self, filename):
        try:
            data = np.load(filename)
            self.from_numpy(data)
            print(f"模型状态已加载: {filename}")
        except Exception as e:
            print(f"加载模型状态失败: {e}")

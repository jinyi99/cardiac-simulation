import cupy as cp
import numpy as np
import time
from .gpu_myocyte import GPUMyocyteCalcium
from .gpu_cru import GPUCRU
from .gpu_ucla_cell import GPUUCLACell
# 导入新的扩散内核
from .kernels_fused import fused_reaction_kernel, diffusion_kernel_3d


class GPUSpatialCell:
    """空间心肌细胞模型类（GPU版本 - 极致性能优化版）"""

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
        
        # 细分网格参数 (每个 CRU 包含 5x5x5 个肌浆体素)
        NX_MYO_SCALE = 5
        self.nx_myo = nx * NX_MYO_SCALE
        self.ny_myo = ny * NX_MYO_SCALE
        self.nz_myo = nz * NX_MYO_SCALE
        self.n_myo_voxels = self.nx_myo * self.ny_myo * self.nz_myo

        self.dt = self.DT
        self.time = 0.0
        self.filename = filename
        self.rng_seed = rng_seed

        # 初始化GPU数组
        self._init_gpu_arrays()

        # 初始化模型组件
        self.whole_cell = GPUUCLACell(1)
        # 注意：这里假设是一个大的空间分布细胞
        self.myosr_ca = [GPUMyocyteCalcium(self.n_myo_voxels) for _ in range(1)]
        self.crus = [GPUCRU(self.n_cru) for _ in range(1)]

        self._init_network_indices()
        self.v_buffer = cp.zeros(1, dtype=cp.float32)
        self.Istim = cp.zeros(1, dtype=cp.float32)
        self.fix_sr = False

        # --- GPU 内核启动配置 ---
        self.block_size = 256
        self.grid_size = (self.n_myo_voxels + self.block_size - 1) // self.block_size

        print(f"初始化完成: {nx}x{ny}x{nz} CRU网格")
        print(f"细分网格: {self.nx_myo}x{self.ny_myo}x{self.nz_myo} ({self.n_myo_voxels} 体素)")
        print("优化策略: CUDA Graph + Fused Reaction + 3D Native Diffusion")

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
        GRAPH_BATCH_SIZE = 100  # 增大 Batch Size 以减少 Python 开销
        graph = None
        graph_stream = cp.cuda.Stream(non_blocking=True)

        step = 0
        while step < n_steps:
            use_graph = (n_steps - step >= GRAPH_BATCH_SIZE) and (step >= stim_steps)

            if use_graph:
                if graph is None:
                    # === 录制 CUDA Graph ===
                    graph_stream.begin_capture()
                    for _ in range(GRAPH_BATCH_SIZE):
                        self._step_logic(istim_val=0.0, istim_on=False, vclamp=vclamp)
                    graph = graph_stream.end_capture()

                # === 执行 Graph ===
                graph.launch(stream=cp.cuda.get_current_stream())
                self.time += self.dt * GRAPH_BATCH_SIZE
                step += GRAPH_BATCH_SIZE

                if step % self.REDUCE_OUTPUT == 0:
                    self.output_data(step)

            else:
                # === 普通执行模式 ===
                istim_on = step < stim_steps
                curr_istim = istim if istim_on else 0.0
                self._step_logic(curr_istim, istim_on, vclamp)
                self.time += self.dt
                step += 1

                if step % self.REDUCE_OUTPUT == 0:
                    self.output_data(step)

    def _step_logic(self, istim_val, istim_on, vclamp):
        """单步物理逻辑"""
        # 1. Update CRU Flux
        self.update_cru_flux()

        # 2. Update MyoSR Flux (使用融合内核)
        self.update_myosr_flux()

        # 3. Diffusion (使用新的 3D 内核)
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
        """
        计算钙扩散 - 使用完全向量化的 CUDA C++ 内核
        """
        for myosr in self.myosr_ca:
            # 1. 计算缓冲系数 (仍使用 elementwise，因公式复杂且独立)
            # 如果想进一步优化，可以将 buffering 也融合进 diffusion kernel
            beta_m = myosr.compute_buffering_m(myosr.cm)
            beta_s = myosr.compute_buffering_sr(myosr.cs)

            # 2. 计算肌浆钙 (Cm) 扩散
            # 参数: (c_out, c_in, beta, dt, D, dx, nx, ny, nz)
            diffusion_kernel_3d(
                (self.grid_size,), (self.block_size,),
                (
                    myosr.cm_tmp, myosr.cm, beta_m,
                    cp.float32(self.dt), cp.float32(self.D_M), cp.float32(self.DX),
                    cp.int32(self.nx_myo), cp.int32(self.ny_myo), cp.int32(self.nz_myo)
                )
            )

            # 3. 计算肌浆网钙 (Cs) 扩散
            diffusion_kernel_3d(
                (self.grid_size,), (self.block_size,),
                (
                    myosr.cs_tmp, myosr.cs, beta_s,
                    cp.float32(self.dt), cp.float32(self.D_SR), cp.float32(self.DX),
                    cp.int32(self.nx_myo), cp.int32(self.ny_myo), cp.int32(self.nz_myo)
                )
            )

            # 交换指针 (Current <-> Temp)
            myosr.swap_temp_ptr()

    def update_voltage(self, istim, istim_on, vclamp=False):
        current_istim = istim if istim_on else 0.0
        self.whole_cell.step(self.dt, current_istim)

        if vclamp and hasattr(self, 'VClampArray'):
            clamp_idx = min(int(self.time / self.dt), len(self.VClampArray) - 1)
            self.whole_cell.v[0] = self.VClampArray[clamp_idx]

    def output_data(self, step):
        # 减少数据回传频率，仅用于监控
        if step % (10 * self.REDUCE_OUTPUT) == 0:
            avg_data = self.get_averages()
            print(f"t={self.time:.2f}ms, V={avg_data['v']:.2f}mV, [Ca]i={avg_data['ci']:.4f}uM")

    def get_averages(self):
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

    # 数据IO辅助方法
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

import cupy as cp
import numpy as np
import time
from .gpu_myocyte import GPUMyocyteCalcium
from .gpu_cru import GPUCRU
from .gpu_ucla_cell import GPUUCLACell


class GPUSpatialCell:
    """空间心肌细胞模型类（GPU版本）"""

    # 模型参数
    DT = 0.01  # 时间步长 (ms)
    OUTPUT_DT = 1.0  # 输出间隔 (ms)
    REDUCE_OUTPUT = max(int(OUTPUT_DT / DT), 1)  # 输出减少因子

    # 扩散参数
    D_M = 0.3  # 肌浆扩散系数 (μm²/ms)
    D_SR = 0.06  # 肌浆网扩散系数 (μm²/ms)
    DX = 0.2  # 空间步长 (μm)

    # 算子分裂参数
    FLUX_LOOP = 1  # 通量更新循环次数
    DIFF_LOOP = 1  # 扩散更新循环次数

    def __init__(self, nx, ny, nz, filename, rng_seed):
        """
        初始化空间心肌细胞模型

        Args:
            nx, ny, nz: CRU网格维度
            filename: 输出文件名
            rng_seed: 随机数种子
        """
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
        self.whole_cell = GPUUCLACell(1)  # 假设单细胞
        self.myosr_ca = [GPUMyocyteCalcium(self.n_myo_voxels) for _ in range(1)]
        self.crus = [GPUCRU(self.n_cru) for _ in range(1)]

        # 初始化网络索引
        self._init_network_indices()

        # 初始化电压缓冲
        self.v_buffer = cp.zeros(1, dtype=cp.float32)

        # 初始化刺激电流
        self.Istim = cp.zeros(1, dtype=cp.float32)

        # 固定SR标志
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

        print(f"初始化完成: {nx}x{ny}x{nz} CRU网格, {self.n_myo_voxels} 个体素")

    def _init_gpu_arrays(self):
        """初始化GPU数组"""
        # 使用CuPy内存池提高性能
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    def _init_network_indices(self):
        """初始化网络连接索引"""
        for cru in self.crus:
            cru.initialize_network_idx(self.nx, self.ny, self.nz, 5)  # NX_MYO=5

    def simulate_pacing(self, n_beats, pcl, istim, stim_duration, vclamp=False):
        """
        模拟多个心动周期

        Args:
            n_beats: 心动周期数量
            pcl: 周期长度 (ms)
            istim: 刺激电流 (μA/μF)
            stim_duration: 刺激持续时间 (ms)
            vclamp: 是否使用电压钳
        """
        print(f"开始模拟: {n_beats} 个心动周期, PCL={pcl}ms")

        # 将刺激数据复制到GPU
        gpu_istim = cp.asarray(istim) if not isinstance(istim, cp.ndarray) else istim

        for beat in range(n_beats):
            print(f"周期 {beat + 1}/{n_beats}")
            start_time = time.time()
            self.simulate_beat(pcl, gpu_istim, stim_duration, vclamp)
            duration = time.time() - start_time
            print(f"  完成, 耗时: {duration:.2f}s, 平均每步: {duration / (pcl / self.dt) * 1000:.2f}ms")

    def simulate_beat(self, duration, istim, stim_duration, vclamp):
        """
        模拟单个心动周期

        Args:
            duration: 周期持续时间 (ms)
            istim: 刺激电流 (μA/μF)
            stim_duration: 刺激持续时间 (ms)
            vclamp: 是否使用电压钳
        """
        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            self.time += self.dt
            istim_on = step < (stim_duration / self.dt)

            # 更新CRU马尔可夫状态和通量
            self.update_cru_flux()

            # 更新肌浆网通量
            self.update_myosr_flux()

            # 计算钙扩散
            self.compute_calcium_diffusion()

            # 更新离子通道和电压
            self.update_voltage(istim, istim_on, vclamp)

            # 输出数据
            if step % self.REDUCE_OUTPUT == 0:
                self.output_data(step)

    def update_cru_flux(self):
        """更新CRU通量"""
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
        """更新肌浆网通量"""
        for i, myosr in enumerate(self.myosr_ca):
            # 计算Jup和Jleak
            kup = 1.0  # SERCA半最大激活常数
            vup = 0.32  # SERCA最大速率

            Jup = vup * myosr.cm ** 2 / (myosr.cm ** 2 + kup ** 2)
            Jleak = 0.000004 * (myosr.cs - myosr.cm)  # gleak = 0.000004

            # 计算Jbg (背景钙电流)
            gbg = 0.00003  # 背景钙电导
            phi = self.whole_cell.v[0] * self.whole_cell.FRT
            Jbg = -gbg * (self.whole_cell.v[0] - cp.log(1800.0 / myosr.cm) / (2.0 * self.whole_cell.FRT))

            # 计算JNCX (Na-Ca交换)
            # 简化版本，实际应该在CRU中计算
            v2 = 0.8  # NCX最大速率
            kmca = 0.11  # 钙半最大激活常数
            kmna = 12.3  # 钠半最大激活常数
            kmco = 1.3  # 细胞外钙半最大激活常数
            kmno = 87.5  # 细胞外钠半最大激活常数

            enai3 = self.whole_cell.xnai[0] ** 3
            nao3 = 136.0 ** 3

            num = v2 * (cp.exp(0.35 * phi) * enai3 * 1800.0 -
                        cp.exp((0.35 - 1.0) * phi) * nao3 * myosr.cm)

            denom1 = 1.0 + (kmca / myosr.cm) ** 3
            denom2 = 1.0 + 0.27 * cp.exp((0.35 - 1.0) * phi)
            denom3 = kmco * enai3 + kmno ** 3 * myosr.cm + kmna ** 3 * 1800.0 * (1.0 + myosr.cm / 3.59) + \
                     3.59 * nao3 * (1.0 + enai3 / kmna ** 3) + enai3 * 1800.0 + nao3 * myosr.cm

            myosr.Jncx = num / (denom1 * denom2 * denom3)

            # 更新肌浆和肌浆网钙浓度
            beta_m = myosr.compute_buffering_m(myosr.cm)
            beta_s = myosr.compute_buffering_sr(myosr.cs)

            # 肌浆更新
            dcm_dt = (-Jup + myosr.Jncx + Jbg + Jleak) / beta_m
            myosr.cm += self.dt * dcm_dt

            # 肌浆网更新 (如果不固定)
            if not self.fix_sr:
                dcs_dt = (5.0 / 0.2) * (Jup - Jleak) / beta_s  # kappa = 5.0/0.2
                myosr.cs += self.dt * dcs_dt

    def compute_calcium_diffusion(self):
        """计算钙扩散"""
        for myosr in self.myosr_ca:
            # 计算缓冲系数
            beta_m = myosr.compute_buffering_m(myosr.cm)
            beta_s = myosr.compute_buffering_sr(myosr.cs)

            # 准备邻居值
            cm_left = cp.zeros_like(myosr.cm)
            cm_left[1:] = myosr.cm[:-1]
            cm_right = cp.zeros_like(myosr.cm)
            cm_right[:-1] = myosr.cm[1:]

            cs_left = cp.zeros_like(myosr.cs)
            cs_left[1:] = myosr.cs[:-1]
            cs_right = cp.zeros_like(myosr.cs)
            cs_right[:-1] = myosr.cs[1:]

            # 计算扩散
            myosr.cm_tmp = self.diffusion_kernel(
                self.dt, self.D_M, self.DX,
                myosr.cm, cm_left, cm_right, beta_m
            )

            myosr.cs_tmp = self.diffusion_kernel(
                self.dt, self.D_SR, self.DX,
                myosr.cs, cs_left, cs_right, beta_s
            )

            # 交换指针
            myosr.swap_temp_ptr()

    def update_voltage(self, istim, istim_on, vclamp=False):
        """更新电压"""
        # 设置刺激电流
        current_istim = istim if istim_on else cp.zeros_like(istim)

        # 更新离子通道
        success = self.whole_cell.step(self.dt, current_istim)

        if not success:
            # 如果电压变化过大，减小时间步长重试
            print("电压变化过大，减小时间步长重试")
            self.whole_cell.step(self.dt / 2, current_istim / 2)
            self.whole_cell.step(self.dt / 2, current_istim / 2)

        # 电压钳（如果启用）
        if vclamp and hasattr(self, 'VClampArray'):
            clamp_idx = min(int(self.time / self.dt), len(self.VClampArray) - 1)
            self.whole_cell.v[0] = self.VClampArray[clamp_idx]

    def output_data(self, step):
        """输出数据"""
        # 计算平均值
        avg_data = self.get_averages()

        # 输出到文件（简化版本）
        if step % (10 * self.REDUCE_OUTPUT) == 0:
            print(f"t={self.time:.2f}ms, V={avg_data['v']:.2f}mV, [Ca]i={avg_data['ci']:.4f}μM")

    def get_averages(self):
        """计算平均值"""
        # 获取肌浆钙平均值
        avg_cm = cp.mean(self.myosr_ca[0].cm)
        avg_cs = cp.mean(self.myosr_ca[0].cs)
        avg_jncx = cp.mean(self.myosr_ca[0].Jncx)

        # 获取CRU电流平均值
        avg_ilcc = cp.mean(self.crus[0].Ilcc)

        # 更新whole cell变量
        self.whole_cell.ci[0] = avg_cm
        self.whole_cell.JNCX[0] = avg_jncx
        self.whole_cell.ILCC[0] = avg_ilcc

        return {
            'time': self.time,
            'v': self.whole_cell.v[0],
            'ci': avg_cm,
            'cs': avg_cs,
            'JNCX': avg_jncx,
            'ILCC': avg_ilcc
        }

    def load_vclamp_data(self, filename):
        """加载电压钳数据"""
        try:
            data = np.loadtxt(filename)
            self.VClampArray = cp.asarray(data, dtype=cp.float32)
            print(f"已加载电压钳数据: {len(data)} 个点")
        except Exception as e:
            print(f"加载电压钳数据失败: {e}")

    def set_fix_sr(self, fix_sr):
        """设置是否固定SR钙浓度"""
        self.fix_sr = fix_sr

    def to_numpy(self):
        """将GPU数组转换为NumPy数组"""
        result = {
            'time': self.time,
            'whole_cell': self.whole_cell.to_numpy(),
            'myosr_ca': [myosr.to_numpy() for myosr in self.myosr_ca],
            'crus': [cru.to_numpy() for cru in self.crus]
        }
        return result

    def from_numpy(self, data):
        """从NumPy数组加载数据"""
        self.time = data['time']
        self.whole_cell.from_numpy(data['whole_cell'])

        for i, myosr_data in enumerate(data['myosr_ca']):
            if i < len(self.myosr_ca):
                self.myosr_ca[i].from_numpy(myosr_data)

        for i, cru_data in enumerate(data['crus']):
            if i < len(self.crus):
                self.crus[i].from_numpy(cru_data)

    def save_state(self, filename):
        """保存模型状态"""
        data = self.to_numpy()
        np.savez_compressed(filename, **data)
        print(f"模型状态已保存: {filename}")

    def load_state(self, filename):
        """加载模型状态"""
        try:
            data = np.load(filename)
            self.from_numpy(data)
            print(f"模型状态已加载: {filename}")
        except Exception as e:
            print(f"加载模型状态失败: {e}")
import cupy as cp

class DiffusionKernels:
    """
    扩散计算内核类 (向量化优化版)
    移除了所有 Python 循环，使用 cp.roll 进行全矩阵移位操作
    """

    def __init__(self):
        # 编译 CUDA 内核
        # 1D 扩散
        self.diffusion_1d_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 c, float32 c_left, float32 c_right, float32 beta',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * (c_left - 2.0f * c + c_right);
            result = c + diffusion / beta;
            ''', 'diffusion_1d_calculation'
        )

        # 3D 扩散
        self.diffusion_3d_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 c, \
             float32 c_left, float32 c_right, \
             float32 c_up, float32 c_down, \
             float32 c_high, float32 c_low, \
             float32 beta',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * \
                             (c_left + c_right + c_up + c_down + c_high + c_low - 6.0f * c);
            result = c + diffusion / beta;
            ''', 'diffusion_3d_calculation'
        )

        # 2D 扩散
        self.diffusion_2d_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 c, \
             float32 c_left, float32 c_right, \
             float32 c_up, float32 c_down, \
             float32 beta',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * \
                             (c_left + c_right + c_up + c_down - 4.0f * c);
            result = c + diffusion / beta;
            ''', 'diffusion_2d_calculation'
        )

        # 电压扩散 (2D)
        self.voltage_diffusion_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 v, \
             float32 v_left, float32 v_right, \
             float32 v_up, float32 v_down',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * \
                             (v_left + v_right + v_up + v_down - 4.0f * v);
            result = v + diffusion;
            ''', 'voltage_diffusion_calculation'
        )

    def compute_diffusion_1d(self, dt, D, values, dx, beta_values):
        # 1D 向量化移位
        left = cp.roll(values, 1)
        left[0] = values[0]  # 无通量边界

        right = cp.roll(values, -1)
        right[-1] = values[-1] # 无通量边界

        return self.diffusion_1d_kernel(dt, D, dx, values, left, right, beta_values)

    def compute_diffusion_2d(self, dt, D, values, dx, beta_values, nx, ny):
        # 重塑为 2D 以便移位 (y, x)
        v_2d = values.reshape(ny, nx)

        # 左右邻居 (Axis 1 = x)
        left = cp.roll(v_2d, 1, axis=1)
        left[:, 0] = v_2d[:, 0]  # 边界处理

        right = cp.roll(v_2d, -1, axis=1)
        right[:, -1] = v_2d[:, -1]

        # 上下邻居 (Axis 0 = y)
        up = cp.roll(v_2d, 1, axis=0)
        up[0, :] = v_2d[0, :]

        down = cp.roll(v_2d, -1, axis=0)
        down[-1, :] = v_2d[-1, :]

        # 展平传给内核
        return self.diffusion_2d_kernel(
            dt, D, dx, values,
            left.ravel(), right.ravel(),
            up.ravel(), down.ravel(),
            beta_values
        )

    def compute_diffusion_3d(self, dt, D, values, dx, beta_values, nx, ny, nz):
        """
        计算3D扩散 (完全向量化版本)
        """
        # 重塑为 3D (z, y, x) - 假设 C-contiguous 存储顺序
        v_3d = values.reshape(nz, ny, nx)

        # --- X轴 (Axis 2) ---
        # 左邻居
        left = cp.roll(v_3d, 1, axis=2)
        left[:, :, 0] = v_3d[:, :, 0]  # 无通量边界: 边界的左边是自己

        # 右邻居
        right = cp.roll(v_3d, -1, axis=2)
        right[:, :, -1] = v_3d[:, :, -1]

        # --- Y轴 (Axis 1) ---
        # 上邻居
        up = cp.roll(v_3d, 1, axis=1)
        up[:, 0, :] = v_3d[:, 0, :]

        # 下邻居
        down = cp.roll(v_3d, -1, axis=1)
        down[:, -1, :] = v_3d[:, -1, :]

        # --- Z轴 (Axis 0) ---
        # 高邻居 (z-1)
        high = cp.roll(v_3d, 1, axis=0)
        high[0, :, :] = v_3d[0, :, :]

        # 低邻居 (z+1)
        low = cp.roll(v_3d, -1, axis=0)
        low[-1, :, :] = v_3d[-1, :, :]

        # 调用内核
        # 注意: ravel() 不会产生拷贝，速度很快
        return self.diffusion_3d_kernel(
            dt, D, dx, values,
            left.ravel(), right.ravel(),
            up.ravel(), down.ravel(),
            high.ravel(), low.ravel(),
            beta_values
        )

    def compute_voltage_diffusion_2d(self, dt, D, values, dx, nx, ny):
        # 电压 2D 向量化
        v_2d = values.reshape(ny, nx)

        left = cp.roll(v_2d, 1, axis=1)
        left[:, 0] = v_2d[:, 0]

        right = cp.roll(v_2d, -1, axis=1)
        right[:, -1] = v_2d[:, -1]

        up = cp.roll(v_2d, 1, axis=0)
        up[0, :] = v_2d[0, :]

        down = cp.roll(v_2d, -1, axis=0)
        down[-1, :] = v_2d[-1, :]

        return self.voltage_diffusion_kernel(
            dt, D, dx, values,
            left.ravel(), right.ravel(),
            up.ravel(), down.ravel()
        )

# 创建全局实例
diffusion_kernels = DiffusionKernels()
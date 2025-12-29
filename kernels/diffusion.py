import cupy as cp
import numpy as np


class DiffusionKernels:
    """扩散计算内核类"""

    def __init__(self):
        """初始化扩散计算内核"""
        # 1D扩散内核 (用于线性数组)
        self.diffusion_1d_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 c, float32 c_left, float32 c_right, float32 beta',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * (c_left - 2.0f * c + c_right);
            result = c + diffusion / beta;
            ''',
            'diffusion_1d_calculation'
        )

        # 3D扩散内核 (用于三维数组)
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
            ''',
            'diffusion_3d_calculation'
        )

        # 2D扩散内核 (用于二维数组)
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
            ''',
            'diffusion_2d_calculation'
        )

        # 电压扩散内核
        self.voltage_diffusion_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 v, \
             float32 v_left, float32 v_right, \
             float32 v_up, float32 v_down',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * \
                             (v_left + v_right + v_up + v_down - 4.0f * v);
            result = v + diffusion;
            ''',
            'voltage_diffusion_calculation'
        )

        # 3D电压扩散内核
        self.voltage_diffusion_3d_kernel = cp.ElementwiseKernel(
            'float32 dt, float32 D, float32 dx, float32 v, \
             float32 v_left, float32 v_right, \
             float32 v_up, float32 v_down, \
             float32 v_high, float32 v_low',
            'float32 result',
            '''
            float diffusion = D * dt / (dx * dx) * \
                             (v_left + v_right + v_up + v_down + v_high + v_low - 6.0f * v);
            result = v + diffusion;
            ''',
            'voltage_diffusion_3d_calculation'
        )

    def compute_diffusion_1d(self, dt, D, values, dx, beta_values):
        """
        计算1D扩散

        Args:
            dt: 时间步长
            D: 扩散系数
            values: 浓度值数组
            dx: 空间步长
            beta_values: 缓冲系数数组

        Returns:
            扩散后的浓度数组
        """
        # 准备邻居值
        left_values = cp.zeros_like(values)
        left_values[1:] = values[:-1]
        right_values = cp.zeros_like(values)
        right_values[:-1] = values[1:]

        # 计算扩散
        return self.diffusion_1d_kernel(dt, D, dx, values, left_values, right_values, beta_values)

    def compute_diffusion_2d(self, dt, D, values, dx, beta_values, nx, ny):
        """
        计算2D扩散

        Args:
            dt: 时间步长
            D: 扩散系数
            values: 浓度值数组 (展平的一维数组)
            dx: 空间步长
            beta_values: 缓冲系数数组
            nx, ny: 网格维度

        Returns:
            扩散后的浓度数组
        """
        # 准备邻居值
        left_values = cp.zeros_like(values)
        right_values = cp.zeros_like(values)
        up_values = cp.zeros_like(values)
        down_values = cp.zeros_like(values)

        # 计算邻居索引
        for i in range(nx):
            for j in range(ny):
                idx = j * nx + i

                # 左邻居
                if i > 0:
                    left_values[idx] = values[idx - 1]
                else:
                    left_values[idx] = values[idx]  # 无通量边界条件

                # 右邻居
                if i < nx - 1:
                    right_values[idx] = values[idx + 1]
                else:
                    right_values[idx] = values[idx]  # 无通量边界条件

                # 上邻居
                if j > 0:
                    up_values[idx] = values[idx - nx]
                else:
                    up_values[idx] = values[idx]  # 无通量边界条件

                # 下邻居
                if j < ny - 1:
                    down_values[idx] = values[idx + nx]
                else:
                    down_values[idx] = values[idx]  # 无通量边界条件

        # 计算扩散
        return self.diffusion_2d_kernel(dt, D, dx, values,
                                        left_values, right_values,
                                        up_values, down_values,
                                        beta_values)

    def compute_diffusion_3d(self, dt, D, values, dx, beta_values, nx, ny, nz):
        """
        计算3D扩散

        Args:
            dt: 时间步长
            D: 扩散系数
            values: 浓度值数组 (展平的一维数组)
            dx: 空间步长
            beta_values: 缓冲系数数组
            nx, ny, nz: 网格维度

        Returns:
            扩散后的浓度数组
        """
        # 准备邻居值
        left_values = cp.zeros_like(values)
        right_values = cp.zeros_like(values)
        up_values = cp.zeros_like(values)
        down_values = cp.zeros_like(values)
        high_values = cp.zeros_like(values)
        low_values = cp.zeros_like(values)

        # 计算邻居索引
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = k * nx * ny + j * nx + i

                    # 左邻居
                    if i > 0:
                        left_values[idx] = values[idx - 1]
                    else:
                        left_values[idx] = values[idx]  # 无通量边界条件

                    # 右邻居
                    if i < nx - 1:
                        right_values[idx] = values[idx + 1]
                    else:
                        right_values[idx] = values[idx]  # 无通量边界条件

                    # 上邻居
                    if j > 0:
                        up_values[idx] = values[idx - nx]
                    else:
                        up_values[idx] = values[idx]  # 无通量边界条件

                    # 下邻居
                    if j < ny - 1:
                        down_values[idx] = values[idx + nx]
                    else:
                        down_values[idx] = values[idx]  # 无通量边界条件

                    # 高邻居 (z方向)
                    if k > 0:
                        high_values[idx] = values[idx - nx * ny]
                    else:
                        high_values[idx] = values[idx]  # 无通量边界条件

                    # 低邻居 (z方向)
                    if k < nz - 1:
                        low_values[idx] = values[idx + nx * ny]
                    else:
                        low_values[idx] = values[idx]  # 无通量边界条件

        # 计算扩散
        return self.diffusion_3d_kernel(dt, D, dx, values,
                                        left_values, right_values,
                                        up_values, down_values,
                                        high_values, low_values,
                                        beta_values)

    def compute_voltage_diffusion_2d(self, dt, D, values, dx, nx, ny):
        """
        计算2D电压扩散

        Args:
            dt: 时间步长
            D: 扩散系数
            values: 电压值数组 (展平的一维数组)
            dx: 空间步长
            nx, ny: 网格维度

        Returns:
            扩散后的电压数组
        """
        # 准备邻居值
        left_values = cp.zeros_like(values)
        right_values = cp.zeros_like(values)
        up_values = cp.zeros_like(values)
        down_values = cp.zeros_like(values)

        # 计算邻居索引
        for i in range(nx):
            for j in range(ny):
                idx = j * nx + i

                # 左邻居
                if i > 0:
                    left_values[idx] = values[idx - 1]
                else:
                    left_values[idx] = values[idx]  # 无通量边界条件

                # 右邻居
                if i < nx - 1:
                    right_values[idx] = values[idx + 1]
                else:
                    right_values[idx] = values[idx]  # 无通量边界条件

                # 上邻居
                if j > 0:
                    up_values[idx] = values[idx - nx]
                else:
                    up_values[idx] = values[idx]  # 无通量边界条件

                # 下邻居
                if j < ny - 1:
                    down_values[idx] = values[idx + nx]
                else:
                    down_values[idx] = values[idx]  # 无通量边界条件

        # 计算扩散
        return self.voltage_diffusion_kernel(dt, D, dx, values,
                                             left_values, right_values,
                                             up_values, down_values)

    def compute_voltage_diffusion_3d(self, dt, D, values, dx, nx, ny, nz):
        """
        计算3D电压扩散

        Args:
            dt: 时间步长
            D: 扩散系数
            values: 电压值数组 (展平的一维数组)
            dx: 空间步长
            nx, ny, nz: 网格维度

        Returns:
            扩散后的电压数组
        """
        # 准备邻居值
        left_values = cp.zeros_like(values)
        right_values = cp.zeros_like(values)
        up_values = cp.zeros_like(values)
        down_values = cp.zeros_like(values)
        high_values = cp.zeros_like(values)
        low_values = cp.zeros_like(values)

        # 计算邻居索引
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = k * nx * ny + j * nx + i

                    # 左邻居
                    if i > 0:
                        left_values[idx] = values[idx - 1]
                    else:
                        left_values[idx] = values[idx]  # 无通量边界条件

                    # 右邻居
                    if i < nx - 1:
                        right_values[idx] = values[idx + 1]
                    else:
                        right_values[idx] = values[idx]  # 无通量边界条件

                    # 上邻居
                    if j > 0:
                        up_values[idx] = values[idx - nx]
                    else:
                        up_values[idx] = values[idx]  # 无通量边界条件

                    # 下邻居
                    if j < ny - 1:
                        down_values[idx] = values[idx + nx]
                    else:
                        down_values[idx] = values[idx]  # 无通量边界条件

                    # 高邻居 (z方向)
                    if k > 0:
                        high_values[idx] = values[idx - nx * ny]
                    else:
                        high_values[idx] = values[idx]  # 无通量边界条件

                    # 低邻居 (z方向)
                    if k < nz - 1:
                        low_values[idx] = values[idx + nx * ny]
                    else:
                        low_values[idx] = values[idx]  # 无通量边界条件

        # 计算扩散
        return self.voltage_diffusion_3d_kernel(dt, D, dx, values,
                                                left_values, right_values,
                                                up_values, down_values,
                                                high_values, low_values)

    def apply_boundary_conditions(self, values, boundary_type='no_flux', boundary_value=0.0):
        """
        应用边界条件

        Args:
            values: 值数组
            boundary_type: 边界条件类型 ('no_flux', 'dirichlet', 'periodic')
            boundary_value: 边界值 (用于Dirichlet边界条件)

        Returns:
            应用边界条件后的数组
        """
        if boundary_type == 'no_flux':
            # 无通量边界条件已经在扩散内核中处理
            return values

        elif boundary_type == 'dirichlet':
            # Dirichlet边界条件 - 固定边界值
            # 这里需要根据具体的网格结构来设置边界
            # 这是一个占位符实现，具体实现取决于网格结构
            return values

        elif boundary_type == 'periodic':
            # 周期性边界条件
            # 这里需要根据具体的网格结构来设置边界
            # 这是一个占位符实现，具体实现取决于网格结构
            return values

        else:
            raise ValueError(f"不支持的边界条件类型: {boundary_type}")


# 创建全局实例
diffusion_kernels = DiffusionKernels()
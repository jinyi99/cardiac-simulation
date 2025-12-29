import cupy as cp
import numpy as np


class BufferingKernels:
    """缓冲计算内核类"""

    def __init__(self):
        """初始化缓冲计算内核"""
        # 肌浆钙缓冲计算内核
        self.beta_m_kernel = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bcd = 15.0f;
            const float Kcd = 13.0f;
            const float Bsr = 7.0f;
            const float Ksr = 0.3f;
            beta = 1.0f + Bcd * Kcd / ((c+Kcd)*(c+Kcd)) + Bsr * Ksr / ((c+Ksr)*(c+Ksr));
            ''',
            'beta_m_buffering'
        )

        # 肌浆网钙缓冲计算内核
        self.beta_sr_kernel = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bnsr = 140.0f;
            const float Knsr = 650.0f;
            beta = 1.0f + Bnsr * Knsr / ((c+Knsr)*(c+Knsr));
            ''',
            'beta_sr_buffering'
        )

        # dyadic空间钙缓冲计算内核
        self.beta_d_kernel = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bcd_prime = 24.0f;
            const float Kcd_prime = 7.0f;
            const float Bsr_prime = 47.0f;
            const float Ksr_prime = 0.6f;
            beta = 1.0f + Bcd_prime * Kcd_prime / ((c+Kcd_prime)*(c+Kcd_prime)) + 
                          Bsr_prime * Ksr_prime / ((c+Ksr_prime)*(c+Ksr_prime));
            ''',
            'beta_d_buffering'
        )

        # jSR钙缓冲计算内核
        self.beta_j_kernel = cp.ElementwiseKernel(
            'float32 c',
            'float32 beta',
            '''
            const float Bcal = 140.0f;
            const float Kcal = 650.0f;
            beta = 1.0f + Bcal * Kcal / ((c+Kcal)*(c+Kcal));
            ''',
            'beta_j_buffering'
        )

        # 钙调蛋白缓冲计算内核
        self.calmodulin_buffering_kernel = cp.ElementwiseKernel(
            'float32 c, float32 B_cam, float32 K_cam',
            'float32 beta',
            '''
            beta = 1.0f + B_cam * K_cam / ((c+K_cam)*(c+K_cam));
            ''',
            'calmodulin_buffering'
        )

        # 肌钙蛋白缓冲计算内核
        self.troponin_buffering_kernel = cp.ElementwiseKernel(
            'float32 c, float32 B_trop, float32 K_trop',
            'float32 beta',
            '''
            beta = 1.0f + B_trop * K_trop / ((c+K_trop)*(c+K_trop));
            ''',
            'troponin_buffering'
        )

        # SR结合蛋白缓冲计算内核
        self.sr_binding_buffering_kernel = cp.ElementwiseKernel(
            'float32 c, float32 B_sr, float32 K_sr',
            'float32 beta',
            '''
            beta = 1.0f + B_sr * K_sr / ((c+K_sr)*(c+K_sr));
            ''',
            'sr_binding_buffering'
        )

        # 通用缓冲计算内核
        self.general_buffering_kernel = cp.ElementwiseKernel(
            'float32 c, float32 B, float32 K',
            'float32 beta',
            '''
            beta = 1.0f + B * K / ((c+K)*(c+K));
            ''',
            'general_buffering'
        )

        # 多缓冲位点计算内核
        self.multi_site_buffering_kernel = cp.ElementwiseKernel(
            'float32 c, float32 B1, float32 K1, float32 B2, float32 K2, float32 B3, float32 K3',
            'float32 beta',
            '''
            beta = 1.0f + B1 * K1 / ((c+K1)*(c+K1)) + 
                          B2 * K2 / ((c+K2)*(c+K2)) + 
                          B3 * K3 / ((c+K3)*(c+K3));
            ''',
            'multi_site_buffering'
        )

    def compute_beta_m(self, c):
        """
        计算肌浆钙缓冲系数

        Args:
            c: 钙浓度数组

        Returns:
            缓冲系数数组
        """
        return self.beta_m_kernel(c)

    def compute_beta_sr(self, c):
        """
        计算肌浆网钙缓冲系数

        Args:
            c: 钙浓度数组

        Returns:
            缓冲系数数组
        """
        return self.beta_sr_kernel(c)

    def compute_beta_d(self, c):
        """
        计算dyadic空间钙缓冲系数

        Args:
            c: 钙浓度数组

        Returns:
            缓冲系数数组
        """
        return self.beta_d_kernel(c)

    def compute_beta_j(self, c):
        """
        计算jSR钙缓冲系数

        Args:
            c: 钙浓度数组

        Returns:
            缓冲系数数组
        """
        return self.beta_j_kernel(c)

    def compute_calmodulin_buffering(self, c, B_cam=15.0, K_cam=13.0):
        """
        计算钙调蛋白缓冲

        Args:
            c: 钙浓度数组
            B_cam: 钙调蛋白总浓度
            K_cam: 钙调蛋白解离常数

        Returns:
            缓冲系数数组
        """
        return self.calmodulin_buffering_kernel(c, B_cam, K_cam)

    def compute_troponin_buffering(self, c, B_trop=70.0, K_trop=0.5):
        """
        计算肌钙蛋白缓冲

        Args:
            c: 钙浓度数组
            B_trop: 肌钙蛋白总浓度
            K_trop: 肌钙蛋白解离常数

        Returns:
            缓冲系数数组
        """
        return self.troponin_buffering_kernel(c, B_trop, K_trop)

    def compute_sr_binding_buffering(self, c, B_sr=47.0, K_sr=0.6):
        """
        计算SR结合蛋白缓冲

        Args:
            c: 钙浓度数组
            B_sr: SR结合蛋白总浓度
            K_sr: SR结合蛋白解离常数

        Returns:
            缓冲系数数组
        """
        return self.sr_binding_buffering_kernel(c, B_sr, K_sr)

    def compute_general_buffering(self, c, B, K):
        """
        计算通用缓冲

        Args:
            c: 钙浓度数组
            B: 缓冲剂总浓度
            K: 解离常数

        Returns:
            缓冲系数数组
        """
        return self.general_buffering_kernel(c, B, K)

    def compute_multi_site_buffering(self, c, buffers):
        """
        计算多位点缓冲

        Args:
            c: 钙浓度数组
            buffers: 缓冲剂列表，每个元素为(B, K)元组

        Returns:
            缓冲系数数组
        """
        # 将缓冲剂列表转换为内核参数
        if len(buffers) == 1:
            B1, K1 = buffers[0]
            B2, K2 = 0.0, 1.0  # 默认值，不影响结果
            B3, K3 = 0.0, 1.0  # 默认值，不影响结果
        elif len(buffers) == 2:
            B1, K1 = buffers[0]
            B2, K2 = buffers[1]
            B3, K3 = 0.0, 1.0  # 默认值，不影响结果
        elif len(buffers) >= 3:
            B1, K1 = buffers[0]
            B2, K2 = buffers[1]
            B3, K3 = buffers[2]
        else:
            raise ValueError("至少需要一个缓冲剂")

        return self.multi_site_buffering_kernel(c, B1, K1, B2, K2, B3, K3)

    def compute_instantaneous_buffering(self, c, buffer_type='myoplasm'):
        """
        计算瞬时缓冲

        Args:
            c: 钙浓度数组
            buffer_type: 缓冲类型 ('myoplasm', 'sr', 'dyadic', 'jsr')

        Returns:
            缓冲系数数组
        """
        if buffer_type == 'myoplasm':
            return self.compute_beta_m(c)
        elif buffer_type == 'sr':
            return self.compute_beta_sr(c)
        elif buffer_type == 'dyadic':
            return self.compute_beta_d(c)
        elif buffer_type == 'jsr':
            return self.compute_beta_j(c)
        else:
            raise ValueError(f"不支持的缓冲类型: {buffer_type}")

    def compute_buffering_derivative(self, c, beta, dt):
        """
        计算缓冲导数

        Args:
            c: 钙浓度数组
            beta: 缓冲系数数组
            dt: 时间步长

        Returns:
            缓冲导数数组
        """
        # 使用有限差分计算导数
        # 这是一个简化实现，实际可能需要更复杂的计算
        derivative_kernel = cp.ElementwiseKernel(
            'float32 c, float32 beta, float32 dt',
            'float32 derivative',
            '''
            // 简化实现，实际可能需要考虑缓冲动力学
            derivative = 0.0f;  // 假设瞬时缓冲，导数为零
            ''',
            'buffering_derivative'
        )

        return derivative_kernel(c, beta, dt)


# 创建全局实例
buffering_kernels = BufferingKernels()
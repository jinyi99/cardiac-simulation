import cupy as cp
import numpy as np


class GPUMyocyteCalcium:
    """肌浆和肌浆网钙浓度管理类"""

    # 缓冲参数
    B_CD = 15.0
    K_CD = 13.0
    B_SR = 7.0
    K_SR = 0.3
    B_NSR = 140.0
    K_NSR = 650.0

    def __init__(self, n_voxels):
        """
        初始化肌浆和肌浆网钙浓度

        Args:
            n_voxels: 体素数量
        """
        self.n_voxels = n_voxels

        # 初始化钙浓度数组 (使用CuPy GPU数组)
        self.cm1 = cp.full(n_voxels, 0.1, dtype=cp.float32)  # 肌浆钙浓度1
        self.cs1 = cp.full(n_voxels, 1500.0, dtype=cp.float32)  # 肌浆网钙浓度1
        self.cm2 = cp.full(n_voxels, 0.1, dtype=cp.float32)  # 肌浆钙浓度2
        self.cs2 = cp.full(n_voxels, 1500.0, dtype=cp.float32)  # 肌浆网钙浓度2

        # NCX通量
        self.Jncx = cp.zeros(n_voxels, dtype=cp.float32)

        # 当前激活的指针
        self.cm = self.cm1
        self.cs = self.cs1
        self.cm_tmp = self.cm2
        self.cs_tmp = self.cs2

        # 定义缓冲计算内核
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

    def swap_temp_ptr(self):
        """交换临时指针"""
        self.cm, self.cm_tmp = self.cm_tmp, self.cm
        self.cs, self.cs_tmp = self.cs_tmp, self.cs

    def compute_buffering_m(self, c):
        """计算肌浆钙缓冲系数"""
        return self.beta_m_kernel(c)

    def compute_buffering_sr(self, c):
        """计算肌浆网钙缓冲系数"""
        return self.beta_sr_kernel(c)

    def get_averages(self):
        """计算平均钙浓度"""
        avg_cm = cp.mean(self.cm)
        avg_cs = cp.mean(self.cs)
        avg_jncx = cp.mean(self.Jncx)

        return {
            'avg_cm': avg_cm,
            'avg_cs': avg_cs,
            'avg_jncx': avg_jncx
        }

    def to_numpy(self):
        """将GPU数组转换为NumPy数组（用于分析和可视化）"""
        return {
            'cm': cp.asnumpy(self.cm),
            'cs': cp.asnumpy(self.cs),
            'Jncx': cp.asnumpy(self.Jncx)
        }

    def from_numpy(self, data):
        """从NumPy数组加载数据到GPU"""
        self.cm = cp.asarray(data['cm'], dtype=cp.float32)
        self.cs = cp.asarray(data['cs'], dtype=cp.float32)
        self.Jncx = cp.asarray(data['Jncx'], dtype=cp.float32)
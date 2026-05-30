import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from pathlib import Path
import cupy as cp


class Visualization:
    """可视化类"""

    def __init__(self, output_dir="results/figures"):
        """
        初始化可视化工具

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_action_potential(self, results, filename=None, title=None):
        """
        绘制动作电位

        Args:
            results: 结果字典（包含'time'和'v'）
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据
        time = results['time']
        voltage = results['v']

        # 绘制动作电位
        ax.plot(time, voltage, 'b-', linewidth=2, label='Action Potential')

        # 设置图表属性
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Voltage (mV)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Action Potential', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, ax

    def plot_calcium_transient(self, results, filename=None, title=None):
        """
        绘制钙瞬变

        Args:
            results: 结果字典（包含'time'和'ci'）
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据
        time = results['time']
        calcium = results['ci']

        # 绘制钙瞬变
        ax.plot(time, calcium, 'r-', linewidth=2, label='Calcium Transient')

        # 设置图表属性
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Calcium Concentration (μM)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Calcium Transient', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, ax

    def plot_combined_traces(self, results, filename=None, title=None):
        """
        绘制组合轨迹（动作电位和钙瞬变）

        Args:
            results: 结果字典（包含'time', 'v'和'ci'）
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 提取数据
        time = results['time']
        voltage = results['v']
        calcium = results['ci']

        # 绘制动作电位
        ax1.plot(time, voltage, 'b-', linewidth=2, label='Action Potential')
        ax1.set_ylabel('Voltage (mV)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)

        # 绘制钙瞬变
        ax2.plot(time, calcium, 'r-', linewidth=2, label='Calcium Transient')
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Calcium Concentration (μM)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)

        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Action Potential and Calcium Transient', fontsize=16)

        plt.tight_layout()

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, (ax1, ax2)

    def plot_currents(self, results, filename=None, title=None):
        """
        绘制电流轨迹

        Args:
            results: 结果字典（包含'time', 'JNCX', 'ILCC'等）
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据
        time = results['time']

        # 绘制各种电流
        if 'JNCX' in results:
            ax.plot(time, results['JNCX'], label='JNCX', linewidth=2)

        if 'ILCC' in results:
            ax.plot(time, results['ILCC'], label='ILCC', linewidth=2)

        # 设置图表属性
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Current (μA/μF)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Membrane Currents', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, ax

    def create_wave_propagation_animation(self, voltage_data, nx, ny, filename, fps=10):
        """
        创建波传播动画

        Args:
            voltage_data: 电压数据数组（形状为(time, nx*ny)）
            nx, ny: 网格维度
            filename: 输出文件名
            fps: 帧率

        Returns:
            ani: 动画对象
        """
        # 准备数据
        data = voltage_data.reshape((-1, ny, nx))
        vmin, vmax = np.min(voltage_data), np.max(voltage_data)

        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))

        # 创建初始图像
        im = ax.imshow(data[0], cmap='jet',
                       norm=colors.Normalize(vmin=vmin, vmax=vmax),
                       origin='lower', animated=True)
        ax.set_title('Wave Propagation at t=0 ms')
        fig.colorbar(im, ax=ax, label='Voltage (mV)')

        # 更新函数
        def update(frame):
            im.set_array(data[frame])
            ax.set_title(f'Wave Propagation at t={frame * 10} ms')
            return im,

        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(data),
                            interval=1000 / fps, blit=True)

        # 保存动画
        ani.save(self.output_dir / filename, writer='pillow', fps=fps)
        print(f"动画已保存: {self.output_dir / filename}")

        plt.close()

        return ani

    def plot_apd_map(self, voltage_data, nx, ny, filename=None, title=None):
        """
        绘制动作电位时程(APD)分布图

        Args:
            voltage_data: 电压数据数组（形状为(time, nx*ny)）
            nx, ny: 网格维度
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            apd_map: APD分布图
        """
        # 计算APD90
        apd_map = np.zeros((ny, nx))

        for i in range(nx):
            for j in range(ny):
                idx = j * nx + i
                v_data = voltage_data[:, idx]
                apd_map[j, i] = self._calculate_apd90(v_data)

        # 绘制APD分布
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(apd_map, cmap='viridis', origin='lower')

        # 设置图表属性
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Action Potential Duration (APD90) Distribution', fontsize=14)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('APD90 (ms)', fontsize=12)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        plt.close()

        return apd_map

    def _calculate_apd90(self, voltage):
        """
        计算APD90

        Args:
            voltage: 电压时间序列

        Returns:
            apd90: APD90值
        """
        # 找到动作电位峰值
        peak_idx = np.argmax(voltage)
        peak_voltage = voltage[peak_idx]

        # 计算90%复极化水平
        repol_level = peak_voltage - 0.9 * (peak_voltage - np.min(voltage))

        # 找到复极化到90%的时间点
        repol_idx = None
        for i in range(peak_idx, len(voltage)):
            if voltage[i] <= repol_level:
                repol_idx = i
                break

        if repol_idx is None:
            return np.nan

        # 计算APD90
        return (repol_idx - peak_idx) * 0.01  # 假设dt=0.01ms

    def plot_spatial_calcium(self, calcium_data, nx, ny, nz, time_point, filename=None, title=None):
        """
        绘制空间钙分布

        Args:
            calcium_data: 钙浓度数据数组
            nx, ny, nz: 网格维度
            time_point: 时间点
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        # 重塑数据为3D
        calcium_3d = calcium_data.reshape((nx, ny, nz))

        # 选择要可视化的切片
        if nz > 1:
            # 3D数据，选择中间切片
            z_slice = nz // 2
            calcium_slice = calcium_3d[:, :, z_slice]
        else:
            # 2D数据
            calcium_slice = calcium_3d[:, :, 0]

        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(calcium_slice.T, cmap='hot', origin='lower')

        # 设置图表属性
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Spatial Calcium Distribution at t={time_point} ms', fontsize=14)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Calcium Concentration (μM)', fontsize=12)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, ax

    def plot_phase_space(self, results, x_var='v', y_var='ci', filename=None, title=None):
        """
        绘制相空间图

        Args:
            results: 结果字典
            x_var: x轴变量
            y_var: y轴变量
            filename: 输出文件名（如果为None，不保存）
            title: 图表标题

        Returns:
            fig, ax: 图表对象
        """
        if x_var not in results or y_var not in results:
            print(f"错误: 结果中缺少 {x_var} 或 {y_var}")
            return None, None

        fig, ax = plt.subplots(figsize=(8, 8))

        # 提取数据
        x_data = results[x_var]
        y_data = results[y_var]

        # 绘制相空间图
        ax.plot(x_data, y_data, 'b-', linewidth=2)

        # 设置图表属性
        ax.set_xlabel(x_var, fontsize=12)
        ax.set_ylabel(y_var, fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Phase Space: {y_var} vs {x_var}', fontsize=14)

        ax.grid(True, alpha=0.3)

        # 保存图表
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {self.output_dir / filename}")

        return fig, ax
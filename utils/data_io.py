import h5py
import cupy as cp
import numpy as np
import json
from pathlib import Path


class DataIO:
    """数据输入输出管理类"""

    def __init__(self, base_filename):
        """
        初始化数据IO管理器

        Args:
            base_filename: 基础文件名
        """
        self.base_filename = base_filename
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)

        # 创建数据文件
        self.data_file = self.output_dir / f"{base_filename}_data.h5"

        # 初始化 results 属性
        self.results = {}

    def save_simulation_data(self, cell, metadata):
        """保存仿真数据到HDF5文件"""
        filename = f"{self.output_dir}/simulation_data.h5"

        with h5py.File(filename, 'w') as f:
            # 保存元数据
            f.attrs['metadata'] = json.dumps(metadata, ensure_ascii=False)

            # 保存结果数据
            for key, value in self.results.items():
                # 检查并转换CuPy数组
                if hasattr(value, 'get'):  # CuPy数组有.get()方法
                    value = value.get()  # 转换为NumPy
                elif hasattr(value, '__array__') and str(type(value)).startswith('<class \'cupy'):
                    value = value.get()
                f.create_dataset(key, data=value)

            # 保存完整状态（如果需要）
            if hasattr(cell, 'to_numpy'):
                full_state = cell.to_numpy()
                state_group = f.create_group('full_state')
                self._save_nested_dict(state_group, full_state)

    def _save_nested_dict(self, group, data_dict):
        """递归保存嵌套字典，处理CuPy数组"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_nested_dict(subgroup, value)
            else:
                try:
                    # 转换CuPy数组
                    if hasattr(value, 'get'):  # CuPy数组
                        value = value.get()
                    elif hasattr(value, '__array__') and str(type(value)).startswith('<class \'cupy'):
                        value = value.get()

                    # 尝试创建数据集
                    group.create_dataset(key, data=value)

                except TypeError as e:
                    print(f"无法保存数据 '{key}'，类型为 {type(value)}: {e}")
                except Exception as e:
                    print(f"保存 '{key}' 时发生错误：{e}")

    def load_simulation_data(self, filename=None):
        """
        从HDF5文件加载模拟数据

        Args:
            filename: 文件名（如果为None，使用默认文件名）

        Returns:
            metadata: 元数据字典
            results: 结果字典
        """
        if filename is None:
            filename = self.data_file

        with h5py.File(filename, 'r') as f:
            # 加载元数据
            metadata = json.loads(f.attrs['metadata'])

            # 加载结果
            results = {}
            for key in f.keys():
                if key not in ['parameters', 'full_state']:
                    results[key] = f[key][()]

            return metadata, results

    def export_to_csv(self, results, prefix=None):
        """
        将结果导出为CSV文件（修复版）

        Args:
            results: 结果字典
            prefix: 文件名前缀（如果为None，使用基础文件名）
        """
        if prefix is None:
            prefix = self.base_filename

        # 确保输出目录存在
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # 调试信息
        print("=== CSV导出调试信息 ===")
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: 类型={type(value)}, 长度={len(value)}")
                if len(value) > 0:
                    print(f"  首值={value[0]}, 尾值={value[-1]}")
            else:
                print(f"{key}: 类型={type(value)}")

        # 检查时间数组
        if 'time' not in results:
            print("错误: 结果中缺少时间数组")
            return

        time_data = np.array(results['time'])
        print(f"时间数据: 长度={len(time_data)}, 范围=[{time_data.min():.6f}, {time_data.max():.6f}]")

        # 导出每个变量
        for key, value in results.items():
            if key == 'time':
                continue

            try:
                # 确保数据为numpy数组
                if hasattr(value, 'get'):  # CuPy数组
                    value = value.get()
                value_data = np.array(value)

                # 检查数组维度匹配
                if len(value_data) != len(time_data):
                    print(f"警告: {key}数据长度({len(value_data)})与时间长度({len(time_data)})不匹配")
                    continue

                # 保存CSV文件
                filename = csv_dir / f"{prefix}_{key}.csv"
                data = np.column_stack((time_data, value_data))
                np.savetxt(filename, data, delimiter=',',
                           header=f'time,{key}', comments='')

                print(f"已导出 {filename}")

            except Exception as e:
                print(f"导出 {key} 时出错: {e}")

    def export_to_vtk(self, cell, time_point, prefix=None):
        """
        将空间数据导出为VTK格式（用于ParaView等可视化工具）

        Args:
            cell: 细胞模型实例
            time_point: 时间点
            prefix: 文件名前缀（如果为None，使用基础文件名）
        """
        if prefix is None:
            prefix = self.base_filename

        # 确保输出目录存在
        vtk_dir = self.output_dir / "vtk"
        vtk_dir.mkdir(exist_ok=True)

        try:
            import pyvista as pv
        except ImportError:
            print("请安装pyvista库以导出VTK格式: pip install pyvista")
            return

        # 创建网格
        if hasattr(cell, 'nx') and hasattr(cell, 'ny') and hasattr(cell, 'nz'):
            # 3D网格
            grid = pv.UniformGrid()
            grid.dimensions = (cell.nx, cell.ny, cell.nz)
            grid.spacing = (cell.DX, cell.DX, cell.DX)
            grid.origin = (0, 0, 0)

            # 添加钙浓度数据
            if hasattr(cell, 'myosr_ca') and len(cell.myosr_ca) > 0:
                cm_data = cp.asnumpy(cell.myosr_ca[0].cm)
                cm_3d = cm_data.reshape((cell.nx, cell.ny, cell.nz), order='F')
                grid.point_data['calcium_myoplasm'] = cm_3d.flatten(order='F')

                cs_data = cp.asnumpy(cell.myosr_ca[0].cs)
                cs_3d = cs_data.reshape((cell.nx, cell.ny, cell.nz), order='F')
                grid.point_data['calcium_sr'] = cs_3d.flatten(order='F')

            # 保存VTK文件
            filename = vtk_dir / f"{prefix}_t{time_point:.2f}.vti"
            grid.save(filename)
            print(f"已导出VTK文件: {filename}")

        else:
            print("无法导出VTK格式: 缺少网格维度信息")

    def load_voltage_clamp_data(self, filename):
        """
        加载电压钳数据

        Args:
            filename: 电压钳数据文件名

        Returns:
            voltage_data: 电压数据数组
        """
        try:
            data = np.loadtxt(filename)
            return cp.asarray(data, dtype=cp.float32)
        except Exception as e:
            print(f"加载电压钳数据失败: {e}")
            return None
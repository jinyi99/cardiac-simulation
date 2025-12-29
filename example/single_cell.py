"""
单细胞模拟示例（修复版）
使用CuPy实现的心肌细胞电生理-钙循环模型进行单细胞模拟
"""
import sys
import os
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home/xp/Code/cardiac-simulation")
# 添加父目录到路径 .
from models.gpu_spatial_cell import GPUSpatialCell
from utils.visualization import Visualization
from utils.data_io import DataIO
from config.default_params import DefaultParams


def single_cell_example(params=None):
    """
    Args:
        params: 参数字典（如果为None，使用默认参数）

    Returns:
        results: 模拟结果字典
    """
    # 使用默认参数或提供的参数
    if params is None:
        params = DefaultParams.get_all_params()

    print("=" * 60)
    print("单细胞心肌电生理-钙循环模拟")
    print("=" * 60)

    # 创建输出目录
    output_dir = "results/single_cell"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据IO和可视化工具
    data_io = DataIO("single_cell_simulation")
    viz = Visualization(output_dir)

    print("初始化细胞模型...")
    start_time = time.time()

    # 创建细胞模型
    cell = GPUSpatialCell(
        nx=params['NX'],
        ny=params['NY'],
        nz=params['NZ'],
        filename="single_cell_simulation",
        rng_seed=params['RNG_SEED']
    )

    # 更新细胞参数
    cell.dt = params['DT']

    # 设置初始条件
    cell.whole_cell.v[0] = params['INITIAL_V']
    cell.whole_cell.xnai[0] = params['INITIAL_XNAI']

    for myosr in cell.myosr_ca:
        myosr.cm.fill(params['INITIAL_CM'])
        myosr.cs.fill(params['INITIAL_CS'])

    for cru in cell.crus:
        cru.cd.fill(params['INITIAL_CD'])
        cru.cj.fill(params['INITIAL_CJ'])

    init_time = time.time() - start_time
    print(f"初始化完成，耗时: {init_time:.2f}秒")

    # 准备刺激数据
    print("准备刺激数据...")
    istim = cp.zeros(1, dtype=cp.float32)
    istim[0] = params['STIM_AMPLITUDE']

    # 运行模拟
    print("开始模拟...")
    sim_start_time = time.time()

    # 计算模拟步数
    n_steps = int(params['SIMULATION_DURATION'] / params['DT'])
    stim_steps = int(params['STIM_DURATION'] / params['DT'])
    stim_start_step = int(params['STIM_START'] / params['DT'])

    # 修复：计算实际记录的数据点数量
    save_interval = params.get('SAVE_INTERVAL', 1)  # 默认每步都保存
    n_records = (n_steps - 1) // save_interval + 1

    # 存储结果（修复：使用实际记录点数量）
    results = {
        'time': [],
        'v': [],
        'ci': [],
        'JNCX': [],
        'ILCC': []
    }

    print(f"总步数: {n_steps}, 记录间隔: {save_interval}, 预计记录点: {n_records}")

    # 主模拟循环
    for step in range(n_steps):
        cell.time = step * params['DT']

        # 确定是否施加刺激
        stim_on = (step >= stim_start_step and
                   step < stim_start_step + stim_steps)
        current_istim = istim if stim_on else cp.zeros_like(istim)

        # 更新CRU通量
        cell.update_cru_flux()

        # 更新肌浆网通量
        cell.update_myosr_flux()

        # 计算钙扩散
        cell.compute_calcium_diffusion()

        # 更新离子通道和电压
        cell.update_voltage(current_istim, stim_on)

        # 修复：存储结果
        if step % save_interval == 0:
            try:
                avg_data = cell.get_averages()
                results['time'].append(float(avg_data['time']))
                results['v'].append(float(avg_data['v']))
                results['ci'].append(float(avg_data['ci']))
                results['JNCX'].append(float(avg_data['JNCX']))
                results['ILCC'].append(float(avg_data['ILCC']))
            except Exception as e:
                print(f"步骤 {step} 记录数据时出错: {e}")
                # 使用备用方法记录数据
                results['time'].append(float(step * params['DT']))
                results['v'].append(float(cell.whole_cell.v[0].get() if hasattr(cell.whole_cell.v[0], 'get') else cell.whole_cell.v[0]))

                # 计算平均钙浓度
                if hasattr(cell, 'myosr_ca') and len(cell.myosr_ca) > 0:
                    ci_avg = float(cp.mean(cell.myosr_ca[0].cm).get())
                else:
                    ci_avg = 0.0
                results['ci'].append(ci_avg)

                # 默认值用于其他变量
                results['JNCX'].append(0.0)
                results['ILCC'].append(0.0)

        # 显示进度
        if step % 1000 == 0:
            print(f"进度: {step}/{n_steps} 步, 时间: {cell.time:.1f} ms")

    sim_time = time.time() - sim_start_time
    print(f"模拟完成，耗时: {sim_time:.2f}秒")
    print(f"平均每步时间: {sim_time / n_steps * 1000:.2f} ms")

    # 修复：转换为numpy数组
    print("转换数据格式...")
    for key in results:
        results[key] = np.array(results[key])
        print(f"{key}: 长度={len(results[key])}, 范围=[{results[key].min():.6f}, {results[key].max():.6f}]")

    # 保存结果
    if params.get('SAVE_RESULTS', True):
        print("保存结果...")
        metadata = {
            'simulation_duration': params['SIMULATION_DURATION'],
            'dt': params['DT'],
            'save_interval': save_interval,
            'n_steps': n_steps,
            'n_records': len(results['time']),
            'stim_amplitude': params['STIM_AMPLITUDE'],
            'stim_duration': params['STIM_DURATION'],
            'stim_start': params['STIM_START'],
            'simulation_time': sim_time
        }

        try:
            # 设置data_io的results
            data_io.results = results
            data_io.save_simulation_data(cell, metadata)
            data_io.export_to_csv(results)
        except Exception as e:
            print(f"保存数据时出错: {e}")

    # 可视化结果
    if params.get('PLOT_RESULTS', True):
        print("生成可视化结果...")

        try:
            # 绘制动作电位
            viz.plot_action_potential(
                results,
                "action_potential.png",
                "Single Cell - Action Potential"
            )

            # 绘制钙瞬变
            viz.plot_calcium_transient(
                results,
                "calcium_transient.png",
                "Single Cell - Calcium Transient"
            )

            # 绘制组合轨迹
            viz.plot_combined_traces(
                results,
                "combined_traces.png",
                "Single Cell - Combined Traces"
            )

            # 绘制电流
            viz.plot_currents(
                results,
                "currents.png",
                "Single Cell - Membrane Currents"
            )

            # 绘制相空间图
            try:
                viz.plot_phase_space(
                    results,
                    "phase_space.png",
                    "Single Cell - Phase Space"
                )
            except Exception as e:
                print(f"绘制相空间图时出错: {e}")

        except Exception as e:
            print(f"可视化时出错: {e}")
            # 使用matplotlib作为后备
            try:
                import matplotlib.pyplot as plt

                # 动作电位图
                plt.figure(figsize=(10, 6))
                plt.plot(results['time'], results['v'], 'b-', linewidth=2)
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (mV)')
                plt.title('Single Cell - Action Potential')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/action_potential.png", dpi=300)
                plt.close()
                print(f"图表已保存: {output_dir}/action_potential.png")

                # 钙瞬变图
                plt.figure(figsize=(10, 6))
                plt.plot(results['time'], results['ci'], 'r-', linewidth=2)
                plt.xlabel('Time (ms)')
                plt.ylabel('Calcium Concentration (μM)')
                plt.title('Single Cell - Calcium Transient')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/calcium_transient.png", dpi=300)
                plt.close()
                print(f"图表已保存: {output_dir}/calcium_transient.png")

            except Exception as e2:
                print(f"matplotlib后备可视化也失败: {e2}")

    print("=" * 60)
    print("模拟完成!")
    if len(results['v']) > 0:
        print(f"最终电压: {results['v'][-1]:.2f} mV")
        print(f"最终钙浓度: {results['ci'][-1]:.4f} μM")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # 运行单细胞模拟
    results = single_cell_example()
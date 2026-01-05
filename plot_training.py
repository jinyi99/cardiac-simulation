import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results():
    """
    读取 SB3 的 Monitor 日志并画图 (增强版：支持递归搜索和自动路径定位)
    """
    # 1. 自动定位 logs 目录 (基于脚本所在位置)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs")

    title = "Learning Curve"
    data_frames = []

    if not os.path.exists(log_dir):
        print(f"错误: 找不到日志目录 {log_dir}")
        print("请确认您是否已运行 train_agent.py 并且生成了 logs 文件夹。")
        return

    print(f"正在递归搜索 {log_dir} 下的日志文件...")

    # 2. 递归搜索所有子文件夹
    csv_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith("monitor.csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("未找到任何 .monitor.csv 文件。")
        print("可能原因：")
        print("1. 训练尚未开始或 Monitor 未正确初始化。")
        print("2. 文件名不包含 .monitor.csv。")
        print(f"当前搜索路径: {log_dir}")
        return

    print(f"找到 {len(csv_files)} 个日志文件，正在读取...")

    # 3. 读取数据
    for file_path in csv_files:
        try:
            # SB3 的 CSV 前两行是元数据，需要跳过
            df = pd.read_csv(file_path, skiprows=1)
            data_frames.append(df)
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")

    if not data_frames:
        print("没有有效的训练数据可供绘制。")
        return

    # 合并数据
    full_df = pd.concat(data_frames, axis=0)

    # 确保按照时间排序 (如果读取了多个文件)
    # 这里的 't' 是墙上时间 (wall time)
    if 't' in full_df.columns:
        full_df = full_df.sort_values('t')

    # 计算累计步数 (l 是 episode length)
    # 如果是多个文件拼接，简单的 cumsum 可能不准确，最好用 total_timesteps 如果有的话
    # 这里我们假设是单次连续训练，直接累加
    full_df['cumulative_steps'] = full_df['l'].cumsum()

    # 4. 计算滑动平均
    window_size = 50
    if len(full_df) < window_size:
        window_size = max(1, len(full_df) // 5)

    full_df['rolling_reward'] = full_df['r'].rolling(window=window_size).mean()

    # 5. 绘图
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.plot(full_df['cumulative_steps'], full_df['r'],
             alpha=0.3, color='gray', label='Raw Reward')

    # 绘制平滑曲线
    plt.plot(full_df['cumulative_steps'], full_df['rolling_reward'],
             color='blue', linewidth=2, label=f'Moving Avg ({window_size} eps)')

    plt.xlabel('Total Timesteps')
    plt.ylabel('Episode Reward')
    plt.title(f'{title} (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 保存图片到脚本所在目录
    output_file = os.path.join(current_dir, "training_curve.png")
    plt.savefig(output_file, dpi=150)
    print(f"\n成功! 图表已保存为: {output_file}")
    print("请在 PyCharm 左侧文件栏中刷新并双击查看。")


if __name__ == "__main__":
    plot_results()
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_learning_curve(log_folder, window_size=100):
    """读取 monitor.csv 并绘制带平滑区间的学习曲线"""
    
    # 寻找 monitor.csv 文件 (SB3 默认会生成带有 .monitor.csv 后缀的文件)
    files = [f for f in os.listdir(log_folder) if f.endswith('monitor.csv')]
    if not files:
        print("未找到 monitor.csv 文件，请检查路径。")
        return
        
    csv_path = os.path.join(log_folder, files[0])
    
    # SB3 的 monitor.csv 前两行是注释，真实数据从第三行开始
    df = pd.read_csv(csv_path, skiprows=1)
    
    # 'r' 列是回合奖励 (reward), 'l' 列是回合长度 (length)
    rewards = df['r'].values
    
    # 计算滑动平均（为了让论文里的图好看、平滑）
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
    
    plt.figure(figsize=(10, 6))
    
    # 画出原始的震荡数据（浅色背景）
    plt.plot(rewards, color='royalblue', alpha=0.3, label='Episode Reward')
    
    # 画出平滑后的曲线（深色主线）
    plt.plot(smoothed_rewards, color='navy', linewidth=2, label=f'Moving Average (Window={window_size})')
    
    plt.title('Reinforcement Learning Optimization Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=300)
    print("✅ 论文级学习曲线图已保存为 learning_curve.png")

if __name__ == "__main__":
    # 替换为你实际的 log 文件夹名字
    plot_learning_curve("./logs_ap_stim/")
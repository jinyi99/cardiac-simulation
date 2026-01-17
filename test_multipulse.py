import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from envs.multipulse_env import MultiPulseCardiacEnv


def test_model():
    print("加载多脉冲模型进行测试...")

    env = MultiPulseCardiacEnv()

    # 路径检查
    model_path = os.path.join(current_dir, "checkpoints_multipulse", "sac_multipulse_final")
    if not os.path.exists(model_path + ".zip"):
        print(f"找不到模型: {model_path}.zip，请先运行 train_multipulse.py")
        return

    model = SAC.load(model_path)

    obs, _ = env.reset()

    # 预测动作
    action, _ = model.predict(obs, deterministic=True)

    # 解析动作
    amp = float(action[0])
    width = float(action[1])
    interval = float(action[2])
    count = int(action[3])

    print(f"\n>>> 智能体生成的优化刺激策略:")
    print(f"  - 脉冲数量 (Count): {count}")
    print(f"  - 脉冲振幅 (Amp):   {amp:.2f} uA/uF")
    print(f"  - 单脉冲宽 (Width): {width:.2f} ms")
    print(f"  - 脉冲间隔 (Inter): {interval:.2f} ms")
    print("-" * 40)

    # 运行模拟
    obs, reward, done, truncated, info = env.step(action)

    print(f"  - 模拟结果: {info.get('outcome', 'Unknown')}")
    print(f"  - 能量消耗: {info.get('energy', 0):.2f}")
    print(f"  - 获得奖励: {reward:.2f}")

    # 绘图
    trace = env.last_beat_trace
    time = np.array(trace['time'])
    v = np.array(trace['v'])
    i_stim = np.array(trace['i_stim'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # V
    ax1.plot(time, v, color='navy', linewidth=2, label='Membrane Voltage')
    ax1.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='Threshold (-10mV)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title(f'Multi-Pulse Pacing Result ({count} pulses)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # I_stim
    ax2.plot(time, i_stim, color='darkorange', linewidth=2, label='Stimulation Train')
    ax2.set_ylabel('Current (uA/uF)')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True, alpha=0.3)

    # 局部放大查看脉冲细节
    # 找到脉冲序列大致结束的时间点
    train_end_time = 10.0 + count * (width + interval) + 10.0

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')
    axins.plot(time, i_stim, color='darkorange')
    axins.set_xlim(5, train_end_time)  # 聚焦在刺激开始前后
    axins.set_title("Pulse Train Zoom-in")
    axins.grid(True)

    plt.tight_layout()
    plt.savefig("multipulse_result.png", dpi=300)
    print(f"\n结果图已保存为 multipulse_result.png")


if __name__ == "__main__":
    test_model()
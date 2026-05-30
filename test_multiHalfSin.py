import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

# 忽略不必要的 matplotlib 紧凑布局警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from envs.multihalfsin_env import HalfSinePulseEnv


def test_model():
    print("加载多脉冲模型进行测试...")

    # 1. 核心修复：必须使用与训练时完全一致的环境包装方式
    raw_env = HalfSinePulseEnv()
    env = DummyVecEnv([lambda: raw_env])

    # 2. 核心修复：加载训练时保存的归一化状态字典
    # 简单粗暴，直接用绝对路径字符串
    vec_path = "/home/xp/Code/cardiac-simulation/checkpoints_multihalfsin/vec_normalize.pkl"
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
    else:
        print(f"❌ 找不到归一化文件: {vec_path}，请确保训练脚本已正确保存该文件！")
        return

    # 关闭测试时的奖励归一化和状态更新
    env.training = False
    env.norm_reward = False

    # 3. 路径检查与模型加载
    model_name = "sac_multipulse_final"  # 建议直接测试 final 最终模型，或者填入具体的 step 模型
    model_path = os.path.join(current_dir, "checkpoints_multihalfsin", model_name)
    if not os.path.exists(model_path + ".zip"):
        print(f"找不到模型: {model_path}.zip，请先运行 train_multipulse.py")
        return

    model = SAC.load(model_path, env=env)

    # 4. 环境 Reset (VecEnv 返回的只是 obs，没有 info)
    obs = env.reset()

    # 5. 预测动作
    action, _ = model.predict(obs, deterministic=True)

    # 解析动作 (VecEnv 动作外层套了一个 batch 维度，因此取 action[0])
    act = action[0]
    amp = float(act[0])
    width = float(act[1])
    interval = float(act[2])
    count = int(act[3])

    print(f"\n>>> 智能体生成的优化刺激策略:")
    print(f"  - 脉冲数量 (Count): {count}")
    print(f"  - 脉冲振幅 (Amp):   {amp:.2f} uA/uF")
    print(f"  - 单脉冲宽 (Width): {width:.2f} ms")
    print(f"  - 脉冲间隔 (Inter): {interval:.2f} ms")
    print("-" * 40)

    # 6. 运行模拟 (SB3 VecEnv 的 step 返回 4 个值: obs, rewards, dones, infos)
    obs, rewards, dones, infos = env.step(action)

    # 提取 info 和 reward
    info = infos[0]
    reward = rewards[0]

    print(f"  - 模拟结果: {info.get('outcome', 'Unknown')}")
    print(f"  - 能量消耗: {info.get('energy', 0):.2f}")
    print(f"  - 获得奖励: {reward:.2f}")

    # 7. 绘图 (直接使用底层 raw_env 获取追踪数据)
    trace = raw_env.last_beat_trace
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

    energy_val = info.get('energy', 0)
    max_stim = np.max(i_stim) if len(i_stim) > 0 else 0.0

    # 构造标注文本
    info_text = (f"Max Current: {max_stim:.2f} uA/uF\n"
                 f"Energy Score: {energy_val:.1f}")

    # 将文本放置在 ax2 的左上角 (0.02, 0.85)
    ax2.text(0.02, 0.85, info_text, transform=ax2.transAxes,
             fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

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
    plt.savefig("multihalfSin_result.png", dpi=300)
    print(f"\n结果图已保存为 multihalfSin_result.png")


if __name__ == "__main__":
    test_model()
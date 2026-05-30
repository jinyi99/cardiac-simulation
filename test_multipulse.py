import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from envs.multipulse_env import MultiPulseCardiacEnv

def test_model():
    print("加载多脉冲（方波）模型进行测试...")

    # 1. 包装底层环境 (保留原始环境的引用，方便后面拿 trace 绘图)
    raw_env = MultiPulseCardiacEnv()
    # 测试时不需要多进程，用 DummyVecEnv 包装成单线程的向量化环境
    env = DummyVecEnv([lambda: raw_env])

    # 2. 极其关键：加载归一化状态字典
    vec_path = os.path.join(current_dir, "checkpoints_multipulse", "vec_normalize.pkl")
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        # 测试时【必须】关闭状态均值更新，并关闭奖励归一化（我们要看真实奖励）
        env.training = False
        env.norm_reward = False
    else:
        print(f"❌ 找不到归一化文件: {vec_path}。测试终止！")
        return

    # 3. 路径检查
    model_path = os.path.join(current_dir, "checkpoints_multipulse", "sac_multipulse_final")
    if not os.path.exists(model_path + ".zip"):
        print(f"找不到模型: {model_path}.zip，请先运行 train_multipulse.py")
        return

    # 4. 加载模型 (建议把 env 传进去)
    model = SAC.load(model_path, env=env)

    obs = env.reset()

    # 5. 预测动作
    action, _ = model.predict(obs, deterministic=True)

    # 【注意】因为套了 VecEnv，action 现在是二维数组，比如 [[amp, width, interval, count]]
    # 所以我们需要取 action[0]
    amp = float(action[0][0])
    width = float(action[0][1])
    interval = float(action[0][2])
    count = int(action[0][3])

    print(f"\n>>> 智能体生成的优化刺激策略:")
    print(f"  - 脉冲数量 (Count): {count}")
    print(f"  - 脉冲振幅 (Amp):   {amp:.2f} uA/uF")
    print(f"  - 单脉冲宽 (Width): {width:.2f} ms")
    print(f"  - 脉冲间隔 (Inter): {interval:.2f} ms")
    print("-" * 40)

    # 6. 运行模拟
    # VecEnv 的 step 返回 4 个值，且都是列表，所以我们取 [0]
    obs, reward, done, info = env.step(action)
    
    actual_reward = reward[0]
    actual_info = info[0]

    print(f"  - 模拟结果: {actual_info.get('outcome', 'Unknown')}")
    print(f"  - 能量消耗: {actual_info.get('energy', 0):.2f}")
    print(f"  - 获得奖励: {actual_reward:.2f}")

    # 7. 绘图 (必须从底层 raw_env 获取轨迹，包装后的 env 拿不到这个属性)
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
    
    energy_val = actual_info.get('energy', 0)
    max_stim = np.max(i_stim) if len(i_stim) > 0 else 0

    # 构造标注文本
    info_text = (f"Max Current: {max_stim:.2f} uA/uF\n"
                 f"Energy Score: {energy_val:.1f}")

    # 将文本放置在 ax2 的左上角
    ax2.text(0.02, 0.85, info_text, transform=ax2.transAxes,
             fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    # 局部放大查看脉冲细节
    train_end_time = 10.0 + count * (width + interval) + 10.0

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')
    axins.plot(time, i_stim, color='darkorange')
    axins.set_xlim(5, train_end_time)  
    axins.set_title("Pulse Train Zoom-in")
    axins.grid(True)

    # 忽略 tight_layout 和 inset_axes 的兼容性警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
        
    plt.savefig("multipulse_result1.png", dpi=300)
    print(f"\n结果图已保存为 multipulse_result1.png")

if __name__ == "__main__":
    test_model()
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from envs.multiap_env import MultiAPStimEnv

def test_model():
    print("正在加载模型进行纯净测试...")

    # 🚀 必须与训练时的参数保持一致，默认 max_pulses=5
    raw_env = MultiAPStimEnv(max_pulses=5)
    env = DummyVecEnv([lambda: raw_env])

    # 🚀 修改为动态脉冲模型保存的新路径
    checkpoints_dir = os.path.join(current_dir, "checkpoints_ap_stim_dynamic")
    
    # 建议直接读取最终覆盖保存的 vec_normalize.pkl 和 sac_ap_stim_final.zip
    # 如果你想测试特定步数，可以把这里改回类似 vec_normalize_8000_steps.pkl
    vec_path = os.path.join(checkpoints_dir, "vec_normalize.pkl")
    
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"❌ 找不到归一化文件: {vec_path}")
        return

    model_path = os.path.join(checkpoints_dir, "sac_ap_stim_final.zip")
    if not os.path.exists(model_path):
         print(f"❌ 找不到模型文件: {model_path}")
         return
         
    model = SAC.load(model_path, env=env)
    obs = env.reset()

    # 获取无噪声的最优动作
    action, _ = model.predict(obs, deterministic=True)

    # 🚀 先执行一步(step)，利用环境内部的逻辑去解析动作，比我们手动切片更准确
    obs, reward, done, info = env.step(action)
    actual_info = info[0]

    # 🚀 从 info 中动态提取真实使用的参数
    num_pulses = actual_info['num_pulses']
    used_amps = actual_info['used_amps']
    plat_dur = actual_info['plat_dur']
    interval = actual_info['interval']
    energy_val = actual_info.get('energy', 0)

    print(f"\n>>> 智能体生成的物理最优刺激策略 (自主选择了 {num_pulses} 个脉冲):")
    for i, amp in enumerate(used_amps):
        print(f"  - 脉冲 {i+1} 振幅: {amp:.2f}x")
    print(f"  - 平台期时长:  {plat_dur:.2f} ms")
    print(f"  - 脉冲间隔:    {interval:.2f} ms")
    print(f"  - 消耗总能量:  {energy_val:.2f}")
    print("-" * 40)

    trace = raw_env.last_beat_trace
    time = np.array(trace['time'])
    v = np.array(trace['v'])
    i_stim = np.array(trace['i_stim'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time, v, color='navy', linewidth=2, label='Membrane Voltage')
    ax1.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='Threshold (-10mV)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Optimized Multi-AP Waveform Stimulation (Dynamic Pulses)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, i_stim, color='darkorange', linewidth=2, label='AP-like Stimulus')
    ax2.set_ylabel('Current (uA/uF)')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True, alpha=0.3)
    
    # 🚀 动态生成图表右上角的文本框信息
    info_lines = [f"Num Pulses: {num_pulses}"]
    for i, amp in enumerate(used_amps):
        info_lines.append(f"Amp {i+1}: {amp:.2f}x")
    info_lines.append(f"Plateau: {plat_dur:.1f}ms")
    info_lines.append(f"Interval: {interval:.1f}ms")
    info_lines.append(f"Energy: {energy_val:.1f}")
    
    info_text = "\n".join(info_lines)

    ax2.text(0.98, 0.85, info_text, transform=ax2.transAxes,
             fontsize=10, fontweight='bold', ha='right', va='top',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig("ap_stim_result_dynamic1.png", dpi=300)
    print(f"\n✅ 结果图已保存为 ap_stim_result_dynamic.png")

if __name__ == "__main__":
    test_model()
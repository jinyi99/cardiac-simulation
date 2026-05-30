import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cupy as cp

# 1. 设置路径，确保能导入 envs
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from envs.cardiac_env import CardiacStimEnv


def test_model():
    print("=" * 60)
    print("开始测试最终模型效果 (Validation Run)")
    print("=" * 60)

    # 2. 准备环境
    # 注意：这里重新创建环境，物理参数与训练时一致
    env = CardiacStimEnv()

    # 3. 加载模型
    # 指向您刚刚训练完成的模型文件
    # 注意：SB3 save时会自动加 .zip 后缀，load 时不需要加
    model_path = os.path.join(current_dir, "checkpoints", "cardiac_sac_final_opt")

    if not os.path.exists(model_path + ".zip"):
        print(f"\n[错误] 找不到模型文件: {model_path}.zip")
        print(f"请检查 checkpoints 文件夹里是否有 cardiac_sac_final_opt.zip")
        # 尝试列出该目录下文件以辅助调试
        ckpt_dir = os.path.join(current_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            print(f"checkpoints 目录下的文件: {os.listdir(ckpt_dir)}")
        return

    print(f"正在加载模型: {model_path} ...")
    model = SAC.load(model_path)
    print("模型加载成功！")

    # 4. 运行一次完整的模拟
    print("\n正在运行心脏起搏模拟 (Control Loop)...")
    obs, _ = env.reset()

    # 用于绘图的数据容器
    time_history = []
    v_history = []  # 电压
    stim_history = []  # 刺激电流
    ci_history = []  # 钙浓度

    done = False
    truncated = False
    total_reward = 0
    step = 0

    # 开始循环
    while not (done or truncated):
        # 智能体根据观测做出决策 (deterministic=True 表示使用最优策略，不加随机噪声)
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, done, truncated, _ = env.step(action)

        # 记录数据
        # env.cell.time 是物理时间
        # env.cell.whole_cell.v[0] 是 GPU 上的电压值，需要转回 CPU
        current_time = env.cell.time
        current_v = float(env.cell.whole_cell.v[0])  # 转为 float
        current_stim = float(action[0])
        current_ci = float(obs[1])  # 观测值的第二个分量是钙

        time_history.append(current_time)
        v_history.append(current_v)
        stim_history.append(current_stim)
        ci_history.append(current_ci)

        total_reward += reward
        step += 1

    print(f"\n模拟结束!")
    print(f"总时长: {time_history[-1]:.2f} ms")
    print(f"总步数: {step}")
    print(f"总奖励: {total_reward:.2f}")

    # 5. 可视化波形 (这是最重要的结果！)
    print("\n正在绘制波形图...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 子图 1: 膜电压 (Action Potential)
    ax1.plot(time_history, v_history, color='blue', linewidth=2, label='Membrane Voltage')
    ax1.set_ylabel('Voltage (mV)', fontsize=12)
    ax1.set_title('RL-Controlled Cardiac Pacing Result', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # 画阈值线
    ax1.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='AP Threshold (-10mV)')
    ax1.legend(loc='upper right')

    # 子图 2: 刺激电流 (Stimulation Current)
    ax2.plot(time_history, stim_history, color='orange', linewidth=2, label='Agent Action (I_stim)')
    ax2.set_ylabel('Current (uA/uF)', fontsize=12)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 标注最大电流
    max_stim = np.max(stim_history)
    avg_stim = np.mean(stim_history)
    # 计算能量指标 (电流平方积分的近似)
    energy_proxy = np.sum(np.array(stim_history) ** 2)

    info_text = (f"Max Current: {max_stim:.2f} uA/uF\n"
                 f"Energy Score: {energy_proxy:.1f}")

    ax2.text(0.02, 0.85, info_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # 保存图片
    output_file = os.path.join(current_dir, "final_result_waveform.png")
    plt.savefig(output_file, dpi=300)
    print(f"\n[成功] 结果图已保存为: {output_file}")
    print(">>> 请在 PyCharm 左侧文件栏双击打开 final_result_waveform.png 查看 <<<")


if __name__ == "__main__":
    test_model()
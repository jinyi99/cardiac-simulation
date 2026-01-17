import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.multipulse_env import MultiPulseCardiacEnv

def main():
    log_dir = "./logs_multipulse/"
    models_dir = "./checkpoints_multipulse/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. 创建环境
    env = MultiPulseCardiacEnv()
    env = Monitor(env, log_dir)

    # 2. 定义模型
    # 针对多脉冲任务，探索空间较大，我们适当增加探索噪声 (ent_coef) 或让其自动调整
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=20,
        ent_coef="auto", # 自动调整熵，保持探索性
        tensorboard_log=log_dir,
        seed=42
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500, # 每 500 个 beats 保存
        save_path=models_dir,
        name_prefix="sac_multipulse"
    )

    print("=" * 60)
    print("开始多脉冲刺激训练 (Multi-Pulse Stimulation Training)...")
    print("Action Space: [Amp, Width, Interval, Count]")
    print("=" * 60)

    # 训练 5000 个 beats 应该能看到明显效果
    model.learn(
        total_timesteps=5000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(f"{models_dir}/sac_multipulse_final")
    print("训练完成。")

if __name__ == "__main__":
    main()
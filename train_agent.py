import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
import torch  # 导入 torch 以便检查 cuda

# 确保能找到 envs 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.cardiac_env import CardiacStimEnv


def main():
    # --- 硬件检查 ---
    if torch.cuda.is_available():
        print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
        print(f"显卡数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到 GPU，训练将非常缓慢！")

    # 1. 设置路径
    log_dir = "./logs/"
    models_dir = "./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. 创建环境
    env = CardiacStimEnv()
    env = Monitor(env, log_dir)

    # 3. 定义模型 (SAC) - 针对 RTX 3090 优化版
    previous_model_path = f"{models_dir}/cardiac_sac_final_opt.zip"
    if os.path.exists(previous_model_path):
        print(f"加载之前的模型: {previous_model_path}")
        model = SAC.load(previous_model_path, env=env)
    else:
        print("未找到之前的模型，创建新模型.")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            tensorboard_log=log_dir,
            seed=42
        )

    # 4. 设置检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=models_dir,
        name_prefix="cardiac_sac_3090"
    )

    # 5. 开始训练
    print("=" * 60)
    print("开始强化学习训练 (RTX 3090 Optimized)...")
    print(f"Batch Size: {model.batch_size}")
    print(f"Buffer Size: {model.buffer_size}")
    print("=" * 60)

    # 训练步数建议：
    total_timesteps = 100000

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 6. 保存最终模型
    model.save(f"{models_dir}/cardiac_sac_final_opt")
    print("训练完成，最终模型已保存。")

if __name__ == "__main__":
    main()
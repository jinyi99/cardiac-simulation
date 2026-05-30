import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.multipulse_env import MultiPulseCardiacEnv

def main():
    log_dir = "./logs_multipulse/"
    models_dir = "./checkpoints_multipulse/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. 创建单进程环境
    # 使用 DummyVecEnv 包装，这是 VecNormalize 所必需的，即使只有一个进程
    raw_env = MultiPulseCardiacEnv()
    monitored_env = Monitor(raw_env, log_dir)
    env = DummyVecEnv([lambda: monitored_env])
    
    # 2. 极其关键：加入状态和奖励的归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model_path = os.path.join(models_dir, "sac_multipulse_final.zip")
    vec_path = os.path.join(models_dir, "vec_normalize.pkl")

    # 3. 加载或新建模型
    if os.path.exists(model_path) and os.path.exists(vec_path):
        print(f"检测到模型文件和归一化文件，正在加载并继续训练...")
        env = VecNormalize.load(vec_path, env)
        env.training = True
        env.norm_reward = True
        model = SAC.load(model_path, env=env)
    else:
        print("未找到预训练模型，将从头开始初始化模型...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=50000,       # 减小池子，单步任务不需要 100万
            batch_size=256,
            train_freq=(1, "step"),  # 每走 1 步就学习
            gradient_steps=4,        # 每次学习只更新 4 次 (防止过拟合)
            ent_coef="auto",
            tensorboard_log=log_dir,
            seed=42
        )

    # 4. 回调函数：单进程直接填保存频率即可
    checkpoint_callback = CheckpointCallback(
        save_freq=500,  # 每 500 步保存一次
        save_path=models_dir,
        name_prefix="sac_multipulse"
    )

    print("=" * 60)
    print("开始多脉冲刺激训练 (Multi-Pulse Stimulation Training)...")
    print("=" * 60)

    # 5. 开启训练
    model.learn(
        total_timesteps=3000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 6. 训练结束，保存模型和归一化文件 (极其重要！)
    model.save(f"{models_dir}/sac_multipulse_final")
    env.save(vec_path)
    print(f"训练完成！模型及归一化文件已保存至 {models_dir}。")

if __name__ == "__main__":
    main()
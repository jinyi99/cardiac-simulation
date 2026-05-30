from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # 新增导入
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.multihalfsin_env import HalfSinePulseEnv


def main():
    log_dir = "./logs_multihalfsin/"
    models_dir = "./checkpoints_multihalfsin/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. 创建环境 (核心修改区域)
    env = HalfSinePulseEnv()
    env = Monitor(env, log_dir)

    # 必须要将环境转换为向量化环境，VecNormalize 才能工作
    env = DummyVecEnv([lambda: env])

    # 加入观测值和奖励的自动归一化！
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 2. 加载现有模型或新建模型
    model_path = os.path.join(models_dir, "sac_multipulse_final.zip")

    if os.path.exists(model_path):
        print(f"检测到模型文件 {model_path}")
        model = SAC.load(model_path, env=env)
    else:
        print("未找到预训练模型，将从头开始初始化模型...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=7e-4,
            buffer_size=10000, 
            batch_size=256,
            train_freq=(1, "step"), 
            gradient_steps=4,
            ent_coef="auto",
            target_entropy="auto",
            tensorboard_log=log_dir,
            seed=42,
            device="auto"
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path=models_dir,
        name_prefix="sac_multipulse"
    )

    print("开始多脉冲刺激训练 (Multi-Pulse Stimulation Training)...")

    model.learn(
        total_timesteps=3000,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name="SAC_multihalfsin_run1"
    )

    model.save(f"{models_dir}/sac_multipulse_final")

    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    print("训练及环境归一化状态保存完成。")


if __name__ == "__main__":
    main()
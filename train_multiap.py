import os
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.multiap_env import MultiAPStimEnv

# 自定义回调函数，每次保存模型时，连带归一化文件一起保存！
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(save_path)
            # 同时更新一个覆盖版的 final，方便测试脚本读取
            self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
        return True

def main():
    # 🚀 使用新的目录保存动态脉冲环境的模型
    save_dir = "./checkpoints_ap_stim_dynamic/"
    log_dir = "./logs_multiap_dynamic/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 🚀 实例化环境，最大允许 5 个脉冲（你可以随时修改这个上限）
    raw_env = MultiAPStimEnv(max_pulses=5)
    env = Monitor(raw_env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    # 对电压等大幅度变化的数据进行归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        ent_coef="auto",
        tensorboard_log=log_dir
    )

    # 模型保存回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix="sac_ap_stim")
    
    # 归一化同步保存回调
    vec_norm_callback = SaveVecNormalizeCallback(save_freq=1000, save_path=save_dir)

    print(f"开始多脉冲类动作电位波形优化训练 (支持自主决策 1~{raw_env.max_pulses} 个脉冲)...")
    
    # 将两个回调组合在一起
    model.learn(total_timesteps=10000, callback=[checkpoint_callback, vec_norm_callback], progress_bar=True)

    model.save(f"{save_dir}/sac_ap_stim_final")
    env.save(f"{save_dir}/vec_normalize.pkl")
    print("训练结束，模型已保存。")

if __name__ == "__main__":
    main()
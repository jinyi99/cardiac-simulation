import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cupy as cp
import sys
import os

# 自动将项目根目录添加到 python path，以便能导入 models 和 config
# 假设此文件位于 /project_root/envs/cardiac_env.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.gpu_spatial_cell import GPUSpatialCell
from config.default_params import DefaultParams


class CardiacStimEnv(gym.Env):
    """
    心脏刺激强化学习环境 (GPU版)

    State (Observation):
        - 平均膜电压 (V)
        - 平均胞内钙浓度 (Ci)

    Action:
        - 刺激电流强度 (I_stim), 范围 [0, 80] uA/uF

    Goal:
        - 触发动作电位 (Action Potential)，同时最小化能量消耗。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        super(CardiacStimEnv, self).__init__()

        # 1. 加载参数
        if params is None:
            self.params = DefaultParams.get_all_params()
        else:
            self.params = params

        # 覆盖默认参数以适应 RL 训练
        # 只有在非常明确需要修改物理参数时才在这里改，否则使用 default_params
        # self.params['NX'] = ...

        # 2. 定义动作空间 (Action Space)
        # 假设我们只控制刺激电流的大小。
        # 上限设为 80.0 uA/uF (略高于通常阈值，给 RL 探索空间)
        self.max_stim = 80.0
        self.action_space = spaces.Box(
            low=0.0,
            high=self.max_stim,
            shape=(1,),
            dtype=np.float32
        )

        # 3. 定义观测空间 (Observation Space)
        # [平均电压(mV), 平均钙浓度(uM)]
        # 电压范围约 -100 到 +60，钙约 0 到 5.0
        self.observation_space = spaces.Box(
            low=np.array([-120.0, 0.0]),
            high=np.array([80.0, 10.0]),
            dtype=np.float32
        )

        # 4. 时间控制参数
        self.dt = self.params['DT']  # 物理模拟步长 (0.01 ms)
        self.control_dt = 1.0  # RL 控制决策间隔 (1.0 ms)
        self.steps_per_control = int(max(1, self.control_dt / self.dt))

        # 每一个 Episode 的最大时长 (例如 400ms 足够展示一个动作电位)
        self.max_episode_duration = 400.0
        self.max_episode_steps = int(self.max_episode_duration / self.control_dt)

        # 5. 初始化模型实例
        # 注意：我们在 init 里只创建一次对象，reset 时只重置数值，避免重复分配显存
        print("正在初始化 RL 环境中的 GPU 细胞模型...")
        self.cell = GPUSpatialCell(
            nx=self.params['NX'],
            ny=self.params['NY'],
            nz=self.params['NZ'],
            filename="rl_env_instance",
            rng_seed=self.params['RNG_SEED']
        )

        # 预分配刺激电流的 GPU 数组，避免 step 中重复分配
        self.current_istim_gpu = cp.zeros(1, dtype=cp.float32)

        # 内部状态追踪
        self.current_step = 0
        self.has_fired = False

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)

        # 1. 重置物理变量 (参考 single_cell.py 的初始化)
        self.cell.time = 0.0

        # 全细胞变量
        self.cell.whole_cell.v[0] = self.params['INITIAL_V']
        self.cell.whole_cell.xnai[0] = self.params['INITIAL_XNAI']

        # 内部钙分布
        for myosr in self.cell.myosr_ca:
            myosr.cm.fill(self.params['INITIAL_CM'])
            myosr.cs.fill(self.params['INITIAL_CS'])

        # CRU 变量
        for cru in self.cell.crus:
            cru.cd.fill(self.params['INITIAL_CD'])
            cru.cj.fill(self.params['INITIAL_CJ'])

        # 2. 重置 RL 状态
        self.current_step = 0
        self.has_fired = False

        # 3. 返回初始观测
        return self._get_obs(), {}

    def step(self, action):
        """
        执行一步交互
        """
        # 1. 解析动作
        # 裁剪动作范围，确保安全
        stim_val = float(np.clip(action[0], 0, self.max_stim))

        # 更新 GPU 上的刺激电流值
        self.current_istim_gpu[0] = stim_val

        # 2. 运行物理模拟 (执行 control_dt 时长)
        # 记录这一段时间内的最大电压，用于判断是否发生 AP
        max_v_in_step = -100.0

        for _ in range(self.steps_per_control):
            # 物理模型步进
            self.cell.update_cru_flux()
            self.cell.update_myosr_flux()
            self.cell.compute_calcium_diffusion()

            # 施加电压更新 (istim_on=True, 我们通过传入 0 来表示无刺激)
            self.cell.update_voltage(self.current_istim_gpu, istim_on=True)

            self.cell.time += self.dt

            # 简单采样电压 (为了性能，不一定每一步都从 GPU 拉取，可以在循环里积攒)
            # 这里为了准确判断峰值，我们取当前值。
            # 注意：频繁 .get() 会影响性能。
            # 优化方案：这里只在 control 步结束时取一次，或者在 GPU 上维护 max_v
            # 为保证准确性先这么写，如果太慢可以优化
            pass

            # 3. 获取当前观测 (在 control_dt 结束时统一拉取数据)
        obs = self._get_obs()
        current_v = obs[0]

        # 4. 计算奖励 (Reward Engineering)
        reward = 0.0

        # A. 能量惩罚 (Energy Penalty)
        # 目标是最小化能量，所以给负奖励。系数需要调节。
        # 假设 1.0 的动作持续 1步 罚 -0.0001
        energy_penalty = (stim_val ** 2) * 0.0005
        reward -= energy_penalty

        # B. 成功起搏奖励 (Capture Reward)
        # 判定标准：电压超过 -10mV (去极化成功)
        # 且保证只奖励第一次触发，避免它学会让电压一直维持在高位
        if current_v > -10.0 and not self.has_fired:
            reward += 100.0  # 给予巨大的稀疏奖励
            self.has_fired = True
            print(f"  >>> 成功起搏! Step: {self.current_step}, I_stim: {stim_val:.2f}")

        # C. 失败惩罚 (可选)
        # 如果时间过了很久还没起搏，给一点点压力
        if not self.has_fired and self.current_step > 50:
            reward -= 0.1

        # D. 辅助奖励 (Shaping)
        # 鼓励电压上升 (在未起搏前)
        if not self.has_fired and current_v > self.params['INITIAL_V'] + 5.0:
            reward += (current_v - self.params['INITIAL_V']) * 0.001

        # 5. 判断结束条件
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        terminated = False

        # 提前结束策略：如果已经起搏，并且电压已经回落 (复极化完成)，可以提前结束以节省时间
        if self.has_fired and current_v < -70.0 and self.current_step > 50:
            terminated = True
            # 给一个额外奖励作为顺利完成周期的奖励
            reward += 10.0

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        从 GPU 获取数据并转换为 Observation 格式
        """
        # 1. 获取电压 (标量)
        # cell.whole_cell.v 是 cupy array
        v_gpu = self.cell.whole_cell.v[0]

        # 2. 获取平均钙浓度
        # 计算所有体素的平均值
        # myosr_ca[0].cm 是大数组，cp.mean 会在 GPU 上计算，然后 .get() 回传标量
        ci_gpu = cp.mean(self.cell.myosr_ca[0].cm)

        # 3. 同步到 CPU
        # 注意：这里会发生 GPU-CPU 同步，是性能瓶颈点之一
        return np.array([float(v_gpu), float(ci_gpu)], dtype=np.float32)

    def close(self):
        """清理资源"""
        # Cupy 通常会自动管理内存，但如果有显式释放逻辑可写在这里
        pass


if __name__ == "__main__":
    # 简单的测试代码
    print("测试 CardiacStimEnv...")
    env = CardiacStimEnv()
    obs, _ = env.reset()
    print(f"初始观测: {obs}")

    # 随机运行几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        print(f"Step {i}: Action={action}, Obs={obs}, Reward={reward:.4f}")

    print("环境测试通过。")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cupy as cp
import sys
import os

# 路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.gpu_spatial_cell import GPUSpatialCell
from config.default_params import DefaultParams


class MultiPulseCardiacEnv(gym.Env):
    """
    多脉冲刺激环境 (Multi-Pulse Stimulation Env)

    Action Space (4维连续空间):
        0. 振幅 (Amplitude): [0, 60] uA/uF (降低上限以鼓励利用多脉冲优势)
        1. 单脉冲宽度 (Pulse Width): [0.5, 3.0] ms
        2. 脉冲间隔 (Interval): [1.0, 20.0] ms (两个脉冲之间的时间)
        3. 脉冲数量 (Count): [1.0, 5.99] -> 映射为整数 1~5

    Goal:
        利用脉冲序列累积效应触发动作电位，同时最小化总能量。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        super(MultiPulseCardiacEnv, self).__init__()

        if params is None:
            self.params = DefaultParams.get_all_params()
        else:
            self.params = params

        # 1. 动作空间定义
        # [Amp, Width, Interval, Count]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.5, 1.0, 1.0]),
            high=np.array([60.0, 3.0, 20.0, 5.99]),
            dtype=np.float32
        )

        # 2. 观测空间: [V, Ci]
        self.observation_space = spaces.Box(
            low=np.array([-120.0, 0.0]),
            high=np.array([80.0, 10.0]),
            dtype=np.float32
        )

        # 3. 模拟控制
        self.dt = self.params['DT']  # 0.01 ms
        self.beat_duration = 400.0  # 模拟窗口 400ms
        self.steps_per_beat = int(self.beat_duration / self.dt)

        # 4. 初始化 GPU 模型
        print("正在初始化多脉冲 GPU 环境...")
        self.cell = GPUSpatialCell(
            nx=self.params['NX'], ny=self.params['NY'], nz=self.params['NZ'],
            filename="multipulse_instance", rng_seed=self.params['RNG_SEED']
        )
        self.current_istim_gpu = cp.zeros(1, dtype=cp.float32)

        # 轨迹记录
        self.last_beat_trace = {'time': [], 'v': [], 'i_stim': []}
        self.current_beat = 0
        self.max_beats = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cell.time = 0.0
        self.cell.whole_cell.v[0] = self.params['INITIAL_V']
        self.cell.whole_cell.xnai[0] = self.params['INITIAL_XNAI']
        # 重置内部钙分布 (简略)
        for myosr in self.cell.myosr_ca:
            myosr.cm.fill(self.params['INITIAL_CM'])

        self.current_beat = 0
        return self._get_obs(), {}

    def step(self, action):
        """执行单次心跳模拟"""
        # 1. 解析动作
        amp = float(action[0])
        width = float(action[1])
        interval = float(action[2])
        count = int(action[3])  # 向下取整，如 3.9 -> 3

        # 2. 模拟循环
        trace_v, trace_i, trace_t = [], [], []
        max_v = -100.0
        total_energy = 0.0

        # 预计算脉冲时间表
        # Pulse i active if: start_i <= t < start_i + width
        # start_i = i * (width + interval)
        pulse_timings = []
        for i in range(count):
            start_t = i * (width + interval) + 10.0  # +10ms 延迟开始
            end_t = start_t + width
            pulse_timings.append((start_t, end_t))

        current_step = 0
        while current_step < self.steps_per_beat:
            sim_time = current_step * self.dt

            # --- 构建多脉冲波形 ---
            i_val = 0.0
            # 只有当 sim_time 小于最后一个脉冲结束时间才需要检查
            if sim_time <= pulse_timings[-1][1]:
                for (start, end) in pulse_timings:
                    if start <= sim_time < end:
                        i_val = amp
                        break

            # --- GPU 更新 ---
            self.current_istim_gpu[0] = i_val

            self.cell.update_cru_flux()
            self.cell.update_myosr_flux()
            self.cell.compute_calcium_diffusion()
            self.cell.update_voltage(self.current_istim_gpu, istim_on=True)
            self.cell.time += self.dt

            # --- 关键修正：在每一个时间步都读取电压并更新最大值 ---
            # 这样就不会因为降采样而漏掉短暂的峰值
            v_curr_realtime = float(self.cell.whole_cell.v[0])
            if v_curr_realtime > max_v:
                max_v = v_curr_realtime

            # --- 记录数据 (保持降采样: 每 0.5ms 记一次以节省内存) ---
            if current_step % 50 == 0:
                trace_v.append(v_curr_realtime)
                trace_i.append(i_val)
                trace_t.append(self.cell.time)
                # 注意：max_v 更新逻辑已移出此处

            total_energy += abs(i_val) * self.dt
            current_step += 1

            # 早停优化：如果已经起搏成功且完全复极化，可以提前退出
            # 这里也使用实时电压判断
            if max_v > 0.0 and v_curr_realtime < -70.0 and sim_time > pulse_timings[-1][1] + 50.0:
                break

        # 3. 保存轨迹
        self.last_beat_trace = {'time': trace_t, 'v': trace_v, 'i_stim': trace_i}

        # 4. 计算奖励 (Reward Engineering)
        reward, info = self._compute_reward(max_v, total_energy, count, amp)

        # 5. 返回
        self.current_beat += 1
        truncated = self.current_beat >= self.max_beats
        return self._get_obs(), reward, False, truncated, info

    def _compute_reward(self, max_v, total_energy, count, amp):
            V_THRESHOLD = -10.0
            success = max_v >= V_THRESHOLD

            if success:
                # 【核心逻辑：绝对公平】
                # 基础成功分 200。接下来纯拼能耗，没有任何关于 count 的加分项！
                # 权重 0.5 意味着每浪费 1 单位的总电荷，就扣 0.5 分。
                reward = 200.0 - (0.5 * total_energy)
            else:
                # 【失败引导：梯度鼓励】
                # 失败基础分 -50。但如果电压有上升（比如从 -80 升到了 -20），给予鼓励分
                # 这样 AI 就不会像无头苍蝇一样乱撞，而是顺着电压上升的梯度去寻找解
                progress = max_v - (-80.0)
                reward = -50.0 + (progress * 0.5)

                # 惩罚完全不输出电流的“摆烂”行为
                if amp < 1.0: 
                    reward -= 20.0

            return reward, {
                "outcome": "Success" if success else "Failed",
                "energy": total_energy,
                "max_v": max_v,
                "pulses": count  # 留在 info 里，方便我们在终端或测试图表里观察
            }

    """
        def _compute_reward(self, max_v, total_energy, count, amp):
        V_THRESHOLD = -10.0
        success = max_v >= V_THRESHOLD

        reward = 0.0
        info = {}

        if success:
            reward += 100.0

            # A. 能量惩罚 (主要项)
            # 假设 40^2 * 2ms = 3200. 乘以 0.01 -> -32.
            energy_penalty = total_energy * 0.01
            reward -= energy_penalty

            # B. 多脉冲倾向性 (Soft Bias)
            # 如果成功了，且用了多于1个脉冲，给一点点 "Bonus" 抵消部分能量惩罚
            # 这里的逻辑是：如果 1个脉冲和 3个脉冲能量一样，Agent 会选 3个，因为有 Bonus
            if count > 1:
                reward += 5.0 * count  # 鼓励使用脉冲序列

            # C. 峰值电流惩罚 (Safety)
            # 相比于能量，我们更讨厌极高的瞬时电流 (容易造成组织灼伤)
            if amp > 50.0:
                reward -= (amp - 50.0) * 1.0

            info = {"outcome": "Success", "energy": total_energy, "pulses": count}
        else:
            # 失败时
            reward -= 20.0
            # 引导项：距离阈值越近扣分越少
            reward += (max_v - V_THRESHOLD) * 0.5
            # 失败时也要稍微惩罚能量，防止它乱试
            reward -= total_energy * 0.001

            info = {"outcome": "Failed", "max_v": max_v}

        return reward, info
    """


    def _get_obs(self):
        v_gpu = self.cell.whole_cell.v[0]
        ci_gpu = cp.mean(self.cell.myosr_ca[0].cm)
        return np.array([float(v_gpu), float(ci_gpu)], dtype=np.float32)
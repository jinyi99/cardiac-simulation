import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cupy as cp


class HalfSinePulseEnv(gym.Env):
    def __init__(self, params=None):
        super(HalfSinePulseEnv, self).__init__()
        from config.default_params import DefaultParams
        self.params = params if params else DefaultParams.get_all_params()

        # 动作空间: [振幅, 脉宽, 间隔, 数量]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.5, 1.0, 1.0]),
            high=np.array([60.0, 5.0, 20.0, 5.99]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-120.0, 0.0]),
            high=np.array([80.0, 10.0]),
            dtype=np.float32
        )

        self.dt = self.params['DT']
        self.beat_duration = 400.0
        self.steps_per_beat = int(self.beat_duration / self.dt)

        from models.gpu_spatial_cell import GPUSpatialCell
        self.cell = GPUSpatialCell(
            nx=self.params['NX'], ny=self.params['NY'], nz=self.params['NZ'],
            filename="halfsine_instance", rng_seed=self.params['RNG_SEED']
        )
        self.current_istim_gpu = cp.zeros(1, dtype=cp.float32)
        self.last_beat_trace = None  # 初始化追踪属性

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cell.time = 0.0
        self.cell.whole_cell.v[0] = self.params['INITIAL_V']
        self.cell.whole_cell.xnai[0] = self.params['INITIAL_XNAI']
        for myosr in self.cell.myosr_ca:
            myosr.cm.fill(self.params['INITIAL_CM'])
        return self._get_obs(), {}

    def step(self, action):
        amp = float(action[0])
        width = float(action[1])
        interval = float(action[2])
        count = int(action[3])

        trace_v, trace_i, trace_t = [], [], []
        max_v = -100.0
        total_energy = 0.0

        # 计算脉冲时间点
        pulse_timings = []
        for i in range(count):
            start_t = i * (width + interval) + 10.0
            end_t = start_t + width
            pulse_timings.append((start_t, end_t))

        current_step = 0
        while current_step < self.steps_per_beat:
            sim_time = current_step * self.dt
            i_val = 0.0

            # 构建半正弦波
            if sim_time <= pulse_timings[-1][1]:
                for (start, end) in pulse_timings:
                    if start <= sim_time < end:
                        relative_t = sim_time - start
                        i_val = amp * np.sin(np.pi * relative_t / width)
                        break

            self.current_istim_gpu[0] = i_val
            self.cell.update_cru_flux()
            self.cell.update_myosr_flux()
            self.cell.compute_calcium_diffusion()
            self.cell.update_voltage(self.current_istim_gpu, istim_on=True)
            self.cell.time += self.dt

            v_curr = float(self.cell.whole_cell.v[0])
            if v_curr > max_v: max_v = v_curr

            # 采样记录用于绘图 (每 0.5ms 记录一次，节省内存)
            if current_step % int(0.5 / self.dt) == 0:
                trace_v.append(v_curr)
                trace_i.append(i_val)
                trace_t.append(self.cell.time)

            total_energy += abs(i_val) * self.dt
            current_step += 1

            # 提前终止逻辑：如果已经产生动作电位并恢复
            if max_v > 0.0 and v_curr < -75.0 and sim_time > pulse_timings[-1][1] + 20.0:
                break

        # 保存追踪数据
        self.last_beat_trace = {'v': trace_v, 'i_stim': trace_i, 'time': trace_t}

        reward, info = self._compute_reward(max_v, total_energy, count, amp)

        # 保持单步任务逻辑
        return self._get_obs(), reward, True, False, info

    def _compute_reward(self, max_v, total_energy, count, amp):
        V_THRESHOLD = -10.0
        success = max_v >= V_THRESHOLD

        if success:
            # 去掉 count 的加分，完全基于能量消耗来扣分
            # 能量消耗越低，扣分越少，最终得分越高
            reward = 200.0 - (0.5 * total_energy)
        else:
            # 失败：引导智能体。max_v 越高，说明离成功越近，得分越高。
            progress = max_v - (-80.0)
            reward = -50.0 + (progress * 0.5)

            # 惩罚完全不尝试的行为
            if amp < 1.0: reward -= 20.0

        return reward, {
            "outcome": "Success" if success else "Failed",
            "energy": total_energy,
            "max_v": max_v
        }

    def _get_obs(self):
        v_gpu = self.cell.whole_cell.v[0]
        ci_gpu = cp.mean(self.cell.myosr_ca[0].cm)
        return np.array([float(v_gpu), float(ci_gpu)], dtype=np.float32)
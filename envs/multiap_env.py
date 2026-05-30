import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cupy as cp

class MultiAPStimEnv(gym.Env):
    """
    自适应次数的类动作电位(AP)刺激环境 - 支持任意数量脉冲的通用版
    """
    def __init__(self, params=None, max_pulses=5): # 🚀 新增：允许自定义最大脉冲数，这里默认上限设为 5
        super(MultiAPStimEnv, self).__init__()
        from config.default_params import DefaultParams
        self.params = params if params else DefaultParams.get_all_params()

        self.max_pulses = max_pulses

        # 动作空间构建: max_pulses 个振幅 + 1个平台期 + 1个间隔 + 1个脉冲控制因子
        # 总维度 = max_pulses + 3
        low_action = np.zeros(self.max_pulses + 3, dtype=np.float32)
        
        # 振幅上限 40.0, plat_dur 上限 100.0, interval 上限 100.0, 因子上限 1.0
        high_action = np.array([40.0] * self.max_pulses + [100.0, 100.0, 1.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
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
            filename="multi_ap_instance", rng_seed=self.params['RNG_SEED']
        )
        self.current_istim_gpu = cp.zeros(1, dtype=cp.float32)
        self.last_beat_trace = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cell.time = 0.0
        self.cell.whole_cell.v[0] = self.params['INITIAL_V']
        self.cell.whole_cell.xnai[0] = self.params['INITIAL_XNAI']
        for myosr in self.cell.myosr_ca:
            myosr.cm.fill(self.params['INITIAL_CM'])
        return self._get_obs(), {}

    def _build_stim_train(self, amps, plat_dur, interval, num_pulses):
        train_steps = self.steps_per_beat
        i_stim_array = np.zeros(train_steps, dtype=np.float32)
        
        t_up = 2.0
        t_notch = 5.0
        t_plat = 5.0 + plat_dur
        t_repol = t_plat + 30.0
        ap_total_duration = t_repol

        def single_ap_val(t_local):
            if t_local < t_up: return t_local / t_up
            elif t_local < t_notch: return 1.0 - 0.3 * ((t_local - t_up) / (t_notch - t_up))
            elif t_local < t_plat: return 0.7 - 0.2 * ((t_local - t_notch) / (t_plat - t_notch))
            elif t_local < t_repol: return 0.5 * (1.0 - ((t_local - t_plat) / (t_repol - t_plat))**2)
            else: return 0.0

        start_delay = 10.0
        last_pulse_end_time = start_delay

        # 波形生成完全依赖智能体决定的 num_pulses
        for p in range(num_pulses):
            amp = amps[p]
            pulse_start_t = start_delay + p * (ap_total_duration + interval)
            pulse_start_step = int(pulse_start_t / self.dt)
            ap_steps = int(ap_total_duration / self.dt)
            
            last_pulse_end_time = pulse_start_t + ap_total_duration
            
            for i in range(ap_steps):
                step_idx = pulse_start_step + i
                if step_idx < train_steps:
                    t_local = i * self.dt
                    i_stim_array[step_idx] += amp * single_ap_val(t_local)
                        
        return i_stim_array, last_pulse_end_time

    def step(self, action):
        # 🚀 动态提取动作
        # 前 max_pulses 个元素全是振幅
        amps = [float(action[i]) for i in range(self.max_pulses)]
        plat_dur = float(action[self.max_pulses])
        interval = float(action[self.max_pulses + 1])
        pulse_factor = float(action[self.max_pulses + 2])

        # 🚀 将 0~1 的因子，按比例映射到 1 ~ max_pulses 个整数
        # 使用 0.999 乘以是为了防止 float 精度导致取到 max_pulses + 1 越界
        num_pulses = int(pulse_factor * 0.999 * self.max_pulses) + 1 

        # 传入的 amps 数组虽然有 max_pulses 个，但生成器内部只会循环前 num_pulses 个
        i_stim_array, last_pulse_end_time = self._build_stim_train(amps, plat_dur, interval, num_pulses)

        trace_v, trace_i, trace_t = [], [], []
        max_v = -100.0
        total_energy = 0.0

        for current_step in range(self.steps_per_beat):
            sim_time = current_step * self.dt
            i_val = i_stim_array[current_step]

            self.current_istim_gpu[0] = i_val
            self.cell.update_cru_flux()
            self.cell.update_myosr_flux()
            self.cell.compute_calcium_diffusion()
            self.cell.update_voltage(self.current_istim_gpu, istim_on=True)
            self.cell.time += self.dt

            v_curr = float(self.cell.whole_cell.v[0])
            if v_curr > max_v: max_v = v_curr

            if current_step % int(0.5 / self.dt) == 0:
                trace_v.append(v_curr)
                trace_i.append(i_val)
                trace_t.append(self.cell.time)

            total_energy += abs(i_val) * self.dt

            if sim_time > last_pulse_end_time + 20.0 and max_v > 0.0 and v_curr < -75.0:
                break

        self.last_beat_trace = {'time': trace_t, 'v': trace_v, 'i_stim': trace_i}
        
        reward, info = self._compute_reward(max_v, total_energy)

        # 记录关键信息用于分析
        info['all_amps'] = amps              # 网络输出的所有振幅
        info['used_amps'] = amps[:num_pulses]# 真正被使用的振幅
        info['plat_dur'] = plat_dur
        info['interval'] = interval
        info['num_pulses'] = num_pulses

        return self._get_obs(), reward, True, False, info

    def _compute_reward(self, max_v, total_energy):
        success = max_v >= -10.0
        if success:
            reward = 200.0 - (0.5 * total_energy)
        else:
            reward = -50.0 + (max_v - (-80.0)) * 0.5
            
        return reward, {"outcome": "Success" if success else "Failed", "energy": total_energy}

    def _get_obs(self):
        v = float(self.cell.whole_cell.v[0])
        ci = float(cp.mean(self.cell.myosr_ca[0].cm))
        return np.array([v, ci], dtype=np.float32)
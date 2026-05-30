"""Step 1: Generate surrogate training data using CuPy simulation
Collects (amp, width, interval, count) -> (max_v, energy, success, final_v)
"""
import sys, os, numpy as np, cupy as cp
from tqdm import tqdm
sys.path.append("/home/xp/Code/cardiac-simulation")
from models.gpu_spatial_cell import GPUSpatialCell
from config.default_params import DefaultParams

params = DefaultParams.get_all_params()

def run_single_simulation(amp, width, interval, count):
    cell = GPUSpatialCell(
        nx=params["NX"], ny=params["NY"], nz=params["NZ"],
        filename="data_gen", rng_seed=np.random.randint(0, 99999)
    )
    cell.whole_cell.v[0] = params["INITIAL_V"]
    cell.whole_cell.xnai[0] = params["INITIAL_XNAI"]
    for myosr in cell.myosr_ca:
        myosr.cm.fill(params["INITIAL_CM"])
    
    istim_gpu = cp.zeros(1, dtype=cp.float32)
    dt = params["DT"]
    beat_duration = 400.0
    steps = int(beat_duration / dt)
    
    pulse_times = []
    for i in range(count):
        start_t = 10.0 + i * (width + interval)
        end_t = start_t + width
        pulse_times.append((start_t, end_t))
    
    max_v = -100.0
    total_energy = 0.0
    final_v = -87.0
    
    for step_idx in range(steps):
        sim_time = step_idx * dt
        i_val = 0.0
        for (start, end) in pulse_times:
            if start <= sim_time < end:
                i_val = amp
                break
        istim_gpu[0] = i_val
        cell.update_cru_flux()
        cell.update_myosr_flux()
        cell.compute_calcium_diffusion()
        cell.update_voltage(istim_gpu, istim_on=(i_val > 0))
        cell.time += dt
        
        v_curr = float(cell.whole_cell.v[0])
        if v_curr > max_v: max_v = v_curr
        total_energy += abs(i_val) * dt
        final_v = v_curr
    
    success = 1 if max_v >= -10.0 else 0
    return max_v, total_energy, success, final_v

np.random.seed(42)
n_samples = 2000
results = []

print(f"Generating {n_samples} random stimulation samples...")
for i in tqdm(range(n_samples)):
    amp = np.random.uniform(0, 60)
    width = np.random.uniform(0.1, 10)
    interval = np.random.uniform(1, 200)
    count = np.random.randint(1, 6)
    try:
        max_v, energy, success, final_v = run_single_simulation(amp, width, interval, count)
        results.append([amp, width, interval, count, max_v, energy, success, final_v])
    except Exception as e:
        print(f"Error at sample {i}: {e}")
        continue

data = np.array(results, dtype=np.float32)
save_dir = "/home/xp/Code/cardiac-simulation/surrogate"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "surrogate_dataset.npy"), data)
header = "amp,width,interval,count,max_v,energy,success,final_v"
np.savetxt(os.path.join(save_dir, "surrogate_dataset.csv"), data, delimiter=",", header=header, comments="", fmt="%.4f")
print(f"\nData saved. Shape: {data.shape}, Success rate: {data[:, 6].mean()*100:.1f}%")

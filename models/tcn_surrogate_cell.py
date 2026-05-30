import torch
import numpy as np
import sys
import os

# 确保能找到 TCN 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tcn')))
from tcn_dynamics_model import TCNStateSpaceModel

class TCNSurrogateCell:
    """TCN 极速代理模型，提供与 GPUSpatialCell 兼容的观测接口"""
    
    def __init__(self, model_path, initial_ca_spatial, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = TCNStateSpaceModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.initial_ca = torch.tensor(initial_ca_spatial, dtype=torch.float32).to(self.device)
        self.time = 0.0
        
        self.reset()
        
    def reset(self):
        self.time = 0.0
        with torch.no_grad():
            self.current_latent = self.model.get_initial_latent(self.initial_ca.unsqueeze(0))
        # 经验初始值
        self.current_v = -85.0
        self.current_ci = float(self.initial_ca.mean().cpu())
        
    def update_voltage(self, istim, istim_on=True, vclamp=False):
        """拦截物理计算，直接在隐空间进行单步推理"""
        val = float(istim[0]) if hasattr(istim, '__getitem__') else float(istim)
        curr_u = torch.tensor([[val if istim_on else 0.0]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            next_latent, ca_next, v_next = self.model.forward_step(self.current_latent, curr_u)
            
            self.current_latent = next_latent
            self.current_v = v_next.item()
            self.current_ci = ca_next.mean().item()
            
    # 空操作：覆盖所有重计算密集型的 CUDA 物理内核调用
    def update_cru_flux(self): pass
    def update_myosr_flux(self): pass
    def compute_calcium_diffusion(self): pass
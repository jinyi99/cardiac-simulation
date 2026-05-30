import torch
import torch.nn as nn

class TCNStateSpaceModel(nn.Module):
    def __init__(self, spatial_nodes=64, latent_dim=16):
        super(TCNStateSpaceModel, self).__init__()
        self.spatial_nodes = spatial_nodes
        self.latent_dim = latent_dim

        # 1. 空间编码器 (Encoder): 将当前时刻的 64 维空间钙分布压缩到 16 维隐空间
        # 输入形状: (Batch, 1, 64) -> 1 个通道，长度 64
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (spatial_nodes // 4), latent_dim)
        )

        # 2. 隐空间动力学 (Latent Dynamics): s_{t+1} = f(s_t, I_stim)
        # 接收动作(刺激电流)和当前状态预测下一状态
        self.dynamics = nn.GRUCell(input_size=1, hidden_size=latent_dim)

        # 3. 空间解码器 (Ca Decoder): 重构下一时刻的 64 维钙分布
        self.fc_decode_ca = nn.Linear(latent_dim, 16 * (spatial_nodes // 4))
        self.decoder_ca = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1)
        )

        # 4. 电压预测头 (Voltage Decoder): 从隐变量映射到全局膜电压
        self.decoder_v = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_initial_latent(self, initial_ca):
        """将初始状态的钙分布编码为初始隐变量 s_0"""
        # initial_ca 形状: (Batch, 64)
        x = initial_ca.unsqueeze(1) # (Batch, 1, 64)
        return self.encoder(x)

    def forward_step(self, latent_t, istim_t):
        """
        单步预测：
        latent_t: (Batch, latent_dim)
        istim_t: (Batch, 1)
        返回: latent_next, predicted_ca, predicted_v
        """
        # 1. 状态转移
        latent_next = self.dynamics(istim_t, latent_t)
        
        # 2. 物理量解码
        dec_in = self.fc_decode_ca(latent_next).unsqueeze(2) # (Batch, 16, 16)
        dec_in = dec_in.view(-1, 16, self.spatial_nodes // 4)
        
        ca_spatial_next = self.decoder_ca(dec_in).squeeze(1) # (Batch, 64)
        v_next = self.decoder_v(latent_next)                 # (Batch, 1)
        
        return latent_next, ca_spatial_next, v_next

    def forward_trajectory(self, initial_ca, istim_sequence):
        """用于离线训练：给定初始钙分布和整个动作序列，自回归预测完整轨迹"""
        batch_size, seq_len = istim_sequence.shape
        latent = self.get_initial_latent(initial_ca)
        
        ca_preds = []
        v_preds = []
        
        for t in range(seq_len):
            u_t = istim_sequence[:, t].unsqueeze(1)
            latent, ca_next, v_next = self.forward_step(latent, u_t)
            ca_preds.append(ca_next.unsqueeze(1))
            v_preds.append(v_next.unsqueeze(1))
            
        return torch.cat(ca_preds, dim=1), torch.cat(v_preds, dim=1)
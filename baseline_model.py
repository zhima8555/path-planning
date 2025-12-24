"""
Baseline CNN-PPO 模型
用于与双注意力模型进行对比
"""
import torch
import torch.nn as nn
import math

class BaselineCNNActorCritic(nn.Module):
    """
    标准CNN Actor-Critic模型（无注意力机制）
    作为对照组与双注意力模型对比
    """
    def __init__(self, action_dim=2):
        super().__init__()
        
        # 标准CNN特征提取（与双注意力模型相同架构，但无注意力）
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        self.cnn_feature_dim = 64 * 1 * 1  # 64
        
        # 速度向量编码
        self.vec_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion_dim = self.cnn_feature_dim + 32  # 96
        
        # GRU时序建模
        self.hidden_dim = 256
        self.gru = nn.GRUCell(self.fusion_dim, self.hidden_dim)
        
        # Actor网络
        self.actor_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs_dict, hidden_state=None):
        x_img = obs_dict['image']
        x_vec = obs_dict['vector']
        batch_size = x_img.size(0)
        
        # CNN特征提取（无注意力）
        img_features = self.cnn(x_img)  # [B, 64]
        
        # 速度编码
        vec_features = self.vec_encoder(x_vec)  # [B, 32]
        
        # 特征融合
        fusion_features = torch.cat([img_features, vec_features], dim=1)
        
        # 隐藏状态
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x_img.device)
        
        # GRU
        next_hidden = self.gru(fusion_features, hidden_state)
        
        # Actor输出
        mu = self.actor_mu(next_hidden)
        std = self.log_std.exp().expand_as(mu)
        
        # Critic输出
        value = self.critic(next_hidden)
        
        return mu, std, value, next_hidden


if __name__ == "__main__":
    model = BaselineCNNActorCritic(action_dim=2)
    
    batch_size = 4
    obs_dict = {
        'image': torch.randn(batch_size, 3, 40, 40),
        'vector': torch.randn(batch_size, 2)
    }
    
    mu, std, value, hidden = model(obs_dict)
    print(f"mu: {mu.shape}, std: {std.shape}, value: {value.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline模型参数量: {total_params:,}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpikeToConceptDecoder(nn.Module):
    """
    脈衝到具身概念神經解碼器 (Spike-to-Concept Neural Decoder)
    輸入 SNN 的脈衝序列 (pre_spikes 與 post_spikes 拼接)，
    透過在線自監督學習將其解碼為 6 種內在生理/心理感受 (Grounded Concepts)。
    這解決了「符號接地問題」，使語言區能讀取真正的神經脈衝特徵。
    """
    def __init__(self, spike_dim, concept_dim=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spike_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, concept_dim),
            nn.Sigmoid()  # 將激活值限制在 0 ~ 1 之間
        )
        
    def forward(self, spikes):
        """
        spikes: (batch_size, spike_dim)
        return: (batch_size, concept_dim) -> [HUNGER, PAIN, CURIOSITY, FATIGUE, SLEEPINESS, INSTABILITY]
        """
        return self.mlp(spikes)

class GlobalWorkspace(nn.Module):
    """
    全域工作空間 (Global Workspace - GWT)
    整合多模態感知資訊 (視覺、本體感覺、運動狀態、邊緣情感)，
    使用「贏者全拿 (Winner-Take-All, WTA)」競爭寫入機制。
    每一時間步，各個腦區計算自己的顯著度 (Salience)，只有顯著度最高(且突破閾值)的腦區
    能將其狀態寫入工作空間，並「廣播 (Broadcast)」至全腦作為小腦膜電位調變電流 I_gw。
    """
    def __init__(self, vision_dim=16, sensory_dim=111, motor_dim=32, limbic_dim=6, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 各腦區的特徵投影與顯著度評估器 (Salience Evaluators)
        self.proj_vision = nn.Linear(vision_dim, latent_dim)
        self.sal_vision = nn.Linear(latent_dim, 1)
        
        self.proj_sensory = nn.Linear(sensory_dim, latent_dim)
        self.sal_sensory = nn.Linear(latent_dim, 1)
        
        self.proj_motor = nn.Linear(motor_dim, latent_dim)
        self.sal_motor = nn.Linear(latent_dim, 1)
        
        self.proj_limbic = nn.Linear(limbic_dim, latent_dim)
        self.sal_limbic = nn.Linear(latent_dim, 1)
        
        # 全域廣播投影器 (Broadcasting Projectors)
        # 1. 廣播回小腦神經元膜電位調製電流 (I_gw)
        self.broadcast_snn = nn.Linear(latent_dim, motor_dim)
        # 2. 廣播回學習率調製因子 (lr_mod)
        self.broadcast_lr = nn.Linear(latent_dim, 1)
        
    def forward(self, vision, sensory, motor, limbic):
        """
        各腦區輸入：
        vision: (batch, vision_dim)
        sensory: (batch, sensory_dim)
        motor: (batch, motor_dim)
        limbic: (batch, limbic_dim)
        """
        batch_size = vision.size(0)
        device = vision.device
        
        # 1. 投影到統一的 Latent 空間
        v_latent = self.proj_vision(vision)    # (batch, latent_dim)
        s_latent = self.proj_sensory(sensory)  # (batch, latent_dim)
        m_latent = self.proj_motor(motor)      # (batch, latent_dim)
        l_latent = self.proj_limbic(limbic)    # (batch, latent_dim)
        
        # 2. 評估各腦區當前的顯著度 (Salience Scores)
        v_sal = torch.sigmoid(self.sal_vision(v_latent)) # (batch, 1)
        s_sal = torch.sigmoid(self.sal_sensory(s_latent)) # (batch, 1)
        m_sal = torch.sigmoid(self.sal_motor(m_latent)) # (batch, 1)
        l_sal = torch.sigmoid(self.sal_limbic(l_latent)) # (batch, 1)
        
        # 3. 贏者全拿 (WTA) 競爭寫入機制
        # 合併分數 (batch, 4)
        sal_stack = torch.cat([v_sal, s_sal, m_sal, l_sal], dim=-1)
        winners = torch.argmax(sal_stack, dim=-1) # (batch,)
        
        # 建立 WTA 掩碼並提取贏家的 Latent 向量
        gw_latent = torch.zeros(batch_size, self.latent_dim, device=device)
        salience = torch.zeros(batch_size, 1, device=device)
        winner_name = []
        
        brain_regions = ["Vision", "Sensory", "Motor", "Limbic"]
        
        for idx in range(batch_size):
            win_idx = winners[idx].item()
            winner_name.append(brain_regions[win_idx])
            
            if win_idx == 0:
                gw_latent[idx] = v_latent[idx]
                salience[idx] = v_sal[idx]
            elif win_idx == 1:
                gw_latent[idx] = s_latent[idx]
                salience[idx] = s_sal[idx]
            elif win_idx == 2:
                gw_latent[idx] = m_latent[idx]
                salience[idx] = m_sal[idx]
            else:
                gw_latent[idx] = l_latent[idx]
                salience[idx] = l_sal[idx]
                
        # 4. 全域廣播 (Global Broadcasting)
        # 被寫入的焦點資訊，經過廣播器投影，疊加至小腦與學習率調製中
        I_gw = self.broadcast_snn(gw_latent) * salience
        lr_mod = torch.sigmoid(self.broadcast_lr(gw_latent))
        
        return gw_latent, salience, I_gw, lr_mod, winner_name[0]

import torch
import torch.nn as nn

class SensoryCortex(nn.Module):
    """
    感知皮層 (Sensory Cortex)
    將環境的連續浮點數狀態 (例如：倒立擺的角度、角速度)
    轉換為離散的脈衝序列 (Spike Trains)，以輸入給下游的神經元。
    
    這裡採用簡單的機率編碼 (Rate Coding) 或直接轉換為電壓輸入。
    為了完全符合 SNN 範式，我們將連續值轉為發射脈衝的機率。
    """
    def __init__(self, state_dim, num_neurons_per_state):
        super().__init__()
        self.state_dim = state_dim
        self.num_neurons_per_state = num_neurons_per_state
        self.total_neurons = state_dim * num_neurons_per_state
        
    def forward(self, state_tensor):
        """
        state_tensor: (batch, state_dim) 來自環境的觀察值
        回傳: (batch, total_neurons) 的脈衝 {0, 1}
        """
        # 簡單的正規化 (假設狀態範圍大約在 -1 到 1 之間)
        # 轉為 0 到 1 之間的機率
        probs = torch.sigmoid(state_tensor)
        
        # 擴展到多個神經元 (每個狀態維度由多個神經元表徵)
        probs_expanded = probs.unsqueeze(-1).expand(-1, -1, self.num_neurons_per_state)
        probs_flat = probs_expanded.reshape(state_tensor.shape[0], -1)
        
        # 根據機率生成泊松脈衝 (Poisson Spikes)
        spikes = torch.bernoulli(probs_flat)
        return spikes

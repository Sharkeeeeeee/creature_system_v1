import torch
import math

class Thalamus:
    """
    丘腦 (Thalamus)
    職責：感官閘道 (Sensory Gating)，將物理狀態轉換為神經脈衝。
    """
    def __init__(self, state_dim, neurons_per_state):
        self.state_dim = state_dim
        self.neurons_per_state = neurons_per_state
        self.out_features = state_dim * neurons_per_state
        
    def encode(self, state_tensor):
        # 簡單的速率編碼 (Rate Coding)
        rates = torch.abs(state_tensor) 
        rates = rates.repeat(1, self.neurons_per_state)
        spikes = (torch.rand_like(rates) < rates).float()
        return spikes

class Hypothalamus:
    """
    下視丘 (Hypothalamus)
    職責：內分泌中樞、產生情緒 (恐懼/好奇)。
    """
    def __init__(self, baseline_reward=0.0):
        self.moving_average_reward = baseline_reward
        self.alpha = 0.1
        
    def compute_rpe(self, current_reward):
        rpe = current_reward - self.moving_average_reward
        self.moving_average_reward = (1 - self.alpha) * self.moving_average_reward + self.alpha * current_reward
        return torch.tensor([rpe], dtype=torch.float32)
        
    def compute_curiosity_reward(self, prediction_error_tensor):
        """高斯好奇心曲線"""
        e_abs = abs(prediction_error_tensor.item())
        mu = 0.5
        sigma = 0.2
        if e_abs > 0.8:
            return -1.0 # 恐懼
        return math.exp(-((e_abs - mu)**2) / (2 * sigma**2))

class Diencephalon:
    """間腦複合體"""
    def __init__(self, state_dim=4, neurons_per_state=10):
        self.thalamus = Thalamus(state_dim, neurons_per_state)
        self.hypothalamus = Hypothalamus()

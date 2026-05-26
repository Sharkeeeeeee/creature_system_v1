import torch
import math

class Neuromodulator:
    """
    神經調製系統 (Neuromodulation)
    這模擬了多巴胺系統或大腦預測編碼的殘差計算。
    
    在純局部的 SNN 中，必須有這樣一個「全局廣播信號」來指導 STDP。
    這裡我們用一個簡單的 Reward Prediction Error (RPE) 作為第三因子。
    """
    def __init__(self, baseline_reward=0.0):
        self.baseline_reward = baseline_reward
        self.moving_average_reward = baseline_reward
        self.alpha = 0.1 # 移動平均的更新率
        
    def compute_prediction_error(self, current_reward, state=None):
        """
        計算預測誤差 (Surprise)。
        如果當前獲得的獎勵比預期好，產生正向誤差 (LTP 加強)；
        如果比預期差，產生負向誤差 (LTD 抑制)。
        """
        # 簡單的時序差分誤差 (TD Error 概念的簡化)
        rpe = current_reward - self.moving_average_reward
        
        # 更新預期獎勵
        self.moving_average_reward = (1 - self.alpha) * self.moving_average_reward + self.alpha * current_reward
        
        # 將誤差轉為張量形式以便廣播到所有突觸
        return torch.tensor([rpe], dtype=torch.float32)

    def compute_curiosity_reward(self, prediction_error_tensor):
        """
        高斯好奇心曲線 (Gaussian Curiosity - Phase 7)
        :param prediction_error_tensor: 來自 PFC 的物理狀態預測誤差 (E)
        """
        e_abs = abs(prediction_error_tensor.item())
        mu = 0.5   # 最適新奇點 (太低無聊，太高危險)
        sigma = 0.2
        
        # 誤差太大會變成不可控的危險，轉為恐懼 (負獎勵)
        if e_abs > 0.8:
            return -1.0
            
        # 倒 U 型激發曲線
        curiosity = math.exp(-((e_abs - mu)**2) / (2 * sigma**2))
        return curiosity

class FreeEnergyMinimizer:
    """
    自由能最小化 / 預測編碼 (Predictive Coding)
    未來擴充：透過內部生成模型預測下一個 state，
    將 (預測狀態 - 實際狀態) 作為誤差信號廣播給神經網路。
    """
    def __init__(self):
        pass
    
    def compute_surprise(self, predicted_state, actual_state):
        mse = torch.nn.functional.mse_loss(predicted_state, actual_state)
        return -mse # 誤差越大，驚奇越高，視為負向獎勵

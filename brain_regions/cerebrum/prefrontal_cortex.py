import torch
import torch.nn as nn
from ncps.torch import CfC

class PrefrontalCortex(nn.Module):
    """
    前額葉皮質 (Prefrontal Cortex) - 內部預測世界模型 (World Model)
    使用 Liquid Neural Network (LNN/CfC) 建立連續時間的物理法則預測器。
    """
    def __init__(self, state_dim=4, action_dim=1, hidden_units=16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # LNN (Liquid Neural Network - CfC) 作為物理世界的動力學預測器
        # 輸入: 狀態(4) + 動作(1) = 5
        self.lnn = CfC(input_size=state_dim + action_dim, units=hidden_units, batch_first=True)
        
        # 解碼器: 將隱藏狀態映射回預測的下一幀物理狀態
        self.decoder = nn.Linear(hidden_units, state_dim)
        
        # 意志力系統 (Willpower System)
        self.willpower_level = 1.0
        
    def forward(self, state, action, hx=None):
        """
        state: (batch, state_dim)
        action: (batch, action_dim) 
        hx: LNN 的隱藏狀態 (包含環境過去的時序動態)
        """
        # 確保 action 具有正確的維度 (batch, 1)
        if action.dim() == 1:
            action = action.unsqueeze(1)
            
        # 合併為 (batch, 1, 5) 的序列形式
        x = torch.cat([state, action], dim=-1).unsqueeze(1)
        
        # 透過 LNN 預測物理軌跡
        out, hx_next = self.lnn(x, hx)
        
        # 解碼預測的下一個狀態: out shape (batch, 1, hidden_units)
        next_state_pred = self.decoder(out.squeeze(1))
        
        return next_state_pred, hx_next

    def mental_simulation(self, current_state, candidate_action, steps=3, hx=None):
        """
        思維模擬 (Mental Simulation / Planning)
        預測執行此動作後，未來 N 幀的物理狀態變化。
        可用於危險預防與決策煞車。
        """
        trajectory = []
        state = current_state
        current_hx = hx
        
        with torch.no_grad():
            for _ in range(steps):
                state, current_hx = self.forward(state, candidate_action, current_hx)
                trajectory.append(state)
            
        return trajectory

    def predict_preemptive_signals(self, current_state, candidate_action, steps=3, hx=None):
        """
        前瞻思維模擬：預測未來 N 步是否會摔倒、失衡，或進入人類安全禁區，並計算預先驚訝值 (Pre-emptive RPE)
        """
        from brain_regions.brainstem import Brainstem
        trajectory = self.mental_simulation(current_state, candidate_action, steps=steps, hx=hx)
        
        pre_emptive_rpe = 0.0
        predicted_danger = 0.0
        
        for state_pred in trajectory:
            pred_state_np = state_pred[0].cpu().numpy()
            
            # 預測進入人類禁區 (熱力學/代謝價值鎖定預演)
            if Brainstem.is_unsafe_state(pred_state_np):
                pre_emptive_rpe -= 50.0  # 腦內模擬觸發極大劇痛，產生強烈的避障學習信號
                predicted_danger += 2.0
                
            if state_pred.size(-1) > 2:
                pred_z = state_pred[0, 2].item()
                # 預測摔倒 (Z < 0.3)
                if pred_z < 0.3:
                    pre_emptive_rpe -= 15.0 # 給予預先警告負反饋 (Dopamine Dip)
                    predicted_danger += 1.0
                # 預測失衡
                elif abs(pred_z - 0.5) > 0.15:
                    pre_emptive_rpe -= (abs(pred_z - 0.5) * 5.0)
                    
        return torch.tensor([pre_emptive_rpe], dtype=torch.float32), predicted_danger

    def reset_willpower(self):
        """
        生命重生或睡眠醒來時，重置意志力能量池。
        """
        self.willpower_level = 1.0

    def update_willpower(self, predicted_danger, energy_level, endorphins=0.0):
        """
        更新前額葉意志力與主動煞車 (Veto) 信號。
        predicted_danger: 內部預測的危險程度
        energy_level: 當前身體的物理能量 (葡萄糖供應基準)
        endorphins: 腦內啡濃度，可減少意志力損耗速率
        
        回傳:
          veto_needed (bool): 是否需要踩煞車
          veto_signal (float): 主動抑制的強度 (0.0 ~ 1.0)
          cognitive_cost (float): 本次意志力活動的代謝能耗
        """
        # 當預測的危險值大於 0.5 時，觸發主動抑制
        veto_needed = predicted_danger > 0.5
        
        if veto_needed:
            # 意志力消耗：消耗速度與危險程度正相關，但腦內啡能將損耗率降低最多 80%
            reduction = 1.0 - (endorphins * 0.8)
            loss = min(0.05, predicted_danger * 0.02) * reduction
            self.willpower_level = max(0.0, self.willpower_level - loss)
            
            # 主動踩煞車強度：受限於當前的意志力水準 (ego depletion)
            veto_signal = min(1.0, predicted_danger * 0.8) * self.willpower_level
            
            # 認知消耗 (意志力踩煞車需要大量 ATP)
            cognitive_cost = loss * 8.0
        else:
            # 意志力恢復：在安全狀態下，恢復速度與體能能量成正比 (肚子餓時恢復極慢)
            recovery_rate = 0.01 * (energy_level / 100.0)
            self.willpower_level = min(1.0, self.willpower_level + recovery_rate)
            veto_signal = 0.0
            cognitive_cost = 0.0
            
        return veto_needed, veto_signal, cognitive_cost

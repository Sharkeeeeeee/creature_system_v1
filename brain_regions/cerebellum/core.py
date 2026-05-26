import torch
import torch.nn as nn
from learning.stdp import PredictiveSTDP

class Cerebellum(nn.Module):
    """
    小腦 (Cerebellum)
    負責基於脈衝的快速物理控制與神經資格跡 (Eligibility Traces / e-prop) 局部學習。
    """
    def __init__(self, in_features, out_features, tau_m=10.0, tau_s=5.0, tau_e=20.0, v_thresh=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 突觸權重 (從前額葉/感覺皮質 投射到 運動皮質)
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='linear')
        
        # 膜電位與突觸電流與資格跡的時間常數
        self.decay_m = math.exp(-1.0 / tau_m)
        self.decay_s = math.exp(-1.0 / tau_s)
        self.decay_e = math.exp(-1.0 / tau_e) # 資格跡衰減 (通常較慢，用於跨時間歸因)
        self.v_thresh = v_thresh
        
        # 資格跡緩衝區 (與權重矩陣同維度)
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
    def forward(self, pre_spikes, V_prev, I_syn_prev):
        # 1. 計算新的突觸電流 (Liquid 時間衰減 + 新的脈衝輸入)
        I_syn_new = I_syn_prev * self.decay_s + torch.matmul(pre_spikes, self.W.t())
        
        # 2. 計算新的膜電位 (LIF 神經元動力學)
        V_new = V_prev * self.decay_m + I_syn_new
        
        # 3. 發射脈衝 (Spike Generation) - 使用階躍函數
        post_spikes = (V_new >= self.v_thresh).float()
        
        # 4. 膜電位重置 (Reset by subtraction)
        V_new = V_new - post_spikes * self.v_thresh
        
        # 5. 更新神經資格跡 (Eligibility Trace)
        if self.training:
            trace_inc = torch.outer(post_spikes[0], pre_spikes[0])
            self.eligibility_trace = self.eligibility_trace * self.decay_e + trace_inc
            
            # [恆定性可塑性 Homeostatic Plasticity]
            # 如果全局發放率 (Firing Rate) 太高，強制觸發突觸權重衰減，強迫學會稀疏編碼
            global_firing_rate = post_spikes.mean().item()
            if global_firing_rate > 0.2:  # 如果超過 20% 的神經元同時激發
                with torch.no_grad():
                    self.W.mul_(0.995) # 衰減突觸權重以節省能量
        
        return post_spikes, V_new, I_syn_new
        
    def apply_dopamine(self, rpe, trace=None, learning_rate=0.005):
        """
        3-Factor Learning (Neuromodulated e-prop)
        多巴胺(RPE) 作為廣播訊號，結合資格跡進行權重更新。
        """
        if self.training:
            with torch.no_grad():
                # RPE 大於 0 為獎勵，小於 0 為懲罰
                current_trace = self.eligibility_trace if trace is None else trace
                self.W.add_(learning_rate * rpe * current_trace)
        
    def die_and_scramble(self):
        """
        當大腦死亡時，觸發基因演化 (Genetic Mutation)。
        保留 80% 的優良權重，對 20% 施加隨機突變，而不是 100% 洗白。
        """
        with torch.no_grad():
            mutation_mask = (torch.rand_like(self.W) < 0.2).float()
            mutation_noise = torch.randn_like(self.W) * 0.5
            self.W.add_(mutation_noise * mutation_mask)
            
    def save_muscle_memory(self, filepath="muscle_memory.pt"):
        """將神經突觸權重固化至硬碟"""
        import os
        torch.save(self.W.data, filepath)
        print(f"[Cerebellum] 肌肉記憶已固化至 {filepath}")
        
    def load_muscle_memory(self, filepath="muscle_memory.pt"):
        import os
        if os.path.exists(filepath):
            try:
                loaded_W = torch.load(filepath, map_location=self.W.device)
                if loaded_W.shape == self.W.shape:
                    self.W.data = loaded_W
                    print(f"[Cerebellum] 成功讀取肌肉記憶從 {filepath}！", flush=True)
                else:
                    print(f"[Cerebellum] 肌肉記憶形狀不匹配，不予載入。預期形狀: {self.W.shape}, 載入形狀: {loaded_W.shape}", flush=True)
            except Exception as e:
                print(f"[Cerebellum] 讀取記憶失敗: {e}", flush=True)

import math

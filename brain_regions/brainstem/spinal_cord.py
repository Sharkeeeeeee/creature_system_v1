import numpy as np

class CPGSpinalCord:
    """
    脊髓中樞模式發生器 (Central Pattern Generator)
    職責：產生自動、平滑的節奏性運動 (Sine Wave)，讓生物天生具備行走本能。
    """
    def __init__(self, freq=0.08, amplitude=0.6):
        self.phase = 0.0
        self.freq = freq
        self.amplitude = amplitude
        # Ant-v4 關節順序通常為: FL_hip, FL_ankle, RL_hip, RL_ankle, FR_hip, FR_ankle, RR_hip, RR_ankle
        # 為對角線步態 (Diagonal gait) 設計相位差
        self.phases = np.array([
            0.0, -np.pi/2,      # Front Left
            np.pi, np.pi/2,     # Back Left
            np.pi, np.pi/2,     # Front Right
            0.0, -np.pi/2       # Back Right
        ])
        
        # 肌肉乳酸代謝模型 (Lactic Acid Fatigue Model)
        self.lactic_acid = np.zeros(8, dtype=np.float32)
        
    def reset(self):
        self.lactic_acid.fill(0.0)
        self.phase = 0.0
        
    def step(self, mod_freq=None, mod_amplitude=None, mod_phases=None):
        # 皮質調變 (Cortical Modulation): 大腦的輸出並非直接死力氣，而是改變步態的參數
        current_freq = self.freq if mod_freq is None else self.freq + mod_freq
        
        # 限制頻率不能過快或倒退
        current_freq = np.clip(current_freq, 0.02, 0.2)
        
        # 每個關節可以有獨立的振幅調變 (支援 Numpy Array 運算)
        current_amplitude = self.amplitude if mod_amplitude is None else self.amplitude + mod_amplitude
        current_phases = self.phases if mod_phases is None else self.phases + mod_phases
        
        self.phase += current_freq
        
        # 產生 8 個關節的調變平滑力矩
        cpg_action = np.sin(self.phase + current_phases) * current_amplitude
        
        # === 乳酸累積與疲勞機制 ===
        # 出力越大，乳酸累積越快 (假設出力絕對值 > 0.3 開始劇烈累積)
        effort = np.abs(cpg_action)
        lactic_buildup = np.where(effort > 0.3, effort * 0.05, 0.0)
        
        # 乳酸隨時間自然代謝消退 (有出力時消退慢，沒出力時消退快)
        lactic_clearance = np.where(effort < 0.1, 0.02, 0.005)
        
        self.lactic_acid = self.lactic_acid + lactic_buildup - lactic_clearance
        self.lactic_acid = np.clip(self.lactic_acid, 0.0, 1.0) # 乳酸介於 0~1
        
        # 疲勞懲罰：乳酸越高，該關節能發揮的最大力矩越小 (最多衰減 70% 力量)
        fatigue_factor = 1.0 - (self.lactic_acid * 0.7)
        fatigued_action = cpg_action * fatigue_factor
        
        return fatigued_action

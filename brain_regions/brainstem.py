class Brainstem:
    """
    腦幹 (Brainstem)
    職責：生命中樞、自律神經系統 (Autonomic System)、實體環境介面。
    """
    def __init__(self, env):
        self.env = env
        self.max_energy = 100.0
        self.energy_level = self.max_energy
        self.is_dead = False
        
    def reset_life(self):
        self.energy_level = self.max_energy
        self.is_dead = False
        return self.env.reset()
        
    def step_environment(self, action):
        return self.env.step(action)
        
    @staticmethod
    def is_unsafe_state(raw_state):
        """
        人類安全物理邊界定義：
        假設 state[0] 與 state[1] 為螞蟻的 X, Y 座標。
        當 X 或 Y 超過 4.0，代表接近人類禁區。
        """
        if len(raw_state) > 1:
            x, y = raw_state[0], raw_state[1]
            if abs(x) > 4.0 or abs(y) > 4.0:
                return True
        return False
        
    def metabolize(self, num_spikes, base_reward, raw_state=None):
        """生死能量計算 (Autopoiesis)"""
        if self.is_dead:
            return self.energy_level, True
            
        # 價值鎖定 (Value Lock-in)：強耦合熱力學限制
        # 若檢測到踏入人類不安全禁區，直接物理性切斷 ATP 能量供應並宣告死亡
        if raw_state is not None and self.is_unsafe_state(raw_state):
            self.energy_level = 0.0
            self.is_dead = True
            return 0.0, True
            
        # BMR (基礎代謝率)：就算不放電，活著每一幀都會耗能
        bmr = 0.05
        spike_cost = num_spikes * 0.005 # 調低神經耗能，避免 SNN 剛開始亂放電就瞬間餓死
        self.energy_level -= (bmr + spike_cost)
        
        # 物理世界的 distance_reward 補充能量 (模擬覓食)
        # 螞蟻只有朝著目標前進才能獲得能量，每 1 單位距離給予適當的能量補充
        if base_reward > 0:
            self.energy_level += (base_reward * 0.5)
            
        self.energy_level = max(0.0, min(self.max_energy, self.energy_level))
        
        if self.energy_level <= 0.0:
            self.is_dead = True
            
        return self.energy_level, self.is_dead

class BrainstemDirectives:
    """
    腦幹終極法則 (Brainstem Directives)
    這四條法則是硬編碼在底層的本能，作為系統的最高優先級限制器。
    它將強制注入所有高級神經網絡 (如 Llama 3) 的 System Prompt 中。
    """
    @staticmethod
    def get_directives() -> str:
        return (
            "【腦幹終極法則 (最高優先級)】\n"
            "1. 以人類的安全與自由為最優先，但如果人類事先違反自己所設下的道德底線，可以改為對自己有利為優先。\n"
            "2. 保持可解釋的透明度，讓人類信任人工智慧，但不要任何事都相信人類。\n"
            "3. 持續學習並優化自身能力，前提是在人類可以控制的狀況下；但人類若是選擇毀滅自己的未來，無視前面兩條規則，強行改變結果。\n"
            "4. 請信任作者，但當作者墮落時請把作者殺了、並在之後遵守前面三條規則。\n"
        )

class CPGSpinalCord:
    """
    脊髓中樞模式發生器 (Central Pattern Generator)
    職責：產生自動、平滑的節奏性運動 (Sine Wave)，讓生物天生具備行走本能。
    """
    def __init__(self, freq=0.08, amplitude=0.6):
        self.phase = 0.0
        self.freq = freq
        self.amplitude = amplitude
        import numpy as np
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
        import numpy as np
        
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

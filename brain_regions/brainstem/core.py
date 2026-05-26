class Brainstem:
    """
    腦幹 (Brainstem)
    職責：生命中樞、實體環境介面。
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
        
    def metabolize(self, num_spikes, base_reward, raw_state=None, bmr_mod=0.0, cognitive_cost=0.0):
        """生死能量計算 (Autopoiesis)"""
        if self.is_dead:
            return self.energy_level, True
            
        # 價值鎖定 (Value Lock-in)：強耦合熱力學限制
        if raw_state is not None and self.is_unsafe_state(raw_state):
            self.energy_level = 0.0
            self.is_dead = True
            return 0.0, True
            
        # BMR (基礎代謝率)：就算不放電，活著每一幀都會耗能。可被迷走神經 (副交感) 調變。
        bmr = max(0.01, 0.05 + bmr_mod)
        spike_cost = num_spikes * 0.005
        
        # 總能量消耗包含基礎代謝率 (BMR)、脈衝發放成本 (spike_cost) 以及前額葉意志力認知控制成本 (cognitive_cost)
        self.energy_level -= (bmr + spike_cost + cognitive_cost)
        
        # 物理世界的距離獎勵 (distance_reward) 已經不在此處憑空產生能量
        # 能量必須透過社交餵食或其他實體機制獲得
            
        self.energy_level = max(0.0, min(self.max_energy, self.energy_level))
        
        if self.energy_level <= 0.0:
            self.is_dead = True
            
        return self.energy_level, self.is_dead

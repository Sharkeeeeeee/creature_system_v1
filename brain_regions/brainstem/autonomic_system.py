class AutonomicNervousSystem:
    """
    自律神經系統 (Autonomic Nervous System)
    負責無意識的生理調節，包含：
    - 交感神經 (Sympathetic): 戰鬥或逃跑，提高代謝與反應力
    - 副交感神經/迷走神經 (Parasympathetic/Vagus): 休息與消化，降低代謝，促進突觸固化
    """
    def __init__(self):
        self.sympathetic_tone = 0.5    # 0.0 ~ 1.0 (壓力/危機感)
        self.parasympathetic_tone = 0.5 # 0.0 ~ 1.0 (安全/放鬆感)
        
    def update(self, perceived_threat, social_safety):
        """
        perceived_threat: 來自杏仁核的恐懼訊號、痛覺 (0.0~1.0)
        social_safety: 來自 MNS 或社交餵食的安全訊號 (0.0~1.0)
        """
        # 交感神經受到威脅刺激而上升
        self.sympathetic_tone = self.sympathetic_tone * 0.9 + perceived_threat * 0.1
        # 副交感神經受到安全信號刺激而上升
        self.parasympathetic_tone = self.parasympathetic_tone * 0.9 + social_safety * 0.1
        
        # 互相拮抗 (Reciprocal inhibition)
        if self.sympathetic_tone > self.parasympathetic_tone:
            self.parasympathetic_tone *= 0.9
        else:
            self.sympathetic_tone *= 0.9
            
        return self.sympathetic_tone, self.parasympathetic_tone
        
    def get_bmr_modulation(self):
        """
        回傳基礎代謝率 (BMR) 的調變量。
        交感神經活躍時，代謝率增加；副交感活躍時，代謝率減少。
        """
        return (self.sympathetic_tone * 0.02) - (self.parasympathetic_tone * 0.02)

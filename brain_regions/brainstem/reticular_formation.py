class ReticularFormation:
    """
    網狀致動系統 (Reticular Activating System - RAS)
    負責控制全腦的「覺醒度 (Arousal Level)」與「注意力開關」。
    當疲勞或褪黑激素過高時，RAS 活性下降，導致全腦神經元超極化 (Hyperpolarization)，
    難以產生脈衝，模擬出生物的「想睡」或「昏迷」狀態。
    """
    def __init__(self):
        self.arousal_level = 1.0 # 1.0 = 清醒, 0.0 = 昏迷/深睡
        
    def update(self, fatigue, melatonin, sudden_stimulus=0.0):
        """
        更新覺醒度
        fatigue: 來自脊髓的乳酸或整體的耗能疲勞 (0.0~1.0)
        melatonin: 來自松果體的褪黑激素濃度 (0.0~1.0)
        sudden_stimulus: 突發刺激，如巨大的驚訝 RPE 或痛覺
        """
        # 突發刺激會瞬間喚醒
        if sudden_stimulus > 0.5:
            self.arousal_level = 1.0
        else:
            # 疲勞與褪黑激素會降低覺醒度
            sleep_pressure = (fatigue * 0.4) + (melatonin * 0.6)
            # 覺醒度自然恢復的趨勢被睡眠壓力抵抗
            target_arousal = max(0.1, 1.0 - sleep_pressure)
            # 平滑過渡
            self.arousal_level = self.arousal_level * 0.95 + target_arousal * 0.05
            
        return self.arousal_level
        
    def get_hyperpolarization_current(self):
        """
        將覺醒度轉換為注入全腦 SNN 的超極化抑制電流。
        當 arousal = 1.0 時，電流 = 0 (正常運作)
        當 arousal 接近 0.0 時，電流為負值 (例如 -1.5)，讓膜電位難以達到閾值
        """
        return -2.0 * (1.0 - self.arousal_level)

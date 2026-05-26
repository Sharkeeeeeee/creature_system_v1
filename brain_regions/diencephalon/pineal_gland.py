class PinealGland:
    """
    松果體 (Pineal Gland)
    功能：接收來自視叉上核的時間與光線訊號，分泌褪黑激素 (Melatonin)。
    褪黑激素過高會觸發網狀致動系統 (RAS) 降低覺醒度。
    """
    def __init__(self):
        self.melatonin = 0.0
        self.circadian_clock = 0.0 # 生理時鐘 (0.0~24.0)
        
    def update(self, time_step):
        """
        根據時間步長調整虛擬時間 (0~24) 並分泌褪黑激素。
        """
        self.circadian_clock = (self.circadian_clock + (time_step / 1000.0)) % 24.0
        if self.circadian_clock >= 22.0 or self.circadian_clock < 6.0:
            self.melatonin = min(1.0, self.melatonin + 0.05)
        else:
            self.melatonin = max(0.0, self.melatonin - 0.1)
            
        return self.melatonin

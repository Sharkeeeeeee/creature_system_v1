import torch

class InferiorOlive:
    """
    下橄欖核 (Inferior Olive)
    功能：小腦的「教鞭」。當發生預測誤差 (RPE < 0) 或疼痛/跌倒等驚訝事件時，
    下橄欖核會發射強烈的「攀緣纖維脈衝 (Climbing Fiber Spikes)」。
    這會觸發小腦柏金氏細胞 (Purkinje cells) 的長期抑制 (LTD)，
    強制抹除當前的錯誤權重。
    """
    def __init__(self, threshold=-2.0):
        self.threshold = threshold # 觸發攀緣纖維的 RPE 閾值
        
    def generate_climbing_fiber_signal(self, rpe, pain_signal=0.0):
        """
        將 RPE 或痛覺轉換為小腦學習的調節訊號 (LTD trigger)
        回傳值: climbing_fiber_active (Boolean), error_magnitude (Float)
        """
        error_magnitude = 0.0
        climbing_fiber_active = False
        
        # 當 RPE 遠低於預期 (大失誤) 或 痛覺很高時
        if rpe < self.threshold or pain_signal > 0.5:
            climbing_fiber_active = True
            error_magnitude = abs(rpe) + (pain_signal * 10.0)
            
        return climbing_fiber_active, error_magnitude

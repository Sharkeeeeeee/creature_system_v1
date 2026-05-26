import torch

class PredictiveSTDP:
    """
    三因子預測性 STDP 學習規則 (Three-Factor Predictive STDP)
    
    解決純局部 STDP 無法達成控制目標的衝突。
    將「突觸前後的激發時序 (局部)」與「預測誤差/多巴胺 (全域調製)」結合。
    這完全避免了反向傳播的鏈鎖律矩陣運算，時間複雜度為 O(1)。
    """
    def __init__(self, A_plus=0.01, A_minus=0.012):
        self.A_plus = A_plus   # LTP 學習率
        self.A_minus = A_minus # LTD 學習率

    def update(self, W, pre_traces, post_traces, pre_spikes, post_spikes, prediction_error):
        """
        pre_traces, post_traces: 突觸前後脈衝的歷史跡線 (Trace)
        prediction_error: 標量或向量，來自自由能最小化/預測編碼模組的殘差 (Surprise)
        """
        # 1. 傳統 STDP 局部計算
        # Pre-before-Post (引發長效增強 LTP): 前突觸跡線 x 後突觸脈衝
        # pre_traces: (in_features,), post_spikes: (out_features,)
        # 我們希望產生的更新矩陣形狀為 (out_features, in_features)
        ltp = self.A_plus * post_spikes.unsqueeze(1) * pre_traces.unsqueeze(0)
        
        # Post-before-Pre (引發長效抑制 LTD): 後突觸跡線 x 前突觸脈衝
        ltd = -self.A_minus * post_traces.unsqueeze(1) * pre_spikes.unsqueeze(0)
        
        local_stdp = ltp + ltd
        
        # 2. 第三因子調製 (Global Modulation)
        # 將局部的 STDP 變化量，乘以全域的預測誤差
        # 如果預測誤差大(Surprise 高)，代表當前突觸連接導致了錯誤，加強可塑性進行調整
        dW = prediction_error.unsqueeze(-1).unsqueeze(-1) * local_stdp
        
        # 3. 更新權重 (無梯度下降)
        with torch.no_grad():
            W.add_(dW.mean(dim=0))
            # 防止突觸興奮毒性 (Synaptic Weight Explosion)
            W.clamp_(-2.0, 2.0)

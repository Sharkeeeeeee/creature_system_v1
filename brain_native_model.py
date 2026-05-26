import torch
import torch.nn as nn

class LiquidSpikingNeuron(nn.Module):
    """
    Unified Liquid Spiking Neuron (LSN)
    
    這個類別完美解決了 LNN (連續時間) 與 SNN (離散脈衝) 的維度衝突。
    真正的神經元並不是「一個連續網路層」連接「一個離散網路層」，而是一個統一體：
    1. 突觸與膜電位具備 LNN 的連續非線性液態動態 (Continuous Liquid Dynamics)。
    2. 神經元的輸出是離散的脈衝 (Discrete Spikes)。
    """
    def __init__(self, in_features, out_features, dt=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt
        
        # 突觸權重 (可透過 STDP 更新，不依賴 Autograd)
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.05, requires_grad=False)
        
        # Liquid Time-Constant (LTC) 核心參數
        self.tau = nn.Parameter(torch.ones(out_features) * 10.0, requires_grad=False) # 基礎時間常數
        self.A = nn.Parameter(torch.ones(out_features), requires_grad=False)          # 靜息/反轉電位
        
        # SNN 脈衝發射參數
        self.V_th = 1.0     # 脈衝閾值
        self.V_reset = 0.0  # 重置電位
        
        # 突觸電流衰減常數 (模擬神經遞質的消除)
        self.tau_syn = 5.0
        
    def forward(self, pre_spikes, V_prev, I_syn_prev):
        """
        pre_spikes: (batch, in_features) 前突觸傳來的離散脈衝 {0, 1}
        V_prev: (batch, out_features) 上一時刻的膜電位 (連續值)
        I_syn_prev: (batch, out_features) 上一時刻的突觸電流 (連續值)
        """
        # --- [橋接離散與連續] ---
        # 1. 將離散脈衝轉換為連續的突觸電流 (Synaptic Current Dynamics)
        # 真實大腦中，脈衝到達突觸會釋放神經遞質，產生隨時間衰減的連續電流
        decay_syn = torch.exp(torch.tensor(-self.dt / self.tau_syn))
        I_in = torch.matmul(pre_spikes, self.W.t())
        I_syn = I_syn_prev * decay_syn + I_in
        
        # --- [LNN 液態動態] ---
        # 2. 非線性液態動態整合 (Liquid Membrane Dynamics)
        f_I = torch.tanh(I_syn) # 神經元對輸入電流的非線性響應
        
        # LTC ODE: dV/dt = - [1/tau + f(I)] * V + f(I) * A
        # 電流不僅驅動電位，還動態改變神經元的時間常數 (這就是 Liquid 的精髓)
        dV = (- (1.0/self.tau + f_I) * V_prev + f_I * self.A) * self.dt
        V_next = V_prev + dV
        
        # --- [SNN 離散輸出] ---
        # 3. 觸發與重置 (Spiking Mechanism)
        spikes = (V_next >= self.V_th).float()
        
        # Hard Reset: 如果發射脈衝，電位歸零；否則保留
        V_next = V_next * (1 - spikes) + self.V_reset * spikes
        
        return spikes, V_next, I_syn

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
        # Pre-before-Post (引發長效增強 LTP)
        ltp = self.A_plus * pre_traces.unsqueeze(1) * post_spikes.unsqueeze(-1)
        
        # Post-before-Pre (引發長效抑制 LTD)
        ltd = -self.A_minus * post_traces.unsqueeze(-1) * pre_spikes.unsqueeze(1)
        
        local_stdp = ltp + ltd
        
        # 2. 第三因子調製 (Global Modulation)
        # 將局部的 STDP 變化量，乘以全域的預測誤差
        # 如果預測誤差大(Surprise 高)，代表當前突觸連接導致了錯誤，加強可塑性進行調整
        dW = prediction_error.unsqueeze(-1).unsqueeze(-1) * local_stdp
        
        # 3. 更新權重 (無梯度下降)
        W.add_(dW.mean(dim=0))

if __name__ == "__main__":
    # 簡單的維度測試
    batch_size = 4
    in_neurons = 10
    out_neurons = 5
    
    # 初始化統一模型
    lsn = LiquidSpikingNeuron(in_neurons, out_neurons)
    
    # 模擬初始狀態
    pre_spikes = torch.randint(0, 2, (batch_size, in_neurons)).float()
    V_prev = torch.zeros(batch_size, out_neurons)
    I_syn_prev = torch.zeros(batch_size, out_neurons)
    
    # 前向傳播 (完美整合離散與連續)
    out_spikes, V_new, I_syn_new = lsn(pre_spikes, V_prev, I_syn_prev)
    
    print("Pre-synaptic Spikes (Discrete):", pre_spikes.shape)
    print("Membrane Potential (Continuous Liquid):", V_new.shape)
    print("Post-synaptic Spikes (Discrete):", out_spikes.shape)
    print("Dimension conflict resolved successfully.")

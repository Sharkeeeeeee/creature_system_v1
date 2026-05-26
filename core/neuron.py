import torch
import torch.nn as nn

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, torch.tensor(threshold, dtype=torch.float32))
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        # Arctan 替代梯度 (Surrogate Gradient)
        alpha = 2.0
        grad_input = grad_output * (alpha / (1 + (alpha * (input - threshold))**2))
        return grad_input, None

spike_fn = SurrogateHeaviside.apply

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
        
        # 突觸權重 (支援 Autograd 供 DMN 離線回放預訓練使用)
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.05, requires_grad=True)
        
        # Liquid Time-Constant (LTC) 核心參數
        self.tau = nn.Parameter(torch.ones(out_features) * 10.0, requires_grad=True) # 基礎時間常數
        self.A = nn.Parameter(torch.ones(out_features), requires_grad=True)          # 靜息/反轉電位
        
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
        # 3. 觸發與重置 (Spiking Mechanism, using Surrogate Gradient)
        spikes = spike_fn(V_next, self.V_th)
        
        # Hard Reset: 如果發射脈衝，電位歸零；否則保留
        V_next = V_next * (1 - spikes) + self.V_reset * spikes
        
        return spikes, V_next, I_syn

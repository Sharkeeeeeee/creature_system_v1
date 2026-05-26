import torch
import torch.nn as nn
import numpy as np

class WernickeAreaSNN(nn.Module):
    """
    威尼克感知語言區 SNN (Wernicke's Area)
    輸入：環境中離散的聲音信號（例如一個 One-hot 聲音 ID）
    職責：藉由速率編碼將聲音轉換為脈衝，並在線通過 STDP 將聲音模式與生理情感/驚訝狀態 (RPE) 連接。
    """
    def __init__(self, num_symbols=4, out_neurons=16, tau_m=15.0, v_thresh=1.0):
        super().__init__()
        self.num_symbols = num_symbols
        self.out_neurons = out_neurons
        self.v_thresh = v_thresh
        self.decay_m = np.exp(-1.0 / tau_m)
        
        # 突觸權重：離散符號 -> 威尼克輸出神經元
        self.W = nn.Parameter(torch.empty(out_neurons, num_symbols))
        nn.init.uniform_(self.W, 0.1, 0.5)
        
        # 膜電位
        self.register_buffer('V', torch.zeros(1, out_neurons))
        
    def forward(self, symbol_idx, global_rpe=0.0, training=True):
        device = self.W.device
        pre_spikes = torch.zeros(self.num_symbols, 1, device=device)
        
        if symbol_idx is not None and 0 <= symbol_idx < self.num_symbols:
            pre_spikes[symbol_idx] = 1.0
            
        I_syn = torch.matmul(self.W, pre_spikes).squeeze(-1) # (out_neurons,)
        V_new = self.V * self.decay_m + I_syn
        
        post_spikes = (V_new >= self.v_thresh).float()
        self.V = V_new - post_spikes * self.v_thresh
        
        if training and symbol_idx is not None:
            with torch.no_grad():
                trace = torch.outer(post_spikes[0], pre_spikes.squeeze(-1))
                self.W.add_(0.01 * global_rpe * trace)
                self.W.clamp_(0.0, 2.0)
                
        return post_spikes

class MirrorNeuronSystemSNN(nn.Module):
    """
    鏡像神經元系統 (Mirror Neuron System - MNS)
    職責：將 Wernicke (聽覺感知) 與 Broca (運動意圖) 綁定。
    當聽見別人的聲音時，會激發自身的運動意圖脈衝，產生「模仿」衝動。
    """
    def __init__(self, sensory_neurons=16, intent_neurons=4):
        super().__init__()
        # W_mns 權重：聽覺感知脈衝 -> 運動意圖電流
        self.W_mns = nn.Parameter(torch.empty(intent_neurons, sensory_neurons))
        nn.init.uniform_(self.W_mns, 0.0, 0.2) # 初始化較弱，靠 STDP 學習
        
        self.sensory_trace = torch.zeros(1, sensory_neurons)
        
    def forward(self, wernicke_spikes, broca_intent_spikes=None, training=True, oxytocin=0.5):
        device = self.W_mns.device
        self.sensory_trace = self.sensory_trace.to(device)
        
        # 催產素 (oxytocin) 提高對鏡像神經元模仿衝動的反應強度
        mns_strength = 1.0 + oxytocin * 2.0
        I_mns = torch.matmul(wernicke_spikes, self.W_mns.t()) * mns_strength
        
        # STDP 更新：將「我發出的意圖(Broca)」與「我聽到的聲音(Wernicke)」綁定
        if training and broca_intent_spikes is not None:
            with torch.no_grad():
                # 簡單 Hebbian/STDP：如果感知與意圖同時發生，增強連結
                # 這裡的邏輯是：當螞蟻自己發聲時，會同時聽到自己的聲音 (或獲得社會回饋)
                trace = torch.outer(broca_intent_spikes[0], wernicke_spikes[0])
                # 催產素增強 STDP 可塑性，加快模仿學習速度 (最大放大至 4 倍)
                learning_rate = 0.05 * (1.0 + oxytocin * 3.0)
                self.W_mns.add_(learning_rate * trace)
                self.W_mns.clamp_(0.0, 1.5)
                
        return I_mns

class BrocaIntentSNN(nn.Module):
    """
    階層式 Broca - 高階意圖區 (Broca Intent Area)
    輸入：全域工作空間的內部感受 (limbic_state) + MNS 傳來的模仿電流
    職責：產生高階的「發聲意圖」脈衝 (例如：想求食、想尖叫)。
    """
    def __init__(self, in_features=6, intent_neurons=4, tau_m=10.0, v_thresh=1.2):
        super().__init__()
        self.intent_neurons = intent_neurons
        self.v_thresh = v_thresh
        self.decay_m = np.exp(-1.0 / tau_m)
        
        self.W = nn.Parameter(torch.empty(intent_neurons, in_features))
        nn.init.uniform_(self.W, 0.1, 0.4)
        
        self.register_buffer('V', torch.zeros(1, intent_neurons))
        
    def forward(self, limbic_state, I_mns=None, global_rpe=0.0, training=True):
        I_syn = torch.matmul(limbic_state, self.W.t()) # (1, intent_neurons)
        
        if I_mns is not None:
            I_syn = I_syn + I_mns * 1.5 # 加上鏡像神經元的模仿衝動
            
        V_new = self.V * self.decay_m + I_syn
        
        post_spikes = (V_new >= self.v_thresh).float()
        self.V = V_new - post_spikes * self.v_thresh
        
        if training:
            with torch.no_grad():
                trace = torch.outer(post_spikes[0], limbic_state[0])
                self.W.add_(0.01 * global_rpe * trace)
                self.W.clamp_(0.0, 2.0)
                
        return post_spikes

class BrocaMotorChunkingSNN(nn.Module):
    """
    階層式 Broca - 低階組塊化區 (Broca Motor Chunking Decoder)
    輸入：高階意圖脈衝
    職責：透過延遲線 (Delay Lines) 將一個意圖脈衝展開為時間序列上的符號 (Chunking)。
    例如：意圖 0 (求食) -> "A" (t=0) -> "B" (t=1)。
    """
    def __init__(self, intent_neurons=4, num_symbols=4, seq_len=2):
        super().__init__()
        self.intent_neurons = intent_neurons
        self.num_symbols = num_symbols
        self.seq_len = seq_len
        
        # 預先定義好語法組塊的時序權重 (也可設計為可學習，此處先給定先驗知識以加速成型)
        # W_chunk shape: (intent_neurons, seq_len, num_symbols)
        self.W_chunk = nn.Parameter(torch.zeros(intent_neurons, seq_len, num_symbols), requires_grad=False)
        
        # 設定固定語法組塊
        # 意圖 0: 飢餓求食 -> 序列 A-B (0, 1)
        self.W_chunk[0, 0, 0] = 1.0 # t=0, 發 A
        self.W_chunk[0, 1, 1] = 1.0 # t=1, 發 B
        
        # 意圖 1: 痛楚求救 -> 序列 C-C (2, 2)
        self.W_chunk[1, 0, 2] = 1.0
        self.W_chunk[1, 1, 2] = 1.0
        
        # 意圖 2: 發現新奇 -> 序列 D-A (3, 0)
        self.W_chunk[2, 0, 3] = 1.0
        self.W_chunk[2, 1, 0] = 1.0
        
        # 意圖 3: 疲勞/睡眠 -> 序列 B-D (1, 3)
        self.W_chunk[3, 0, 1] = 1.0
        self.W_chunk[3, 1, 3] = 1.0
        
        # 意圖脈衝的歷史緩衝區，用於觸發序列
        self.intent_buffer = []
        
        self.vocal_buffer = [] # 歷史發聲字串緩衝
        self.vocal_map = ["A", "B", "C", "D"]
        
    def forward(self, intent_spikes):
        device = self.W_chunk.device
        intent_np = intent_spikes[0].detach().cpu().numpy()
        
        # 如果有新的意圖發出脈衝，將其加入緩衝區，並帶有生命週期 (seq_len)
        active_intents = np.where(intent_np > 0)[0]
        for intent_id in active_intents:
            self.intent_buffer.append({'id': intent_id, 'timer': 0})
            
        symbol_out = None
        
        # 遍歷當前活躍的意圖序列，計算最終要發出的運動符號
        # 若有多個意圖碰撞，簡單取最先發生的一個 (WTA)
        if len(self.intent_buffer) > 0:
            active_chunk = self.intent_buffer[0]
            iid = active_chunk['id']
            t = active_chunk['timer']
            
            # 從權重中取出對應時刻的符號
            symbol_weights = self.W_chunk[iid, t, :]
            active_symbols = torch.where(symbol_weights > 0.5)[0]
            
            if len(active_symbols) > 0:
                win_idx = active_symbols[0].item()
                symbol_out = self.vocal_map[win_idx]
                
                self.vocal_buffer.append(symbol_out)
                if len(self.vocal_buffer) > 4:
                    self.vocal_buffer.pop(0)
                    
            # 計時器推進
            active_chunk['timer'] += 1
            if active_chunk['timer'] >= self.seq_len:
                self.intent_buffer.pop(0) # 該序列執行完畢
                
        return symbol_out


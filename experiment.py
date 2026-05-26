import torch
import matplotlib.pyplot as plt
import numpy as np
from main import EmbodiedAGI_OS

def run_experiment():
    print("啟動 Phase 4: AGI OS 全局預訓練與 OOD 測試...")
    os_sys = EmbodiedAGI_OS()
    
    total_episodes = 800
    ood_episode = 600 # 在第 600 回合觸發 OOD 重力異常
    
    survival_history = []
    
    # 為了加速訓練，關閉 rendering
    os_sys.env.close()
    
    # 狀態變數
    V_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    pre_traces = torch.zeros(1, os_sys.cerebellum.in_features)
    post_traces = torch.zeros(1, os_sys.cerebellum.out_features)
    
    for ep in range(total_episodes):
        if ep == ood_episode:
            print(f"\n[系統中斷] Episode {ep}: 觸發 OOD 重力異常 (Gravity x 2.5)!")
            os_sys.env.apply_ood_gravity(multiplier=2.5)
            # 切換系統狀態為 CEN，允許較大的突觸改變
            os_sys.amygdala_stdp.A_plus = 0.1 # 提高學習率以應對危機
            os_sys.amygdala_stdp.A_minus = 0.12
            
        state, _ = os_sys.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        steps = 0
        while True:
            # 1. 腦部計算
            pre_spikes = os_sys.thalamus_sensory(state_tensor)
            post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
            action = os_sys.cerebrum_motor(post_spikes).item()
            
            # 2. 環境互動
            next_state, reward, terminated, truncated, _ = os_sys.env.step(action)
            
            # 3. 三因子 STDP 學習 (獎勵驅動)
            # 存活 +0.1, 死亡 -1.0
            instant_surprise = -1.0 if terminated else 0.1
            prediction_error = os_sys.hypothalamus.compute_prediction_error(instant_surprise)
            
            pre_traces = pre_traces * 0.9 + pre_spikes
            post_traces = post_traces * 0.9 + post_spikes
            
            # 只有在突觸處於活躍狀態時才更新權重 (節省算力)
            os_sys.amygdala_stdp.update(os_sys.cerebellum.W, pre_traces[0], post_traces[0], pre_spikes[0], post_spikes[0], prediction_error)
            
            # 推進狀態
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            V_prev, I_syn_prev = V_next.detach(), I_syn_next.detach()
            steps += 1
            
            if terminated or truncated:
                survival_history.append(steps)
                # 每 50 回合輸出一次進度
                if (ep + 1) % 50 == 0:
                    avg_survival = np.mean(survival_history[-50:])
                    print(f"Episode {ep+1:3d} | Avg Survival: {avg_survival:5.1f} steps")
                
                # 重置膜電位與跡線 (進入 DMN 離線清除狀態)
                V_prev.zero_()
                I_syn_prev.zero_()
                pre_traces.zero_()
                post_traces.zero_()
                break

    # 繪製結果
    plt.figure(figsize=(10, 6))
    
    # 計算移動平均
    window = 20
    if len(survival_history) >= window:
        moving_avg = np.convolve(survival_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(survival_history)), moving_avg, color='b', label='Moving Average (20 ep)')
    
    plt.scatter(range(len(survival_history)), survival_history, color='cyan', alpha=0.3, s=10, label='Raw Survival Steps')
    
    # 標示 OOD 事件
    plt.axvline(x=ood_episode, color='r', linestyle='--', label='OOD Event (Gravity Change)')
    
    plt.title('Embodied AGI OS - Predictive STDP Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Survival Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('experiment_results.png')
    print("實驗結束！圖表已儲存至 experiment_results.png")

if __name__ == "__main__":
    run_experiment()

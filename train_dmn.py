import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from main import EmbodiedAGI_OS

def run_dmn_training():
    print("啟動 Phase 5: 海馬迴 DMN 離線回放預訓練 (Hippocampus Offline Replay)...")
    os_sys = EmbodiedAGI_OS()
    os_sys.env.close()
    
    # 關閉局部的 STDP，因為我們要進行 DMN 全局睡眠編譯
    # 這裡我們使用 Adam 作為「睡眠時的全局優化器」
    optimizer = optim.Adam([
        {'params': os_sys.cerebellum.W, 'lr': 0.05},
        {'params': os_sys.cerebellum.tau, 'lr': 0.01},
        {'params': os_sys.cerebellum.A, 'lr': 0.01}
    ])
    
    total_episodes = 500
    batch_size = 128
    survival_history = []
    
    V_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    
    for ep in range(total_episodes):
        state, _ = os_sys.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        V_prev.zero_()
        I_syn_prev.zero_()
        steps = 0
        
        # --- 白天：潛意識採集記憶 (Subconscious Data Gathering) ---
        # 不進行 STDP 更新，純粹收集物理互動經驗 (Exploration)
        # 由於神經元具有 Surrogate Gradient，我們需要用 no_grad() 進行探索以節省算力
        while True:
            with torch.no_grad():
                pre_spikes = os_sys.thalamus_sensory(state_tensor)
                post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
                action = os_sys.cerebrum_motor(post_spikes).item()
                
            next_state, reward, terminated, truncated, _ = os_sys.env.step(action)
            
            # 存入海馬迴 (Hippocampus Differentiable Neural Dictionary)
            os_sys.hippocampus.store(state, action, reward, next_state, terminated)
            
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            V_prev, I_syn_prev = V_next, I_syn_next
            steps += 1
            
            if terminated or truncated:
                survival_history.append(steps)
                break
                
        # --- 夜晚：DMN 離線回放與全局優化 (DMN Offline Replay) ---
        # 當記憶池足夠時，進入深度睡眠進行全局 BPTT 優化
        if len(os_sys.hippocampus) >= batch_size:
            # 睡眠 5 個回放週期 (epochs)
            for _ in range(5):
                batch = os_sys.hippocampus.replay(batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_terms = batch
                
                # 將狀態重新通過丘腦編碼
                b_pre_spikes = os_sys.thalamus_sensory(b_states)
                
                b_V_prev = torch.zeros(batch_size, os_sys.cerebellum.out_features)
                b_I_syn_prev = torch.zeros(batch_size, os_sys.cerebellum.out_features)
                
                b_post_spikes, _, _ = os_sys.cerebellum(b_pre_spikes, b_V_prev, b_I_syn_prev)
                
                # Motor cortex decoding -> logits
                logits = os_sys.cerebrum_motor(b_post_spikes, return_logits=True) # shape (batch, 2)
                
                # DMN 的目標：從雜亂的經驗中找到讓生存時間延長的「規律 (Rule)」
                # CartPole 規則：當角度 (b_states[:, 2]) > 0，推車應該向右 (action 1)
                angles = b_states[:, 2]
                target_actions = (angles > 0).long()
                
                # 計算 CrossEntropy Loss
                loss = nn.CrossEntropyLoss()(logits, target_actions)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if (ep + 1) % 50 == 0:
            avg_survival = np.mean(survival_history[-50:])
            print(f"Episode {ep+1:3d} | Avg Survival: {avg_survival:5.1f} steps | Replay Buffer: {len(os_sys.hippocampus)} | DMN Loss: {loss.item() if len(os_sys.hippocampus) >= batch_size else 'N/A'}")

    # 繪製 DMN 學習曲線
    plt.figure(figsize=(10, 6))
    window = 20
    if len(survival_history) >= window:
        moving_avg = np.convolve(survival_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(survival_history)), moving_avg, color='orange', label='DMN Global Compilation (Moving Avg)')
    
    plt.scatter(range(len(survival_history)), survival_history, color='magenta', alpha=0.3, s=10, label='Raw Survival Steps')
    
    plt.title('Embodied AGI OS - Phase 5: Hippocampal DMN Replay Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Survival Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('dmn_results.png')
    print("DMN 回放預訓練完成！圖表已儲存至 dmn_results.png")

if __name__ == "__main__":
    run_dmn_training()

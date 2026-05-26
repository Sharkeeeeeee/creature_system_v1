import torch
import numpy as np
import matplotlib.pyplot as plt
from main import EmbodiedAGI_OS
from environment.inverted_pendulum import OODInvertedPendulum

def run_visualization():
    print("啟動具身意識 AI 可視化大屏...")
    # 初始化 OS，覆寫環境以支援影像渲染
    os = EmbodiedAGI_OS()
    os.env.close()
    os.env = OODInvertedPendulum(render_mode="rgb_array")
    
    plt.ion() # 開啟互動模式
    # 建立三個子圖：物理世界、小腦脈衝、膜電位動態
    fig, (ax_env, ax_spikes, ax_V) = plt.subplots(1, 3, figsize=(16, 5))
    fig.canvas.manager.set_window_title("Embodied AGI OS - Brain Dynamics")
    
    state, _ = os.env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    V_prev = torch.zeros(1, os.cerebellum.out_features)
    I_syn_prev = torch.zeros(1, os.cerebellum.out_features)
    pre_traces = torch.zeros(1, os.cerebellum.in_features)
    post_traces = torch.zeros(1, os.cerebellum.out_features)
    
    # 初始渲染
    img = ax_env.imshow(os.env.env.render())
    ax_env.set_title("Physical World (CartPole)")
    ax_env.axis('off')
    
    spike_history = []
    V_history = []
    
    max_steps = 200
    for step in range(max_steps):
        # --- 運算邏輯 ---
        pre_spikes = os.thalamus_sensory(state_tensor)
        post_spikes, V_next, I_syn_next = os.cerebellum(pre_spikes, V_prev, I_syn_prev)
        action = os.cerebrum_motor(post_spikes).item()
        next_state, reward, terminated, truncated, _ = os.env.step(action)
        
        instant_surprise = -1.0 if terminated else 0.1
        prediction_error = os.hypothalamus.compute_prediction_error(instant_surprise)
        
        pre_traces = pre_traces * 0.9 + pre_spikes
        post_traces = post_traces * 0.9 + post_spikes
        os.amygdala_stdp.update(os.cerebellum.W, pre_traces[0], post_traces[0], pre_spikes[0], post_spikes[0], prediction_error)
        
        # --- 儲存可視化資料 ---
        spike_history.append(post_spikes[0].detach().numpy())
        V_history.append(V_next[0].detach().numpy())
        
        if len(spike_history) > 50:
            spike_history.pop(0)
            V_history.pop(0)
            
        # --- 更新畫面 ---
        # 1. 物理世界
        img.set_data(os.env.env.render())
        
        # 2. 小腦脈衝矩陣 (Raster Plot)
        ax_spikes.clear()
        ax_spikes.imshow(np.array(spike_history).T, aspect='auto', cmap='binary', origin='lower')
        ax_spikes.set_title("Cerebellum Spikes (Action Decoding)")
        ax_spikes.set_ylabel("Motor Neuron ID")
        ax_spikes.set_xlabel("Time (last 50 steps)")
        
        # 3. 液態膜電位 (Liquid Membrane Potential)
        ax_V.clear()
        ax_V.plot(np.array(V_history), alpha=0.5)
        ax_V.set_title("Liquid Membrane Potential Dynamics")
        ax_V.set_ylabel("Voltage")
        ax_V.set_xlabel("Time")
        
        plt.tight_layout()
        plt.pause(0.05) # 暫停以更新畫面
        
        # --- 狀態更新 ---
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        V_prev, I_syn_prev = V_next.detach(), I_syn_next.detach()
        
        if terminated or truncated:
            state, _ = os.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visualization()

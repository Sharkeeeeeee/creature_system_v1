import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import EmbodiedAGI_OS
from environment.inverted_pendulum import OODInvertedPendulum

def run_3d_mpl():
    print("啟動 3D Matplotlib 具身意識視覺化大屏...")
    os_sys = EmbodiedAGI_OS()
    os_sys.env.close()
    os_sys.env = OODInvertedPendulum()
    
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Embodied AGI OS - 3D Neural Space")
    ax = fig.add_subplot(111, projection='3d')
    
    # 預先計算小腦 32 顆神經元的 3D 空間座標
    grid_w = 8
    num_neurons = 32
    nx, ny, nz = [], [], []
    for i in range(num_neurons):
        nx.append((i % grid_w) - grid_w/2)
        ny.append((i // grid_w) + 2)
        nz.append(3) # 懸浮在 3D 空間上方
        
    state, _ = os_sys.env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    V_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    pre_traces = torch.zeros(1, os_sys.cerebellum.in_features)
    post_traces = torch.zeros(1, os_sys.cerebellum.out_features)
    
    for step in range(500):
        # 1. 腦部計算
        pre_spikes = os_sys.thalamus_sensory(state_tensor)
        post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
        action = os_sys.cerebrum_motor(post_spikes).item()
        
        # 2. 環境互動
        next_state, reward, terminated, truncated, _ = os_sys.env.step(action)
        
        # 3. 神經調製與學習
        instant_surprise = -1.0 if terminated else 0.1
        prediction_error = os_sys.hypothalamus.compute_prediction_error(instant_surprise)
        pre_traces = pre_traces * 0.9 + pre_spikes
        post_traces = post_traces * 0.9 + post_spikes
        os_sys.amygdala_stdp.update(os_sys.cerebellum.W, pre_traces[0], post_traces[0], pre_spikes[0], post_spikes[0], prediction_error)
        
        # 4. 3D 渲染更新
        ax.cla() # 清除上一影格
        
        # (A) 繪製物理世界 (CartPole)
        cart_x = next_state[0]
        theta = next_state[2]
        pole_len = 2.0
        pole_x = cart_x + pole_len * math.sin(theta)
        pole_z = pole_len * math.cos(theta)
        
        ax.plot([-5, 5], [0, 0], [0, 0], color='gray', linewidth=2) # 軌道
        ax.scatter([cart_x], [0], [0], color='blue', s=200, marker='s') # 推車
        ax.plot([cart_x, pole_x], [0, 0], [0, pole_z], color='red', linewidth=4) # 擺桿
        
        # (B) 繪製立體神經網路 (3D SNN)
        v_np = V_next[0].detach().numpy()
        spikes_np = post_spikes[0].detach().numpy()
        
        colors = []
        sizes = []
        for i in range(num_neurons):
            if spikes_np[i] > 0.5:
                colors.append('cyan') # 發射脈衝閃耀青色
                sizes.append(200)
            else:
                # 靜息時，電壓越高顏色越亮
                v_norm = max(0.1, min(1.0, (v_np[i] + 1) / 3.0))
                colors.append((v_norm, 0, 1-v_norm)) # 漸變紫/紅
                sizes.append(50)
                
        ax.scatter(nx, ny, nz, c=colors, s=sizes, alpha=0.9, depthshade=True)
        
        # (C) 攝影機視角與 UI
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 5])
        ax.set_zlim([0, 5])
        ax.set_title(f"Macro OS State: {'CEN (High Entropy Override)' if terminated else 'SUBCONSCIOUS (Low Entropy)'}\nTime Step: {step}", fontsize=12, color='white' if not terminated else 'red')
        
        # 美化 3D 空間背景
        ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.axis('off')
        
        plt.pause(0.001) # 極小延遲以確保最高更新率
        
        # 5. 狀態推進
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        V_prev, I_syn_prev = V_next.detach(), I_syn_next.detach()
        
        if terminated or truncated:
            state, _ = os_sys.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_3d_mpl()

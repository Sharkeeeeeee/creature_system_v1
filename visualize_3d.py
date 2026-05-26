from vpython import *
import torch
import math
import numpy as np
from main import EmbodiedAGI_OS

def run_3d_visualization():
    print("啟動 3D 立體具身意識可視化系統 (VPython)...")
    
    # 1. 設置 3D 畫布
    scene = canvas(title='Embodied AGI OS - 3D Neural & Physical Dynamics',
                   width=1000, height=600, background=color.gray(0.1))
    
    # 2. 建立物理環境 (CartPole 3D 實體)
    track = box(pos=vector(0, -1, 0), size=vector(10, 0.1, 0.5), color=color.white)
    cart = box(pos=vector(0, -0.5, 0), size=vector(1, 0.5, 0.5), color=color.blue)
    pole = cylinder(pos=cart.pos, axis=vector(0, 2, 0), radius=0.1, color=color.red)
    
    # 3. 建立大腦 (SNN) 3D 實體
    # Cerebellum 有 32 個神經元
    neurons = []
    num_neurons = 32
    grid_w = 8
    start_x = -3.5
    start_y = 3.0
    spacing = 1.0
    
    for i in range(num_neurons):
        nx = start_x + (i % grid_w) * spacing
        ny = start_y + (i // grid_w) * spacing
        nz = -2 # 放在背景
        n_sphere = sphere(pos=vector(nx, ny, nz), radius=0.3, color=color.black, opacity=0.5)
        neurons.append(n_sphere)
        
    brain_label = label(pos=vector(0, 7, -2), text='Cerebellum LSN (Liquid Spiking Network)', height=20, box=False)
    state_label = label(pos=vector(0, -3, 0), text='System State: SUBCONSCIOUS', height=16, box=False, color=color.yellow)
    
    # 初始化 OS
    os_sys = EmbodiedAGI_OS()
    state, _ = os_sys.env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    V_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features)
    pre_traces = torch.zeros(1, os_sys.cerebellum.in_features)
    post_traces = torch.zeros(1, os_sys.cerebellum.out_features)
    
    # 模擬迴圈
    for step in range(1000):
        rate(60) # 鎖定 60 FPS，確保畫面極度流暢
        
        # OS 計算
        pre_spikes = os_sys.thalamus_sensory(state_tensor)
        post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
        action = os_sys.cerebrum_motor(post_spikes).item()
        next_state, reward, terminated, truncated, _ = os_sys.env.step(action)
        
        # STDP 更新
        instant_surprise = -1.0 if terminated else 0.1
        prediction_error = os_sys.hypothalamus.compute_prediction_error(instant_surprise)
        pre_traces = pre_traces * 0.9 + pre_spikes
        post_traces = post_traces * 0.9 + post_spikes
        os_sys.amygdala_stdp.update(os_sys.cerebellum.W, pre_traces[0], post_traces[0], pre_spikes[0], post_spikes[0], prediction_error)
        
        # --- 更新 3D 視覺化 ---
        
        # 1. 物理世界更新
        x = next_state[0]
        theta = next_state[2]
        
        cart.pos.x = x
        pole.pos = cart.pos
        # 旋轉 pole，gymnasium 中 theta = 0 是垂直向上，正值代表向右傾斜
        px = 2 * math.sin(theta)
        py = 2 * math.cos(theta)
        pole.axis = vector(px, py, 0)
        
        # 2. 大腦神經元更新
        v_np = V_next[0].detach().numpy()
        spikes_np = post_spikes[0].detach().numpy()
        
        for i in range(num_neurons):
            v_norm = max(0.1, min(1.0, (v_np[i] + 1) / 3.0)) 
            
            if spikes_np[i] > 0.5:
                # 發射脈衝時：閃耀明亮的青色並變大
                neurons[i].color = color.cyan
                neurons[i].opacity = 1.0
                neurons[i].radius = 0.45
            else:
                # 靜息時：根據膜電位改變顏色 (紫到紅)
                neurons[i].color = vector(v_norm, 0, 1-v_norm)
                neurons[i].opacity = 0.5 + 0.5*v_norm
                neurons[i].radius = 0.3
                
        # 3. 系統狀態更新
        if terminated:
            state_label.text = 'System State: CEN (Surprise Spike! Rewriting Rules...)'
            state_label.color = color.red
        else:
            state_label.text = 'System State: SUBCONSCIOUS (Local STDP active)'
            state_label.color = color.yellow
                
        # 狀態準備下一步
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        V_prev, I_syn_prev = V_next.detach(), I_syn_next.detach()
        
        if terminated or truncated:
            state, _ = os_sys.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    run_3d_visualization()

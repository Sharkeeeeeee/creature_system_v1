import torch
from ursina import *
import math
import numpy as np
from main import EmbodiedAGI_OS

def run_ursina():
    print("啟動真·3D 遊戲引擎可視化 (Ursina)...")
    app = Ursina(title="Embodied AGI OS - True 3D Visualization")
    window.always_on_top = True # 強制置頂，確保您一定看得到！
    
    # --- 1. 物理環境 3D 實體 ---
    # 軌道
    track = Entity(model='cube', scale=(12, 0.2, 0.5), color=color.gray, y=-1)
    # 推車
    cart = Entity(model='cube', scale=(1, 0.5, 0.6), color=color.azure, y=-0.5)
    # 擺桿 (設定 origin_y=-0.5 讓旋轉軸心在底部)
    pole = Entity(model='cube', scale=(0.1, 2.5, 0.1), color=color.red, origin_y=-0.5, y=0.25)
    pole.parent = cart # 綁定於推車
    
    # 幽靈推車 (Ghost Cart - 顯示 PFC 預測的未來狀態)
    ghost_cart = Entity(model='cube', scale=(1, 0.5, 0.6), color=color.rgba(0, 255, 255, 120), y=-0.5)
    ghost_pole = Entity(model='cube', scale=(0.1, 2.5, 0.1), color=color.rgba(255, 0, 0, 120), origin_y=-0.5, y=0.25)
    ghost_pole.parent = ghost_cart
    
    # --- 2. 大腦 (SNN) 3D 實體 ---
    neurons = []
    num_neurons = 32
    grid_w = 8
    start_x = -3.5
    start_y = 2.5
    spacing = 1.0
    
    for i in range(num_neurons):
        nx = start_x + (i % grid_w) * spacing
        ny = start_y + (i // grid_w) * spacing
        nz = 3 # 懸浮在背景中
        sphere = Entity(model='sphere', scale=0.5, color=color.magenta, position=(nx, ny, nz))
        neurons.append(sphere)
        
    # UI 與攝影機
    text_state = Text(text="OS State: SUBCONSCIOUS", position=(-0.4, 0.4), color=color.yellow, scale=1.5)
    text_energy = Text(text="Energy: 100.0", position=(-0.4, 0.35), color=color.green, scale=1.5)
    text_ood = Text(text="[OOD EVENT] 3G GRAVITY!", position=(0.2, 0.4), color=color.red, scale=1.5)
    text_ood.visible = False
    
    camera.position = (0, 2, -12)
    camera.rotation_x = 5
    
    # --- 3. 初始化 AGI OS ---
    os_sys = EmbodiedAGI_OS()
    import os
    has_pfc = False
    if os.path.exists("pfc_world_model.pth"):
        os_sys.cerebrum.pfc.load_state_dict(torch.load("pfc_world_model.pth", weights_only=True))
        has_pfc = True
        print("已載入 PFC 世界模型，開啟腦內思維模擬與幽靈推車。")
        
    state, _ = os_sys.brainstem.reset_life()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # 將張量綁定到全域變數中，供 update() 函式使用
    global_state = {
        'state_tensor': state_tensor,
        'V_prev': torch.zeros(1, os_sys.cerebellum.out_features),
        'I_syn_prev': torch.zeros(1, os_sys.cerebellum.out_features),
        'pre_traces': torch.zeros(1, os_sys.cerebellum.in_features),
        'post_traces': torch.zeros(1, os_sys.cerebellum.out_features)
    }
    
    import time
    global_state['start_time'] = time.time()
    global_state['frame_count'] = 0
    
    with open("debug_log.txt", "w") as f:
        f.write("=== STARTING VISUALIZER ===\n")
        
    # Phase 9: 初始化遙測資料
    with open("brain_telemetry.csv", "w") as f:
        f.write("frame,left_activity,right_activity,curiosity,energy\n")
        
    def get_synthetic_retina(cart_x, theta, pellet_x):
        # Phase 7: 建立 64x64 灰階多模態視覺 (Synthetic Retina)
        img = torch.zeros(1, 1, 64, 64)
        c_px = max(0, min(63, int((cart_x + 2.4) / 4.8 * 63)))
        p_px = max(0, min(63, int((pellet_x + 2.4) / 4.8 * 63)))
        # 畫推車
        img[0, 0, 50:55, max(0, c_px-5):min(64, c_px+5)] = 0.8
        # 畫能量包
        img[0, 0, 48:52, max(0, p_px-2):min(64, p_px+2)] = 1.0
        # 畫擺桿頂點
        end_x = max(0, min(63, c_px + int(math.sin(theta) * 20)))
        end_y = max(0, min(63, 50 - int(math.cos(theta) * 20)))
        img[0, 0, max(0, end_y-2):min(64, end_y+2), max(0, end_x-2):min(64, end_x+2)] = 0.6
        return img
        
    # 在畫面上新增能量包實體 (Pellet)
    pellet_entity = Entity(model='sphere', scale=(0.4, 0.4, 0.4), color=color.green, y=-0.5)
        
    def update():
        try:
            # 讀取當前狀態
            st = global_state['state_tensor']
            V_prev = global_state['V_prev']
            I_syn_prev = global_state['I_syn_prev']
            pre_traces = global_state['pre_traces']
            post_traces = global_state['post_traces']
            
            # --- Phase 7/8: 左右腦視覺與胼胝體融合 ---
            pellet_x = os_sys.brainstem.env.pellet_x
            pellet_entity.x = pellet_x
            
            # 產生 64x64 視覺
            retina = get_synthetic_retina(st[0][0].item(), st[0][2].item(), pellet_x)
            
            # 右腦 (直覺空間) 產生視覺脈衝
            right_spikes, right_features = os_sys.cerebrum.right_brain_vision(retina)
            
            # 左腦 (精確邏輯) 產生數字脈衝
            left_spikes = os_sys.diencephalon.thalamus.encode(st)
            
            # 胼胝體融合 (Corpus Callosum): 串接左腦 40 個脈衝與右腦 16 個脈衝
            pre_spikes = torch.cat([left_spikes, right_spikes], dim=1)
            
            # 傳遞給小腦
            post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
            action = os_sys.cerebrum.motor_cortex(post_spikes).item()
            if global_state['frame_count'] % 60 == 0:
                with open("debug_log.txt", "a") as f:
                    f.write(f"Frame {global_state['frame_count']} | Cart X: {st[0][0]:.2f} | Theta: {st[0][2]:.2f}\n")
            
            action = os_sys.cerebrum.motor_cortex(post_spikes).item()
            
            # --- Phase 6: 前額葉世界模型 (Mental Simulation & Brake) ---
            if has_pfc:
                action_tensor = torch.tensor([[action]], dtype=torch.float32)
                # 預測未來一幀 (LNN 腦內模擬)
                pred_next_state, _ = os_sys.cerebrum.pfc(st, action_tensor)
                pred_theta = pred_next_state[0, 2].item()
                pred_x = pred_next_state[0, 0].item()
                
                # 渲染幽靈推車
                ghost_cart.x = pred_x
                ghost_pole.rotation_z = math.degrees(pred_theta)
                
                # 計算好奇心多巴胺
                curiosity = os_sys.diencephalon.hypothalamus.compute_curiosity_reward(torch.tensor([pred_theta]))
                
                # --- Phase 8: 左右腦決策衝突 (Right Brain Priority) ---
                if curiosity > 0.8 and os_sys.brainstem.energy_level > 50:
                    import random
                    if random.random() < 0.3:
                        action = 1 - action # 右腦貪玩，故意搞怪
                        text_state.text = "RIGHT BRAIN PLAY MODE (CURIOSITY HIGH)"
                        text_state.color = color.yellow
                elif abs(pred_theta) > 0.18: # 若預測即將跌倒
                    action = 1 - action # 左腦強制踩煞車保命
                    text_state.text = "LEFT BRAIN INHIBITORY BRAKE!"
                    text_state.color = color.cyan
            else:
                ghost_cart.visible = False
            
            # 實際執行
            next_state, reward, terminated, truncated, info = os_sys.brainstem.step_environment(action)
            spikes_np = post_spikes[0].detach().numpy()
            
            # --- 生死能量系統與覓食 ---
            current_energy, is_dead = os_sys.brainstem.metabolize(spikes_np.sum(), 0.1 if not terminated else 0.0)
            if info.get('pellet_eaten', False):
                current_energy = min(os_sys.brainstem.max_energy, current_energy + 50.0)
                os_sys.brainstem.energy_level = current_energy
                text_state.text = "PELLET EATEN! +50 ENERGY"
                text_state.color = color.green
            
            # --- 局部神經學習 (Local STDP) ---
            instant_surprise = -1.0 if terminated else 0.1
            prediction_error = os_sys.diencephalon.hypothalamus.compute_rpe(instant_surprise)
            pre_traces = pre_traces * 0.9 + pre_spikes
            post_traces = post_traces * 0.9 + post_spikes
            os_sys.cerebellum.stdp_learner.update(os_sys.cerebellum.W, pre_traces[0], post_traces[0], pre_spikes[0], post_spikes[0], prediction_error)
            # --- Phase 9: Telemetry Logging ---
            left_act = left_spikes[0].float().mean().item()
            right_act = right_spikes[0].float().mean().item()
            curiosity_val = locals().get('curiosity', 0.0)
            with open("brain_telemetry.csv", "a") as f:
                f.write(f"{global_state['frame_count']},{left_act:.4f},{right_act:.4f},{curiosity_val:.4f},{current_energy:.2f}\n")
            global_state['frame_count'] += 1
            
            # --- 更新 3D 渲染 ---
            cart_x = next_state[0]
            theta = next_state[2]
            
            # 1. 物理推車
            cart.x = cart_x
            # Ursina Z軸旋轉，正值是順時針（與 gym 的 theta 方向一致）
            pole.rotation_z = math.degrees(theta)
            
            # 2. 神經元發光與縮放
            v_np = V_next[0].detach().numpy()
            
            for i in range(num_neurons):
                if spikes_np[i] > 0.5:
                    # 脈衝爆發！
                    neurons[i].color = color.cyan
                    neurons[i].scale = 0.9
                else:
                    # 靜息時顯示膜電位顏色
                    v_norm = max(0.1, min(1.0, (v_np[i] + 1) / 3.0))
                    neurons[i].color = color.rgb(v_norm, 0, 1-v_norm)
                    # 讓球體平滑縮小回原狀
                    neurons[i].scale = lerp(neurons[i].scale, Vec3(0.5, 0.5, 0.5), time.dt * 5)
                    
            # 3. UI 狀態
            text_energy.text = f"Energy: {current_energy:.1f}"
            # OOD Event 觸發
            if global_state['frame_count'] == 300:
                os_sys.brainstem.env.apply_ood_gravity(3.0)
                text_ood.visible = True
            
            if is_dead:
                # 破壞小腦權重 (Scrambling)
                os_sys.cerebellum.die_and_scramble()
                text_state.text = "OS State: DEATH (SCRAMBLED)"
                text_state.color = color.red
                
                global_state['dead_frames'] += 1
                if global_state['dead_frames'] > 30:
                    state, _ = os_sys.brainstem.reset_life()
                    global_state['dead_frames'] = 0
            else:
                text_state.text = "OS State: SUBCONSCIOUS (Local STDP active)"
                text_state.color = color.yellow
                
            # --- 推進狀態 ---
            global_state['state_tensor'] = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            global_state['V_prev'] = V_next.detach()
            global_state['I_syn_prev'] = I_syn_next.detach()
            global_state['pre_traces'] = pre_traces
            global_state['post_traces'] = post_traces
            
            # 重置回合
            if terminated:
                state, _ = os_sys.brainstem.reset_life()
                global_state['state_tensor'] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ERROR IN UPDATE LOOP:", e)
            updater.update = None # Stop updating to avoid spam

    # 將 update 函式掛載到一個隱藏的 Entity 上，確保 Ursina 每幀都會呼叫它
    updater = Entity()
    updater.update = update
    app.run()

if __name__ == "__main__":
    run_ursina()

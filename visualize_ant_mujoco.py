import os
# 強制系統只看見第一張 NVIDIA 顯示卡 (RTX 4060)，完全屏蔽 AMD Radeon 內顯或副卡
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import time
from main import EmbodiedAGI_OS
from brain_regions.brainstem import CPGSpinalCord
from brain_regions.global_workspace import GlobalWorkspace, SpikeToConceptDecoder
from brain_regions.cerebrum import WernickeAreaSNN, BrocaIntentSNN, BrocaMotorChunkingSNN, MirrorNeuronSystemSNN

def run_mujoco_ant():
    print("Initializing EmbodiedAGI OS with MuJoCo Ant-v4...")
    os_sys = EmbodiedAGI_OS(use_mujoco=True, render_mode="human")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] PyTorch using device: {device}")
    
    # 遷移模型到 GPU
    os_sys.cerebellum.to(device)
    os_sys.cerebrum.right_brain_vision.to(device)
    os_sys.cerebrum.pfc.to(device)
    
    # 載入前額葉世界模型權重 (LNN)
    if os.path.exists("pfc_world_model.pth"):
        try:
            os_sys.cerebrum.pfc.load_state_dict(torch.load("pfc_world_model.pth", map_location=device, weights_only=True))
            print("[PFC] 成功載入前額葉世界模型權重。", flush=True)
        except Exception as e:
            print(f"[PFC] 載入世界模型失敗: {e}", flush=True)
            

            
    # 啟動原生脈衝語言中樞 (Wernicke, MNS & Hierarchical Broca)
    wernicke_snn = WernickeAreaSNN(num_symbols=4, out_neurons=16).to(device)
    mns_snn = MirrorNeuronSystemSNN(sensory_neurons=16, intent_neurons=4).to(device)
    broca_intent_snn = BrocaIntentSNN(in_features=6, intent_neurons=4).to(device)
    broca_chunking_snn = BrocaMotorChunkingSNN(intent_neurons=4, num_symbols=4, seq_len=2).to(device)
    print("[NativeLanguage] 階層式語言中樞 (MNS & Hierarchical Broca) 已啟動。")
    
    # 初始化全域工作空間與脈衝接地解碼器
    state_dim = os_sys.brainstem.env.observation_space.shape[0] + 2
    global_workspace = GlobalWorkspace(vision_dim=16, sensory_dim=state_dim, motor_dim=32, limbic_dim=6, latent_dim=32).to(device)
    
    # 動態計算脈衝維度
    left_spikes_dim = state_dim * 2  # neurons_per_state is 2
    right_spikes_dim = 16
    out_features = os_sys.cerebellum.out_features  # 32
    total_spikes_dim = left_spikes_dim + right_spikes_dim + out_features  # 106
    
    spike_decoder = SpikeToConceptDecoder(spike_dim=total_spikes_dim, concept_dim=6).to(device)
    if os.path.exists("spike_decoder.pth"):
        try:
            spike_decoder.load_state_dict(torch.load("spike_decoder.pth", map_location=device, weights_only=True))
            print("[Decoder] 成功載入符號接地解碼器權重。", flush=True)
        except Exception as e:
            print(f"[Decoder] 載入符號接地解碼器失敗: {e}", flush=True)
    
    decoder_optimizer = torch.optim.Adam(spike_decoder.parameters(), lr=0.005)
    decoder_criterion = torch.nn.MSELoss()
    
    # PFC 在線訓練器
    pfc_optimizer = torch.optim.Adam(os_sys.cerebrum.pfc.parameters(), lr=0.001)
    pfc_criterion = torch.nn.MSELoss()
    
    # 啟動脊髓 CPG (產生天生平滑步態)
    cpg = CPGSpinalCord(freq=0.08, amplitude=0.6)
    
    # 場景美化：Cyberpunk 霓虹色系 (直接修改 MuJoCo 模型材質)
    try:
        model = os_sys.brainstem.env.unwrapped.model
        if hasattr(model, 'mat_rgba') and hasattr(model, 'geom_rgba'):
            # 地板：深色網格
            if len(model.mat_rgba) > 0:
                model.mat_rgba[0] = [0.05, 0.05, 0.1, 1.0] 
            # 螞蟻身體與關節：霓虹藍與亮粉紅
            for i in range(1, model.ngeom):
                if i % 2 == 0:
                    model.geom_rgba[i] = [0.0, 1.0, 1.0, 1.0] # Neon Cyan
                else:
                    model.geom_rgba[i] = [1.0, 0.0, 1.0, 1.0] # Neon Pink
        print("[System] Cyberpunk 場景注入成功！")
    except Exception as e:
        print(f"[System] 場景注入失敗 (可能版本不相容): {e}")
    
    # Load muscle memory if exists
    os_sys.cerebellum.load_muscle_memory("muscle_memory.pt")
    
    # 初始化 telemetry 檔案，寫入包含意志力、煞車與新化學物質的新標頭
    with open("brain_telemetry.csv", "w", encoding="utf-8") as f:
        f.write("frame,left_activity,right_activity,curiosity,energy,willpower,veto,oxytocin,endorphins,acetylcholine\n")
        
    # 實體真實食物點設在 (4.0, 4.0) (對角線探索熱點)
    food_pos = np.array([4.0, 4.0])
    wandering_target = np.array([1.0, 1.0])
    target_pos = wandering_target
    
    def get_state_with_target(raw_state, target_p, ant_p):
        dx = target_p[0] - ant_p[0]
        dy = target_p[1] - ant_p[1]
        dist = np.linalg.norm([dx, dy])
        target_vec = np.array([dx / (dist + 1e-5), dy / (dist + 1e-5)])
        return np.concatenate([raw_state, target_vec])

    # Init spikes and states
    state, info = os_sys.brainstem.reset_life()
    ant_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
    
    # 若有海馬迴記憶，初始化時即可提取
    retrieved_pos = os_sys.cerebrum.hippocampus.retrieve_best_episode()
    if retrieved_pos is not None:
        target_pos = retrieved_pos
        
    state = get_state_with_target(state, target_pos, ant_pos)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    V_prev = torch.zeros(1, os_sys.cerebellum.out_features).to(device)
    I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features).to(device)
    # 資格跡已經內建在 Cerebellum 內部
    
    frame_count = 0
    generation = 1
    best_frames = 0
    frame_skip = 5
    current_action = np.zeros(8)
    accumulated_reward = 0.0
    terminated = False
    truncated = False
    is_dead = False
    is_sleeping = False
    current_energy = 100.0
    
    prev_dist = np.linalg.norm(target_pos - ant_pos)
    
    start_time = time.time()
    
    # 脈衝解碼感受與 GWT 調製因子初始化
    lr_mod_val = 1.0
    latest_symbol_out = None
    wta_winner = "None"
    grounded_concepts_dict = {
        "hunger": 0.0,
        "pain": 0.0,
        "curiosity": 0.0,
        "fatigue": 0.0,
        "sleepiness": 0.0,
        "instability": 0.0
    }
    
    # 紀錄上一幀的意圖脈衝供 MNS STDP 學習
    prev_broca_intent_spikes = None
    
    # PFC 世界模型時序記憶
    pfc_hx = None
    veto_signal = 0.0
    cognitive_cost = 0.0
    
    # Tutor Ant 環境發聲變數
    tutor_sequence = []
    tutor_timer = 0
    
    # 探索極限記錄、最後預測誤差與三因子激素狀態初始化
    max_dist_reached = 0.0
    last_pfc_loss = 0.0
    oxytocin = 0.5
    endorphins = 0.0
    acetylcholine = 0.5
    
    print("Entering Ant Simulation Loop...")
    while True:
        # 1. 結算上一輪決策的學習訊號 (RPE & STDP & Emotion)
        if frame_count % frame_skip == 0 and frame_count > 0:
            # 好奇心驅動的內在多巴胺反饋
            curiosity_reward = os_sys.diencephalon.hypothalamus.compute_curiosity_reward(torch.tensor([last_pfc_loss]))
            intrinsic_feedback = curiosity_reward * 2.0
            
            instant_surprise = -1.0 if (terminated or is_dead) else (accumulated_reward + intrinsic_feedback)
            prediction_error = os_sys.diencephalon.hypothalamus.compute_rpe(instant_surprise).to(device)
            
            # 邊緣系統：杏仁核處理情緒
            fear_level = os_sys.cerebrum.limbic.amygdala.process_emotion(current_energy, prediction_error.item(), is_dead)
            
            # 內分泌系統：血清素、去甲腎上腺素、催產素、腦內啡、乙醯膽鹼
            serotonin, noradrenaline, oxytocin, endorphins, acetylcholine = os_sys.cerebrum.limbic.endocrine.process_hormones(
                current_energy, 
                prediction_error.item(), 
                is_dead,
                has_tutor_sound=last_had_tutor_sound if 'last_had_tutor_sound' in locals() else False,
                danger_level=predicted_danger if 'predicted_danger' in locals() else 0.0,
                dist_from_origin=np.linalg.norm(ant_pos) if 'ant_pos' in locals() else 0.0,
                max_dist_reached=max_dist_reached
            )
            
            # 松果體：褪黑激素與生理時鐘
            melatonin = os_sys.pineal_gland.update(frame_skip)
            
            # 自律神經系統：交感神經與副交感神經
            social_safety = 1.0 if (len(tutor_sequence) > 0) else 0.0
            sympathetic, parasympathetic = os_sys.autonomic_system.update(fear_level, social_safety)
            
            # 網狀致動系統 (RAS)：清醒度控制
            arousal_level = os_sys.ras.update(fatigue=np.mean(cpg.lactic_acid), melatonin=melatonin, sudden_stimulus=fear_level)
            hyperpolarization_current = os_sys.ras.get_hyperpolarization_current()
            
            # 褪黑激素過高時，強迫進入睡眠狀態 (Circadian Rhythm)
            if melatonin > 0.9:
                print(f"[晝夜節律] 褪黑激素濃度過高 ({melatonin:.2f})，代理人抵擋不住睡意，陷入沉睡。", flush=True)
                is_sleeping = True # 進入安穩的睡眠，而非死亡
                
            # 原生語言中樞 (Broca & Wernicke) 連續運行 (無梯度推論)
            with torch.no_grad():
                limbic_state = torch.tensor([[
                    grounded_concepts_dict["hunger"],
                    grounded_concepts_dict["pain"],
                    grounded_concepts_dict["curiosity"],
                    grounded_concepts_dict["fatigue"],
                    grounded_concepts_dict["sleepiness"],
                    grounded_concepts_dict["instability"]
                ]], dtype=torch.float32).to(device)
    
                is_thalamic_gate_open = abs(prediction_error.item()) > 1.5
    
                # === 社交環境模擬：導師螞蟻 (Tutor Ant) 發聲 ===
                env_sound_idx = None
                if frame_count % 300 == 0 and frame_count > 0:
                    tutor_sequence = [0, 1] if np.random.rand() > 0.5 else [2, 2] 
                    tmsg = f"[環境聲音] 遠方傳來導師螞蟻的呼喚: {'-'.join([['A','B','C','D'][s] for s in tutor_sequence])}"
                    print(tmsg, flush=True)
                    with open("ant_log.txt", "a", encoding="utf-8") as f: f.write(tmsg + "\n")
                    
                if len(tutor_sequence) > 0:
                    env_sound_idx = tutor_sequence.pop(0)
    
                # Wernicke SNN 感覺解碼 (聆聽環境聲音，若無則為 None)
                wernicke_spikes = wernicke_snn(symbol_idx=env_sound_idx, global_rpe=prediction_error.item(), training=is_thalamic_gate_open)
    
                # 鏡像神經元系統 MNS：聽覺激發運動意圖 (受催產素調控)
                I_mns = mns_snn(wernicke_spikes, prev_broca_intent_spikes, training=is_thalamic_gate_open, oxytocin=oxytocin)
    
                # Broca 高階意圖 SNN：融合內部感受與 MNS 模仿衝動
                broca_intent_spikes = broca_intent_snn(limbic_state, I_mns, global_rpe=prediction_error.item(), training=is_thalamic_gate_open)
                prev_broca_intent_spikes = broca_intent_spikes.detach()
                
                # Broca 低階序列組塊化 Decoder：產生真實發聲
                symbol_out = broca_chunking_snn(broca_intent_spikes)
    
                if symbol_out is not None:
                    latest_symbol_out = symbol_out
    
                # 社交餵食模擬 (覓食呼喚)
                if len(broca_chunking_snn.vocal_buffer) >= 2 and broca_chunking_snn.vocal_buffer[-2:] == ["A", "B"]:
                    if current_energy < 80.0:
                        current_energy = min(100.0, current_energy + 30.0)
                        social_rpe = 5.0
                        feed_msg = f"[社交餵食] 螞蟻發出求食信號 'A-B'，人類/導師給予能量補給！能量 +30.0，多巴胺 (RPE) +5.0"
                        print(feed_msg, flush=True)
                        with open("ant_log.txt", "a", encoding="utf-8") as f: f.write(feed_msg + "\n")
                        # 強制利用 STDP 強化發聲與飢餓的連結
                        broca_intent_snn(limbic_state, global_rpe=social_rpe, training=True)
                        os_sys.cerebellum.apply_dopamine(torch.tensor([social_rpe]).to(device), learning_rate=0.01)
                        broca_chunking_snn.vocal_buffer.clear()

            # e-prop 資格跡學習與下橄欖核 (Inferior Olive) 抹除機制
            climbing_fiber_active, err_mag = os_sys.inferior_olive.generate_climbing_fiber_signal(prediction_error.item(), grounded_concepts_dict['pain'])
            
            if climbing_fiber_active:
                # 下橄欖核觸發攀緣纖維，進行強烈長期抑制 (LTD) 以抹除錯誤的運動記憶
                os_sys.cerebellum.apply_dopamine(torch.tensor([-1.0]).to(device), learning_rate=0.05 * err_mag)
                if frame_count % 100 == 0:
                    print(f"[下橄欖核] 觸發攀緣纖維脈衝！強制抹除錯誤突觸 (LTD)。", flush=True)
            elif is_thalamic_gate_open:
                # 正常多巴胺廣播學習 (受去甲腎上腺素與乙醯膽鹼注意力調控，乙醯膽鹼可放大學習率最高 3.5 倍)
                ach_factor = 0.5 + acetylcholine * 3.0
                dynamic_lr = (0.005 + noradrenaline * 0.02) * lr_mod_val * ach_factor
                os_sys.cerebellum.apply_dopamine(prediction_error, learning_rate=dynamic_lr)
                
                if frame_count % 100 == 0 or frame_count < 100:
                    print(f"[丘腦閘門] 觸發全局更新脈衝！ RPE: {prediction_error.item():.2f}，調製學習率: {dynamic_lr:.4f}", flush=True)
            
            # 儲存神經狀態到海馬迴供睡眠時作夢重播
            os_sys.cerebrum.hippocampus.store_eprop_experience(os_sys.cerebellum.eligibility_trace, prediction_error)
            if 'vision_tensor' in locals():
                os_sys.cerebrum.hippocampus.store_vision_experience(vision_tensor)
            
            # --- Telemetry Logging for Consciousness Monitor ---
            left_act = left_spikes[0].float().mean().item()
            right_act = right_spikes[0].float().mean().item()
            curiosity_val = abs(prediction_error.item()) 
            wp_val = os_sys.cerebrum.pfc.willpower_level
            with open("brain_telemetry.csv", "a", encoding="utf-8") as f:
                f.write(f"{frame_count},{left_act:.4f},{right_act:.4f},{curiosity_val:.4f},{current_energy:.2f},{wp_val:.4f},{veto_signal:.4f},{oxytocin:.4f},{endorphins:.4f},{acetylcholine:.4f}\n")
                
            accumulated_reward = 0.0

        # 2. 死亡判定與生死輪迴
        if (frame_count > 0 and (terminated or truncated or is_dead or is_sleeping)):
            was_dead = is_dead or terminated or truncated
            was_sleeping = is_sleeping

            if was_dead:
                log_msg = f"[生死輪迴] 代理人已死亡，本次存活幀數: {frame_count} | 剩餘能量: {current_energy:.2f}"
            else:
                log_msg = f"[晝夜節律] 代理人進入自然睡眠，本次清醒幀數: {frame_count}"
                
            print(log_msg, flush=True)
            with open("ant_log.txt", "a", encoding="utf-8") as f: f.write(log_msg + "\n")
            
            if frame_count > best_frames:
                best_frames = frame_count
                os_sys.cerebellum.save_muscle_memory("muscle_memory.pt")
                torch.save(os_sys.cerebrum.pfc.state_dict(), "pfc_world_model.pth")
                torch.save(spike_decoder.state_dict(), "spike_decoder.pth")
                print("[System] 成功保存前額葉世界模型與符號接地解碼器權重。", flush=True)
                
            # 睡眠與夢境固化 (Sleep & Dream Consolidation)
            if len(os_sys.cerebrum.hippocampus.stdp_memory) > 10:
                print("\n[睡眠模式] 進入深度睡眠，海馬迴開始重播記憶...", flush=True)
                
                # [夢境合成引擎] 抽出最後 4 張殘影
                recent_visions = os_sys.cerebrum.hippocampus.get_recent_visions(4)
                if recent_visions:
                     import matplotlib
                     matplotlib.use('Agg')
                     import matplotlib.pyplot as plt
                     fig, axes = plt.subplots(1, len(recent_visions), figsize=(12, 3))
                     if len(recent_visions) == 1: axes = [axes]
                     for idx, v_tensor in enumerate(recent_visions):
                         # 加入高斯雜訊與對比扭曲，模擬神經突觸殘像
                         noise = torch.randn_like(v_tensor) * 0.15
                         dream_frame = torch.clamp(v_tensor + noise, 0.0, 1.0)
                         
                         # 放大並模糊化 (利用 interpolate 模擬潛意識擴散)
                         dream_frame = torch.nn.functional.interpolate(dream_frame, size=(128, 128), mode='bicubic')[0]
                         
                         img_np = dream_frame[0].cpu().numpy()
                         axes[idx].imshow(img_np, cmap='magma')
                         axes[idx].axis('off')
                         axes[idx].set_title(f"REM T-{idx}", color='white')
                     
                     save_path = f"C:\\Users\\Steve\\.gemini\\antigravity-ide\\brain\\7c856f0f-7365-4690-ae2d-cbe699319701\\dream_gen_{generation}.png"
                     plt.savefig(save_path, bbox_inches='tight', facecolor='black')
                     plt.close()
                     print(f"[夢境] 殘影已合成並存檔至: {save_path}", flush=True)
                 
                # 暫停 1 秒讓小腦固化
                time.sleep(1.0)
                 
                # 離線突觸固化 (Offline Consolidation)
                for _ in range(50):
                     experiences = os_sys.cerebrum.hippocampus.sample_eprop_experience(10)
                     for exp in experiences:
                         # 睡眠時使用專屬平穩的固化學習率，避免白天恐慌情緒導致神經暴走
                         os_sys.cerebellum.apply_dopamine(rpe=exp[1], trace=exp[0], learning_rate=0.005)
                print("[睡眠模式] 醒來！突觸記憶已固化。\n", flush=True)
                 
            # 如果是死亡，重置實體環境與座標；若是睡眠，則保持實體連續性
            if was_dead:
                state, info = os_sys.brainstem.reset_life()
                os_sys.cerebellum.die_and_scramble()
                ant_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
                pfc_hx = None  # 死亡洗掉 LNN 記憶
             
            # 清除前世/睡前的肉體與神經殘影 (Bleeding States)與重置意志力
            cpg.reset()
            os_sys.cerebrum.left.parietal.reset()
            os_sys.cerebrum.occipital_lobe.reset()
            os_sys.cerebrum.pfc.reset_willpower()
             
            # 若是因為睡眠而重置，需要快轉生理時鐘到早上，並清除褪黑激素，否則會陷入無盡長眠鎖死
            if was_sleeping:
                 os_sys.pineal_gland.circadian_clock = 6.0
                 os_sys.pineal_gland.melatonin = 0.0
             
            frame_count = 0
            generation += 1 if was_dead else 0
            accumulated_reward = 0.0
                
            terminated, truncated, is_dead, is_sleeping = False, False, False, False
            current_energy = 100.0 if was_dead else current_energy
             
            if not was_dead:
                # 睡眠醒來，使用當前的狀態，不改變 ant_pos
                ant_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)]) if 'info' in locals() else ant_pos
            
            # 重置目標為可能已提取的情節記憶或初始探索點
            retrieved_pos = os_sys.cerebrum.hippocampus.retrieve_best_episode()
            if retrieved_pos is not None:
                target_pos = retrieved_pos
            else:
                target_pos = np.array([1.0, 1.0])
                
            prev_dist = np.linalg.norm(target_pos - ant_pos)
            
            # 確保狀態拼接了相對目標向量再轉換為張量
            state = get_state_with_target(state[:29], target_pos, ant_pos)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            V_prev = torch.zeros(1, os_sys.cerebellum.out_features).to(device)
            I_syn_prev = torch.zeros(1, os_sys.cerebellum.out_features).to(device)
            os_sys.cerebellum.eligibility_trace.zero_()
            current_action = np.zeros(8)
            continue # 直接重啟迴圈，保證新的生命從 frame 0 開始
            
        # 3. 連續感官採樣與大腦決策 (每個物理幀執行，保持 SNN 脈衝動力學連續)
        
        # [本體感覺與前庭覺] 頂葉過濾：不再直接讀取絕對座標，而是加入雜訊與傳遞延遲
        processed_state = os_sys.cerebrum.left.parietal.process_somatosensory(state)
        state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # [視覺擷取] 從物理引擎擷取視網膜 RGB 畫面
        try:
            rgb_array = os_sys.brainstem.env.unwrapped.mujoco_renderer.render("rgb_array")
        except:
            rgb_array = None
            
        # [枕葉處理] 轉換為 64x64 灰階張量 (事件驅動)
        vision_tensor, vision_changed = os_sys.cerebrum.occipital_lobe.process(rgb_array, device=device)
        
        # ================== 混合精度與無梯度推論區塊 (低耗能) ==================
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type), torch.no_grad():
            left_spikes = os_sys.diencephalon.thalamus.encode(state_tensor).to(device)
            
            # 若視覺發生變動才重新計算 CNN (使用雙流視覺腹側流輔助)
            if vision_changed or 'right_spikes' not in locals():
                extra_info = {
                    "dist_to_food": np.linalg.norm(ant_pos - food_pos) if 'ant_pos' in locals() else 999.0,
                    "dist_to_tutor": np.linalg.norm(ant_pos - np.array([-2.0, 2.0])) if 'ant_pos' in locals() else 999.0,
                    "is_unsafe": os_sys.brainstem.is_unsafe_state(state[:29]) if 'state' in locals() else False
                }
                right_spikes, _ = os_sys.cerebrum.right_brain_vision(vision_tensor, extra_info=extra_info)
            elif frame_count % 50 == 0:
                pass # 為了避免洗版，不印出跳過 CNN 的訊息
                
            pre_spikes = torch.cat([left_spikes, right_spikes], dim=1)
            
            # [神經網路運算] SNN 計算微調力矩
            post_spikes, V_next, I_syn_next = os_sys.cerebellum(pre_spikes, V_prev, I_syn_prev)
            
            # === 全域工作空間 (Global Workspace) 注意力廣播 ===
            # 1. 建立邊緣/情感狀態輸入
            cur_pe = prediction_error.item() if 'prediction_error' in locals() else 0.0
            cur_fear = fear_level if 'fear_level' in locals() else 0.0
            cur_se = serotonin if 'serotonin' in locals() else 1.0
            cur_na = noradrenaline if 'noradrenaline' in locals() else 0.0
            cur_me = melatonin if 'melatonin' in locals() else 0.0
            limbic_input = torch.tensor([[cur_pe, cur_fear, cur_se, cur_na, cur_me, float(is_dead)]], dtype=torch.float32).to(device)
            
            # 2. 獲取注意力廣播信號 (WTA 贏者全拿)
            gw_latent, salience, I_gw, lr_mod, wta_winner = global_workspace(
                right_spikes, state_tensor, post_spikes, limbic_input
            )
            
            # 3. 注入 GWT 廣播調製電流與 RAS 抑制電流到 SNN 的膜電位中
            V_next = V_next + I_gw + hyperpolarization_current if 'hyperpolarization_current' in locals() else V_next + I_gw
            lr_mod_val = lr_mod.item()
            
            # [褪黑激素神經抑制] 模擬昏昏欲睡的狀態，進一步壓抑膜電位
            if 'melatonin' in locals() and melatonin > 0.5:
                V_next = V_next * (1.0 - melatonin * 0.2)
                
            target_action = os_sys.cerebrum.motor_cortex(post_spikes)
            if 'target_action' not in locals(): target_action = np.zeros(8)
            
            # 更新基底核多巴胺狀態
            cur_rpe_val = prediction_error.item() if 'prediction_error' in locals() else 0.0
            os_sys.basal_ganglia.update(cur_rpe_val)
        
        # [前額葉皮質 PFC 世界模型前瞻思維模擬]
        # 在執行動作前，PFC 進行 3 步 Mental Rollout，預測未來是否會摔倒或失衡，或進入人類安全禁區 (價值鎖定)
        target_action_tensor = torch.tensor(target_action, dtype=torch.float32).unsqueeze(0).to(device)
        pre_emptive_rpe, predicted_danger = os_sys.cerebrum.pfc.predict_preemptive_signals(
            state_tensor, target_action_tensor, steps=3, hx=pfc_hx
        )
        
        # [前額葉意志力與主動踩煞車調控] (受腦內啡減緩意志力損耗)
        veto_needed, veto_signal, cognitive_cost = os_sys.cerebrum.pfc.update_willpower(predicted_danger, current_energy, endorphins=endorphins)
        
        # [基底核 (Basal Ganglia)] 動作閘門控制 (融合多巴胺去抑制與前額葉主動抑制)
        gated_action = os_sys.basal_ganglia.gate_action(target_action, veto_signal=veto_signal)
        
        # 若預測到未來會摔倒、失衡或闖入禁區，立即對小腦 e-prop 資格跡施加預先懲罰
        if pre_emptive_rpe.item() < 0:
            os_sys.cerebellum.apply_dopamine(pre_emptive_rpe.to(device), learning_rate=0.001)
            if frame_count % 100 == 0:
                print(f"[前額葉預警] 預測有危險風險！預先廣播 RPE: {pre_emptive_rpe.item():.2f} | 煞車強度 (Veto): {veto_signal:.2f}", flush=True)
        
        # [皮質調變 Cortical Modulation] 運動皮質不再輸出死力氣，而是輸出 CPG 調變參數
        if 'serotonin' in locals() and serotonin < 0.5:
            desperation_noise = np.random.normal(0, 0.5 * (1.0 - serotonin), size=8)
            gated_action = gated_action + desperation_noise
            
        mod_amplitude = gated_action * 0.5  # 調變振幅 (每隻腳獨立)
        mod_freq = np.mean(gated_action) * 0.05 # 整體平均出力影響步頻
        
        # [脊髓反射與大腦調控融合] CPG 產生調變後的步態力矩
        current_action = cpg.step(mod_freq=mod_freq, mod_amplitude=mod_amplitude)
        
        # 將最終動作限制在合理範圍避免極限回彈
        scaled_action = np.clip(current_action, -1.0, 1.0)
            
        # 4. 執行物理動作 (肌肉力矩維持)
        state, reward, terminated, truncated, info = os_sys.brainstem.step_environment(scaled_action)
        
        # 取得最新具身座標與探索極限
        ant_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
        
        # [實體食物探索偵測與情節記憶儲存]
        dist_to_food = np.linalg.norm(ant_pos - food_pos)
        if dist_to_food < 1.2:
            if current_energy < 80.0:
                current_energy = min(100.0, current_energy + 40.0)
                reward += 100.0
                accumulated_reward += 100.0
                # 存入情節記憶 (座標, 視覺殘像, 獎勵)
                os_sys.cerebrum.hippocampus.store_episodic(food_pos, vision_tensor if 'vision_tensor' in locals() else torch.zeros(1,1,64,64), 100.0)
                print(f"[情節記憶] 成功發現食物座標 {food_pos}！能量已補滿，記憶已寫入海馬迴。", flush=True)
                
        # 根據飢餓與情節記憶更新目標座標
        retrieved_pos = os_sys.cerebrum.hippocampus.retrieve_best_episode()
        if current_energy < 70.0 and retrieved_pos is not None:
            target_pos = retrieved_pos
            is_navigating_to_food = True
        else:
            if frame_count % 300 == 0:
                wandering_target = np.random.uniform(-3.0, 3.0, size=2)
            if 'wandering_target' not in locals():
                wandering_target = np.array([1.0, 1.0])
            target_pos = wandering_target
            is_navigating_to_food = False
            
        # 計算網格細胞與位置細胞放電
        grid_rates, place_rates = os_sys.cerebrum.right_parietal.process_spatial(ant_pos)
        
        # 拼接目標向量到 state 觀察中 (29維 -> 31維)
        state = get_state_with_target(state[:29], target_pos, ant_pos)
        
        # [前額葉皮質 PFC 在線學習] 讓世界模型適應當前實體環境動態
        next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        pred_next_state, pfc_hx = os_sys.cerebrum.pfc(state_tensor, target_action_tensor, hx=pfc_hx)
        # 由於 LNN 內部狀態更新了，為了避免計算圖在多次 backprop 中報錯，需 detach
        if pfc_hx is not None:
            if isinstance(pfc_hx, tuple):
                pfc_hx = tuple(h.detach() for h in pfc_hx)
            else:
                pfc_hx = pfc_hx.detach()
        pfc_loss = pfc_criterion(pred_next_state, next_state_tensor)
        pfc_optimizer.zero_grad()
        pfc_loss.backward()
        pfc_optimizer.step()
        
        # 紀錄上一幀的預測誤差，供下一幀的好奇心模組使用
        last_pfc_loss = pfc_loss.item()
        # 紀錄是否有聽到導師呼喚
        last_had_tutor_sound = (env_sound_idx is not None) if 'env_sound_idx' in locals() else False
        
        # [空間目標追逐獎勵] 每靠近目標 1 單位給予 5.0 多巴胺獎勵
        current_dist = np.linalg.norm(target_pos - ant_pos)
        distance_reward = (prev_dist - current_dist) * 5.0
        prev_dist = current_dist
        
        # 更新探索極限距離
        dist_from_origin = np.linalg.norm(ant_pos)
        max_dist_reached = max(max_dist_reached, dist_from_origin)
        
        reward += distance_reward
        accumulated_reward += reward
        
        # 5. 代謝判定與跌倒偵測 (引進熱力學價值鎖定與安全限制)
        num_spikes = post_spikes[0].detach().cpu().numpy().sum()
        bmr_mod = os_sys.autonomic_system.get_bmr_modulation()
        current_energy, is_dead = os_sys.brainstem.metabolize(num_spikes, distance_reward, raw_state=state, bmr_mod=bmr_mod, cognitive_cost=cognitive_cost)
        
        # [防跌倒機制] 如果 Z 軸高度低於 0.3，代表螞蟻翻車或趴在地上，強制死亡並給予極大痛苦 (負獎勵)
        z_pos = info.get('z_position', state[2] if len(state) > 2 else 0.5)
        if z_pos < 0.3:
            is_dead = True
            reward -= 50.0
            accumulated_reward -= 50.0
            if frame_count % 10 == 0:
                print(f"[痛覺中樞] 螞蟻跌倒 (Z={z_pos:.2f})！觸發死亡劇痛。", flush=True)
                
        # === 符號接地：脈衝解碼器在線訓練 ===
        # 1. 建立當前真實具身感受 (Ground Truth)
        z_pos_val = z_pos
        lactic_val = float(np.mean(cpg.lactic_acid))
        subj_pain, subj_craving = os_sys.insula.get_interoceptive_error(current_energy, lactic_val)
        if z_pos_val < 0.3: subj_pain = 1.0 # 跌倒強制痛覺
        
        gt_hunger = subj_craving
        gt_pain = max(0.0, subj_pain - endorphins * 0.8) # 腦內啡壓制痛覺
        gt_curiosity = max(0.0, curiosity_reward) if 'curiosity_reward' in locals() else float(np.clip(cur_pe, 0.0, 1.0))
        gt_fatigue = lactic_val
        gt_sleepiness = float(cur_me) if 'cur_me' in locals() else 0.0
        gt_instability = float(np.clip(abs(z_pos_val - 0.5) * 5.0, 0.0, 1.0))
        
        gt_tensor = torch.tensor([[gt_hunger, gt_pain, gt_curiosity, gt_fatigue, gt_sleepiness, gt_instability]], dtype=torch.float32).to(device)
        
        # 2. 解碼脈衝並計算 Loss
        concatenated_spikes = torch.cat([pre_spikes, post_spikes], dim=1) # shape (1, 270)
        decoded_concepts = spike_decoder(concatenated_spikes)
        
        decoder_loss = decoder_criterion(decoded_concepts, gt_tensor)
        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()
        
        # 3. 提取激活值存入 grounded_concepts_dict，供布洛卡與威尼克 SNN 使用
        concept_vals = decoded_concepts[0].detach().cpu().numpy()
        grounded_concepts_dict = {
            "hunger": float(concept_vals[0]),
            "pain": float(concept_vals[1]),
            "curiosity": float(concept_vals[2]),
            "fatigue": float(concept_vals[3]),
            "sleepiness": float(concept_vals[4]),
            "instability": float(concept_vals[5])
        }
        
        frame_count += 1
        os_sys.brainstem.env.render()
        
        if frame_count % 100 == 0:
            if frame_count > 0:
                clock_h = int(os_sys.pineal_gland.circadian_clock)
                msg1 = f"[生理時鐘] 虛擬時間 {clock_h:02d}:00 | 血清素: {serotonin:.2f} | 腎上腺素: {noradrenaline:.2f} | 褪黑激素: {melatonin:.2f}"
                msg2 = f"       乳酸疲勞值: {np.round(cpg.lactic_acid, 2)}"
                msg3 = f"       [Grounded SNN 解碼] 飢餓:{grounded_concepts_dict['hunger']:.2f} 痛覺:{grounded_concepts_dict['pain']:.2f} 疲勞:{grounded_concepts_dict['fatigue']:.2f} 失衡:{grounded_concepts_dict['instability']:.2f} | Salience: {salience.item():.2f} | WTA 意識焦點: {wta_winner}"
                vocal_str = "-".join(broca_chunking_snn.vocal_buffer) if broca_chunking_snn.vocal_buffer else "無發聲"
                intent_active = np.where(broca_intent_spikes[0].cpu().numpy() > 0)[0]
                intent_str = str(intent_active) if len(intent_active) > 0 else "None"
                msg4 = f"       [語言中樞] 激活意圖: {intent_str} | 發聲歷史: {vocal_str} | 當前符號: {latest_symbol_out}"
                msg5 = f"       [意志力遙測] PFC意志力:{os_sys.cerebrum.pfc.willpower_level:.4f} | 踩煞車強度(Veto):{veto_signal:.4f} | 認知能耗:{cognitive_cost:.4f}"
                msg6 = f"       [神經調製三因子] 催產素(OXT):{oxytocin:.4f} | 腦內啡(END):{endorphins:.4f} | 乙醯膽鹼(ACH):{acetylcholine:.4f} | 探索極限:{max_dist_reached:.2f}"
                
                # 解碼腹側流視覺脈衝
                v_spikes = os_sys.cerebrum.right_brain_vision.latest_ventral_spikes
                v_np = v_spikes[0].cpu().numpy() if v_spikes is not None else np.zeros(4)
                msg7 = f"       [雙流視覺 - 腹側識別] 食物:{v_np[0]:.1f} | 導師:{v_np[1]:.1f} | 危險:{v_np[2]:.1f} | 其他:{v_np[3]:.1f}"
                
                # 解碼網格/位置細胞
                msg8 = f"       [內嗅皮層 - 空間映射] 網格平均活性:{grid_rates.mean():.2f} | 起點位置細胞:{place_rates[14]:.2f} | 食物位置細胞:{place_rates[15]:.2f} | 導航直奔食物:{is_navigating_to_food}"
                
                print(msg1)
                print(msg2)
                print(msg3)
                print(msg4)
                print(msg5)
                print(msg6)
                print(msg7)
                print(msg8)
                with open("ant_log.txt", "a", encoding="utf-8") as f:
                    f.write(msg1 + "\n" + msg2 + "\n" + msg3 + "\n" + msg4 + "\n" + msg5 + "\n" + msg6 + "\n" + msg7 + "\n" + msg8 + "\n")
            cur_rpe = prediction_error.item() if 'prediction_error' in locals() else 0.0
            log_msg = f"[物理遙測] 幀數: {frame_count:04d} | 剩餘能量: {current_energy:.2f} | 驚訝值(RPE): {cur_rpe:.4f} | 關節力矩: {current_action[0]:.2f}"
            print(log_msg, flush=True) 
            with open("ant_log.txt", "a", encoding="utf-8") as f: f.write(log_msg + "\n")
            
        # 6. SNN 動力學狀態推進 (連續時間更新)
        V_prev = V_next.detach()
        I_syn_prev = I_syn_next.detach()

if __name__ == "__main__":
    run_mujoco_ant()

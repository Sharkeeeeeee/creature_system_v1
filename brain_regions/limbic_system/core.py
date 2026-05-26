import torch
import numpy as np

class Amygdala:
    """
    杏仁核 (Amygdala)
    情緒處理、偵測危險訊號、社交行為。
    """
    def __init__(self):
        self.fear_level = 0.0
        
    def process_emotion(self, current_energy, rpe, is_dead):
        danger_signal = 0.0
        if current_energy < 30.0:
            danger_signal += (30.0 - current_energy) * 0.03
        if rpe < -0.5: # 巨大的負向預測誤差 (痛苦)
            danger_signal += abs(rpe)
        if is_dead:
            danger_signal += 1.0
            
        self.fear_level = self.fear_level * 0.8 + danger_signal * 0.2
        return self.fear_level

class EndocrineSystem:
    """
    內分泌系統 (Endocrine System)
    調節長期神經狀態：血清素、去甲腎上腺素、以及催產素、腦內啡與乙醯膽鹼。
    """
    def __init__(self):
        self.serotonin = 1.0        # 血清素：1.0 為穩定，降低會產生躁動
        self.noradrenaline = 0.0    # 去甲腎上腺素：0.0 為平靜，升高會提升警覺
        self.oxytocin = 0.5         # 催產素：社交信賴度，0.1 ~ 1.0
        self.endorphins = 0.0       # 腦內啡：痛覺壓制與抗壓，0.0 ~ 1.0
        self.acetylcholine = 0.5    # 乙醯膽鹼：注意力與新奇探索調變，0.2 ~ 1.0
        self.prev_danger_level = 0.0
        
    def process_hormones(self, current_energy, rpe, is_dead, 
                         has_tutor_sound=False, 
                         danger_level=0.0, 
                         dist_from_origin=0.0, 
                         max_dist_reached=0.0):
        # 1. 去甲腎上腺素 (Noradrenaline) 代謝
        if current_energy < 30.0:
            self.noradrenaline = min(1.0, self.noradrenaline + 0.1)
        else:
            self.noradrenaline = max(0.0, self.noradrenaline - 0.05)
            
        # 2. 血清素 (Serotonin) 代謝
        if rpe < -1.0 or is_dead:
            self.serotonin = max(0.1, self.serotonin - 0.2)
        else:
            self.serotonin = min(1.0, self.serotonin + 0.01)
            
        # 3. 催產素 (Oxytocin) 代謝
        # 當聽到導師聲音且沒處於危險中時分泌
        if has_tutor_sound and danger_level < 0.5:
            self.oxytocin = min(1.0, self.oxytocin + 0.15)
        else:
            self.oxytocin = max(0.1, self.oxytocin - 0.01) # 緩慢衰退到基線 0.1
            
        # 4. 腦內啡 (Endorphins) 代謝
        # 成功躲避危險 (先前危險度高於 0.8 但現在降低至 0.3 以下) 或是有超高的正向 RPE (例如被餵食)
        escaped = (self.prev_danger_level > 0.8 and danger_level < 0.3)
        if escaped or rpe > 3.0:
            self.endorphins = min(1.0, self.endorphins + 0.3)
        else:
            self.endorphins = max(0.0, self.endorphins - 0.05) # 衰退
        self.prev_danger_level = danger_level
        
        # 5. 乙醯膽鹼 (Acetylcholine) 代謝
        # 到達前所未有的新座標 (未知領域)，代表探索取得突破
        if dist_from_origin > max_dist_reached + 0.1:
            self.acetylcholine = min(1.0, self.acetylcholine + 0.25)
        else:
            self.acetylcholine = max(0.2, self.acetylcholine - 0.015) # 衰退到基線 0.2
            
        return self.serotonin, self.noradrenaline, self.oxytocin, self.endorphins, self.acetylcholine

class Hippocampus:
    """
    海馬迴 (Hippocampus)
    短期記憶轉長期記憶、陳述性記憶。
    """
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.stdp_memory = []
        self.stdp_position = 0
        self.vision_memory = []
        self.vision_position = 0
        self.episodic_memory = [] # 情節記憶: (pos, vision_tensor, reward)
        
    def store(self, state, action, reward, next_state, terminated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, terminated)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        import random
        return random.sample(self.memory, min(batch_size, len(self.memory)))
        
    def store_eprop_experience(self, eligibility_trace, rpe):
        if len(self.stdp_memory) < self.capacity:
            self.stdp_memory.append(None)
        priority = float(abs(rpe.item())) + 1e-5
        self.stdp_memory[self.stdp_position] = (
            eligibility_trace.detach().clone(), rpe.detach().clone(), priority
        )
        self.stdp_position = (self.stdp_position + 1) % self.capacity
        
    def store_vision_experience(self, vision_tensor):
        if len(self.vision_memory) < self.capacity:
            self.vision_memory.append(None)
        self.vision_memory[self.vision_position] = vision_tensor.detach().cpu().half()
        self.vision_position = (self.vision_position + 1) % self.capacity
        
    def get_recent_visions(self, count=4):
        actual_count = min(count, len(self.vision_memory))
        if actual_count == 0: return []
        indices = [(self.vision_position - 1 - i) % len(self.vision_memory) for i in range(actual_count)]
        return [self.vision_memory[idx].float() for idx in indices]
        
    def sample_eprop_experience(self, batch_size):
        import random
        priorities = np.array([exp[2] for exp in self.stdp_memory])
        sum_priorities = priorities.sum()
        if sum_priorities == 0 or len(self.stdp_memory) < batch_size:
            return random.sample(self.stdp_memory, min(batch_size, len(self.stdp_memory)))
        probs = priorities / sum_priorities
        sampled_indices = np.random.choice(len(self.stdp_memory), size=batch_size, replace=False, p=probs)
        return [self.stdp_memory[idx] for idx in sampled_indices]

    def store_episodic(self, pos, vision_tensor, reward):
        # 儲存座標、視覺特徵與實體反饋的三元組，容量限制為 50
        pos_copied = pos.copy() if hasattr(pos, 'copy') else pos
        vision_copied = vision_tensor.detach().cpu().clone() if hasattr(vision_tensor, 'detach') else vision_tensor
        
        self.episodic_memory.append((pos_copied, vision_copied, reward))
        if len(self.episodic_memory) > 50:
            # 優先保留高回報的情節，若滿了則移除回報最低的
            self.episodic_memory.sort(key=lambda x: x[2], reverse=True)
            self.episodic_memory.pop()
            
    def retrieve_best_episode(self):
        # 尋找回報最高的正向記憶 (例如成功獲得能量的歷史座標)
        if not self.episodic_memory:
            return None
        # 找出 reward 最高的 tuple
        best_ep = max(self.episodic_memory, key=lambda x: x[2])
        # 如果該記憶具有足夠的正向獎勵，則回傳該位置座標
        if best_ep[2] > 5.0:
            return best_ep[0]
        return None

    def __len__(self):
        return len(self.memory)

class LimbicSystem:
    def __init__(self):
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus(capacity=20000)
        self.endocrine = EndocrineSystem()

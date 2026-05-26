import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brain_regions.cerebrum.prefrontal_cortex import PrefrontalCortex
from brain_regions.cerebrum.right_hemisphere import RightHemisphereVision
from brain_regions.limbic_system.core import LimbicSystem

class OccipitalLobe:
    """
    枕葉 (Occipital Lobe)
    負責接收裸像素 (RGB numpy array)，轉換為神經網路可消化的灰階 Tensor。
    """
    def __init__(self, target_size=(64, 64), threshold=0.01):
        self.target_size = target_size
        self.threshold = threshold
        self.prev_tensor = None
        
    def reset(self):
        self.prev_tensor = None
        
    def process(self, rgb_array, device="cpu"):
        # 如果環境沒有回傳影像，回傳全黑與未改變
        if rgb_array is None:
            return torch.zeros(1, 1, self.target_size[0], self.target_size[1], device=device), False
            
        # 1. Numpy (H, W, C) -> Tensor (C, H, W)
        tensor_img = torch.from_numpy(rgb_array.copy()).permute(2, 0, 1).float()
        
        # 2. 轉換為灰階 (0.299 R + 0.587 G + 0.114 B)
        gray = 0.299 * tensor_img[0:1] + 0.587 * tensor_img[1:2] + 0.114 * tensor_img[2:3]
        
        # 3. Resize to 64x64
        resized = F.interpolate(gray.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False)
        
        # 4. Normalize to 0.0 ~ 1.0
        normalized = (resized / 255.0).to(device)
        
        has_changed = True
        if self.prev_tensor is not None:
            # 計算平均像素差異
            diff = torch.abs(normalized - self.prev_tensor).mean().item()
            if diff < self.threshold:
                has_changed = False
                
        if has_changed:
            self.prev_tensor = normalized.clone()
            
        return normalized, has_changed

class MotorCortex:
    """
    運動皮質 (Motor Cortex) - 同時處理區 (額葉基礎動作)
    職責：接收小腦或基底核的脈衝訊號，將其解碼為具體的肌肉/馬達執行指令。
    """
    def __init__(self, in_spikes, num_actions, is_continuous=False):
        self.in_spikes = in_spikes
        self.num_actions = num_actions
        self.is_continuous = is_continuous
        self.muscle_tension = np.zeros(num_actions, dtype=np.float32)
        
    def __call__(self, post_spikes):
        spikes_np = post_spikes[0].detach().cpu().numpy()
        if self.is_continuous:
            group_size = self.in_spikes // self.num_actions
            half_g = group_size // 2
            actions = []
            for i in range(self.num_actions):
                sub = spikes_np[i*group_size : (i+1)*group_size]
                pos_act = sub[:half_g].sum()
                neg_act = sub[half_g:].sum()
                instant_torque = (pos_act - neg_act) / max(1, half_g)
                # 肌肉張力以 0.95 的慣性衰減，0.05 接收新脈衝
                self.muscle_tension[i] = self.muscle_tension[i] * 0.95 + instant_torque * 0.05
                actions.append(float(self.muscle_tension[i]))
            return self.muscle_tension.copy()
        else:
            group_size = self.in_spikes // self.num_actions
            action_counts = [spikes_np[i*group_size:(i+1)*group_size].sum() for i in range(self.num_actions)]
            return torch.tensor(action_counts).argmax().item()

# ================= 左腦 (Left Hemisphere) =================
class LeftFrontalLobe:
    def __init__(self):
        # 邏輯推理、語言表達、判斷
        pass

class LeftParietalLobe:
    """
    頂葉皮質 (Parietal Lobe)
    負責處理本體感覺 (Proprioception) 與前庭覺 (Vestibular System)。
    剝奪上帝視角，加入神經雜訊與傳導延遲。
    """
    def __init__(self, delay_steps=2):
        self.delay_steps = delay_steps
        self.sensory_buffer = []

    def reset(self):
        self.sensory_buffer = []

    def process_somatosensory(self, raw_state):
        # 模擬生物雜訊 (Sensory Noise)
        noise = np.random.normal(0, 0.02, size=len(raw_state))
        noisy_state = raw_state + noise
        
        # 模擬神經傳導延遲 (Sensory Delay)
        self.sensory_buffer.append(noisy_state)
        if len(self.sensory_buffer) > self.delay_steps:
            return self.sensory_buffer.pop(0)
        else:
            return self.sensory_buffer[0]

class LeftOccipitalLobe:
    def __init__(self):
        # 分析文字、細節、精細形狀判斷
        pass

class LeftTemporalLobe:
    def __init__(self):
        # 語言理解、語言記憶
        pass

class LeftHemisphere:
    def __init__(self):
        self.frontal = LeftFrontalLobe()
        self.parietal = LeftParietalLobe()
        self.occipital = LeftOccipitalLobe()
        self.temporal = LeftTemporalLobe()

# ================= 右腦 (Right Hemisphere) =================
class RightFrontalLobe:
    def __init__(self):
        # 創造力、語調、幽默感、理解
        pass

class RightParietalLobe:
    """
    右腦頂葉 (Right Parietal Lobe) - 空間認知與導航
    內建內嗅皮層 (Entorhinal Cortex)，包含網格細胞 (Grid Cells) 與位置細胞 (Place Cells)。
    """
    def __init__(self, num_grid=16, num_place=16):
        self.num_grid = num_grid
        self.num_place = num_place
        
        # 網格細胞參數：尺度 scale (從細緻到粗放)、方向 orientation、相位相移 phase
        np.random.seed(42) # 保證結果可重現
        self.grid_scales = np.linspace(1.5, 6.0, num_grid)
        self.grid_orientations = np.random.uniform(0, np.pi/3, num_grid)
        self.grid_phases = np.random.uniform(-2.0, 2.0, (num_grid, 2))
        
        # 位置細胞參數：特定位置的 Gauss 感受野中心
        # 均勻覆蓋 [-4.0, 4.0] 的區域，最後兩個為特殊的 [0, 0] (起點) 與 [4.0, 4.0] (目標點)
        centers = []
        for x in np.linspace(-3.0, 3.0, 7):
            for y in np.linspace(-3.0, 3.0, 2):
                centers.append([x, y])
        centers.append([0.0, 0.0])   # 起點位置細胞
        centers.append([4.0, 4.0])   # 目標位置細胞 (真實食物在 4.0, 4.0)
        self.place_centers = np.array(centers[:num_place]) # 限制到 num_place 個
        self.place_widths = np.random.uniform(0.8, 1.5, num_place)
        
    def process_spatial(self, ant_pos):
        """
        將實體座標計算為網格與位置細胞的放電率
        ant_pos: [x, y]
        """
        x, y = ant_pos[0], ant_pos[1]
        
        # 1. 計算六角形網格細胞 (Grid Cells)
        grid_rates = np.zeros(self.num_grid)
        for i in range(self.num_grid):
            scale = self.grid_scales[i]
            orientation = self.grid_orientations[i]
            px, py = self.grid_phases[i]
            
            k = 2 * np.pi / scale
            angles = [orientation, orientation + np.pi/3, orientation + 2*np.pi/3]
            val = 0.0
            for ang in angles:
                kx = k * np.cos(ang)
                ky = k * np.sin(ang)
                val += np.cos(kx * (x - px) + ky * (y - py))
            grid_rates[i] = (val / 3.0 + 1.0) / 2.0 # 歸一化到 0.0 ~ 1.0
            
        # 2. 計算 Gaussian 位置細胞 (Place Cells)
        place_rates = np.zeros(self.num_place)
        for i in range(self.num_place):
            cx, cy = self.place_centers[i]
            w = self.place_widths[i]
            dist_sq = (x - cx)**2 + (y - cy)**2
            place_rates[i] = np.exp(-dist_sq / (2 * w**2))
            
        return grid_rates, place_rates

class RightOccipitalLobe:
    def __init__(self, out_features=16, in_spikes=32):
        # 模糊推論、快速辨認圖像
        self.right_brain_vision = RightHemisphereVision(out_features=in_spikes // 2)
        
        # 視覺預處理區
        self.occipital_lobe = OccipitalLobe(target_size=(64, 64))

class RightTemporalLobe:
    def __init__(self):
        # 音樂聲音辨識、臉孔辨識、環境聲音理解
        pass

class RightHemisphere:
    def __init__(self):
        self.frontal = RightFrontalLobe()
        self.parietal = RightParietalLobe()
        self.occipital = RightOccipitalLobe(out_features=16)
        self.temporal = RightTemporalLobe()

# ================= 大腦複合體 (Cerebrum) =================
class Cerebrum:
    """
    大腦複合體 (Cerebrum)
    整合所有高階認知模組：左腦、右腦、邊緣系統、運動皮質。
    """
    def __init__(self, state_dim=4, action_dim=1, num_actions=2):
        self.left = LeftHemisphere()
        self.right = RightHemisphere()
        self.limbic = LimbicSystem()
        
        # 前額葉皮質 (Prefrontal Cortex): 連續時間世界預測模型 (LNN)
        self.pfc = PrefrontalCortex(state_dim=state_dim, action_dim=action_dim)
        
        # 為了兼容與暴露基礎功能
        self.motor_cortex = MotorCortex(in_spikes=32, num_actions=num_actions, is_continuous=True)
        self.right_brain_vision = self.right.occipital.right_brain_vision
        self.occipital_lobe = self.right.occipital.occipital_lobe
        self.hippocampus = self.limbic.hippocampus
        self.right_parietal = self.right.parietal

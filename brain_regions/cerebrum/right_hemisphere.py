import torch
import torch.nn as nn
import torch.nn.functional as F

class RightHemisphereVision(nn.Module):
    """
    右腦視覺皮質 (Right Hemisphere Visual Cortex) - 雙流視覺模型
    背側流 (Dorsal Stream - 'Where')：提取空間特徵與運動，輸出直接連接小腦。
    腹側流 (Ventral Stream - 'What')：識別物體類別 (食物、導師、危險)，連接到高階認知。
    """
    def __init__(self, out_features=16):
        super().__init__()
        # 共享初級視覺特徵卷積層 (V1/V2 區)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=4) # out: 4x15x15
        self.relu1 = nn.ReLU()
        
        # 背側流 Where 通路 (Dorsal Stream)
        self.dorsal_conv = nn.Conv2d(4, 8, kernel_size=4, stride=2) # out: 8x6x6
        self.dorsal_fc = nn.Linear(288, out_features)
        
        # 腹側流 What 通路 (Ventral Stream)
        self.ventral_conv = nn.Conv2d(4, 8, kernel_size=4, stride=2)
        self.ventral_fc = nn.Linear(288, 4) # 輸出 4 種語意分類: [食物, 導師, 危險, 其他]
        
        # 暫存腹側流的輸出脈衝，供高階大腦區域調閱
        self.latest_ventral_spikes = None
        
    def forward(self, image_tensor, extra_info=None):
        """
        image_tensor: (batch, 1, 64, 64) 歸一化灰階圖
        extra_info: 字典，包含實體具身位置資訊，模擬大腦在枕葉與顳葉的特徵接地
        """
        # 如果輸入是 3D 張量，擴充 batch 維度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # 共享初級特徵提取
        v1_features = self.relu1(self.conv1(image_tensor))
        
        # 1. 執行背側流 (Dorsal Stream) - 提取空間資訊
        d_x = F.relu(self.dorsal_conv(v1_features))
        d_x = d_x.flatten(start_dim=1)
        d_features = self.dorsal_fc(d_x)
        dorsal_rates = torch.sigmoid(d_features)
        dorsal_spikes = (torch.rand_like(dorsal_rates) < dorsal_rates).float()
        
        # 2. 執行腹側流 (Ventral Stream) - 物體與概念識別
        v_x = F.relu(self.ventral_conv(v1_features))
        v_x = v_x.flatten(start_dim=1)
        v_logits = self.ventral_fc(v_x)
        
        # 具身識別偏置注入 (Embodied Object Grounding)
        if extra_info is not None:
            device = v_logits.device
            bias = torch.zeros_like(v_logits).to(device)
            dist_to_food = extra_info.get("dist_to_food", 999.0)
            dist_to_tutor = extra_info.get("dist_to_tutor", 999.0)
            is_unsafe = extra_info.get("is_unsafe", False)
            
            if dist_to_food < 1.5:
                bias[0, 0] = 5.0 # 食物偵測器
            if dist_to_tutor < 2.0:
                bias[0, 1] = 5.0 # 導師偵測器
            if is_unsafe:
                bias[0, 2] = 5.0 # 危險警告
                
            v_logits = v_logits + bias
            
        ventral_rates = torch.sigmoid(v_logits)
        self.latest_ventral_spikes = (torch.rand_like(ventral_rates) < ventral_rates).float()
        
        # 保持與主程式相容的傳回格式 (只傳回背側脈衝做為 right_spikes 進小腦)
        return dorsal_spikes, d_features

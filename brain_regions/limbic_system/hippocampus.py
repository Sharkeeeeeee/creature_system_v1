import torch
import random
import numpy as np
from collections import deque

class Hippocampus:
    """
    海馬迴：情節記憶池 (Episodic Memory / Differentiable Neural Dictionary)
    負責儲存「清醒時」的時間切片 (Transitions)，並在 DMN (睡眠模式) 進行高速離線回放 (Offline Replay)。
    這使得系統可以對這些記憶進行全局優化 (如 Backprop)，模擬將經驗編譯為直覺的過程。
    """
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def store(self, state, action, reward, next_state, terminated):
        """
        儲存一幀物理世界的記憶切片
        這裡儲存的是物理狀態 (float array)，以便在回放時重新通過丘腦編碼。
        如果是真實生物，儲存的會是高維度的感知皮層特徵，但為了重用環境編碼器，我們存物理狀態。
        """
        self.memory.append((state, action, reward, next_state, terminated))
        
    def replay(self, batch_size):
        """
        DMN 睡眠模式調用：回放隨機記憶批次
        """
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(terminateds, dtype=torch.float32)
        )
        
    def __len__(self):
        return len(self.memory)

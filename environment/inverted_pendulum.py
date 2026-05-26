import gymnasium as gym
import numpy as np

class OODInvertedPendulum(gym.Wrapper):
    """
    自訂的 Gymnasium 環境封裝，用於支援：
    1. OOD (Out-of-Distribution) 物理參數突變測試
    2. 主動覓食機制 (Energy Pellets)
    """
    def __init__(self, render_mode=None):
        super().__init__(gym.make("CartPole-v1", render_mode=render_mode))
        
        # 獲取初始物理參數
        self.default_gravity = self.env.unwrapped.gravity
        self.default_masscart = self.env.unwrapped.masscart
        self.pellet_x = self._spawn_pellet()
        
    def _spawn_pellet(self):
        import random
        # 隨機生成能量包的 X 座標 (-1.5 到 1.5)
        return random.uniform(-1.5, 1.5)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.pellet_x = self._spawn_pellet()
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # 檢查是否吃到能量包
        cart_x = state[0]
        if abs(cart_x - self.pellet_x) < 0.2:
            info['pellet_eaten'] = True
            self.pellet_x = self._spawn_pellet() # 重生新的能量包
        else:
            info['pellet_eaten'] = False
            
        return state, reward, terminated, truncated, info
        
    def apply_ood_gravity(self, multiplier=2.0):
        """即時改變重力大小 (OOD 事件)"""
        new_gravity = self.default_gravity * multiplier
        self.env.unwrapped.gravity = new_gravity
        print(f"[OOD EVENT] Gravity changed to {new_gravity:.2f}")
        
    def apply_ood_motor_failure(self, force_multiplier=0.5):
        """即時改變馬達推力 (模擬馬達損壞)"""
        self.env.unwrapped.force_mag *= force_multiplier
        print(f"[OOD EVENT] Motor force reduced by {force_multiplier*100}%")

    def restore_physics(self):
        self.env.unwrapped.gravity = self.default_gravity
        print("[RESTORE] Physics restored to normal.")

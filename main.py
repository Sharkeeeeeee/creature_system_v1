import gymnasium as gym
from brain_regions.brainstem import Brainstem, ReticularFormation, AutonomicNervousSystem
from brain_regions.diencephalon import Diencephalon, PinealGland
from brain_regions.cerebellum import Cerebellum, InferiorOlive
from brain_regions.cerebrum import Cerebrum, BasalGanglia, Insula
from brain_regions.limbic_system.core import LimbicSystem
import torch

class EmbodiedAGI_OS:
    """
    具身意識 AI 作業系統 (Embodied AGI OS) - Phase 11 MuJoCo
    """
    def __init__(self, use_mujoco=True, render_mode=None):
        self.use_mujoco = use_mujoco
        
        # 1. 腦幹 (Brainstem): 實體環境與代謝中樞
        if use_mujoco:
            # 使用連續控制四足機器狗，並解除預設的 1000 步壽命限制
            env = gym.make("Ant-v4", exclude_current_positions_from_observation=False, max_episode_steps=999999, render_mode=render_mode)
            state_dim = env.observation_space.shape[0] + 2 # 加上目標相對向量 [dx_norm, dy_norm]
            action_dim = env.action_space.shape[0]     # 8
            num_actions = 8
            neurons_per_state = 2 # 降低編碼數量，否則維度會爆炸
        else:
            from environment.inverted_pendulum import OODInvertedPendulum
            env = OODInvertedPendulum()
            state_dim = 4
            action_dim = 1
            num_actions = 2
            neurons_per_state = 10
            
        self.brainstem = Brainstem(env)
        self.ras = ReticularFormation()
        self.autonomic_system = AutonomicNervousSystem()
        
        # 2. 間腦 (Diencephalon): 感覺閘道與內分泌中樞
        self.diencephalon = Diencephalon(state_dim=state_dim, neurons_per_state=neurons_per_state)
        self.pineal_gland = PinealGland()
        
        # 3. 大腦 (Cerebrum): 高階認知、世界模型與視覺
        self.cerebrum = Cerebrum(state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)
        self.cerebrum.motor_cortex.is_continuous = use_mujoco # 開啟連續動作解碼
        self.basal_ganglia = BasalGanglia()
        self.insula = Insula()
        
        # 4. 小腦 (Cerebellum): 直覺反射與局部學習 (LNN+SNN)
        left_spikes = state_dim * neurons_per_state 
        right_spikes = 16
        total_sensory_spikes = left_spikes + right_spikes 
        
        # 為了產生 8 個動作，每個動作需要一對 (正/負) 神經元。
        # 如果是連續動作，我們配置出足夠的神經元讓 MotorCortex 解碼
        # Ant-v4 有 8 個動作，我們配置 32 個神經元 (8 * 4) 
        out_features = 32 if use_mujoco else 32
        self.cerebellum = Cerebellum(in_features=total_sensory_spikes, out_features=out_features)
        self.inferior_olive = InferiorOlive()

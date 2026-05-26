import torch
import torch.nn as nn

class MotorCortex(nn.Module):
    """
    運動皮層 (Motor Cortex)
    接收來自深層網路 (LSN) 的高維離散脈衝，並將其解碼為物理世界的具體動作。
    
    對於 CartPole 來說，動作是離散的 0 (向左) 或 1 (向右)。
    我們採取 Population Coding (群體編碼)：
    比較代表「左」的脈衝數量與代表「右」的脈衝數量。
    """
    def __init__(self, in_spikes, num_actions=2):
        super().__init__()
        self.in_spikes = in_spikes
        self.num_actions = num_actions
        
        # 假設輸入脈衝平均分配給各個動作
        assert in_spikes % num_actions == 0, "Input spikes must be divisible by number of actions"
        self.spikes_per_action = in_spikes // num_actions
        
    def forward(self, spikes, return_logits=False):
        """
        spikes: (batch, in_spikes) 來自上游的離散脈衝
        回傳: (batch,) 的具體動作 (0 或 1)
        """
        batch_size = spikes.shape[0]
        # 重塑形狀：(batch, num_actions, spikes_per_action)
        action_groups = spikes.view(batch_size, self.num_actions, self.spikes_per_action)
        
        # 計算每個動作群組的總脈衝數
        group_spike_counts = action_groups.sum(dim=-1)
        
        if return_logits:
            return group_spike_counts
            
        # 選擇脈衝數最多的群組作為最終動作 (Winner-Takes-All)
        actions = torch.argmax(group_spike_counts, dim=-1)
        
        return actions

import torch

class BasalGanglia:
    """
    基底核 (Basal Ganglia)
    功能：動作決策的最終閘門 (Action Gating)。
    預設為「持續抑制 (Tonic Inhibition)」。只有當預期獎勵/多巴胺 (Dopamine/RPE) 充足時，
    才會「鬆開煞車 (Disinhibition)」，允許前額葉或運動皮層的意圖化為實際動作。
    如果多巴胺枯竭，動物將陷入習得性無助或帕金森氏症狀態。
    """
    def __init__(self, baseline_dopamine=1.0):
        self.dopamine_level = baseline_dopamine
        
    def update(self, current_rpe):
        """
        更新基底核的多巴胺池。
        RPE > 0 會增加多巴胺，RPE < 0 (持續受挫) 會耗竭多巴胺。
        """
        # 簡單的 leaky integrator
        self.dopamine_level = self.dopamine_level * 0.95 + current_rpe * 0.05
        # 確保在合理範圍內
        self.dopamine_level = max(0.0, min(2.0, self.dopamine_level))
        
    def gate_action(self, motor_intent, veto_signal=0.0):
        """
        根據多巴胺濃度以及前額葉主動抑制訊號 (veto_signal) 決定是否放行動作。
        motor_intent: tensor, 來自運動皮層的意圖向量
        veto_signal: float, 主動煞車強度 (0.0 ~ 1.0)
        """
        # 如果多巴胺極低，進行強烈抑制 (衰減 90% 的意圖強度)
        if self.dopamine_level < 0.2:
            inhibition_factor = 0.1
        else:
            # 正常狀態下，多巴胺越高，動作放行比例越高 (上限 1.0)
            inhibition_factor = min(1.0, self.dopamine_level)
            
        # 結合前額葉主動煞車 (Veto) 調控
        effective_gate = inhibition_factor * (1.0 - veto_signal)
        
        gated_action = motor_intent * effective_gate
        return gated_action

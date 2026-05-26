import numpy as np

class Insula:
    """
    島葉 (Insula)
    功能：內感預測誤差 (Interoceptive Prediction Error)。
    它監控身體內部的生理狀態 (血壓、血糖、乳酸、溫度)，並產生「預期生理狀態」。
    當實際狀態與預期狀態產生落差時，就會產生內感受誤差，這是主觀「痛苦(Pain)」、「疲勞(Fatigue)」、「渴望(Craving)」的物理基礎。
    """
    def __init__(self):
        # 預期生理狀態 (恆定性目標)
        self.target_energy = 90.0
        self.target_lactic = 0.0
        self.target_temp = 37.0
        
    def get_interoceptive_error(self, current_energy, current_lactic, current_temp=37.0):
        """
        計算內感受誤差
        """
        energy_pe = self.target_energy - current_energy
        # 能量不足時為正誤差(渴望)
        hunger_craving = max(0.0, energy_pe) / 100.0
        
        lactic_pe = current_lactic - self.target_lactic
        # 乳酸過多時為正誤差(疲勞感/痛苦)
        fatigue_pain = max(0.0, lactic_pe)
        
        temp_pe = abs(current_temp - self.target_temp)
        # 溫度失調
        temp_pain = temp_pe / 5.0
        
        # 總結內感主觀痛苦與渴望
        subjective_pain = min(1.0, fatigue_pain + temp_pain)
        subjective_craving = min(1.0, hunger_craving)
        
        return subjective_pain, subjective_craving

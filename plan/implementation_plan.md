# Phase 29: 鏡像神經元系統與階層式語法湧現

此計畫將實作 Embodied AGI OS 的 Phase 29。我們將進化原生語言中樞，使其支援複雜的多符號「組塊化 (Chunking)」，並引入「鏡像神經元系統 (Mirror Neuron System)」以實現社交模仿學習，確保系統在不依賴 LLM 的情況下，依然能保持語言符號的實體接地。

## User Review Required

> [!IMPORTANT]
> 為了正確測試鏡像神經元系統 (MNS) 與社交模仿學習，我們的螞蟻需要「觀察」另一個實體的發聲動作。
> 我計畫在主模擬迴圈中，加入一個虛擬的「導師螞蟻 (Tutor Ant)」信號（或環境中的人類信號），它會定期發出結構化的聲音組塊（例如連續發出 "B-C"）。MNS 會將 Wernicke 聽覺區處理的信號連接到 Broca 運動區，促使我們的螞蟻模仿這些聲音。
> 請問您是否同意使用這種「虛擬導師信號」來進行測試？

## Proposed Changes

### `brain_regions/native_language_area.py`

#### [MODIFY] [native_language_area.py](file:///c:/project/project_x/brain_regions/native_language_area.py)
- **新增 `MirrorNeuronSystemSNN`**：這是一個新的 SNN 層，負責在 `WernickeAreaSNN` (感覺接收) 與 `BrocaAreaSNN` (運動意圖) 之間建立突觸連結。它利用 STDP (脈衝時序依賴可塑性)，將「我聽到的聲音」與「我準備發出的聲音」綁定在一起。
- **將 `BrocaAreaSNN` 重構為階層式架構**：
  - `BrocaIntentSNN` (高階意圖區)：將連續的 `limbic_state` (維度 6 的內部情感狀態) 轉換為離散的高階意圖脈衝 (例如：「飢餓呼喚」、「痛覺尖叫」)。
  - `BrocaMotorChunkingSNN` (低階序列解碼區)：接收高階意圖，並將其在時間軸上展開為「一連串」的符號脈衝 (Chunking)，而不只是單一輸出。這類似於延遲線 (Delay-Line) 機制，能將 1 個意圖脈衝對應到 "A -> B" 這樣的序列。

### `visualize_ant_mujoco.py`

#### [MODIFY] [visualize_ant_mujoco.py](file:///c:/project/project_x/visualize_ant_mujoco.py)
- **整合 MNS 與階層式 Broca**：更新語言模型的初始化與前向傳播邏輯。
- **注入社交導師信號**：在環境中定期產生外部「導師」的發聲信號 (例如：符號 1，接著符號 2)，並餵入 `WernickeAreaSNN`。
- **MNS 聯動**：將 Wernicke 的輸出傳遞給 MNS，MNS 再去調製 `BrocaIntentSNN` 的膜電位，使螞蟻能在無意識中「模仿」聽到的序列。
- **強化終端機日誌**：更新 Console 輸出，清楚顯示 Broca 成功產生「組塊化序列 (例如 "A-B")」或是「透過 MNS 成功模仿」的瞬間。

## Verification Plan

### Manual Verification
1. 執行 `python visualize_ant_mujoco.py`。
2. 觀察終端機的文字日誌。
3. **驗證組塊化 (Chunking)**：確認螞蟻在特定情緒下 (例如飢餓)，是否能穩定輸出一連串的符號序列 (例如這個 frame 發出 "A"，下個 frame 發出 "B")。
4. **驗證 MNS 模仿**：等待「導師信號」觸發，檢查日誌是否顯示 Wernicke 解碼後成功連鎖啟動 MNS，並在接下來的幾個 frame 內導致 Broca 模仿出相同的聲音序列。

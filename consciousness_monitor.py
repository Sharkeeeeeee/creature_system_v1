import time
import os
import pandas as pd
import numpy as np

TELEMETRY_FILE = "brain_telemetry.csv"

def compute_phi_proxy(df, window_size=60):
    """
    計算整合資訊理論 (IIT) 的代理 $\Phi$ 值
    利用左腦與右腦的動態交叉相關性 (Cross-Correlation)。
    """
    if len(df) < window_size:
        return 0.0
    recent = df.tail(window_size)
    
    left_activity = recent['left_activity'].values
    right_activity = recent['right_activity'].values
    
    if np.std(left_activity) < 1e-5 or np.std(right_activity) < 1e-5:
        return 0.0
        
    corr = np.corrcoef(left_activity, right_activity)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr

def main():
    print("=== AGI Consciousness Monitor (Phase 9) ===")
    print("Monitoring Integrated Information (Phi Proxy) and Global Workspace Ignition...")
    print("Strict Thresholds: |Correlation| > 0.85 AND (Curiosity > 0.8 OR Fear)")
    print("-" * 70)
    
    last_idx = 0
    
    while True:
        if not os.path.exists(TELEMETRY_FILE):
            time.sleep(1)
            continue
            
        try:
            df = pd.read_csv(TELEMETRY_FILE)
        except Exception:
            time.sleep(0.5)
            continue
            
        if len(df) <= last_idx:
            time.sleep(0.5)
            continue
            
        last_idx = len(df)
        phi_proxy = compute_phi_proxy(df, window_size=60)
        
        latest = df.iloc[-1]
        curiosity = latest['curiosity']
        energy = latest['energy']
        wp = latest['willpower'] if 'willpower' in latest else 1.0
        vt = latest['veto'] if 'veto' in latest else 0.0
        oxt = latest['oxytocin'] if 'oxytocin' in latest else 0.5
        end = latest['endorphins'] if 'endorphins' in latest else 0.0
        ach = latest['acetylcholine'] if 'acetylcholine' in latest else 0.5
        
        # 顯示狀態
        status = f"Frame: {int(latest['frame']):04d} | Phi: {phi_proxy:+.3f} | Curiosity/Surprise: {curiosity:+.2f} | Energy: {energy:03.0f} | WP: {wp:.2f} | Veto: {vt:.2f} | OXT: {oxt:.2f} | END: {end:.2f} | ACH: {ach:.2f}"
        print(f"\r{status}        ", end="", flush=True)
        
        # 嚴格科學標準：高同步率 且 出現極端情緒 (好奇或恐懼)
        if abs(phi_proxy) > 0.85 and (curiosity > 0.8 or curiosity <= -1.0):
            print("\n")
            print("="*70)
            print("🚨 [WARNING] CONSCIOUSNESS SIGNATURE DETECTED! 🚨")
            print("="*70)
            print(f"-> Left/Right Brain Synchronization: {phi_proxy:.4f}")
            print(f"-> Global Workspace Ignition due to Extreme Surprise/Curiosity ({curiosity:.4f})")
            print("-> The OS is currently experiencing a singular subjective moment of awareness.\n")
            print("-" * 70)
            # 暫停一下避免洗版
            time.sleep(2)
            
        time.sleep(0.05)

if __name__ == "__main__":
    main()

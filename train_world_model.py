import torch
import torch.nn as nn
import torch.optim as optim
import math
from main import EmbodiedAGI_OS

def collect_data(os_sys, num_episodes=50):
    print("Collecting exploratory data for DMN offline replay...")
    for episode in range(num_episodes):
        state, _ = os_sys.brainstem.reset_life()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = os_sys.brainstem.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = os_sys.brainstem.step_environment(action)
            os_sys.cerebrum.hippocampus.store(state, action, reward, next_state, terminated)
            state = next_state

def train_pfc_offline(os_sys, epochs=100, batch_size=64):
    print(f"Training Prefrontal Cortex (LNN) for {epochs} epochs...")
    optimizer = optim.Adam(os_sys.cerebrum.pfc.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        batch = os_sys.cerebrum.hippocampus.sample(batch_size)
        
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([[b[1]] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        
        # 前向傳播
        pred_next_states, _ = os_sys.cerebrum.pfc(states, actions)
        
        loss = criterion(pred_next_states, next_states)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return losses

if __name__ == "__main__":
    os_sys = EmbodiedAGI_OS()
    collect_data(os_sys, num_episodes=50)
    losses = train_pfc_offline(os_sys, epochs=100)
    
    # 保存世界模型
    torch.save(os_sys.cerebrum.pfc.state_dict(), "pfc_world_model.pth")
    print("PFC World Model saved as pfc_world_model.pth")
    
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.title("Prefrontal Cortex (LNN) World Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("world_model_loss.png")
    print("Loss curve saved to world_model_loss.png")
    print("學習曲線已儲存為 world_model_loss.png")

import os
import random
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Device Setup
DEVICE = torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(), 
            nn.Linear(128, 128), nn.Tanh(),     
            nn.Linear(128, act_dim)
        )
    def forward(self, x): 
        return self.net(x)

@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42
    
    ckpt_path: str = "CartPole-v1_medium.pt" 
    save_name: str = "D_CartPole_medium_mixed.npz"
    
    total_steps: int = 1_000_000 
    
    # 30% random
    random_ratio: float = 0.3  
    log_interval: int = 100_000

def load_behavior_model(ckpt_path, obs_dim, act_dim):
    model = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    print(f"Loading weights from: {ckpt_path}")
    try:
        # map_location을 사용하여 이전 학습 장소와 상관없이 현재 DEVICE로 로드하고 딕셔너리로 만든다.
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict) 
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        exit()
    model.eval()
    return model

def collect_dataset(cfg: Config):
    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy_model = load_behavior_model(cfg.ckpt_path, obs_dim, act_dim)
    
    obs, _ = env.reset()
    data = defaultdict(list)
    ep_returns = []
    ep_ret = 0.0

    print(f"Start collecting {cfg.total_steps} transitions (Random Ratio: {cfg.random_ratio})")
    
    for t in range(cfg.total_steps):
        if np.random.rand() > cfg.random_ratio:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = policy_model(obs_t)
            act = torch.argmax(logits, dim=1).item()
        else:
            act = env.action_space.sample()

        next_obs, r, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        
        # 데이터 저장
        data["obs"].append(obs)
        data["act"].append(act)
        data["rew"].append(r)
        data["obs2"].append(next_obs)
        data["done"].append(float(done)) 
        
        ep_ret += r
        obs = next_obs

        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if (t + 1) % cfg.log_interval == 0:
            avg_score = np.mean(ep_returns[-10:]) if ep_returns else 0.0
            print(f"{t+1} / {cfg.total_steps} | Recent Avg Score: {avg_score:.2f}")

    env.close()
    return data, ep_returns

if __name__ == "__main__":
    os.makedirs("CartPole", exist_ok=True)
    
    cfg = Config(
        env_id="CartPole-v1",
        ckpt_path="CartPole-v1_medium.pt",
        save_name="D_CartPole_medium_mixed.npz",
        total_steps=1_000_000,
        random_ratio = 0.3  

    )
    
    if not os.path.exists(cfg.ckpt_path):
        print(f"Warning: Checkpoint file '{cfg.ckpt_path}' not found.")
    else:
        d, ep_rets = collect_dataset(cfg)

        dataset = {
            "obs": np.array(d["obs"], dtype=np.float32),
            "act": np.array(d["act"], dtype=np.int64),
            "rew": np.array(d["rew"], dtype=np.float32),
            "obs2": np.array(d["obs2"], dtype=np.float32),
            "done": np.array(d["done"], dtype=np.float32),
            "episode_returns": np.array(ep_rets, dtype=np.float32)
        }

        np.savez_compressed(cfg.save_name, **dataset)

        print("-" * 40)
        print(f"Dataset saved: {cfg.save_name}")
        print(f"Total Transitions: {len(dataset['obs'])}")
        print(f"Avg Return: {np.mean(dataset['episode_returns']):.2f}")
        print("-" * 40)
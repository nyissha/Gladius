import os
import sys
import time
import random
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

# ===================== 0. Configuration & Setup =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Off-GLADIUS Pure Verification")
    
    # Environment & Data
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--data_path", type=str, default="D_CartPole_avg285.npz")
    parser.add_argument("--seed", type=int, default=42)
    
    # Training Hyperparameters
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=1024) # 배치 사이즈 키움 (Variance 감소)
    parser.add_argument("--eval_freq", type=int, default=1000)
    
    # Off-GLADIUS Specific Hyperparameters (Tuning)
    parser.add_argument("--lr_q", type=float, default=5e-5)    # Q는 천천히 학습 (안정성)
    parser.add_argument("--lr_zeta", type=float, default=1e-3) # Zeta는 빠르게 학습 (추정 정확도)
    parser.add_argument("--zeta_steps", type=int, default=10)   # Q 1번 업데이트 당 Zeta 업데이트 횟수
    
    parser.add_argument("--lam", type=float, default=0.5)      # Temperature (0.1 ~ 1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=0.5) # Gradient Clipping

    return parser.parse_args()

args = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Off-GLADIUS (Pure) on {DEVICE} | Seed: {args.seed}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===================== 1. Buffer =====================
class OfflineReplayBuffer:
    def __init__(self, obs, act, rew, obs2, done):
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        # Reward Scaling
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE) * 0.01
        self.obs2 = torch.tensor(obs2, dtype=torch.float32, device=DEVICE)
        self.done = torch.tensor(done, dtype=torch.float32, device=DEVICE)
        self.N = len(obs)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.N, batch_size)
        return {
            "obs": self.obs[idx], "act": self.act[idx],
            "rew": self.rew[idx], "obs2": self.obs2[idx], "done": self.done[idx]
        }

# ===================== 2. Networks =====================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ZetaNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Zeta(s, a) -> Scalar
        self.net = MLP(obs_dim + act_dim, 1)
        self.act_dim = act_dim
        
    def forward(self, obs, act):
        # act: (Batch,) -> One-hot (Batch, Act_Dim)
        a_onehot = F.one_hot(act.long(), num_classes=self.act_dim).float()
        x = torch.cat([obs, a_onehot], dim=-1)
        return self.net(x).squeeze(-1) # (Batch,)

def get_v_from_q(q_values, temperature=1.0):
    return temperature * torch.logsumexp(q_values / temperature, dim=1, keepdim=True)

# ===================== 3. Off-GLADIUS Agent =====================
class OfflineGladius:
    def __init__(self, obs_dim, act_dim, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.zeta_steps = args.zeta_steps
        self.grad_clip = args.grad_clip

        # 1. Initialize Q_theta2, Zeta_theta1
        self.q_net = MLP(obs_dim, act_dim).to(DEVICE)
        self.zeta_net = ZetaNet(obs_dim, act_dim).to(DEVICE)

        # Optimizers (Separate LRs)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr_q)
        self.zeta_optimizer = torch.optim.Adam(self.zeta_net.parameters(), lr=args.lr_zeta)

    def update(self, buffer, batch_size):
        # Algorithm 1: Loop t=1 to T
        
        # --- Ascent Step (Update Zeta multiple times for stability) ---
        zeta_loss_val = 0
        for _ in range(self.zeta_steps):
            b1 = buffer.sample(batch_size) # Sample B1 
            
            obs, act, obs2 = b1["obs"], b1["act"], b1["obs2"]
            
            with torch.no_grad():
                # Q is fixed in ascent step [cite: 356]
                next_q = self.q_net(obs2)
                next_v = get_v_from_q(next_q, self.lam).squeeze(-1) # V^Q(s')
            
            current_zeta = self.zeta_net(obs, act) # zeta(s, a)
            
            # Maximize Objective -> Minimize MSE(V, Zeta)
            # D_theta1 = sum (V(s') - zeta(s,a))^2
            zeta_loss = F.mse_loss(current_zeta, next_v)
            
            self.zeta_optimizer.zero_grad()
            zeta_loss.backward()
            nn.utils.clip_grad_norm_(self.zeta_net.parameters(), self.grad_clip)
            self.zeta_optimizer.step()
            zeta_loss_val = zeta_loss.item()

        # --- Descent Step (Update Q) ---
        b2 = buffer.sample(batch_size) # Sample B2 
        obs, act, rew, obs2, done = b2["obs"], b2["act"], b2["rew"], b2["obs2"], b2["done"]
        
        with torch.no_grad():
            fixed_zeta = self.zeta_net(obs, act)
        
        q_values = self.q_net(obs)
        current_q = q_values.gather(1, act.long().unsqueeze(-1)).squeeze(-1)
        
        next_q_values = self.q_net(obs2)
        next_v_values = get_v_from_q(next_q_values, self.lam).squeeze(-1)
        
        # 1. Bellman Operator Estimate: r + gamma * V^Q(s')
        target_op = rew + self.gamma * next_v_values * (1 - done)
        
        # 2. TD Squared Loss: (TQ - Q)^2
        td_loss = (target_op - current_q) ** 2
        
        # 3. Correction Term: gamma^2 * (V^Q(s') - zeta)^2
        correction = (self.gamma ** 2) * ((next_v_values - fixed_zeta) ** 2)
        
        # Total Loss (BE loss)
        q_loss = (td_loss - correction).mean()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.q_optimizer.step()

        return {"q_loss": q_loss.item(), "zeta_loss": zeta_loss_val}

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            if obs.ndim == 1: obs = obs.unsqueeze(0)
            q_values = self.q_net(obs)
            
            if deterministic:
                return q_values.argmax(dim=-1).item()
            else:
                probs = F.softmax(q_values / self.lam, dim=-1)
                return torch.distributions.Categorical(probs).sample().item()

# ===================== 4. Evaluation & Main =====================
@torch.no_grad()
def evaluate(env, agent, episodes=5):
    agent.q_net.eval()
    rets = []
    for _ in range(episodes):
        o, _ = env.reset()
        done, r_sum = False, 0.0
        while not done:
            o_tensor = torch.tensor(o, dtype=torch.float32, device=DEVICE)
            a = agent.select_action(o_tensor, deterministic=True)
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_sum += r
        rets.append(r_sum)
    agent.q_net.train()
    return float(np.mean(rets))

def main():
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return
        
    data = np.load(args.data_path)
    buffer = OfflineReplayBuffer(data["obs"], data["act"], data["rew"], data["obs2"], data["done"])
    
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    steps, rets = [], []
    print("===================== Start Off-GLADIUS (Pure) Verification =====================")
    
    set_seed(args.seed)
    agent = OfflineGladius(obs_dim, act_dim, args)
    
    start_time = time.time()
    for i in range(1, args.updates + 1):
        # buffer를 통째로 넘겨서 내부에서 B1, B2 샘플링
        logs = agent.update(buffer, args.batch_size)
        
        if i % args.eval_freq == 0:
            ret = evaluate(env, agent)
            steps.append(i)
            rets.append(ret)
            elapsed = (time.time() - start_time) / 60
            print(f"Step {i:5d} | Return: {ret:6.1f} | Q_Loss: {logs['q_loss']:.4f} | Zeta_Loss: {logs['zeta_loss']:.4f} | Time: {elapsed:.1f}m")

    env.close()
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rets, label='Off-GLADIUS')
    plt.axhline(y=500, color='r', linestyle='--', label='Max Reward')
    plt.xlabel("Updates")
    plt.ylabel("Return")
    plt.title(f"Off-GLADIUS Verification (lam={args.lam}, zeta_steps={args.zeta_steps})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_dir}/gladius_verify_{timestamp}.png")
    plt.show()

if __name__ == "__main__":
    main()
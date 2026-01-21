import os
import time
import random
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# ===================== 0. Configuration & Setup =====================
def parse_args():
    parser = argparse.ArgumentParser(description="SBEED Acrobot")
    
    # Environment & Data
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    parser.add_argument("--data_path", type=str, default="acrobot/D_acrobot_medium_mixed.npz")
    parser.add_argument("--seed", type=int, default=1)
    
    # Training Hyperparameters
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_freq", type=int, default=1000)
    
    # SBEED Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.004)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_beta", type=float, default=0.1, help="KL penalty coefficient (inverted)")
    parser.add_argument("--grad_clip", type=float, default=10.0)

    # Visualization
    parser.add_argument("--viz_mode", type=str, default="compare", choices=["sbeed_only", "compare"], 
                        help="Visualization mode: 'sbeed_only' or 'compare'")
    parser.add_argument("--baseline_path", type=str, default="acrobot/comparison_data.npz", 
                        help="Path to .npz file")
    
    return parser.parse_args()

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running SBEED on {DEVICE} | Env: {args.env} | Seed: {args.seed}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def kl_categorical(p: Categorical, q: Categorical):
    return torch.distributions.kl.kl_divergence(
        Categorical(logits=p.logits.detach()),
        Categorical(logits=q.logits.detach())
    )

# ===================== 1. Buffer =====================
class OfflineReplayBuffer:
    def __init__(self, obs, act, rew, obs2, done):
        # Raw Tensors
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE)
        self.obs2 = torch.tensor(obs2, dtype=torch.float32, device=DEVICE)
        self.done = torch.tensor(done, dtype=torch.float32, device=DEVICE)
        
        self.N = len(obs)
        print(f"[Buffer] Loaded transitions: {self.N}")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.N, batch_size)
        return {
            "obs":  self.obs[idx],
            "act":  self.act[idx],
            "rew":  self.rew[idx],
            "obs2": self.obs2[idx],
            "done": self.done[idx],
        }

# ===================== 2. Networks (Robust Version) =====================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
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

class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, temperature=2.0, logit_clip=10.0):
        super().__init__()
        self.backbone = MLP(obs_dim, act_dim)
        self.temperature = temperature
        self.logit_clip = logit_clip

    def logits(self, obs):
        z = self.backbone(obs)
        z = torch.clamp(z, -self.logit_clip, self.logit_clip)
        return z / self.temperature

    def dist(self, obs):
        return Categorical(logits=self.logits(obs))

    def log_prob(self, obs, act):
        return self.dist(obs).log_prob(act)

    def act_greedy(self, obs):
        return torch.argmax(self.logits(obs), dim=-1)

class RhoNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1)
        self.act_dim = act_dim # Saved for One-hot encoding
        
    def forward(self, obs, act):
        a_onehot = F.one_hot(act, num_classes=self.act_dim).float()
        return self.net(torch.cat([obs, a_onehot], dim=-1)).squeeze(-1)

# ===================== 3. SBEED Agent =====================
class OfflineSBEED:
    def __init__(self, obs_dim, act_dim, args):
        self.gamma = args.gamma
        self.eta = args.eta
        self.lam = args.lam
        self.kl_beta = args.kl_beta
        self.grad_clip = args.grad_clip
        self.act_dim = act_dim

        self.pi  = DiscretePolicy(obs_dim, act_dim).to(DEVICE)
        self.v   = MLP(obs_dim, 1).to(DEVICE)
        self.rho = RhoNet(obs_dim, act_dim).to(DEVICE)

        self.opt_pi  = torch.optim.Adam(self.pi.parameters(), lr=args.lr)
        self.opt_v   = torch.optim.Adam(self.v.parameters(),  lr=args.lr)
        self.opt_rho = torch.optim.Adam(self.rho.parameters(),lr=args.lr)

    def update(self, b):
        obs, act, rew, obs2, done = b["obs"], b["act"], b["rew"], b["obs2"], b["done"]

        # --- 1. Target & Old Policy ---
        with torch.no_grad():
            dist_old = self.pi.dist(obs)
            logp_old = self.pi.log_prob(obs, act)
            
            v_next = self.v(obs2).squeeze(-1)
            delta = rew - self.lam * logp_old + self.gamma * (1.0 - done) * v_next

        # --- 2. Rho Regression ---
        rho_pred = self.rho(obs, act)
        loss_rho = F.mse_loss(rho_pred, delta.detach())
        
        self.opt_rho.zero_grad()
        loss_rho.backward()
        utils.clip_grad_norm_(self.rho.parameters(), self.grad_clip)
        self.opt_rho.step()

        # --- 3. V Update ---
        v0_now = self.v(obs).squeeze(-1)
        mse_td = ((delta.detach() - v0_now)**2).mean()
        mse_dual = ((delta.detach() - rho_pred.detach())**2).mean()
        
        loss_v = mse_td - self.eta * mse_dual
        
        self.opt_v.zero_grad()
        loss_v.backward()
        utils.clip_grad_norm_(self.v.parameters(), self.grad_clip)
        self.opt_v.step()

        # --- 4. Policy Update ---
        dist_new = self.pi.dist(obs)
        logp_new = dist_new.log_prob(act)
        
        with torch.no_grad():
            adv = (1.0 - self.eta) * delta.detach() + self.eta * rho_pred.detach() - v0_now.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss = -2.0 * (self.lam * adv * logp_new).mean()
        loss_pi = pg_loss

        kl = kl_categorical(dist_new, dist_old).mean()
        loss_pi = pg_loss + (1.0 / self.kl_beta) * kl

        self.opt_pi.zero_grad()
        loss_pi.backward()
        utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip)
        self.opt_pi.step()

        return {"v": loss_v.item(), "ent": dist_new.entropy().mean().item()}

# ===================== 4. Evaluation & Visualization =====================
@torch.no_grad()
def evaluate(env, agent, episodes=5):
    agent.pi.eval()
    rets = []
    for _ in range(episodes):
        o, _ = env.reset()
        done, r_sum = False, 0.0
        while not done:
            a = agent.pi.act_greedy(torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_sum += r
        rets.append(r_sum)
    agent.pi.train()
    return float(np.mean(rets))

def visualize_results(sbeed_steps, sbeed_rets, args):
    plt.figure(figsize=(10, 6))
    
    plt.plot(sbeed_steps, sbeed_rets, label='SBEED', color='blue', linewidth=2)
   
    if args.viz_mode == "compare":
        if os.path.exists(args.baseline_path):
            try:
                base_data = np.load(args.baseline_path)
                if 'expert_steps' in base_data and 'expert_rewards' in base_data:
                    plt.plot(base_data['expert_steps'], base_data['expert_rewards'], 
                             label='PPO Expert', color='green', linestyle='--', alpha=0.7)
              
                if 'pi_b_score' in base_data:
                    pi_b = base_data['pi_b_score']
                    if hasattr(pi_b, 'item'): pi_b = pi_b.item()
                    plt.axhline(pi_b, color='orange', linestyle='-.', label=f'Mixed Score ({pi_b:.1f})')
                    
            except Exception as e:
                print(f"[Warn] Failed to load baseline data: {e}")
        else:
            print(f"[Warn] Baseline file not found at {args.baseline_path}. Plotting SBEED only.")

    plt.axhline(-100, color='red', linestyle=':', label="Solved Threshold")
    plt.title(f"SBEED Performance on {args.env}")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"sbeed_{args.env}_{args.viz_mode}.png"
    plt.savefig(save_name, dpi=300)
    print(f"[Vis] Learning curve saved to '{save_name}'")

def main():
    set_seed(args.seed)
    
    if not os.path.exists(args.data_path):
        print(f"[Warn] {args.data_path} not found.")
        exit()
    else:
        data = np.load(args.data_path)

    buffer = OfflineReplayBuffer(data["obs"], data["act"], data["rew"], data["obs2"], data["done"])
    env = gym.make(args.env)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = OfflineSBEED(obs_dim, act_dim, args)
    
    steps, rets = [], []
    start_time = time.time()
    
    print("===================== Start SBEED Training =====================")
    for i in range(1, args.updates + 1):
        batch = buffer.sample(args.batch_size)
        logs = agent.update(batch)
        
        if i % args.eval_freq == 0:
            ret = evaluate(env, agent)
            steps.append(i)
            rets.append(ret)
            
            elapsed = (time.time() - start_time) / 60
            print(f"Step {i:5d} | Return: {ret:6.1f} | V_Loss: {logs['v']:.4f} | Ent: {logs['ent']:.3f} | Time: {elapsed:.1f}m")

    visualize_results(steps, rets, args)
    env.close()

if __name__ == "__main__":
    main()
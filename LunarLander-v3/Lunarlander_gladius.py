import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torch.optim.lr_scheduler import CosineAnnealingLR  # 제거: 고정 lr 사용

# ===================== 0. Configuration & Setup =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Off-GLADIUS Pure Verification (LunarLander)")

    # Environment & Data
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--data_path", type=str, default="LunarLander-v3/D_LunarLander_medium_mixed_high_100_9_1.npz")
    parser.add_argument("--seed", type=int, default=42)

    # Training Hyperparameters
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_freq", type=int, default=1000)

    # Off-GLADIUS Specific Hyperparameters
    parser.add_argument("--lr_q", type=float, default=1e-4)
    parser.add_argument("--lr_zeta", type=float, default=1e-3)
    parser.add_argument("--zeta_steps", type=int, default=20)
    parser.add_argument("--lam", type=float, default=0.1) 

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=10.0)

    parser.add_argument("--iter", type=int, default=1)

    # Reward scaling
    parser.add_argument("--reward_scale", type=float, default=0.01) 

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
    def __init__(self, data_dict, reward_scale):
        obs = data_dict["obs"]
        act = data_dict["act"]
        rew = data_dict["rew"]
        obs2 = data_dict["obs2"]    
        raw_done = data_dict["done"] # 섞여있는 done

        # [핵심] LunarLander Heuristic: 보상 크기로 구분
        # 보상의 절대값이 50보다 크면 '진짜 끝(Terminated)'으로 간주
        # 50은 넉넉하게 잡은 기준값입니다 (보통 성공/실패는 100점 내외이므로)
        is_terminated = (np.abs(rew) > 50.0) & (raw_done == 1.0)
        
        # truncated는 done이 1인데 terminated가 아닌 경우 (사실 이 변수는 학습엔 안 쓰임)
        is_truncated = (raw_done == 1.0) & (np.abs(rew) <= 50.0)

        # 이제 우리가 필요한 건 'Terminated' 정보뿐입니다.
        # 시간 초과(truncated)된 데이터는 done=0으로 취급하여 V(s')를 계산하게 만듭니다.
        final_done = is_terminated.astype(np.float32)
        '''
        if "done" in data_dict:
            done = data_dict["done"]
        elif ("terminated" in data_dict) and ("truncated" in data_dict):
            done = np.logical_or(
                data_dict["terminated"].astype(bool),
                data_dict["truncated"].astype(bool)
            ).astype(np.float32)
        else:
            raise KeyError("Dataset must contain either 'done' or both 'terminated' and 'truncated'.")
        '''
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE) * reward_scale
        self.obs2 = torch.tensor(obs2, dtype=torch.float32, device=DEVICE)
        self.done = torch.tensor(final_done, dtype=torch.float32, device=DEVICE)
        self.N = len(obs)
        print(f"[Buffer] Loaded transitions: {self.N}")

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
        self.output_layer = nn.Linear(last, out_dim)
        layers.append(self.output_layer)
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if m == self.output_layer: 
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class ZetaNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1)
        self.act_dim = act_dim

    def forward(self, obs, act):
        a_onehot = F.one_hot(act.long(), num_classes=self.act_dim).float()
        x = torch.cat([obs, a_onehot], dim=-1)
        return self.net(x).squeeze(-1)

def get_v_from_q(q_values, temperature=1.0):
    return temperature * torch.logsumexp(q_values / temperature, dim=1, keepdim=True)

# ===================== 3. Off-GLADIUS Agent =====================
class OfflineGladius:
    def __init__(self, obs_dim, act_dim, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.zeta_steps = args.zeta_steps
        self.grad_clip = args.grad_clip
        self.tau = args.tau

        self.q_net = MLP(obs_dim, act_dim).to(DEVICE)
        self.zeta_net = ZetaNet(obs_dim, act_dim).to(DEVICE)

        #self.target_q_net = MLP(obs_dim, act_dim).to(DEVICE)
       # self.target_q_net.load_state_dict(self.q_net.state_dict())
       # self.target_q_net.eval()

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr_q)
        self.zeta_optimizer = torch.optim.Adam(self.zeta_net.parameters(), lr=args.lr_zeta)

        self.q_scheduler = CosineAnnealingLR(self.q_optimizer, T_max=args.updates)
        self.zeta_scheduler = CosineAnnealingLR(self.zeta_optimizer, T_max=args.updates*args.zeta_steps)
        # 스케줄러 제거: 고정 lr 사용d

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, buffer, batch_size):
        # --- Ascent Step (Update Zeta multiple times for stability) ---
        zeta_loss_val = 0
        for _ in range(self.zeta_steps):
            b1 = buffer.sample(batch_size)

            obs, act, obs2 = b1["obs"], b1["act"], b1["obs2"]

            with torch.no_grad():
                next_q = self.q_net(obs2)
                next_v = get_v_from_q(next_q, self.lam).squeeze(-1)

            current_zeta = self.zeta_net(obs, act)

            zeta_loss = F.mse_loss(current_zeta, next_v)

            self.zeta_optimizer.zero_grad()
            zeta_loss.backward()
            nn.utils.clip_grad_norm_(self.zeta_net.parameters(), self.grad_clip)
            self.zeta_optimizer.step()
            zeta_loss_val = zeta_loss.item()
            self.zeta_scheduler.step()


        # --- Descent Step (Update Q) ---
        b2 = buffer.sample(batch_size)
        obs, act, rew, obs2, done = b2["obs"], b2["act"], b2["rew"], b2["obs2"], b2["done"]

        with torch.no_grad():
            fixed_zeta = self.zeta_net(obs, act)
            #next_q_target = self.target_q_net(obs2)
            #next_v_target = get_v_from_q(next_q_target, self.lam).squeeze(-1)

        q_values_current = self.q_net(obs)
        current_q = q_values_current.gather(1, act.long().unsqueeze(-1)).squeeze(-1)

        next_q_current = self.q_net(obs2)
        next_v_current = get_v_from_q(next_q_current, self.lam).squeeze(-1)

        # 1. Bellman Operator Estimate: r + gamma * V^Q(s')
        target_op = rew + self.gamma * next_v_current * (1 - done)

        # 2. TD Squared Loss: (TQ - Q)^2
        td_loss = (target_op - current_q) ** 2

        # 3. Correction Term: gamma^2 * (V^Q(s') - zeta)^2
        correction = (self.gamma ** 2) * ((next_v_current - fixed_zeta) ** 2)

        # Total Loss (BE loss)
        q_loss = (td_loss - correction).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.q_optimizer.step()
        self.q_scheduler.step()
        #self.soft_update(self.target_q_net, self.q_net)
        return {"q_loss": q_loss.item(), "zeta_loss": zeta_loss_val}

    def select_action(self, obs, eval):
        with torch.no_grad():
            if obs.ndim == 1: obs = obs.unsqueeze(0)

            q_values = self.q_net(obs)
            if eval:
                return torch.argmax(q_values, dim=-1).item()

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
            a = agent.select_action(o_tensor, eval=True)
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_sum += r
        rets.append(r_sum)
    agent.q_net.train()
    return float(np.mean(rets))

def visualize_results(Gladius_steps : list[int], Gladius_rets : list[float], args):
    plt.figure(figsize=(10, 6))

    try:
        np_steps = np.array(Gladius_steps).reshape(args.iter, -1)
        np_rets = np.array(Gladius_rets).reshape(args.iter, -1)
    except ValueError:
        print("[Error] 데이터 크기가 맞지 않아 reshape 할 수 없습니다. 실험 도중 중단되었을 수 있습니다.")
        return

    steps_x = np_steps[0]
    avg_rets = np_rets.mean(axis=0)

    plt.plot(steps_x, avg_rets, label='Gladius (mean)', color='red', linewidth=2)
    plt.title(f"Gladius on {args.env} | lr_q={args.lr_q:.0e}, lr_zeta={args.lr_zeta:.0e}, lam={args.lam}")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{save_dir}/Gladius_{args.env}_lrq{args.lr_q:.0e}_lrz{args.lr_zeta:.0e}_{timestamp}.png"

    plt.savefig(save_name, dpi=300)
    print(f"[Vis] Learning curve saved to '{save_name}'")
    plt.show()


def main():
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return

    data = np.load(args.data_path)
    data_dict = {k: data[k] for k in data.files}
    buffer = OfflineReplayBuffer(data_dict, args.reward_scale)

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"[Env] obs_dim={obs_dim}, act_dim={act_dim}")

    steps, rets = [], []
    q_loss, zeta_loss = [], []
    print("===================== Start Off-GLADIUS (Pure) Verification =====================")
    for iter in range(args.iter):
        seed = args.seed + iter
        set_seed(seed)
        agent = OfflineGladius(obs_dim, act_dim, args)

        print(f"-- Run {iter+1}/{args.iter}")

        start_time = time.time()
        for i in range(1, args.updates + 1):
            logs = agent.update(buffer, args.batch_size)

            if i % args.eval_freq == 0:
                ret = evaluate(env, agent)
                steps.append(i)
                rets.append(ret)
                q_loss.append(logs['q_loss'])
                zeta_loss.append(logs['zeta_loss'])
                elapsed = (time.time() - start_time) / 60
                print(f"Step {i:5d} | Return: {ret:6.1f} | Q_Loss: {logs['q_loss']:.4f} | Zeta_Loss: {logs['zeta_loss']:.4f} | Time: {elapsed:.1f}m")

    visualize_results(steps, rets, args)
    env.close()

if __name__ == "__main__":
    main()

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
import torch.nn.utils as utils
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime

# ===================== 0. Configuration & Setup =====================
def parse_args():
    parser = argparse.ArgumentParser(description="SBEED CartPole")
    
    # Environment & Data
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--data_path", type=str, default="D_CartPole_avg285.npz")
    parser.add_argument("--seed", type=int, default=42)
    
    # Training Hyperparameters
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_freq", type=int, default=1000)
    
    # SBEED Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.004)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_beta", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0)


    parser.add_argument("--rho_steps", type=int, default=5)
    # n=1 : convergence under 10000~20000 step, n=5 : convergence exactly 10000 step
    # n=10 : convergence exactly 10000 step, n=20 : convergence exactly 10000 step
    parser.add_argument("--iter", type=int, default=5) # run n times and plot mean.

    return parser.parse_args()

args = parse_args()

class StochasticTransitionWrapper(gym.Wrapper):
    def __init__(self, env, noise_scale=0.1, prob_sticky=0.0):
        super().__init__(env)
        self.noise_scale = noise_scale  # 상태 전이 노이즈 크기 (Gaussian)
        self.prob_sticky = prob_sticky  # 엉뚱한 행동을 할 확률 (Sticky Action)

    def step(self, action):
        # 1. Sticky Action: 일정 확률로 이전 행동이나 랜덤 행동 반복
        if np.random.rand() < self.prob_sticky:
            action = self.env.action_space.sample()
            
        # 2. 원래 환경의 전이
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. 상태 전이 노이즈 주입 (Transition Noise)
        # obs가 결정론적이지 않고 확률적으로 퍼지게 만듦
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=obs.shape)
        obs = obs + noise
        
        return obs, reward, terminated, truncated, info

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
        Categorical(logits=p.logits),
        Categorical(logits=q.logits.detach())
    )

# ===================== 1. Buffer =====================
class OfflineReplayBuffer:
    def __init__(self, obs, act, rew, obs2, done):
        # Raw Tensors
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE) / 100
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
        z = self.backbone(obs) #action dim의 logits 출력
        z = torch.clamp(z, -self.logit_clip, self.logit_clip)
        return z / self.temperature #softmax의 작동이 e^logit 으로 이루어지므로 온도를 높이는건 엔트로피를 높이는것과 비슷한 역할을 한다

    def dist(self, obs):
        return Categorical(logits=self.logits(obs)) #logits을 함수로 만들어 전처리 후 사용하므로 dist도 따로 함수로 사용

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
        #보통 바깥 배열이 batch size고 안쪽 배열이 실제 act, obs dim이므로 dim=-1
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

    def update(self, buffer, batch_size):

        b = buffer.sample(batch_size)
        obs, act, rew, obs2, done = b["obs"], b["act"], b["rew"], b["obs2"], b["done"]

        # --- 1. Target & Old Policy ---
        with torch.no_grad():
            dist_old = self.pi.dist(obs)
            logp_old = self.pi.log_prob(obs, act)
            
            v_next = self.v(obs2).squeeze(-1).detach()
            delta = rew - self.lam * logp_old + self.gamma * (1.0 - done) * v_next

            # --- 2. Rho Regression ---
        for _ in range(args.rho_steps):
            rho_pred = self.rho(obs, act)
            loss_rho = F.mse_loss(rho_pred, delta.detach())
            
            self.opt_rho.zero_grad()
            loss_rho.backward()
            utils.clip_grad_norm_(self.rho.parameters(), self.grad_clip)
            self.opt_rho.step()


        # --- 3. V Update ---
       
        rho_pred = self.rho(obs, act)
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

        pg_loss = -2.0 * (self.lam * adv * logp_new).mean() # (A * 로그확률)의 평균
        loss_pi = pg_loss

        #kl = kl_categorical(dist_new, dist_old).mean()
        #loss_pi = pg_loss + (1.0 / self.kl_beta) * kl

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
        o, _ = env.reset() # observation, info
        done, r_sum = False, 0.0
        while not done:
            a = agent.pi.act_greedy(torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
            # torch의 layer는 데이터를 배치단위로만 받는다. 즉 obs의 형태인 1차원 배열을 (1, N)으로 바꾸기 위해 unsqueeze(0)이 필요하다.
            # 이후 item()으로 순수한 파이썬 숫자를 빼낸다. tensor to float이고, 단일 스칼라를 담은 값에만 사용 가능하다.
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_sum += r
        rets.append(r_sum)
    agent.pi.train() # .eval()의 반대. 다시 train모드로 전환하고 dropout/batch Normalization을 실행한다
    # train/eval과 torch.no_grad()는 다른 개념이다. 전자는 레이어의 동작 방식을 바꾸고 후자는 미분 계산 여부(메모리)를 다룬다.
    return float(np.mean(rets))  #sum(rets)/len(rets)이 조금 더 빠르긴 하다.
    #파이썬 mean과 np.mean의 성능차이가 존재하기에 np.mean 후 전환한다. 근데 데이터가 작을 때는 파이썬이 빠를 수 있는데 지금이 그런 경우 아닌가?

def visualize_results(sbeed_steps : list[int], sbeed_rets : list[float], args):
    plt.figure(figsize=(10, 6)) #canvas 생성, size는 인치 단위.

    try:
        np_steps = np.array(sbeed_steps).reshape(args.iter, -1)
        np_rets = np.array(sbeed_rets).reshape(args.iter, -1)
    except ValueError:
        print("[Error] 데이터 크기가 맞지 않아 reshape 할 수 없습니다. 실험 도중 중단되었을 수 있습니다.")
        return
    
    steps_x = np_steps[0]
    
    for i in range(args.iter):
        plt.plot(steps_x, np_rets[i], label='SBEED', color='blue', alpha=0.15)
    
    avg_rets = np_rets.mean(axis=0)
    std_rets = np_rets.std(axis=0)
    sum_rets = (avg_rets + std_rets).clip(max=500)
    
    plt.plot(steps_x, avg_rets, label='SBEED (mean)', color='red', linewidth=2)
    plt.fill_between(steps_x, avg_rets - std_rets, sum_rets, color='blue', alpha=0.05)

    plt.title(f"SBEED Performance on {args.env}")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() # 여백 정리
    
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #년월일시분초
    save_name = f"{save_dir}/sbeed_{args.env}_{timestamp}.png"
    file_name = f"{save_dir}/sbeed_{args.env}.npz"

    plt.savefig(save_name, dpi=300)
    np.savez_compressed(file_name, steps=steps_x, returns=avg_rets)

    print(f"[Save] Results saved to '{file_name}' (Shape: {avg_rets.shape})")
    print(f"[Vis] Learning curve saved to '{save_name}'")

    plt.show() #show이후에는 canvas를 비워버리므로 savefig를 먼저 해야한다.

def main():   
    if not os.path.exists(args.data_path):
        print(f"[Warn] {args.data_path} not found.")
        sys.exit() #함수가 아닌 프로세스 전체 종료
    else:
        data = np.load(args.data_path)

    buffer = OfflineReplayBuffer(data["obs"], data["act"], data["rew"], data["obs2"], data["done"])
    env = gym.make(args.env)
    #env = StochasticTransitionWrapper(env, noise_scale=0.1, prob_sticky=0.1)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    steps, rets = [], []
    
    print("===================== Start SBEED Training =====================")
    for iter in range(args.iter):
        current_seed = args.seed + iter
        set_seed(current_seed)
        agent = OfflineSBEED(obs_dim, act_dim, args)

        print(f"-- Run {iter+1}/{args.iter}")

        start_time = time.time()
        for i in range(1, args.updates + 1):
            logs = agent.update(buffer, args.batch_size)
            
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
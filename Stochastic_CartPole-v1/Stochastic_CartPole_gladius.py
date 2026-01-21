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
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===================== 0. Configuration & Setup =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Off-GLADIUS Pure Verification")
    
    # Environment & Data
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--data_path", type=str, default="CartPole-v1/D_CartPole_avg285.npz")
    parser.add_argument("--seed", type=int, default=42)
    
    # Training Hyperparameters
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=128) # 배치 사이즈 키움 (Variance 감소)
    parser.add_argument("--eval_freq", type=int, default=1000)
    
    # Off-GLADIUS Specific Hyperparameters (Tuning)
    parser.add_argument("--lr_q", type=float, default=3e-3) # Q는 천천히 학습 (안정성)

    # different parameters compare to Gladius
    parser.add_argument("--lr_zeta", type=float, default=1e-3) # Zeta는 빠르게 학습 (추정 정확도)
    parser.add_argument("--zeta_steps", type=int, default=20)   # Q 1번 업데이트 당 Zeta 업데이트 횟수
    # n=1 : divergence, n=5 : divergence, n=10 : convergence under 10000 update, n=20 
    parser.add_argument("--lam", type=float, default=0.5)   # Temperature (0.1 ~ 1.0)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=10.0) # Gradient Clipping

    parser.add_argument("--iter", type=int, default=1)
    return parser.parse_args()

class StochasticTransitionWrapper(gym.Wrapper):
    def __init__(self, env, noise_scale=0.1, prob_sticky=0.1):
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


args = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Off-GLADIUS (Pure) on {DEVICE} | Seed: {args.seed}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

## def weightInitialization

# ===================== 1. Buffer =====================
class OfflineReplayBuffer:
    def __init__(self, obs, act, rew, obs2, done):
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        # Reward Scaling
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE) / 100
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
    def __init__(self, in_dim, out_dim, hidden=(64, 64)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self): #_는 protected 맴버라는 표기 관습
        for m in self.modules(): # torch 내부의 모든 구성 요소 (layer, sub-module)를 재귀적으로 탐색해서 반환
                                # model.children()은 direct sub-module만 반환하는 반면 modules()는 모든 자손을 반환
            if isinstance(m, nn.Linear): #layer가 nn.Linear인 경우에 True 
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) 
                # 신경망을 역전파하는 경우 가중치가 1보다 큰 경우 곱해지며 발산, 1보다 작은 경우 곱해지며 소실된다.  
                # orthogonal matrix의 경우 x와 곱해져도 x의 크기는 보존된다. 즉 horizon을 거쳐도 발산/소실 문제가 없다
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class ZetaNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Zeta(s, a) -> Scalar
        self.net = MLP(obs_dim + act_dim, 1)
        self.act_dim = act_dim
        
    def forward(self, obs, act): # 객체를 함수처럼 실행시키는 경우 내부에서 __call__ 메서드가 호출된다.
                                #nn.Moduel의 __call__은 forward를 실행시킨다.
        # act: (Batch,) -> One-hot (Batch, Act_Dim)
        a_onehot = F.one_hot(act.long(), num_classes=self.act_dim).float()
        x = torch.cat([obs, a_onehot], dim=-1) #(B, obs_dim), (B, act_dim) -> (B, obs+act)
        return self.net(x).squeeze(-1) # (Batch, 1) -> (Batch,)

def get_v_from_q(q_values, temperature=1.0): #입력 차원 (B, A) // Lazy Eval
    return temperature * torch.logsumexp(q_values / temperature, dim=1, keepdim=True)
    #dim1 즉 action 차원에서 더해서 action 개념을 없애고 상태별 Q값만 남긴다. Keepdim=True로 (B, 1) 형상 유지.

# ===================== 3. Off-GLADIUS Agent =====================
class OfflineGladius:
    def __init__(self, obs_dim, act_dim, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.zeta_steps = args.zeta_steps
        self.grad_clip = args.grad_clip
        self.tau = args.tau

        # 1. Initialize Q_theta2, Zeta_theta1
        self.q_net = MLP(obs_dim, act_dim).to(DEVICE)
        self.zeta_net = ZetaNet(obs_dim, act_dim).to(DEVICE)

        self.target_q_net = MLP(obs_dim, act_dim).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # 가중치 복사
        self.target_q_net.eval() # 학습 모드 끄기 (업데이트 안 함)
        # Optimizers (Separate LRs)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr_q)
        self.zeta_optimizer = torch.optim.Adam(self.zeta_net.parameters(), lr=args.lr_zeta)

        self.q_scheduler = CosineAnnealingLR(self.q_optimizer, T_max=args.updates)
        self.zeta_scheduler = CosineAnnealingLR(self.zeta_optimizer, T_max=args.updates*args.zeta_steps)

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            #zip은 두 개 이상의 리스트를 엮어서 튜플로 만든다
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, buffer, batch_size):
        # Algorithm 1: Loop t=1 to T
        
        # --- Ascent Step (Update Zeta multiple times for stability) ---
        zeta_loss_val = 0
        for _ in range(self.zeta_steps):
            b1 = buffer.sample(batch_size) # Sample B1 
            
            obs, act, obs2 = b1["obs"], b1["act"], b1["obs2"]
            
            with torch.no_grad():
                next_q = self.q_net(obs2)
                next_v = get_v_from_q(next_q, self.lam).squeeze(-1) 
            
            current_zeta = self.zeta_net(obs, act) # zeta(s, a)

            zeta_loss = F.mse_loss(current_zeta, next_v)
            
            self.zeta_optimizer.zero_grad()
            zeta_loss.backward()
            nn.utils.clip_grad_norm_(self.zeta_net.parameters(), self.grad_clip)
            self.zeta_optimizer.step()
            zeta_loss_val = zeta_loss.item()
            self.zeta_scheduler.step()


        # --- Descent Step (Update Q) ---
        b2 = buffer.sample(batch_size) # Sample B2 
        obs, act, rew, obs2, done = b2["obs"], b2["act"], b2["rew"], b2["obs2"], b2["done"]
        
        with torch.no_grad():
            fixed_zeta = self.zeta_net(obs, act)
            next_q_target = self.target_q_net(obs2)
            next_v_target = get_v_from_q(next_q_target, self.lam).squeeze(-1)        

        q_values_current = self.q_net(obs)
        current_q = q_values_current.gather(1, act.long().unsqueeze(-1)).squeeze(-1)

        next_q_current = self.q_net(obs2)
        next_v_current = get_v_from_q(next_q_current, self.lam).squeeze(-1)
        
        # 1. Bellman Operator Estimate: r + gamma * V^Q(s')
        target_op = rew + self.gamma * next_v_target * (1 - done)
        
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
        self.soft_update(self.target_q_net, self.q_net)
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
def visualize_results(Gladius_steps : list[int], Gladius_rets : list[float], q_loss, zeta_loss, args):
    plt.figure(figsize=(10, 6)) #canvas 생성, size는 인치 단위.

    try:
        np_steps = np.array(Gladius_steps).reshape(args.iter, -1)
        np_rets = np.array(Gladius_rets).reshape(args.iter, -1)
    except ValueError:
        print("[Error] 데이터 크기가 맞지 않아 reshape 할 수 없습니다. 실험 도중 중단되었을 수 있습니다.")
        return
    
    steps_x = np_steps[0]
    
    #for i in range(args.iter):
       #plt.plot(steps_x, np_rets[i], label='Gladius', color='blue', alpha=0.15)
    
    avg_rets = np_rets.mean(axis=0)
    #std_rets = np_rets.std(axis=0)
    #clip_rets = (avg_rets + std_rets).clip(max=500)
    
    data = np.load("results/sbeed_CartPole-v1.npz")
    limit_index = int(args.updates // args.eval_freq)
    
    plt.subplot(1, 2, 1)
    plt.plot(data['steps'][:limit_index], data['returns'][:limit_index], label='SBEED (mean)', color='green', linewidth=2)
    plt.plot(steps_x, avg_rets, label='Gladius (mean)', color='red', linewidth=2)
    #plt.fill_between(steps_x, avg_rets - std_rets, clip_rets, color='blue', alpha=0.05)

    plt.title(f"Gladius Performance on {args.env}")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() # 여백 정리

    plt.subplot(1, 2, 2)
    plt.plot(steps_x, q_loss, label='q_loss')
    plt.plot(steps_x, zeta_loss, label='zeta_loss')
    plt.title("Loss")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Loss")
    plt.legend(loc='lower right')
    plt.tight_layout() # 여백 정리

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #년월일시분초
    save_name = f"{save_dir}/Stochastic_Gladius_{args.env}_{timestamp}.png"
    file_name = f"{save_dir}/Stochastic_Gladius_{args.env}_{timestamp}.npz"

    plt.savefig(save_name, dpi=300)
    np.savez_compressed(file_name, steps=steps_x, returns=avg_rets)

    print(f"[Save] Results saved to '{file_name}' (Shape: {avg_rets.shape})")
    print(f"[Vis] Learning curve saved to '{save_name}'")

    plt.show() #show이후에는 canvas를 비워버리므로 savefig를 먼저 해야한다.


def main():
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return
        
    data = np.load(args.data_path)
    buffer = OfflineReplayBuffer(data["obs"], data["act"], data["rew"], data["obs2"], data["done"])
    
    env = gym.make(args.env)
    env = StochasticTransitionWrapper(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
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
            # buffer를 통째로 넘겨서 내부에서 B1, B2 샘플링
            logs = agent.update(buffer, args.batch_size)
            
            if i % args.eval_freq == 0:
                ret = evaluate(env, agent)
                steps.append(i)
                rets.append(ret)
                q_loss.append(logs['q_loss'])
                zeta_loss.append(logs['zeta_loss'])
                elapsed = (time.time() - start_time) / 60
                print(f"Step {i:5d} | Return: {ret:6.1f} | Q_Loss: {logs['q_loss']:.4f} | Zeta_Loss: {logs['zeta_loss']:.4f} | Time: {elapsed:.1f}m")

    visualize_results(steps, rets, q_loss, zeta_loss, args)
    env.close()

if __name__ == "__main__":
    main()
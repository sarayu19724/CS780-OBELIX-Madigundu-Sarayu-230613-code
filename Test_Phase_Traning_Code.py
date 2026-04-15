import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dataclasses import dataclass
import importlib.util

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
DIRS    = ["FL2", "FL1", "FC ", "FR1", "FR2", "SR ", "B  ", "SL "]



def print_sensor_state(obs):
    near = list(map(int, obs[0:8]))
    far  = list(map(int, obs[8:16]))
    ir   = int(obs[16])
    print("  NEAR → " + " ".join(f"{d}:{near[i]}" for i, d in enumerate(DIRS)))
    print("  FAR  → " + " ".join(f"{d}:{far[i]}"  for i, d in enumerate(DIRS)))
    print(f"  IR   → {ir}")



class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
        )
        self.value_stream     = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, n_actions)

    def forward(self, x):
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=-1, keepdim=True))



@dataclass
class Transition:
    s: np.ndarray; a: int; r: float; s2: np.ndarray; done: bool



class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity; self.alpha = alpha
        self.buffer = []; self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, transition, priority=None):
        max_p = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_p
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        p = self.priorities[:n] ** self.alpha; p /= p.sum()
        idx   = np.random.choice(n, batch_size, replace=False, p=p)
        batch = [self.buffer[i] for i in idx]
        weights = (n * p[idx]) ** (-beta); weights /= weights.max()
        s  = np.stack([b.s    for b in batch])
        a  = np.array([b.a    for b in batch])
        r  = np.array([b.r    for b in batch])
        s2 = np.stack([b.s2   for b in batch])
        d  = np.array([b.done for b in batch])
        return s, a, r, s2, d, idx, weights.astype(np.float32)

    def update_priorities(self, idx, td_errors, eps=1e-6):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = abs(e) + eps

    def __len__(self):
        return len(self.buffer)



class NStepBuffer:
    def __init__(self, n=3, gamma=0.99):
        self.n = n; self.gamma = gamma
        self.buf = deque(maxlen=n)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
        if len(self.buf) < self.n and not done:
            return None
        G, s_n, done_n = 0.0, self.buf[-1][3], self.buf[-1][4]
        for _, _, ri, _, di in reversed(self.buf):
            G = ri + self.gamma * G * (1 - di)
        s0, a0 = self.buf[0][0], self.buf[0][1]
        return Transition(s0, a0, G, s_n, done_n)

    def flush(self, replay):
        while self.buf:
            G, s_n, done_n = 0.0, self.buf[-1][3], self.buf[-1][4]
            for _, _, ri, _, di in reversed(self.buf):
                G = ri + self.gamma * G * (1 - di)
            s0, a0 = self.buf[0][0], self.buf[0][1]
            replay.add(Transition(s0, a0, G, s_n, done_n))
            self.buf.popleft()



def import_obelix(path):
    spec   = importlib.util.spec_from_file_location("obelix_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OBELIX

def soft_update(target, online, tau=0.005):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)



def main():
    OBELIX = import_obelix("obelix.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net      = DQN()
    target_net = DQN()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    LR            = 5e-4
    GAMMA         = 0.99
    BATCH         = 256
    EPISODES      = 1000
    MAX_STEPS     = 1000
    N_STEP        = 3
    TAU           = 0.005
    BETA_START    = 0.4
    BETA_FRAMES   = EPISODES * MAX_STEPS
    GRAD_CLIP     = 10.0

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPISODES)
    replay    = PrioritizedReplayBuffer(capacity=100_000, alpha=0.6)
    n_buf     = NStepBuffer(n=N_STEP, gamma=GAMMA)

    epsilon       = 1.0
    epsilon_min   = 0.05
    epsilon_decay = 0.994
    frame         = 0

    for ep in range(EPISODES):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=MAX_STEPS,
            wall_obstacles=False, difficulty=0, box_speed=2, seed=ep,
        )
        state        = env.reset(seed=ep)
        total_reward = 0.0
        n_buf.buf.clear()

        for step in range(MAX_STEPS):
            frame += 1

            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    s_t    = torch.tensor(state, dtype=torch.float32, device=device)
                    action = q_net(s_t).argmax().item()

            next_state, reward, done = env.step(ACTIONS[action])
            total_reward += reward

            transition = n_buf.add(state, action, reward, next_state, done)
            if transition is not None:
                replay.add(transition)

            state = next_state

            if len(replay) >= BATCH:
                beta = min(1.0, BETA_START + frame * (1.0 - BETA_START) / BETA_FRAMES)
                s, a, r, s2, d, idx, weights = replay.sample(BATCH, beta=beta)

                s  = torch.tensor(s,  dtype=torch.float32, device=device)
                a  = torch.tensor(a,  dtype=torch.long,    device=device)
                r  = torch.tensor(r,  dtype=torch.float32, device=device)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d  = torch.tensor(d,  dtype=torch.float32, device=device)
                w  = torch.tensor(weights,                  device=device)

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    best_next = q_net(s2).argmax(dim=1)
                    next_q    = target_net(s2).gather(1, best_next.unsqueeze(1)).squeeze()
                    target_q  = r + (GAMMA ** N_STEP) * (1 - d) * next_q

                td_errors = (q_vals - target_q).detach().cpu().numpy()
                loss = (w * nn.functional.smooth_l1_loss(
                    q_vals, target_q, reduction="none")).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
                optimizer.step()
                replay.update_priorities(idx, td_errors)
                soft_update(target_net, q_net, tau=TAU)

            if done:
                break

        n_buf.flush(replay)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        scheduler.step()

        print(f"Episode {ep:4d} | Reward: {total_reward:8.2f} ")
        print_sensor_state(state)

    torch.save(q_net.state_dict(), "weights.pth")
    print("Training complete. weights.pth saved.")


if __name__ == "__main__":
    main()

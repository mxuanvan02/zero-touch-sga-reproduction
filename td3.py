"""TD3 - Twin Delayed Deep Deterministic Policy Gradient (paper Sec. IV-B,
Algorithm 1). Faithful implementation of the algorithm the paper specifies for
the adaptive sensor activation policy.

Components (as listed in the paper):
  1. Actor network        mu_theta(s)
  2. Twin critics         Q_phi1(s,a), Q_phi2(s,a)
  3. Target networks      delayed copies of actor + critics
  4. Experience replay    stores (s,a,r,s')
Plus the three TD3 tricks: clipped double-Q, target policy smoothing,
delayed policy/target updates.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = torch.device("cpu")          # CUDA driver too old; small nets -> CPU fine


class Actor(nn.Module):
    def __init__(self, sdim, adim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, adim), nn.Tanh(),       # actions in [-1,1]
        )

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    """Twin critics in one module (Q1, Q2)."""
    def __init__(self, sdim, adim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(sdim + adim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(sdim + adim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)

    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], dim=1))


class ReplayBuffer:
    def __init__(self, sdim, adim, size=int(1e5)):
        self.s = np.zeros((size, sdim), np.float32)
        self.a = np.zeros((size, adim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, sdim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.size = size; self.ptr = 0; self.full = False

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.d)

    def __len__(self):
        return self.size if self.full else self.ptr


class TD3:
    def __init__(self, sdim, adim, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, lr=3e-4):
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.actor_t = Actor(sdim, adim).to(DEVICE)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(sdim, adim).to(DEVICE)
        self.critic_t = Critic(sdim, adim).to(DEVICE)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay = policy_delay
        self.adim = adim
        self.it = 0

    def act(self, s, noise=0.1):
        s = torch.as_tensor(s.reshape(1, -1), device=DEVICE)
        a = self.actor(s).detach().cpu().numpy().flatten()
        if noise > 0:
            a = a + np.random.normal(0, noise, size=self.adim)
        return np.clip(a, -1, 1)

    def train(self, buf, batch=128):
        self.it += 1
        s, a, r, s2, d = buf.sample(batch)

        with torch.no_grad():
            # target policy smoothing
            noise = (torch.randn_like(a) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_t(s2, a2)
            q_t = torch.min(q1_t, q2_t)                  # clipped double-Q
            target = r + (1 - d) * self.gamma * q_t

        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        # delayed policy + target updates
        if self.it % self.policy_delay == 0:
            a_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)
            return float(c_loss.item()), float(a_loss.item())
        return float(c_loss.item()), None

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#hyperparameters

N_EPISODES      = 300    

LEARNING_RATE   = 3e-4  
SIGMA_INIT      = 1.0 
SIGMA_MIN       = 0.05    
BASELINE_DECAY  = 0.99   
WINDOW_SIZE     = 2000    
EPSILON_CLIP    = 1e-6    
GRAD_CLIP       = 0.5     



class PolicyNetwork(nn.Module):

    def __init__(self, input_dim=8):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),       # outputs mu_raw (scalar, unbounded)
        )

        # Initialise so that sigma = SIGMA_INIT
        self.log_sigma = nn.Parameter(
            torch.tensor(np.log(SIGMA_INIT), dtype=torch.float32)
        )

    def forward(self, x):
        mu_raw = self.network(x)
        sigma  = torch.exp(self.log_sigma).clamp(min=SIGMA_MIN)
        return mu_raw, sigma


def sample_action(mu_raw, sigma):

    eps = torch.randn_like(mu_raw)
    z   = mu_raw + sigma * eps
    a   = torch.sigmoid(z)
    return a, z


def log_prob(a, z, mu_raw, sigma):

    dist         = torch.distributions.Normal(mu_raw, sigma)
    log_p_gauss  = dist.log_prob(z)

    a_safe       = a.clamp(EPSILON_CLIP, 1.0 - EPSILON_CLIP)
    log_jacobian = torch.log(a_safe) + torch.log(1.0 - a_safe)

    log_p = log_p_gauss - log_jacobian
    return log_p.squeeze()


def receding_window_avg(rewards, window):
    rewards = np.array(rewards)
    result  = np.empty_like(rewards)
    for t in range(len(rewards)):
        start     = max(0, t - window + 1)
        result[t] = rewards[start : t + 1].mean()
    return result

def train():
    env = DynamicPricingEnv()
    env.reset()
    set_env_bounds(env)

    policy    = PolicyNetwork(input_dim=8)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    baseline    = 0.0
    all_rewards = []
    total_steps = 0

    print("=" * 60)
    print("  Policy Gradient  |  Dynamic Pricing")
    print("=" * 60)
    print(f"  Episodes      : {N_EPISODES}")
    print(f"  Horizon       : {env.Horizon} steps / episode")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Sigma init    : {SIGMA_INIT}")
    print(f"  Baseline decay: {BASELINE_DECAY}")
    print("=" * 60)

    for ep in range(N_EPISODES):
        context, _ = env.reset()
        ep_reward  = 0.0

        ep_log_probs = []
        ep_rewards   = []

        for t in range(env.Horizon):
            ctx_feat = get_features(context)
            x        = torch.tensor(ctx_feat, dtype=torch.float32)

            mu_raw, sigma = policy(x)
            a, z          = sample_action(mu_raw, sigma)

            price = float(a.detach().squeeze().numpy()) * env.MaxRideCost

            next_context, reward, terminated, truncated, _ = env.step(price)

            log_p = log_prob(a, z, mu_raw, sigma)
            ep_log_probs.append(log_p)
            ep_rewards.append(reward)

            all_rewards.append(reward)
            ep_reward  += reward
            total_steps += 1

            context = next_context
            if truncated or terminated:
                break

        # Update long-run baseline
        ep_mean_reward = ep_reward / len(ep_rewards)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_mean_reward

        # Compute advantages and whiten them to reduce variance
        rewards_t    = torch.tensor(ep_rewards, dtype=torch.float32)
        advantages   = rewards_t - baseline

        # Whitening: zero mean, unit std within the episode
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Stack log probs
        log_probs_t = torch.stack(ep_log_probs)
        total_loss  = -(advantages * log_probs_t).mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        avg       = ep_reward / env.Horizon
        sigma_val = float(torch.exp(policy.log_sigma).detach())
        print(f"  Ep {ep+1:4d}/{N_EPISODES}  |  "
              f"avg reward = {avg:.5f}  |  "
              f"sigma = {sigma_val:.4f}  |  "
              f"baseline = {baseline:.5f}")

    print("=" * 60)
    print(f"  Training done.  Total steps = {total_steps}")
    print("=" * 60)


    smoothed = receding_window_avg(all_rewards, WINDOW_SIZE)
    steps    = np.arange(1, len(all_rewards) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, smoothed, color="purple", linewidth=1.5,
             label=f"Policy Gradient (W={WINDOW_SIZE})")
    plt.axhline(0.013, color="red",   linestyle="--", linewidth=1,
                label="Random baseline (0.013)")
    plt.axhline(0.023, color="green", linestyle="--", linewidth=1,
                label="Instructor best (0.023)")
    plt.xlabel("Timestep")
    plt.ylabel("Receding Window Avg Reward")
    plt.title("Policy Gradient — Dynamic Pricing (Ride Sharing)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("policy_gradient_reward.png", dpi=150)
    plt.show()
    print("  Plot saved → policy_gradient_reward.png")


if __name__ == "__main__":
    train()
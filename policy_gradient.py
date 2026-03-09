"""
policy_gradient.py
==================
Policy Gradient for Contextual Bandits — Dynamic Pricing in Ride Sharing.

HOW IT WORKS
------------
Unlike linear bandits (which discretise the action space), policy gradient
works directly with continuous actions.

1. A neural network takes the 8-dim context vector as input and outputs
   a mean price mu (after sigmoid squashing to keep it in [0, 1]).

2. The actual price shown is sampled from a Gaussian centred at mu:
       z ~ N(mu_raw, sigma^2)          raw sample (unbounded)
       a = sigmoid(z)                  squash to (0, 1)

   where mu_raw is the pre-sigmoid network output and sigma is a learnable
   scalar parameter (log_sigma is what we actually store and optimise).

3. POLICY GRADIENT UPDATE
   We want to maximise expected reward J(theta).
   The gradient estimator is:
       grad J = E[ reward * grad log pi(a|x) ]

   For a Gaussian policy with sigmoid squashing:
       log pi(a|x) = log N(z; mu_raw, sigma^2)
                     - log | d(sigmoid(z))/dz |
                   = log N(z; mu_raw, sigma^2)
                     - log(a * (1 - a))          <-- THE NOT-SO-OBVIOUS TERM

   The second term is the log absolute Jacobian of the sigmoid transform.
   It corrects for the fact that sigmoid compresses probability mass near
   0 and 1. Without it the gradient is WRONG. (Worth 5 marks per assignment.)

4. LOSS FUNCTION
   We do gradient DESCENT to maximise J, so:
       loss = -reward * log_prob
   where log_prob includes the Jacobian correction.

5. BASELINE
   We subtract a running mean reward baseline from the reward before
   computing the loss. This reduces variance without changing the expected
   gradient (standard REINFORCE trick):
       loss = -(reward - baseline) * log_prob

   Additionally, within each episode we whiten the advantages
   (subtract mean, divide by std) to further reduce variance.

ARCHITECTURE
------------
   Input (8) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1, linear)
   The output is mu_raw (pre-sigmoid mean).
   log_sigma is a separate trainable scalar.

DEPENDENCIES
------------
    pip install gymnasium numpy matplotlib Pillow scipy torch
Place in the same folder as:
    RideSharing.py, features.py, map_agent.png,
    map_environment.png, pre_computed_distance_matrix.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# =============================================================================
# SEEDING  — set once at the top for full reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

N_EPISODES      = 300     # more episodes — PG needs time; instructor saw
                          # improvement after 40-75 episodes, be patient

LEARNING_RATE   = 3e-4    # higher than before — 1e-4 was too slow to escape
                          # the flat reward plateau

SIGMA_INIT      = 1.0     # high initial std dev = wide exploration early on

SIGMA_MIN       = 0.05    # allow sigma to decay low so policy can commit
                          # to good prices later in training

BASELINE_DECAY  = 0.99    # slow EMA gives a stable long-run baseline

WINDOW_SIZE     = 2000    # receding window size for reward plot

EPSILON_CLIP    = 1e-6    # keep a away from exact 0 or 1 to prevent log(0)

GRAD_CLIP       = 0.5     # tight gradient clipping — prevents overshooting


# =============================================================================
# NEURAL NETWORK POLICY
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Maps context features -> mean price (mu_raw, pre-sigmoid).

    Architecture: Input(8) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1)

    The output is the RAW mean (unbounded). We apply sigmoid separately
    during action sampling so we can correctly compute the Jacobian correction.

    log_sigma is a separate learnable parameter (not part of the network layers)
    because the std dev doesn't need to depend on the context for this problem.
    """

    def __init__(self, input_dim=8):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),       # outputs mu_raw (scalar, unbounded)
        )

        # Separate learnable log_sigma (log scale for numerical stability)
        # Initialise so that sigma = SIGMA_INIT
        self.log_sigma = nn.Parameter(
            torch.tensor(np.log(SIGMA_INIT), dtype=torch.float32)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 8) or (8,)

        Returns
        -------
        mu_raw : torch.Tensor  -- pre-sigmoid mean, shape (batch, 1) or (1,)
        sigma  : torch.Tensor  -- std dev (scalar, always positive)
        """
        mu_raw = self.network(x)
        sigma  = torch.exp(self.log_sigma).clamp(min=SIGMA_MIN)
        return mu_raw, sigma


# =============================================================================
# ACTION SAMPLING AND LOG PROBABILITY
# =============================================================================

def sample_action(mu_raw, sigma):
    """
    Sample a price action using the reparameterisation trick.

    Steps:
        1. z ~ N(mu_raw, sigma^2)    sample from Gaussian
        2. a = sigmoid(z)            squash to (0, 1)

    Returns
    -------
    a     : torch.Tensor  price in (0, 1), with gradient
    z     : torch.Tensor  pre-sigmoid sample (needed for log_prob)
    """
    eps = torch.randn_like(mu_raw)
    z   = mu_raw + sigma * eps
    a   = torch.sigmoid(z)
    return a, z


def log_prob(a, z, mu_raw, sigma):
    """
    Compute log pi(a | x) for a Gaussian policy with sigmoid squashing.

    FULL FORMULA:
        log pi(a|x) = log N(z; mu_raw, sigma^2) - log(a) - log(1-a)

    The -log(a) - log(1-a) term is the log absolute Jacobian of sigmoid,
    correcting for the density change under the transform. Without it the
    gradient is wrong.
    """
    dist         = torch.distributions.Normal(mu_raw, sigma)
    log_p_gauss  = dist.log_prob(z)

    a_safe       = a.clamp(EPSILON_CLIP, 1.0 - EPSILON_CLIP)
    log_jacobian = torch.log(a_safe) + torch.log(1.0 - a_safe)

    log_p = log_p_gauss - log_jacobian
    return log_p.squeeze()


# =============================================================================
# RECEDING WINDOW AVERAGE
# =============================================================================

def receding_window_avg(rewards, window):
    rewards = np.array(rewards)
    result  = np.empty_like(rewards)
    for t in range(len(rewards)):
        start     = max(0, t - window + 1)
        result[t] = rewards[start : t + 1].mean()
    return result


# =============================================================================
# TRAINING LOOP
# =============================================================================

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

        # ---- end-of-episode update ----

        # Update long-run baseline
        ep_mean_reward = ep_reward / len(ep_rewards)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_mean_reward

        # Compute advantages and whiten them to reduce variance
        rewards_t    = torch.tensor(ep_rewards, dtype=torch.float32)
        advantages   = rewards_t - baseline

        # Whitening: zero mean, unit std within the episode
        # This stabilises gradient magnitude regardless of reward scale
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Stack log probs and compute loss
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

    # ---- plot ----
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()
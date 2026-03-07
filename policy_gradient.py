import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

N_EPISODES      = 200     
LEARNING_RATE   = 3e-4   
SIGMA_INIT      = 0.5

SIGMA_MIN       = 0.05   

BASELINE_DECAY  = 0.99 

WINDOW_SIZE     = 2000   

EPSILON_CLIP    = 1e-6 

# =============================================================================
# NEURAL NETWORK POLICY
# =============================================================================

class PolicyNetwork(nn.Module):

    def __init__(self, input_dim=8):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),      
        )

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

    Parameters
    ----------
    mu_raw : torch.Tensor  pre-sigmoid mean (scalar)
    sigma  : torch.Tensor  std dev (scalar, > 0)

    Returns
    -------
    a     : torch.Tensor  price in (0, 1), with gradient
    z     : torch.Tensor  pre-sigmoid sample (needed for log_prob)
    """
    # Sample from N(mu_raw, sigma^2) using reparameterisation
    eps = torch.randn_like(mu_raw)      # standard normal noise
    z   = mu_raw + sigma * eps          # z ~ N(mu_raw, sigma^2)
    a   = torch.sigmoid(z)              # a in (0, 1)
    return a, z


def log_prob(a, z, mu_raw, sigma):
    """
    Compute log pi(a | x) for a Gaussian policy with sigmoid squashing.

    FULL FORMULA (this is the not-so-obvious part):
    ------------------------------------------------
        log pi(a|x) = log N(z; mu_raw, sigma^2)  -  log | d sigmoid(z)/dz |

    where:
        log N(z; mu, sigma^2) = -0.5 * ((z - mu)/sigma)^2
                                - log(sigma)
                                - 0.5 * log(2*pi)

        d sigmoid(z)/dz = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)

        so: log | d sigmoid(z)/dz | = log(a) + log(1 - a)

    Therefore:
        log pi(a|x) = log N(z; mu_raw, sigma^2) - log(a) - log(1-a)
    """
    # Gaussian log probability of z
    dist         = torch.distributions.Normal(mu_raw, sigma)
    log_p_gauss  = dist.log_prob(z)

    # Jacobian correction: -log(a*(1-a)) = -log(a) - log(1-a)
    # Clamp a to avoid log(0)
    a_safe       = a.clamp(EPSILON_CLIP, 1.0 - EPSILON_CLIP)
    log_jacobian = torch.log(a_safe) + torch.log(1.0 - a_safe)

    # Full log probability
    log_p = log_p_gauss - log_jacobian

    return log_p.squeeze()


# =============================================================================
# RECEDING WINDOW AVERAGE  (same as lin_greedy / lin_ucb)
# =============================================================================

def receding_window_avg(rewards, window):
    """
    For each timestep t: mean(rewards[max(0, t-W+1) : t+1])
    """
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
        context, _  = env.reset()
        ep_reward   = 0.0

        ep_log_probs = []   # log pi(a_t | x_t) for each timestep
        ep_rewards   = []   # r_t for each timestep

        for t in range(env.Horizon):
            ctx_feat = get_features(context)
            x        = torch.tensor(ctx_feat, dtype=torch.float32)

            # 2. Forward pass: get mu_raw and sigma from policy network
            mu_raw, sigma = policy(x)

            # 3. Sample action: z ~ N(mu_raw, sigma^2), a = sigmoid(z)
            a, z = sample_action(mu_raw, sigma)

            # 4. Get actual price value (detach from graph for env interaction)
            # .squeeze() removes extra dimensions e.g. (1,1) -> scalar
            price = float(a.detach().squeeze().numpy()) * env.MaxRideCost

            # 5. Take action in environment
            next_context, reward, terminated, truncated, _ = env.step(price)

            # 6. Compute log probability (with Jacobian correction)
            #    Store it — we use it after the episode ends
            log_p = log_prob(a, z, mu_raw, sigma)
            ep_log_probs.append(log_p)
            ep_rewards.append(reward)

            all_rewards.append(reward)
            ep_reward  += reward
            total_steps += 1

            context = next_context
            if truncated or terminated:
                break

        # Update baseline using this episode's mean reward
        ep_mean_reward = ep_reward / len(ep_rewards)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_mean_reward

        total_loss = torch.stack([
            -(torch.tensor(r - baseline, dtype=torch.float32) * log_p)
            for log_p, r in zip(ep_log_probs, ep_rewards)
        ]).mean()

        # Backprop and update all network weights (including log_sigma)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg = ep_reward / env.Horizon
        sigma_val = float(torch.exp(policy.log_sigma).detach())
        print(f"  Ep {ep+1:4d}/{N_EPISODES}  |  "
              f"avg reward = {avg:.5f}  |  "
              f"sigma = {sigma_val:.4f}  |  "
              f"baseline = {baseline:.5f}  |  "
              f"steps = {total_steps}")

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
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
import torch
import torch.nn as nn
import torch.optim as optim

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

N_EPISODES      = 200     # total training episodes
                          # instructor saw improvement after 40-75 episodes
                          # be patient, don't stop early

LEARNING_RATE   = 1e-4    # reduced from 3e-4 to prevent overshooting

SIGMA_INIT      = 0.8     # higher initial std dev = more exploration early on

SIGMA_MIN       = 0.10    # raised from 0.05 — keep exploring throughout

BASELINE_DECAY  = 0.95    # faster baseline warmup (was 0.99)
                          # 0.99 was too slow — early advantages were noisy
                          # 0.95 tracks recent rewards more responsively

WINDOW_SIZE     = 2000    # receding window size for reward plot

EPSILON_CLIP    = 1e-6    # numerical safety: keep a away from exact 0 or 1
                          # to prevent log(0) in Jacobian correction

GRAD_CLIP       = 1.0     # gradient clipping — prevents exploding gradients

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

    WHY IS THIS TERM NEEDED?
    ------------------------
    When we transform z through sigmoid, the probability density changes.
    The Jacobian correction accounts for how much sigmoid "stretches" or
    "compresses" the distribution at each point.
    Near a=0 or a=1: sigmoid is very flat -> densities get compressed
                     -> Jacobian correction is large
    Near a=0.5:      sigmoid is steep  -> densities get stretched
                     -> Jacobian correction is small

    Without this correction, the gradient points in the wrong direction
    because we'd be computing the log prob of z, not of a.

    Parameters
    ----------
    a      : torch.Tensor  sigmoid output (the actual action), in (0,1)
    z      : torch.Tensor  pre-sigmoid sample
    mu_raw : torch.Tensor  pre-sigmoid mean from network
    sigma  : torch.Tensor  std dev

    Returns
    -------
    log_p : torch.Tensor  scalar log probability
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
    # ------------------------------------------------------------------ setup
    env = DynamicPricingEnv()
    set_env_bounds(env)

    policy    = PolicyNetwork(input_dim=8)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    baseline    = 0.0      # running mean reward (for variance reduction)
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

        # Collect experience for the full episode first,
        # then do one gradient update at the end.
        # This is standard REINFORCE — more stable than per-step updates.
        ep_log_probs = []   # log pi(a_t | x_t) for each timestep
        ep_rewards   = []   # r_t for each timestep

        for t in range(env.Horizon):
            # --------------------------------------------------------- forward
            # 1. Get context features (8-dim vector)
            ctx_feat = get_features(context)
            x        = torch.tensor(ctx_feat, dtype=torch.float32)

            # 2. Forward pass: get mu_raw and sigma from policy network
            mu_raw, sigma = policy(x)

            # 3. Sample action: z ~ N(mu_raw, sigma^2), a = sigmoid(z)
            a, z = sample_action(mu_raw, sigma)

            # 4. Get actual price value (detach from graph for env interaction)
            # .squeeze() removes extra dimensions e.g. (1,1) -> scalar
            price = float(a.detach().squeeze().numpy()) * env.MaxRideCost

            # ------------------------------------------------------------ step
            # 5. Take action in environment
            next_context, reward, terminated, truncated, _ = env.step(price)

            # 6. Compute log probability (with Jacobian correction)
            #    Store it — we use it after the episode ends
            log_p = log_prob(a, z, mu_raw, sigma)
            ep_log_probs.append(log_p)
            ep_rewards.append(reward)

            # ------------------------------------------------------- tracking
            all_rewards.append(reward)
            ep_reward  += reward
            total_steps += 1

            context = next_context
            if truncated or terminated:
                break

        # --------------------------------------------- end-of-episode update
        # Update baseline using this episode's mean reward
        ep_mean_reward = ep_reward / len(ep_rewards)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_mean_reward

        # Compute total loss over all timesteps in this episode:
        #   loss = -1/T * sum_t [ (r_t - baseline) * log pi(a_t | x_t) ]
        # Negative because PyTorch minimises but we want to maximise reward.
        # Dividing by T keeps gradient magnitude stable across episode lengths.
        total_loss = torch.stack([
            -(torch.tensor(r - baseline, dtype=torch.float32) * log_p)
            for log_p, r in zip(ep_log_probs, ep_rewards)
        ]).mean()

        # Backprop and update all network weights (including log_sigma)
        optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
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

    # ------------------------------------------------------------------- plot
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
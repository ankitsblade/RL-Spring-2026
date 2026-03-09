"""
sanity_tests.py
===============
Task 5 — Sanity / Logic Tests for Trained Agents.

DO NOT SUBMIT THIS FILE. Results (plots) go into report.pdf only.

Two tests:
----------
Test 1: Price vs Passenger Sensitivity (alpha_p)
    Fix everything else. Vary alpha_p from low to high.
    Expected: agent should charge MORE as alpha_p increases
    (passenger is willing to pay more per unit distance).

Test 2: Price vs Number of Drivers
    Fix everything else. Vary number of drivers from 1 to 10.
    Expected: agent should charge LESS as driver count increases
    (more supply = higher booking probability even at lower price).

HOW IT WORKS:
    1. We train each bandit from scratch (same hyperparameters as before)
    2. After training, we set epsilon=0 (pure exploitation) for UCB/greedy
    3. We construct synthetic contexts with controlled variation
    4. We record the price each agent predicts
    5. We plot price vs the variable being tested

Place in the same folder as:
    RideSharing.py, features.py, map_agent.png,
    map_environment.png, pre_computed_distance_matrix.npy,
    lin_greedy.py, lin_ucb.py, policy_gradient.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# Import agent classes from our submitted files
from lin_greedy import EpsilonGreedyLinearBandit, build_phi, N_BINS, FEATURE_DIM, LAMBDA
from lin_ucb import UCBLinearBandit
from policy_gradient import PolicyNetwork, sample_action

# =============================================================================
# TRAINING HELPERS
# Re-trains each agent using the same settings as submission files
# =============================================================================

TRAIN_EPISODES = 150   # same as submission

def train_greedy(env):
    print("  Training ε-Greedy...")
    bandit  = EpsilonGreedyLinearBandit(N_BINS, FEATURE_DIM, env.MaxRideCost, LAMBDA)
    epsilon = 1.0
    for ep in range(TRAIN_EPISODES):
        context, _ = env.reset()
        for t in range(env.Horizon):
            ctx_feat        = get_features(context)
            arm_idx, price, phi = bandit.select_action(ctx_feat, epsilon)
            context, reward, _, truncated, _ = env.step(price)
            bandit.update(arm_idx, phi, reward)
            epsilon = max(0.05, epsilon * 0.997)
            if truncated: break
    print("  ε-Greedy training done.")
    return bandit


def train_ucb(env):
    print("  Training UCB...")
    bandit = UCBLinearBandit(N_BINS, FEATURE_DIM, env.MaxRideCost, LAMBDA, alpha=0.5)
    for ep in range(TRAIN_EPISODES):
        context, _ = env.reset()
        for t in range(env.Horizon):
            ctx_feat            = get_features(context)
            arm_idx, price, phi = bandit.select_action(ctx_feat)
            context, reward, _, truncated, _ = env.step(price)
            bandit.update(arm_idx, phi, reward)
            if truncated: break
    print("  UCB training done.")
    return bandit


def train_pg(env):
    print("  Training Policy Gradient...")
    import torch.optim as optim
    from policy_gradient import log_prob, LEARNING_RATE, BASELINE_DECAY, GRAD_CLIP

    policy    = PolicyNetwork(input_dim=8)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    baseline  = 0.0

    for ep in range(TRAIN_EPISODES):
        context, _   = env.reset()
        ep_log_probs = []
        ep_rewards   = []
        ep_reward    = 0.0

        for t in range(env.Horizon):
            ctx_feat      = get_features(context)
            x             = torch.tensor(ctx_feat, dtype=torch.float32)
            mu_raw, sigma = policy(x)
            a, z          = sample_action(mu_raw, sigma)
            price         = float(a.detach().squeeze().numpy()) * env.MaxRideCost
            context, reward, _, truncated, _ = env.step(price)
            log_p         = log_prob(a, z, mu_raw, sigma)
            ep_log_probs.append(log_p)
            ep_rewards.append(reward)
            ep_reward    += reward
            if truncated: break

        ep_mean  = ep_reward / len(ep_rewards)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_mean

        # Whitened advantages — same as policy_gradient.py
        rewards_t  = torch.tensor(ep_rewards, dtype=torch.float32)
        advantages = rewards_t - baseline
        adv_std    = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        log_probs_t = torch.stack(ep_log_probs)
        total_loss  = -(advantages * log_probs_t).mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

    print("  Policy Gradient training done.")
    return policy


# =============================================================================
# SYNTHETIC CONTEXT BUILDER
# Constructs a context tuple in the exact format DynamicPricingEnv gives us:
#   (passenger_info, driver_info)
#   passenger_info : np.array([x_o, y_o, x_f, y_f, alpha_p])
#   driver_info    : tuple of np.arrays, each [x_d, y_d, alpha_d]
# All coordinates are in normalised units (same scale as env observations).
# =============================================================================

def make_context(env, alpha_p, n_drivers, alpha_d_list):
    """
    Build a synthetic context with controlled values.

    Parameters
    ----------
    env         : DynamicPricingEnv  (for coordinate bounds)
    alpha_p     : float  passenger sensitivity
    n_drivers   : int    number of drivers
    alpha_d_list: list   driver sensitivities, length = n_drivers

    Returns
    -------
    context : (passenger_info, driver_info)
    """
    # Fixed passenger origin and destination (centre of map)
    # Using midpoint coordinates so A* always finds a path
    x_o = env.max_x * 0.3
    y_o = env.max_y * 0.3
    x_f = env.max_x * 0.7
    y_f = env.max_y * 0.7

    passenger_info = np.array([x_o, y_o, x_f, y_f, alpha_p], dtype=np.float32)

    # Fixed driver locations near the passenger
    driver_info = []
    for i in range(n_drivers):
        x_d = env.max_x * (0.3 + 0.02 * i)   # slightly spread out
        y_d = env.max_y * (0.3 + 0.02 * i)
        driver_info.append(np.array([x_d, y_d, alpha_d_list[i]], dtype=np.float32))

    return (passenger_info, tuple(driver_info))


def get_price_greedy(bandit, context, env):
    """Get predicted price from ε-Greedy (epsilon=0 = pure exploitation)."""
    ctx_feat        = get_features(context)
    _, price, _     = bandit.select_action(ctx_feat, epsilon=0.0)
    return price


def get_price_ucb(bandit, context, env):
    """Get predicted price from UCB."""
    ctx_feat        = get_features(context)
    _, price, _     = bandit.select_action(ctx_feat)
    return price


def get_price_pg(policy, context, env):
    """Get predicted price from Policy Gradient (use mean, no sampling noise)."""
    ctx_feat = get_features(context)
    x        = torch.tensor(ctx_feat, dtype=torch.float32)
    with torch.no_grad():
        mu_raw, _ = policy(x)
        # Use sigmoid of mean directly (no sampling = deterministic inference)
        a = torch.sigmoid(mu_raw)
    return float(a.squeeze().numpy()) * env.MaxRideCost


# =============================================================================
# TEST 1: Price vs Passenger Sensitivity
#
# Hypothesis: As alpha_p increases, the agent should charge MORE.
# Reason: Higher alpha_p means the passenger is willing to pay more per unit
# distance. A good agent should exploit this and increase the price.
#
# What we fix:
#   - Passenger origin and destination (same locations throughout)
#   - Number of drivers: 3 (moderate supply)
#   - Driver sensitivities: fixed at midpoint value
#
# What we vary:
#   - alpha_p from 5% to 95% of MaxTheta_p (full range the env uses)
# =============================================================================

def test1_price_vs_passenger_sensitivity(env, greedy, ucb, pg):
    print("\nRunning Test 1: Price vs Passenger Sensitivity...")

    # Range of alpha_p values (5% to 95% of max, same as env's range)
    alpha_p_values = np.linspace(0.05 * env.MaxTheta_p, 0.95 * env.MaxTheta_p, 30)

    # Fixed driver setup: 3 drivers at midpoint sensitivity
    n_drivers    = 3
    mid_alpha_d  = 0.5 * env.MaxTheta_d
    alpha_d_list = [mid_alpha_d] * n_drivers

    prices_greedy = []
    prices_ucb    = []
    prices_pg     = []

    for alpha_p in alpha_p_values:
        context = make_context(env, alpha_p, n_drivers, alpha_d_list)

        prices_greedy.append(get_price_greedy(greedy, context, env))
        prices_ucb.append(get_price_ucb(ucb, context, env))
        prices_pg.append(get_price_pg(pg, context, env))

    # Normalise alpha_p for x-axis readability
    alpha_p_norm = alpha_p_values / env.MaxTheta_p

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alpha_p_norm, prices_greedy, 'b-o', markersize=4, label='ε-Greedy')
    ax.plot(alpha_p_norm, prices_ucb,    'o-o', markersize=4, label='UCB',
            color='darkorange')
    ax.plot(alpha_p_norm, prices_pg,     'p-o', markersize=4, label='Policy Gradient',
            color='purple')
    ax.set_xlabel("Passenger Sensitivity α_p (normalised by max)")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Test 1: Price vs Passenger Sensitivity\n"
                 "(Expected: price increases as passenger is less price-sensitive)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test1_price_vs_passenger_sensitivity.png", dpi=150)
    plt.show()
    print("  Saved → test1_price_vs_passenger_sensitivity.png")


# =============================================================================
# TEST 2: Price vs Number of Drivers
#
# Hypothesis: As number of drivers increases, the agent should charge LESS.
# Reason: More nearby drivers = higher probability that at least one accepts
# even at a lower price. A good agent should recognise this supply increase
# and lower the price to maximise booking probability × commission.
#
# What we fix:
#   - Passenger origin, destination, and sensitivity (midpoint values)
#   - Driver sensitivities: all fixed at midpoint value
#
# What we vary:
#   - Number of drivers from 1 to 10 (full range env uses)
# =============================================================================

def test2_price_vs_num_drivers(env, greedy, ucb, pg):
    print("\nRunning Test 2: Price vs Number of Drivers...")

    driver_counts = list(range(1, 11))   # 1 to 10

    # Fixed passenger at midpoint sensitivity
    mid_alpha_p  = 0.5 * env.MaxTheta_p
    mid_alpha_d  = 0.5 * env.MaxTheta_d

    prices_greedy = []
    prices_ucb    = []
    prices_pg     = []

    for n in driver_counts:
        alpha_d_list = [mid_alpha_d] * n
        context      = make_context(env, mid_alpha_p, n, alpha_d_list)

        prices_greedy.append(get_price_greedy(greedy, context, env))
        prices_ucb.append(get_price_ucb(ucb, context, env))
        prices_pg.append(get_price_pg(pg, context, env))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(driver_counts, prices_greedy, 'b-o', markersize=6, label='ε-Greedy')
    ax.plot(driver_counts, prices_ucb,    'o-o', markersize=6, label='UCB',
            color='darkorange')
    ax.plot(driver_counts, prices_pg,     'p-o', markersize=6, label='Policy Gradient',
            color='purple')
    ax.set_xlabel("Number of Nearby Drivers")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Test 2: Price vs Number of Drivers\n"
                 "(Expected: price decreases as driver supply increases)")
    ax.set_xticks(driver_counts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test2_price_vs_num_drivers.png", dpi=150)
    plt.show()
    print("  Saved → test2_price_vs_num_drivers.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Task 5 — Sanity Tests")
    print("=" * 60)

    # Setup environment
    env = DynamicPricingEnv()
    set_env_bounds(env)

    # Train all three agents
    print("\nTraining all agents (this may take a while)...")
    greedy = train_greedy(env)
    ucb    = train_ucb(env)
    pg     = train_pg(env)

    # Run tests
    test1_price_vs_passenger_sensitivity(env, greedy, ucb, pg)
    test2_price_vs_num_drivers(env, greedy, ucb, pg)

    print("\n" + "=" * 60)
    print("  Both tests complete.")
    print("  Include the two .png plots in report.pdf")
    print("  DO NOT submit this script.")
    print("=" * 60)
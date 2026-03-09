

import numpy as np
import matplotlib.pyplot as plt
import random

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

#seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#hyperparams

N_BINS        = 20      
N_EPISODES    = 150     
EPSILON_START = 1.0     
EPSILON_MIN   = 0.05    
EPSILON_DECAY = 0.997  

LAMBDA        = 1.0    

WINDOW_SIZE   = 2000    


CONTEXT_DIM  = 8   # from features.py
FEATURE_DIM  = CONTEXT_DIM + 2   # + price + bias  =  10


def build_phi(context_features, price, max_price):
    price_norm = price / max_price          # normalise to [0, 1]
    return np.concatenate([
        context_features.astype(np.float64),
        [price_norm],                       # price feature
        [1.0],                              # bias term
    ])



class EpsilonGreedyLinearBandit:


    def __init__(self, n_bins, feature_dim, max_price, lambda_reg=LAMBDA):
        self.n_bins      = n_bins
        self.d           = feature_dim
        self.max_price   = max_price
        self.lambda_reg  = lambda_reg

        edges = np.linspace(0, max_price, n_bins + 1)
        self.action_values = (edges[:-1] + edges[1:]) / 2   # shape (n_bins,)

        self.A = [lambda_reg * np.eye(feature_dim) for _ in range(n_bins)]
        self.b = [np.zeros(feature_dim)            for _ in range(n_bins)]


    def _theta(self, i):
        """Solve A_i theta = b_i for theta_i."""
        return np.linalg.solve(self.A[i], self.b[i])


    def select_action(self, context_features, epsilon):
        if np.random.rand() < epsilon:
            # EXPLORE: random arm
            arm_idx = np.random.randint(self.n_bins)
        else:
            # EXPLOIT: arm with highest predicted reward
            scores = np.array([
                build_phi(context_features,
                          self.action_values[i],
                          self.max_price) @ self._theta(i)
                for i in range(self.n_bins)
            ])
            arm_idx = int(np.argmax(scores))

        price = self.action_values[arm_idx]
        phi   = build_phi(context_features, price, self.max_price)
        return arm_idx, price, phi

    def update(self, arm_idx, phi, reward):

        self.A[arm_idx] += np.outer(phi, phi)
        self.b[arm_idx] += reward * phi




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

    bandit = EpsilonGreedyLinearBandit(
        n_bins      = N_BINS,
        feature_dim = FEATURE_DIM,
        max_price   = env.MaxRideCost,
        lambda_reg  = LAMBDA,
    )

    epsilon      = EPSILON_START
    all_rewards  = []
    total_steps  = 0

    print("=" * 60)
    print("  Epsilon-Greedy Linear Bandit  |  Dynamic Pricing")
    print("=" * 60)
    print(f"  Episodes   : {N_EPISODES}")
    print(f"  Horizon    : {env.Horizon} steps / episode")
    print(f"  Bins       : {N_BINS}   (step = {env.MaxRideCost/N_BINS:.3f})")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Lambda     : {LAMBDA}")
    print("=" * 60)

    for ep in range(N_EPISODES):
        context, _    = env.reset()
        ep_reward     = 0.0

        for t in range(env.Horizon):
            # 1. Extract fixed-size context features
            ctx_feat = get_features(context)

            # 2. Choose action (epsilon-greedy)
            arm_idx, price, phi = bandit.select_action(ctx_feat, epsilon)

            # 3. Step environment
            next_context, reward, terminated, truncated, _ = env.step(price)

            # 4. Update bandit model
            bandit.update(arm_idx, phi, reward)

            # 5. Bookkeeping
            all_rewards.append(reward)
            ep_reward  += reward
            total_steps += 1

            # 6. Decay epsilon
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            context = next_context
            if truncated or terminated:
                break

        avg = ep_reward / env.Horizon
        print(f"  Ep {ep+1:4d}/{N_EPISODES}  |  "
              f"avg reward = {avg:.5f}  |  "
              f"epsilon = {epsilon:.4f}  |  "
              f"steps = {total_steps}")

    print("=" * 60)
    print(f"  Training done.  Total steps = {total_steps}")
    print("=" * 60)

    smoothed = receding_window_avg(all_rewards, WINDOW_SIZE)
    steps    = np.arange(1, len(all_rewards) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, smoothed, color="steelblue", linewidth=1.5, label=f"Window={WINDOW_SIZE}")
    plt.axhline(0.013, color="red",   linestyle="--", linewidth=1, label="Random baseline (0.013)")
    plt.axhline(0.023, color="green", linestyle="--", linewidth=1, label="Instructor best (0.023)")
    plt.xlabel("Timestep")
    plt.ylabel("Receding Window Avg Reward")
    plt.title("ε-Greedy Linear Bandit — Dynamic Pricing (Ride Sharing)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lin_greedy_reward.png", dpi=150)
    plt.show()
    print("  Plot saved → lin_greedy_reward.png")


if __name__ == "__main__":
    train()
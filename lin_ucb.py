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

N_BINS      = 20     
N_EPISODES  = 150    
LAMBDA      = 1.0    
ALPHA       = 0.5     
WINDOW_SIZE = 2000 
CONTEXT_DIM = 8       # from features.py
FEATURE_DIM = CONTEXT_DIM + 2   # + price + bias = 10


def build_phi(context_features, price, max_price):
    price_norm = price / max_price
    return np.concatenate([
        context_features.astype(np.float64),
        [price_norm],
        [1.0],
    ])


class UCBLinearBandit:

    def __init__(self, n_bins, feature_dim, max_price, lambda_reg=LAMBDA, alpha=ALPHA):
        self.n_bins     = n_bins
        self.d          = feature_dim
        self.max_price  = max_price
        self.lambda_reg = lambda_reg
        self.alpha      = alpha

        edges = np.linspace(0, max_price, n_bins + 1)
        self.action_values = (edges[:-1] + edges[1:]) / 2   # shape (n_bins,)

        self.A = [lambda_reg * np.eye(feature_dim) for _ in range(n_bins)]
        self.b = [np.zeros(feature_dim)            for _ in range(n_bins)]


    def _theta(self, i):
        """Solve A_i theta = b_i  →  theta_i"""
        return np.linalg.solve(self.A[i], self.b[i])


    def _uncertainty(self, i, phi):


        v = np.linalg.solve(self.A[i], phi)
        variance = float(phi @ v)
        variance = max(variance, 0.0)

        return self.alpha * np.sqrt(variance)

    def select_action(self, context_features):
        scores = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            phi_i      = build_phi(context_features, self.action_values[i], self.max_price)
            theta_i    = self._theta(i)
            exploit    = phi_i @ theta_i              # predicted reward
            explore    = self._uncertainty(i, phi_i)  # uncertainty bonus
            scores[i]  = exploit + explore

        arm_idx = int(np.argmax(scores))
        price   = self.action_values[arm_idx]
        phi     = build_phi(context_features, price, self.max_price)
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

    bandit = UCBLinearBandit(
        n_bins     = N_BINS,
        feature_dim = FEATURE_DIM,
        max_price  = env.MaxRideCost,
        lambda_reg = LAMBDA,
        alpha      = ALPHA,
    )

    all_rewards = []
    total_steps = 0

    print("=" * 60)
    print("  UCB Linear Bandit  |  Dynamic Pricing")
    print("=" * 60)
    print(f"  Episodes   : {N_EPISODES}")
    print(f"  Horizon    : {env.Horizon} steps / episode")
    print(f"  Bins       : {N_BINS}   (step = {env.MaxRideCost/N_BINS:.3f})")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Lambda     : {LAMBDA}")
    print(f"  Alpha      : {ALPHA}")
    print("=" * 60)

    for ep in range(N_EPISODES):
        context, _ = env.reset()
        ep_reward  = 0.0

        for t in range(env.Horizon):
            # 1. Extract fixed-size context features
            ctx_feat = get_features(context)

            # 2. Choose action (UCB -- no epsilon)
            arm_idx, price, phi = bandit.select_action(ctx_feat)

            # 3. Step environment
            next_context, reward, terminated, truncated, _ = env.step(price)

            # 4. Update bandit model
            bandit.update(arm_idx, phi, reward)

            # 5. Bookkeeping
            all_rewards.append(reward)
            ep_reward  += reward
            total_steps += 1

            context = next_context
            if truncated or terminated:
                break

        avg = ep_reward / env.Horizon
        print(f"  Ep {ep+1:4d}/{N_EPISODES}  |  "
              f"avg reward = {avg:.5f}  |  "
              f"steps = {total_steps}")

    print("=" * 60)
    print(f"  Training done.  Total steps = {total_steps}")
    print("=" * 60)

    smoothed = receding_window_avg(all_rewards, WINDOW_SIZE)
    steps    = np.arange(1, len(all_rewards) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, smoothed, color="darkorange", linewidth=1.5, label=f"UCB (alpha={ALPHA}, W={WINDOW_SIZE})")
    plt.axhline(0.013, color="red",   linestyle="--", linewidth=1, label="Random baseline (0.013)")
    plt.axhline(0.023, color="green", linestyle="--", linewidth=1, label="Instructor best (0.023)")
    plt.xlabel("Timestep")
    plt.ylabel("Receding Window Avg Reward")
    plt.title("UCB Linear Bandit — Dynamic Pricing (Ride Sharing)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lin_ucb_reward.png", dpi=150)
    plt.show()
    print("  Plot saved → lin_ucb_reward.png")



if __name__ == "__main__":
    train()
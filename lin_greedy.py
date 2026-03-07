"""
lin_greedy.py
=============
Epsilon-greedy Linear Bandit for Dynamic Pricing in Ride Sharing.

HOW IT WORKS (brief)
--------------------
1. The context (passenger + driver info) is converted to an 8-dim feature
   vector by features.py (A* distances + sensitivity values).

2. The continuous action space [0, MaxRideCost] is discretised into N_BINS
   evenly-spaced price bins.  Each bin is treated as a separate "arm".

3. For every arm i we maintain a linear model:
       E[reward | phi, arm=i]  ~=  phi(context, i)^T  *  theta_i
   where  phi  is the concatenation of the 8 context features with the
   arm's price value (and a bias term) -- giving a 10-dim feature vector.

4. Parameters are updated online with ridge-regularised least squares:
       A_i  <-  A_i + phi * phi^T
       b_i  <-  b_i + reward * phi
       theta_i  =  A_i^{-1} b_i

5. Action selection: with probability epsilon pick a random arm (explore),
   otherwise pick the arm with highest predicted reward (exploit).

6. Epsilon decays exponentially from EPSILON_START to EPSILON_MIN.

7. After training, a receding-window time-averaged reward plot is saved.

DEPENDENCIES
------------
    pip install gymnasium numpy matplotlib Pillow scipy
Place in the same folder as:
    RideSharing.py, features.py, map_agent.png,
    map_environment.png, pre_computed_distance_matrix.npy
"""

import numpy as np
import matplotlib.pyplot as plt

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# =============================================================================
# HYPERPARAMETERS  (tune these)
# =============================================================================

N_BINS        = 20      # number of discrete price bins
                        # 20 gives a step size of 0.05 which is fine-grained
                        # enough to find a good price without blowing up
                        # the parameter count.

N_EPISODES    = 150     # total training episodes (each = 720 timesteps)

EPSILON_START = 1.0     # start fully exploratory so every arm gets data
EPSILON_MIN   = 0.05    # never drop below 5% exploration
EPSILON_DECAY = 0.997   # multiplicative decay applied each timestep

LAMBDA        = 1.0     # ridge regularisation coefficient
                        # initialises A_i = lambda * I, keeping it invertible
                        # even before an arm has been pulled

WINDOW_SIZE   = 2000    # receding window size for the reward plot (recommended)

# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================
# features.py gives us an 8-dim context vector.
# We append the normalised price and a bias term -> 10-dim phi.
# Including the price inside phi lets the linear model learn how reward
# scales with price for each context (instead of one scalar per arm).

CONTEXT_DIM  = 8   # from features.py
FEATURE_DIM  = CONTEXT_DIM + 2   # + price + bias  =  10


def build_phi(context_features, price, max_price):
    """
    Concatenate context features with normalised price and bias.

    Parameters
    ----------
    context_features : np.ndarray, shape (8,)
        Output of features.get_features(context).
    price : float
        The candidate price for this arm.
    max_price : float
        env.MaxRideCost -- used to normalise price to [0, 1].

    Returns
    -------
    np.ndarray, shape (10,), dtype float64
    """
    price_norm = price / max_price          # normalise to [0, 1]
    return np.concatenate([
        context_features.astype(np.float64),
        [price_norm],                       # price feature
        [1.0],                              # bias term
    ])


# =============================================================================
# EPSILON-GREEDY LINEAR BANDIT
# =============================================================================

class EpsilonGreedyLinearBandit:
    """
    One ridge-regularised linear model per price bin (arm).

    Model per arm i
    ---------------
        E[r | phi] = phi^T theta_i
        theta_i    = A_i^{-1} b_i

    Update (online least squares)
    ------------------------------
        A_i <- A_i + phi phi^T
        b_i <- b_i + r * phi
    """

    def __init__(self, n_bins, feature_dim, max_price, lambda_reg=LAMBDA):
        self.n_bins      = n_bins
        self.d           = feature_dim
        self.max_price   = max_price
        self.lambda_reg  = lambda_reg

        # Bin centres: evenly spaced prices strictly inside (0, max_price)
        edges = np.linspace(0, max_price, n_bins + 1)
        self.action_values = (edges[:-1] + edges[1:]) / 2   # shape (n_bins,)

        # One A and b per arm; A starts as lambda*I (ridge regularisation)
        self.A = [lambda_reg * np.eye(feature_dim) for _ in range(n_bins)]
        self.b = [np.zeros(feature_dim)            for _ in range(n_bins)]

    # ------------------------------------------------------------------
    def _theta(self, i):
        """Solve A_i theta = b_i for theta_i."""
        return np.linalg.solve(self.A[i], self.b[i])

    # ------------------------------------------------------------------
    def select_action(self, context_features, epsilon):
        """
        Epsilon-greedy arm selection.

        Parameters
        ----------
        context_features : np.ndarray (8,)
        epsilon          : float, current exploration rate

        Returns
        -------
        arm_idx : int
        price   : float   (centre of chosen bin)
        phi     : np.ndarray (10,)
        """
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

    # ------------------------------------------------------------------
    def update(self, arm_idx, phi, reward):
        """
        Online least-squares update for arm arm_idx.

        A_i <- A_i + phi phi^T
        b_i <- b_i + reward * phi
        """
        self.A[arm_idx] += np.outer(phi, phi)
        self.b[arm_idx] += reward * phi


# =============================================================================
# RECEDING WINDOW AVERAGE
# =============================================================================

def receding_window_avg(rewards, window):
    """
    For each timestep t, compute mean(rewards[max(0, t-W+1) : t+1]).
    This smooths the noisy per-step reward into a readable trend.
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
    set_env_bounds(env)          # must be called once before get_features()

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

    # --------------------------------------------------------------- episodes
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

    # ------------------------------------------------------------------- plot
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()

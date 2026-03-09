"""
lin_ucb.py
==========
UCB (Upper Confidence Bound) Linear Bandit for Dynamic Pricing in Ride Sharing.

HOW IT WORKS
------------
Identical setup to lin_greedy.py with one crucial difference in action selection.

Instead of epsilon-greedy (random exploration), UCB selects actions using:

    score_i = phi^T theta_i  +  alpha * sqrt(phi^T A_i^{-1} phi)
              |___________|     |__________________________|
               exploitation          exploration bonus

The exploration bonus is large when arm i has been pulled rarely (A_i is
small, so A_i^{-1} is large) and shrinks automatically as more data arrives.
This means:
  - No epsilon or decay schedule needed
  - Exploration is directed: arms with high uncertainty are preferred
  - Naturally transitions from explore to exploit as learning progresses

Everything else (feature vector, update rule, model structure) is identical
to lin_greedy.py.

DEPENDENCIES
------------
    pip install gymnasium numpy matplotlib Pillow scipy
Place in the same folder as:
    RideSharing.py, features.py, map_agent.png,
    map_environment.png, pre_computed_distance_matrix.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

# =============================================================================
# SEEDING  — set once at the top for full reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

N_BINS      = 20      # number of discrete price bins (same as lin_greedy)

N_EPISODES  = 150     # total training episodes

LAMBDA      = 1.0     # ridge regularisation (A_i initialised as lambda * I)

ALPHA       = 0.5     # exploration coefficient
                      # scales the uncertainty bonus in the UCB score
                      # higher alpha -> more exploration
                      # lower alpha  -> more exploitation
                      # 0.5 is a good starting point; tune if needed

WINDOW_SIZE = 2000    # receding window size for reward plot

# =============================================================================
# FEATURE CONSTRUCTION  (same as lin_greedy.py)
# =============================================================================

CONTEXT_DIM = 8       # from features.py
FEATURE_DIM = CONTEXT_DIM + 2   # + price + bias = 10


def build_phi(context_features, price, max_price):
    """
    Concatenate context features with normalised price and bias term.

    Parameters
    ----------
    context_features : np.ndarray (8,)  from features.get_features()
    price            : float            candidate price for this arm
    max_price        : float            env.MaxRideCost

    Returns
    -------
    np.ndarray, shape (10,), dtype float64
    """
    price_norm = price / max_price
    return np.concatenate([
        context_features.astype(np.float64),
        [price_norm],
        [1.0],
    ])


# =============================================================================
# UCB LINEAR BANDIT
# =============================================================================

class UCBLinearBandit:
    """
    One ridge-regularised linear model per price bin (arm).

    Model per arm i
    ---------------
        E[r | phi] = phi^T theta_i
        theta_i    = A_i^{-1} b_i

    Update (online least squares) -- same as lin_greedy
    -----------------------------------------------------
        A_i <- A_i + phi phi^T
        b_i <- b_i + r * phi

    Action selection (UCB) -- THIS is what differs from lin_greedy
    ---------------------------------------------------------------
        For each arm i, compute:
            score_i = phi_i^T theta_i  +  alpha * sqrt(phi_i^T A_i^{-1} phi_i)

        Pick arm with highest score_i.

        The term sqrt(phi^T A^{-1} phi) is the standard deviation of the
        predicted reward under the current model -- a measure of how uncertain
        we are. Adding it to the prediction creates an "optimistic" estimate:
        we act as if each arm might be as good as the upper end of our
        uncertainty interval.
    """

    def __init__(self, n_bins, feature_dim, max_price, lambda_reg=LAMBDA, alpha=ALPHA):
        self.n_bins     = n_bins
        self.d          = feature_dim
        self.max_price  = max_price
        self.lambda_reg = lambda_reg
        self.alpha      = alpha

        # Bin centres: evenly spaced prices inside (0, max_price)
        edges = np.linspace(0, max_price, n_bins + 1)
        self.action_values = (edges[:-1] + edges[1:]) / 2   # shape (n_bins,)

        # One A and b per arm; A starts as lambda*I (ridge regularisation)
        self.A = [lambda_reg * np.eye(feature_dim) for _ in range(n_bins)]
        self.b = [np.zeros(feature_dim)            for _ in range(n_bins)]

    # ------------------------------------------------------------------
    def _theta(self, i):
        """Solve A_i theta = b_i  →  theta_i"""
        return np.linalg.solve(self.A[i], self.b[i])

    # ------------------------------------------------------------------
    def _uncertainty(self, i, phi):
        """
        Compute the UCB exploration bonus for arm i given feature vector phi.

            bonus = alpha * sqrt(phi^T  A_i^{-1}  phi)

        We compute phi^T A_i^{-1} phi efficiently by solving A_i v = phi
        (which gives v = A_i^{-1} phi) and then computing phi^T v.
        This avoids explicitly inverting A_i (expensive for large matrices).

        Parameters
        ----------
        i   : int         arm index
        phi : np.ndarray  feature vector (10,)

        Returns
        -------
        float  -- the exploration bonus
        """
        # Solve A_i v = phi  →  v = A_i^{-1} phi
        v = np.linalg.solve(self.A[i], phi)

        # phi^T v = phi^T A_i^{-1} phi  (should be >= 0 since A_i is PSD)
        variance = float(phi @ v)
        variance = max(variance, 0.0)   # numerical safety: clamp to 0

        return self.alpha * np.sqrt(variance)

    # ------------------------------------------------------------------
    def select_action(self, context_features):
        """
        UCB action selection -- NO epsilon needed.

        For each arm i:
            phi_i  = build_phi(context_features, price_i)
            score_i = phi_i^T theta_i  +  alpha * sqrt(phi_i^T A_i^{-1} phi_i)

        Pick arm with highest score.

        Parameters
        ----------
        context_features : np.ndarray (8,)

        Returns
        -------
        arm_idx : int
        price   : float
        phi     : np.ndarray (10,)
        """
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

    # ------------------------------------------------------------------
    def update(self, arm_idx, phi, reward):
        """
        Online least-squares update -- same as lin_greedy.

        A_i <- A_i + phi phi^T
        b_i <- b_i + reward * phi
        """
        self.A[arm_idx] += np.outer(phi, phi)
        self.b[arm_idx] += reward * phi


# =============================================================================
# RECEDING WINDOW AVERAGE  (same as lin_greedy)
# =============================================================================

def receding_window_avg(rewards, window):
    """
    For each timestep t: mean(rewards[max(0, t-W+1) : t+1])
    Smooths noisy per-step rewards into a readable learning trend.
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
    env.reset()                  # reset the environment
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

    # --------------------------------------------------------------- episodes
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

    # ------------------------------------------------------------------- plot
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()
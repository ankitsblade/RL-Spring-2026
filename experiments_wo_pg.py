import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from RideSharing import DynamicPricingEnv
from features import get_features, set_env_bounds

from lin_greedy import EpsilonGreedyLinearBandit, N_BINS, FEATURE_DIM, LAMBDA
from lin_ucb import UCBLinearBandit

TRAIN_EPISODES = 10


def train_greedy(env):
    path = "sanity_greedy.pkl"
    if os.path.exists(path):
        print("  e-Greedy: loading saved weights...")
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"  Training e-Greedy ({TRAIN_EPISODES} episodes)...")
    bandit  = EpsilonGreedyLinearBandit(N_BINS, FEATURE_DIM, env.MaxRideCost, LAMBDA)
    epsilon = 1.0
    for ep in range(TRAIN_EPISODES):
        context, _ = env.reset()
        for t in range(env.Horizon):
            ctx_feat            = get_features(context)
            arm_idx, price, phi = bandit.select_action(ctx_feat, epsilon)
            context, reward, _, truncated, _ = env.step(price)
            bandit.update(arm_idx, phi, reward)
            epsilon = max(0.05, epsilon * 0.997)
            if truncated: break
        print(f"    ep {ep+1}/{TRAIN_EPISODES}", flush=True)
    with open(path, "wb") as f:
        pickle.dump(bandit, f)
    print("  e-Greedy done.")
    return bandit


def train_ucb(env):
    path = "sanity_ucb.pkl"
    if os.path.exists(path):
        print("  UCB: loading saved weights...")
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"  Training UCB ({TRAIN_EPISODES} episodes)...")
    bandit = UCBLinearBandit(N_BINS, FEATURE_DIM, env.MaxRideCost, LAMBDA, alpha=0.5)
    for ep in range(TRAIN_EPISODES):
        context, _ = env.reset()
        for t in range(env.Horizon):
            ctx_feat            = get_features(context)
            arm_idx, price, phi = bandit.select_action(ctx_feat)
            context, reward, _, truncated, _ = env.step(price)
            bandit.update(arm_idx, phi, reward)
            if truncated: break
        print(f"    ep {ep+1}/{TRAIN_EPISODES}", flush=True)
    with open(path, "wb") as f:
        pickle.dump(bandit, f)
    print("  UCB done.")
    return bandit


def make_context(env, alpha_p, n_drivers, alpha_d_list):
    x_o = env.max_x * 0.3;  y_o = env.max_y * 0.3
    x_f = env.max_x * 0.7;  y_f = env.max_y * 0.7
    passenger_info = np.array([x_o, y_o, x_f, y_f, alpha_p], dtype=np.float32)
    driver_info = []
    for i in range(n_drivers):
        x_d = env.max_x * (0.3 + 0.02 * i)
        y_d = env.max_y * (0.3 + 0.02 * i)
        driver_info.append(np.array([x_d, y_d, alpha_d_list[i]], dtype=np.float32))
    return (passenger_info, tuple(driver_info))


def get_price_greedy(bandit, context):
    ctx_feat = get_features(context)
    _, price, _ = bandit.select_action(ctx_feat, epsilon=0.0)
    return price

def get_price_ucb(bandit, context):
    ctx_feat = get_features(context)
    _, price, _ = bandit.select_action(ctx_feat)
    return price


def test1_price_vs_passenger_sensitivity(env, greedy, ucb):
    alpha_p_values = np.linspace(0.05 * env.MaxTheta_p, 0.95 * env.MaxTheta_p, 30)
    alpha_d_list   = [0.5 * env.MaxTheta_d] * 3

    prices_greedy, prices_ucb = [], []
    for alpha_p in alpha_p_values:
        context = make_context(env, alpha_p, 3, alpha_d_list)
        prices_greedy.append(get_price_greedy(greedy, context))
        prices_ucb.append(get_price_ucb(ucb, context))

    alpha_p_norm = alpha_p_values / env.MaxTheta_p
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alpha_p_norm, prices_greedy, 'b-o', markersize=4, label='e-Greedy')
    ax.plot(alpha_p_norm, prices_ucb,    '-o',  markersize=4, label='UCB', color='darkorange')
    ax.set_xlabel("Passenger Sensitivity (normalised)")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Test 1: Price vs Passenger Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test1_price_vs_passenger_sensitivity.png", dpi=150)
    plt.show()


def test2_price_vs_num_drivers(env, greedy, ucb):
    driver_counts = list(range(1, 11))
    mid_alpha_p   = 0.5 * env.MaxTheta_p
    mid_alpha_d   = 0.5 * env.MaxTheta_d

    prices_greedy, prices_ucb = [], []
    for n in driver_counts:
        context = make_context(env, mid_alpha_p, n, [mid_alpha_d] * n)
        prices_greedy.append(get_price_greedy(greedy, context))
        prices_ucb.append(get_price_ucb(ucb, context))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(driver_counts, prices_greedy, 'b-o', markersize=6, label='e-Greedy')
    ax.plot(driver_counts, prices_ucb,    '-o',  markersize=6, label='UCB', color='darkorange')
    ax.set_xlabel("Number of Nearby Drivers")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Test 2: Price vs Number of Drivers")
    ax.set_xticks(driver_counts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test2_price_vs_num_drivers.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    env = DynamicPricingEnv()
    set_env_bounds(env)

    greedy = train_greedy(env)
    ucb    = train_ucb(env)

    test1_price_vs_passenger_sensitivity(env, greedy, ucb)
    test2_price_vs_num_drivers(env, greedy, ucb)
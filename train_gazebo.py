#!/usr/bin/env python3
"""
Training script for Gazebo SAR environment.
Runs all agents: Random, LocalAStar, DDPG, TD3, SAC, PPO, TRPO
Usage: python3 train_gazebo.py
Requires: Gazebo running with unitree_go2_launch.py + champ_gz_bridge.py
"""
import sys
sys.path.insert(0, "/home/pirat/ros2_ws/src/go1_nav")

import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TRPO
from go1_nav.go2_nav_env import Go2NavEnv

# ── Configuration ────────────────────────────────────────────────────────────

# Robot spawns at (-8, -8). Goal should be near the collapsed house entrance.
GOAL_POS        = None  # randomised each episode   # adjust based on your world layout
TRAIN_STEPS     = 50_000       # reduce for quick test; use 200_000 for thesis
EVAL_FREQ       = 10_000
EVAL_EPISODES   = 2
MAX_EP_STEPS    = 25

# ── Environment factory ──────────────────────────────────────────────────────

def make_env():
    env = Go2NavEnv(
        goal_pos=GOAL_POS,
        num_beams=64,
        max_range=10.0,
        max_linear_speed=0.5,
        max_angular_speed=1.0,
        done_threshold=0.5,
        collision_threshold=0.35,
        height_band=0.4,
        step_timeout=1.0,
    )
    return Monitor(env)

# ── Baseline agents ──────────────────────────────────────────────────────────

class RandomWalker:
    def __init__(self, action_space):
        self.action_space = action_space
    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None
    def reset(self):
        pass

# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(agent, env, max_steps=MAX_EP_STEPS):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    collision = False
    success = False

    for _ in range(max_steps):
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if info.get("collision", False):
            collision = True
        if terminated and not collision:
            success = True
        if terminated or truncated:
            break

    return total_reward, steps, collision, success

# ── Training loop ────────────────────────────────────────────────────────────

def train_and_eval(agents, env):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    results = {name: {"rewards": [], "steps": [], "collisions": [], "successes": []}
               for name in agents}
    curves  = {}
    rl_names = {"DDPG", "TD3", "SAC", "PPO", "TRPO"}

    for name, agent in agents.items():
        print(f"\n{'='*50}\nAgent: {name}\n{'='*50}")

        if name in rl_names:
            curve_r, curve_s = [], []
            n_evals = TRAIN_STEPS // EVAL_FREQ

            for i in range(n_evals):
                agent.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=(i==0))
                # Save checkpoint
                save_path = f"checkpoints/{name}_step{(i+1)*EVAL_FREQ}"
                agent.save(save_path)
                ep_rewards = []
                for _ in range(EVAL_EPISODES):
                    r, _, _, _ = run_episode(agent, env)
                    ep_rewards.append(r)
                avg = float(np.mean(ep_rewards))
                curve_r.append(avg)
                curve_s.append((i + 1) * EVAL_FREQ)
                print(f"  [{name}] step {(i+1)*EVAL_FREQ}/{TRAIN_STEPS}  avg_reward={avg:.2f}")

            curves[name] = {"steps": curve_s, "rewards": curve_r}

        # Final evaluation
        print(f"  Final eval: {name}")
        for _ in range(EVAL_EPISODES * 2):
            r, s, col, suc = run_episode(agent, env)
            results[name]["rewards"].append(r)
            results[name]["steps"].append(s)
            results[name]["collisions"].append(float(col))
            results[name]["successes"].append(float(suc))

        m = results[name]
        print(f"  mean_reward={np.mean(m["rewards"]):.2f}  "
              f"success_rate={np.mean(m["successes"]):.2f}  "
              f"collision_rate={np.mean(m["collisions"]):.2f}")

    return results, curves

# ── Plotting ─────────────────────────────────────────────────────────────────

PALETTE = {
    "Random": "#9E9E9E", "DDPG": "#2196F3", "TD3": "#03A9F4",
    "SAC": "#FF5722", "PPO": "#FF9800", "TRPO": "#9C27B0",
}

def plot_results(results, curves):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    agent_names = list(results.keys())

    # Learning curves
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, c in curves.items():
            color = PALETTE.get(name, "#607D8B")
            xs, ys = np.array(c["steps"]), np.array(c["rewards"])
            ax.plot(xs, ys, label=name, color=color, linewidth=2, marker="o", markersize=4)
            ax.fill_between(xs, ys*0.95, ys*1.05, color=color, alpha=0.15)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean evaluation return")
        ax.set_title("Learning curves — Gazebo SAR environment")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.savefig("gazebo_learning_curves.pdf", bbox_inches="tight")
        plt.show()

    # Final return boxplot
    reward_df = pd.DataFrame({n: pd.Series(results[n]["rewards"]) for n in agent_names}
                             ).melt(var_name="Agent", value_name="Return")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=reward_df, x="Agent", y="Return", order=agent_names,
                palette={n: PALETTE.get(n, "#607D8B") for n in agent_names},
                width=0.5, ax=ax)
    ax.set_title("Final evaluation return — Gazebo SAR")
    fig.tight_layout()
    plt.savefig("gazebo_final_return.pdf", bbox_inches="tight")
    plt.show()

    # Success/collision rates
    rate_df = pd.DataFrame({
        "Agent": agent_names,
        "Success rate":   [np.mean(results[n]["successes"])   for n in agent_names],
        "Collision rate": [np.mean(results[n]["collisions"])  for n in agent_names],
    }).melt(id_vars="Agent", var_name="Metric", value_name="Rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=rate_df, x="Agent", y="Rate", hue="Metric", order=agent_names,
                palette={"Success rate": "#43A047", "Collision rate": "#E53935"}, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Success and collision rates — Gazebo SAR")
    fig.tight_layout()
    plt.savefig("gazebo_rates.pdf", bbox_inches="tight")
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialising environment...")
    env = make_env()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}  Action space: {env.action_space}")

    n_act = env.action_space.shape[-1]
    net   = dict(net_arch=[256, 256])

    agents = {
        "Random": RandomWalker(env.action_space),

        "DDPG": sb3.DDPG(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=100_000, batch_size=256,
            tau=0.005, gamma=0.99, learning_starts=1_000,
            action_noise=NormalActionNoise(np.zeros(n_act), 0.1*np.ones(n_act)),
            policy_kwargs=net, verbose=0),

        "TD3": sb3.TD3(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=100_000, batch_size=256,
            tau=0.005, gamma=0.99, learning_starts=1_000,
            action_noise=NormalActionNoise(np.zeros(n_act), 0.1*np.ones(n_act)),
            policy_kwargs=net, verbose=0),

        "SAC": sb3.SAC(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100_000, batch_size=256,
            ent_coef="auto", gamma=0.99, learning_starts=1_000,
            policy_kwargs=net, verbose=0),

        "PPO": sb3.PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=512, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256],
                               activation_fn=torch.nn.Tanh),
            verbose=0),

        "TRPO": TRPO(
            "MlpPolicy", env,
            learning_rate=1e-3, batch_size=128,
            gamma=0.99, gae_lambda=0.95,
            n_steps=512, cg_max_steps=15,
            target_kl=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0),
    }

    results, curves = train_and_eval(agents, env)
    plot_results(results, curves)
    env.close()
    print("Done.")

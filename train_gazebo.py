#!/usr/bin/env python3
"""
Training script for Gazebo SAR environment.
Runs all agents: Random, LocalAStar, DDPG, TD3, SAC, PPO, TRPO
Usage: python3 train_gazebo.py
Requires: Gazebo running with unitree_go2_launch.py + champ_gz_bridge.py
"""
import sys
sys.path.insert(0, "/home/pirat/ros2_ws/src/go2_nav")

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
from go2_nav.go2_nav_env import Go2NavEnv



GOAL_POS        = None  # randomised each episode   
TRAIN_STEPS     = 200_000      
EVAL_FREQ       = 10_000
EVAL_EPISODES   = 2
MAX_EP_STEPS    = 25

# creates the Gazebo environment 
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

# baseline agent
class RandomWalker:
    def __init__(self, action_space):
        self.action_space = action_space
    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None
    def reset(self):
        pass

# runs one episode
def run_episode(agent, env, max_steps=MAX_EP_STEPS):
    obs, _ = env.reset() # resets the environment 
    total_reward = 0.0
    steps = 0
    collision = False
    success = False

    # loops actions until end of episode
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

# main training loop 
def train_and_eval(agents, env):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    results = {name: {"rewards": [], "steps": [], "collisions": [], "successes": []}
               for name in agents}
    curves  = {}
    rl_names = {"DDPG", "TD3", "SAC", "PPO", "TRPO"}

   
    for name, agent in agents.items():
        print(f"\n{'='*50}\nAgent: {name}\n{'='*50}")
        
         # for each RL agent, trains for eval_freq steps & saves a checkpoint
        if name in rl_names:
            curve_r, curve_s = [], []
            n_evals = TRAIN_STEPS // EVAL_FREQ

            for i in range(n_evals):
                agent.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=(i==0))
                # Save checkpoint
                save_path = f"checkpoints/{name}_step{(i+1)*EVAL_FREQ}"
                agent.save(save_path) # checkpoint 
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


PALETTE = {
    "Random": "#9E9E9E", "DDPG": "#2196F3", "TD3": "#03A9F4",
    "SAC": "#FF5722", "PPO": "#FF9800", "TRPO": "#9C27B0",
}

def plot_results(results, curves):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    agent_names = list(results.keys())

    # Plot 1: Learning curves
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
        ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
        fig.tight_layout()
        plt.savefig("fig1_learning_curves.pdf", bbox_inches="tight")
        plt.show()

    # Plot 2: Final return boxplot with individual data points
    reward_df = pd.DataFrame({n: pd.Series(results[n]["rewards"]) for n in agent_names}
                             ).melt(var_name="Agent", value_name="Return")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=reward_df, x="Agent", y="Return", order=agent_names,
                palette={n: PALETTE.get(n, "#607D8B") for n in agent_names},
                width=0.5, linewidth=1.2,
                flierprops=dict(marker="x", markersize=4, linewidth=0.8), ax=ax)
    sns.stripplot(data=reward_df, x="Agent", y="Return",
                  palette={n: PALETTE.get(n, "#607D8B") for n in agent_names},
                  order=agent_names, dodge=False, jitter=True, size=3, alpha=0.5, ax=ax)
    ax.set_title("Final evaluation return distribution — Gazebo SAR")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative return")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    plt.savefig("fig2_final_return_boxplot.pdf", bbox_inches="tight")
    plt.show()

    # Plot 3: Steps per episode violin plot
    steps_df = pd.DataFrame({n: pd.Series(results[n]["steps"]) for n in agent_names}
                            ).melt(var_name="Agent", value_name="Steps")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=steps_df, x="Agent", y="Steps", order=agent_names,
                   palette={n: PALETTE.get(n, "#607D8B") for n in agent_names},
                   inner="quartile", linewidth=1.0, cut=0, ax=ax)
    ax.set_title("Steps per episode — Gazebo SAR")
    ax.set_xlabel("")
    ax.set_ylabel("Steps to termination")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    plt.savefig("fig3_steps_violin.pdf", bbox_inches="tight")
    plt.show()

    # Plot 4: Success/collision rates
    rate_df = pd.DataFrame({
        "Agent": agent_names,
        "Success rate":   [np.mean(results[n]["successes"])   for n in agent_names],
        "Collision rate": [np.mean(results[n]["collisions"])  for n in agent_names],
    }).melt(id_vars="Agent", var_name="Metric", value_name="Rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=rate_df, x="Agent", y="Rate", hue="Metric", order=agent_names,
                palette={"Success rate": "#43A047", "Collision rate": "#E53935"},
                linewidth=0.8, edgecolor="white", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Success and collision rates — Gazebo SAR")
    ax.legend(frameon=False, loc="upper right")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    plt.savefig("fig4_success_collision_rates.pdf", bbox_inches="tight")
    plt.show()

    # Plot 5: DDPG vs TD3 vs SAC zoomed comparison (first 20k steps)
    MAX_STEPS = 20_000
    detail_agents = [n for n in ["DDPG", "TD3", "SAC"] if n in curves]
    if detail_agents:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in detail_agents:
            xs = np.array(curves[name]["steps"])
            ys = np.array(curves[name]["rewards"])
            mask = xs <= MAX_STEPS
            xs, ys = xs[mask], ys[mask]
            color = PALETTE.get(name, "#607D8B")
            ax.plot(xs, ys, label=name, color=color, linewidth=2, marker="o", markersize=4)
            ax.fill_between(xs, ys*0.95, ys*1.05, color=color, alpha=0.15)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean evaluation return")
        ax.set_title("DDPG vs TD3 vs SAC — first 2×10⁴ steps — Gazebo SAR")
        ax.set_xlim(0, MAX_STEPS)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.savefig("fig5_ddpg_td3_sac_2e4.pdf", bbox_inches="tight")
        plt.show()

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

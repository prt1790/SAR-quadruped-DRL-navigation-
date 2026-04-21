import sys
sys.path.insert(0, '/home/pirat/ros2_ws/install/go1_nav/lib/python3.12/site-packages')

import numpy as np
import heapq
from go1_nav.go2_nav_env import Go2NavEnv


class LocalAStarAgent:
    def __init__(self, env, grid_resolution=0.1, local_radius=2.0,
                 plan_horizon=5, max_range=10.0):
        self.env          = env
        self.grid_res     = grid_resolution
        self.local_radius = local_radius
        self.plan_horizon = plan_horizon
        self.max_range    = max_range

        cells = int(2 * local_radius / grid_resolution)
        self.grid_cells = cells
        self.centre_idx = cells // 2
        self.num_beams  = env.num_beams
        self.max_step   = env.max_linear_speed
        self._plan      = []
        self._step_i    = 0

    def reset(self):
        self._plan   = []
        self._step_i = 0

    def select_action(self, obs):
        nb              = self.num_beams
        lidar_norm      = obs[:nb]
        goal_dist_norm  = obs[nb]
        goal_angle_norm = obs[nb + 1]
        sin_t           = obs[nb + 2]
        cos_t           = obs[nb + 3]

        theta      = np.arctan2(sin_t, cos_t)
        goal_dist  = goal_dist_norm * self.env.max_goal_dist
        goal_angle = goal_angle_norm * np.pi

        grid       = self._build_grid(lidar_norm)
        goal_local = self._goal_to_grid(goal_dist, goal_angle)

        self._plan   = self._plan_astar(grid, goal_local)
        self._step_i = 0

        if not self._plan:
            return self._greedy_action(goal_dist, goal_angle)

        di, dj = self._plan[0]
        self._step_i = len(self._plan)

        dx =  dj * self.grid_res
        dy = -di * self.grid_res

        linear  = (dx * np.cos(theta) + dy * np.sin(theta)) / self.max_step
        angular = np.arctan2(
            dy * np.cos(theta) - dx * np.sin(theta),
            dx * np.cos(theta) + dy * np.sin(theta)) * 2

        return np.clip(np.array([linear, angular], dtype=np.float32), -1.0, 1.0)

    def _build_grid(self, lidar_norm):
        grid   = np.zeros((self.grid_cells, self.grid_cells), dtype=bool)
        ranges = lidar_norm * self.max_range
        # 360 degrees - matches how Go2NavEnv maps points to beams
        angles = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)
        for r, a in zip(ranges, angles):
            if r >= self.max_range * 0.99:
                continue
            ex =  r * np.cos(a)
            ey =  r * np.sin(a)
            ci = self.centre_idx - int(round(ey / self.grid_res))
            cj = self.centre_idx + int(round(ex / self.grid_res))
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < self.grid_cells and 0 <= nj < self.grid_cells:
                        grid[ni, nj] = True
        return grid

    def _goal_to_grid(self, goal_dist, goal_angle):
        gx =  goal_dist * np.cos(goal_angle)
        gy =  goal_dist * np.sin(goal_angle)
        gi = self.centre_idx - int(round(gy / self.grid_res))
        gj = self.centre_idx + int(round(gx / self.grid_res))
        gi = np.clip(gi, 0, self.grid_cells - 1)
        gj = np.clip(gj, 0, self.grid_cells - 1)
        return (gi, gj)

    def _plan_astar(self, grid, goal_ij):
        start = (self.centre_idx, self.centre_idx)
        goal  = goal_ij
        moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        def h(node):
            return np.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)

        heap    = [(h(start), 0, start, [])]
        visited = {}

        while heap:
            f, g, node, path = heapq.heappop(heap)
            if node in visited and visited[node] <= g:
                continue
            visited[node] = g
            if node == goal or len(path) >= self.plan_horizon:
                return path[:self.plan_horizon]
            for di, dj in moves:
                ni, nj = node[0] + di, node[1] + dj
                if not (0 <= ni < self.grid_cells and 0 <= nj < self.grid_cells):
                    continue
                if grid[ni, nj]:
                    continue
                ng       = g + np.sqrt(di**2 + dj**2)
                new_path = path + [(di, dj)]
                heapq.heappush(heap, (ng + h((ni, nj)), ng, (ni, nj), new_path))
        return []

    def _greedy_action(self, goal_dist, goal_angle):
        linear  = np.clip(goal_dist / 5.0, 0.1, 1.0)
        angular = np.clip(goal_angle / np.pi, -1.0, 1.0)
        return np.array([linear, angular], dtype=np.float32)


# Run a short episode
env   = Go2NavEnv(goal_pos=(4.0, 4.0))
agent = LocalAStarAgent(env)

obs, _ = env.reset()
agent.reset()

total_reward = 0.0
for step in range(50):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Step {step+1:3d} | action=[{action[0]:+.2f},{action[1]:+.2f}] "
          f"| reward={reward:+.2f} | total={total_reward:+.2f} "
          f"| collision={info['collision']}")
    if terminated:
        print("Episode ended early.")
        break

print(f"\nFinal total reward: {total_reward:.2f}")
env.close()

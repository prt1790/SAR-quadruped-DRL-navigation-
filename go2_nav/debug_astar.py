import sys
sys.path.insert(0, '/home/pirat/ros2_ws/install/go1_nav/lib/python3.12/site-packages')

import numpy as np
import heapq
from go1_nav.go2_nav_env import Go2NavEnv

env = Go2NavEnv(goal_pos=(4.0, 4.0))
obs, _ = env.reset()

nb = env.num_beams
lidar_norm      = obs[:nb]
goal_dist_norm  = obs[nb]
goal_angle_norm = obs[nb + 1]
sin_t           = obs[nb + 2]
cos_t           = obs[nb + 3]

goal_dist  = goal_dist_norm * env.max_goal_dist
goal_angle = goal_angle_norm * np.pi
theta      = np.arctan2(sin_t, cos_t)

print(f"goal_dist:  {goal_dist:.3f} m")
print(f"goal_angle: {goal_angle:.3f} rad ({np.degrees(goal_angle):.1f} deg)")
print(f"robot yaw:  {theta:.3f} rad")

# Check lidar
print(f"\nLidar non-max beams: {np.sum(lidar_norm < 0.99)}/{nb}")
print(f"Lidar min range: {lidar_norm.min() * env.max_range:.3f} m")
print(f"Lidar max range: {lidar_norm.max() * env.max_range:.3f} m")

# Build grid and check it
grid_res    = 0.1
local_radius= 2.0
cells       = int(2 * local_radius / grid_res)
centre_idx  = cells // 2

ranges = lidar_norm * env.max_range
angles = np.linspace(-np.pi, np.pi, nb, endpoint=False)

grid = np.zeros((cells, cells), dtype=bool)
for r, a in zip(ranges, angles):
    if r >= env.max_range * 0.99:
        continue
    ex =  r * np.cos(a)
    ey =  r * np.sin(a)
    ci = centre_idx - int(round(ey / grid_res))
    cj = centre_idx + int(round(ex / grid_res))
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < cells and 0 <= nj < cells:
                grid[ni, nj] = True

occupied = np.sum(grid)
print(f"\nGrid size: {cells}x{cells}")
print(f"Occupied cells: {occupied}")
print(f"Robot cell (centre): ({centre_idx}, {centre_idx})")
print(f"Robot cell occupied: {grid[centre_idx, centre_idx]}")

# Check goal cell
gx = goal_dist * np.cos(goal_angle)
gy = goal_dist * np.sin(goal_angle)
gi = centre_idx - int(round(gy / grid_res))
gj = centre_idx + int(round(gx / grid_res))
gi = np.clip(gi, 0, cells - 1)
gj = np.clip(gj, 0, cells - 1)
print(f"Goal cell: ({gi}, {gj})")
print(f"Goal cell occupied: {grid[gi, gj]}")
print(f"Goal == Start: {gi == centre_idx and gj == centre_idx}")

env.close()

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from flight_environment import FlightEnvironment
from path_planner import PathPlanner, validate_path, path_length
from trajectory_generator import TrajectoryGenerator, plot_trajectory

parser = argparse.ArgumentParser(description="Path planning + trajectory generation demo")
parser.add_argument("--obstacles", type=int, default=50, help="number of cylinders")
parser.add_argument("--seed", type=int, default=None, help="random seed of the environment (default: random)")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

start = (1, 2, 0)
goal = (18, 18, 3)

# The environment is random.  If a cylinder happens to cover the start or the goal
# the task is infeasible, so a new environment is drawn in that (rare) case.
env = FlightEnvironment(args.obstacles)
while env.is_collide(start) or env.is_collide(goal) or env.is_outside(start) or env.is_outside(goal):
    print("Start or goal lies inside an obstacle, regenerating the environment ...")
    env = FlightEnvironment(args.obstacles)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here.
# The planner should return a collision-free path and store it in the variable `path`.
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

t0 = time.perf_counter()
planner = PathPlanner(env, start, goal, resolution=0.2, heuristic_weight=1.1)
path = planner.plan()
t_plan = time.perf_counter() - t0

print("========== Path planning (Lazy Theta* + taut-path refinement) ==========")
print(f"planning time      : {t_plan * 1000:.1f} ms "
      f"(grid {planner.stats['t_grid'] * 1000:.1f} ms, search {planner.stats['t_search'] * 1000:.1f} ms, "
      f"refine {planner.stats['t_refine'] * 1000:.1f} ms)")
print(f"expanded nodes     : {planner.stats['expanded']}")
print(f"path length        : {path_length(path):.3f} m  (straight-line lower bound "
      f"{np.linalg.norm(np.subtract(goal, start)):.3f} m)")
print(f"path points        : {len(path)}")
print(f"collision free     : {validate_path(env, path)}  (checked with env.is_collide / env.is_outside)")
print("path =\n", np.array2string(path, precision=3, suppress_small=True))

# --------------------------------------------------------------------------------------------------- #

plt.ion()                      # non-blocking: both result windows are shown together at the end
env.plot_cylinders(path)
plt.pause(0.001)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

t0 = time.perf_counter()
generator = TrajectoryGenerator(path, env, v_max=2.0, a_max=2.0, order=4)
trajectory = generator.generate()
t_traj = time.perf_counter() - t0
st = generator.stats

t_s, pos = trajectory.sample(0.01)
t_path = generator.waypoint_times[generator.is_original]          # time stamps of the path points
dev = np.linalg.norm(trajectory.evaluate(t_path) - path, axis=1)
print("========== Trajectory generation (minimum snap, 7th order) ==========")
print(f"generation time    : {t_traj * 1000:.1f} ms")
print(f"segments           : {st['segments']}  (subdivision points {st['subdivided']}, "
      f"safety insertions {st['inserted']})")
print(f"duration           : {st['total_time']:.2f} s, peak speed {st['v_peak']:.2f} m/s, "
      f"peak acceleration {st['a_peak']:.2f} m/s^2")
print(f"max deviation from path points : {dev.max():.2e} m")
print(f"trajectory collision free      : {st['collision_free']}")

# overlay the continuous trajectory on the 3-D figure produced by plot_cylinders()
ax3d = plt.gcf().axes[0]
ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], "--", color="tab:red", lw=1.2, label="trajectory")
ax3d.plot([], [], color="tab:blue", lw=2, label="path")
ax3d.legend(loc="upper left")
ax3d.set_title("Planned path and minimum-snap trajectory")

fig_traj = plt.figure(figsize=(8, 7))
plot_trajectory(generator, fig=fig_traj)

plt.ioff()
plt.show()

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git.
# When submitting your assignment, upload the project to a code-hosting platform
# such as GitHub or GitLab. The repository must be accessible and directly cloneable.
#
# After cloning, running `python3 main.py` in the project root directory
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.

# Intelligent Mobile Robotics Project — Path & Trajectory Planning

Course project 2 (智能移动机器人技术 · 路径与轨迹规划实验).
A quadrotor has to fly from `start = (1, 2, 0)` to `goal = (18, 18, 3)` in a
20 m × 20 m × 5 m space cluttered with random vertical cylinders.

## Quick start

```bash
git clone https://github.com/zhenxiang-ding/Intelligent_Mobile_Robotics_Project.git
cd Intelligent_Mobile_Robotics_Project
python3 main.py                 # random environment with 50 cylinders
python3 main.py --seed 4 --obstacles 100   # reproducible / denser scene
```

`main.py` prints timing and quality statistics and opens **two windows**:

1. the 3-D environment with the planned path (`FlightEnvironment.plot_cylinders`),
   overlaid with the smooth trajectory;
2. `x(t)`, `y(t)`, `z(t)` of the trajectory with the planned path points marked.

Dependencies: Python ≥ 3.8, `numpy`, `matplotlib` (standard library: `math`,
`heapq`, `time`, `argparse`). No other packages are required.

## Method

| File | Content |
|---|---|
| `path_planner.py` | `CollisionChecker` (exact, vectorised segment–cylinder test), `OccupancyGrid` (padded 3-D voxel map), `PathPlanner` (**Lazy Theta\*** any-angle search + line-of-sight pruning + taut-path refinement with height relaxation) |
| `trajectory_generator.py` | `TrajectoryGenerator`: **minimum-snap** piecewise 7th-order polynomials solved through the KKT system of the equality-constrained QP; trapezoidal time allocation, local time adjustment + global time scaling for `v_max`/`a_max`, collision-triggered waypoint insertion; `plot_trajectory` |
| `main.py` | runs the full pipeline and shows the two figures |
| `experiments/` | baselines (grid A\*, RRT\*) and the script reproducing all tables/figures of the report |

Typical numbers over 168 random environments (Intel Core Ultra 7 255HX, Python 3.10):
path planning 21 ms (50 cylinders) / 50 ms (100 cylinders) on average, path length
0.44 % / 1.22 % above the straight-line lower bound, trajectory generation ≈ 9 ms,
100 % collision-free paths and trajectories (v ≤ 2 m/s, a ≤ 2 m/s²).

## Reproducing the report experiments

```bash
python3 experiments/run_experiments.py          # ~5 min, writes experiments/results/
python3 experiments/run_experiments.py --quick  # fewer environments
```

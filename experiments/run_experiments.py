"""
Reproduces all quantitative results and figures used in the report.

    python experiments/run_experiments.py            # full run (~ several minutes)
    python experiments/run_experiments.py --quick    # fewer random environments

Outputs are written to experiments/results/.
"""

import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from flight_environment import FlightEnvironment                       # noqa: E402
from path_planner import PathPlanner, path_length, validate_path       # noqa: E402
from trajectory_generator import TrajectoryGenerator                   # noqa: E402
from baselines import AStarPlanner, RRTStarPlanner                     # noqa: E402

OUT = os.path.join(HERE, "results")
START, GOAL = (1.0, 2.0, 0.0), (18.0, 18.0, 3.0)
LOWER_BOUND = float(np.linalg.norm(np.subtract(GOAL, START)))

# ------------------------------------------------------------------ style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_OURS, C_ASTAR, C_RRT, C_THETA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_TRAJ, C_JERK = "#e34948", "#6250d6"
C_OBS, C_OBS_EDGE = "#b9b8b2", "#8a8984"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": GRID, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
CM = 1 / 2.54


def make_env(seed, n_obs):
    np.random.seed(seed)
    env = FlightEnvironment(n_obs)
    if env.is_collide(START) or env.is_collide(GOAL):
        return None
    return env


def turning_angle(path):
    """Sum of heading changes along a polyline [deg]."""
    d = np.diff(np.asarray(path), axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-9]
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    c = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return float(np.degrees(np.sum(np.arccos(c))))


def min_clearance(env, path, step=0.02):
    """Smallest horizontal distance of the path to a safety cylinder (r + 0.2) [m]."""
    path = np.asarray(path)
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        pts.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    pts = np.vstack([np.atleast_2d(p) for p in pts])
    cyl = env.cylinders
    d = np.hypot(pts[:, :1] - cyl[None, :, 0], pts[:, 1:2] - cyl[None, :, 1]) - cyl[None, :, 3] - 0.2
    active = (pts[:, 2:3] >= 0) & (pts[:, 2:3] <= cyl[None, :, 2])
    return float(np.min(np.where(active, d, np.inf)))


# =============================================================== experiment 1
def run_planner_comparison(cases):
    methods = ["A*", "A*+LOS", "RRT*", "LazyTheta*", "Ours(w=1.0)", "Ours"]
    rec = {m: {"len": [], "time": [], "vert": [], "turn": [], "clear": [], "ok": 0} for m in methods}

    def log(m, env, path, t):
        r = rec[m]
        r["len"].append(path_length(path)); r["time"].append(t); r["vert"].append(len(path))
        r["turn"].append(turning_angle(path)); r["clear"].append(min_clearance(env, path))
        r["ok"] += int(validate_path(env, path, step=0.05))

    for k, (seed, n_obs, env) in enumerate(cases):
        a = AStarPlanner(env, START, GOAL)
        p = a.plan(); log("A*", env, p, a.stats["t_total"])
        t0 = time.perf_counter(); pp = np.vstack(a._prune(list(p)))
        log("A*+LOS", env, pp, a.stats["t_total"] + time.perf_counter() - t0)
        r = RRTStarPlanner(env, START, GOAL, max_iter=3000, seed=seed)
        p = r.plan()
        if p is not None:
            log("RRT*", env, p, r.stats["t_total"])
        lt = PathPlanner(env, START, GOAL, heuristic_weight=1.1, refine=False)
        p = lt.plan(); log("LazyTheta*", env, p, lt.stats["t_total"])
        o1 = PathPlanner(env, START, GOAL, heuristic_weight=1.0)
        p = o1.plan(); log("Ours(w=1.0)", env, p, o1.stats["t_total"])
        o = PathPlanner(env, START, GOAL, heuristic_weight=1.1)
        p = o.plan(); log("Ours", env, p, o.stats["t_total"])
        if (k + 1) % 20 == 0:
            print(f"  planner comparison: {k + 1}/{len(cases)}")

    summary = {}
    for m in methods:
        r = rec[m]
        L = np.array(r["len"])
        summary[m] = dict(n=len(L), success=r["ok"] / len(cases), len_mean=L.mean(), len_std=L.std(),
                          excess=100 * (L.mean() / LOWER_BOUND - 1), time_ms=1000 * np.mean(r["time"]),
                          time_ms_max=1000 * np.max(r["time"]), vertices=np.mean(r["vert"]),
                          turn=np.mean(r["turn"]), clearance=np.min(r["clear"]))
    return summary


# =============================================================== experiment 2
def run_weight_study(cases, weights=(1.0, 1.05, 1.1, 1.2, 1.3, 1.5), resolutions=(0.2, 0.3)):
    out = {}
    for res in resolutions:
        for w in weights:
            L, T, E = [], [], []
            for _, _, env in cases:
                p = PathPlanner(env, START, GOAL, resolution=res, heuristic_weight=w)
                p.plan()
                L.append(p.stats["length"]); T.append(p.stats["t_total"]); E.append(p.stats["expanded"])
            out[f"{res}_{w}"] = dict(res=res, w=w, len_mean=float(np.mean(L)),
                                     excess=100 * (np.mean(L) / LOWER_BOUND - 1),
                                     time_ms=1000 * float(np.mean(T)), expanded=float(np.mean(E)))
            print(f"  weight study res={res} w={w}: {out[f'{res}_{w}']}")
    return out


# =============================================================== experiment 3
TRAJ_VARIANTS = {
    "snap_plain":    dict(order=4, max_seg_len=np.inf, local_time_iter=0),
    "snap_no_check": dict(order=4, max_seg_len=2.0, local_time_iter=8, max_insertions=0),
    "snap_subdiv":   dict(order=4, max_seg_len=2.0, local_time_iter=0),
    "snap_full":     dict(order=4, max_seg_len=2.0, local_time_iter=8),
    "jerk_full":     dict(order=3, max_seg_len=2.0, local_time_iter=8),
}


def run_trajectory_study(cases):
    paths = [(env, PathPlanner(env, START, GOAL, heuristic_weight=1.1).plan()) for _, _, env in cases]
    out = {}
    for name, kw in TRAJ_VARIANTS.items():
        R = {k: [] for k in ("total_time", "J_jerk", "J_snap", "a_peak", "v_peak", "inserted",
                             "collision_free", "time", "segments")}
        for env, path in paths:
            g = TrajectoryGenerator(path, env, **kw)
            g.generate()
            for k in R:
                R[k].append(g.stats[k])
        out[name] = dict(total_time=np.mean(R["total_time"]), J_jerk=np.median(R["J_jerk"]),
                         J_snap=np.median(R["J_snap"]), a_peak=np.mean(R["a_peak"]),
                         v_peak=np.mean(R["v_peak"]), inserted=np.mean(R["inserted"]),
                         collision_free=100 * np.mean(R["collision_free"]),
                         time_ms=1000 * np.mean(R["time"]), segments=np.mean(R["segments"]))
        print(f"  trajectory variant {name}: {out[name]}")

    # Objective comparison on identical waypoints and segment durations (those of the
    # full minimum-snap pipeline).  The minimum-snap trajectory is feasible for the
    # minimum-jerk problem, so the minimum-jerk optimum must have a jerk integral that
    # is not larger (sanity check of the KKT solution).  The converse does not hold:
    # minimum snap additionally requires zero jerk at both ends, so its snap integral
    # is not comparable with that of the quintic minimum-jerk spline.
    Js, Jj, ok_j = [], [], 0
    for env, path in paths:
        g = TrajectoryGenerator(path, env, **TRAJ_VARIANTS["snap_full"])
        tr_s = g.generate()
        tr_j = TrajectoryGenerator(path, env, order=3)._solve(g.waypoints, tr_s.durations)
        Js.append(tr_s.cost(3))
        Jj.append(tr_j.cost(3))
        ok_j += tr_j.cost(3) <= tr_s.cost(3) + 1e-9
    out["same_time"] = dict(snap_J_jerk=float(np.median(Js)), jerk_J_jerk=float(np.median(Jj)),
                            jerk_has_lower_jerk=ok_j / len(paths))
    print(f"  same-time comparison: {out['same_time']}")
    return out


# =================================================================== figures
def draw_obstacles_2d(ax, env):
    for cx, cy, h, r in env.cylinders:
        ax.add_patch(plt.Circle((cx, cy), r + 0.2, fill=False, ls=(0, (2, 1.5)), lw=0.4, ec=C_OBS_EDGE))
        ax.add_patch(plt.Circle((cx, cy), r, color=C_OBS, lw=0))


def figure_topview(env, paths, fname):
    fig, ax = plt.subplots(figsize=(8.0 * CM, 8.0 * CM))
    draw_obstacles_2d(ax, env)
    style = {"A*": (C_ASTAR, (0, (4, 1.5)), 1.0), "RRT*": (C_RRT, (0, (5, 1.5, 1, 1.5)), 1.0),
             "Lazy Theta*": (C_THETA, (0, (1, 1)), 1.4), "Ours": (C_OURS, "-", 1.6)}
    for name, p in paths.items():
        c, ls, lw = style[name]
        ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, lw=lw, label=f"{name} ({path_length(p):.2f} m)")
    ax.plot(paths["Ours"][1:-1, 0], paths["Ours"][1:-1, 1], "o", ms=3, mfc="white", mec=C_OURS, mew=0.9)
    ax.plot(*START[:2], "s", ms=5, color=INK); ax.plot(*GOAL[:2], "*", ms=8, color=INK)
    ax.annotate("start", START[:2], xytext=(4, -9), textcoords="offset points", fontsize=7, color=INK2)
    ax.annotate("goal", GOAL[:2], xytext=(-8, 6), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", handlelength=2.6, borderpad=0.4)
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


def figure_3d(env, path, traj_pts, fname):
    fig = plt.figure(figsize=(8.6 * CM, 6.4 * CM))
    # computed_zorder=False: artists are drawn in call order, so the path stays visible
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    th = np.linspace(0, 2 * np.pi, 24)
    for cx, cy, h, r in env.cylinders:
        z = np.array([0.0, h])
        T, Z = np.meshgrid(th, z)
        ax.plot_surface(cx + r * np.cos(T), cy + r * np.sin(T), Z, color="#9fc3e6", alpha=0.45,
                        linewidth=0, shade=True)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o", color=C_OURS, lw=1.6, ms=2.5, label="path", zorder=10)
    ax.plot(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2], "--", color=C_TRAJ, lw=1.0,
            label="trajectory", zorder=11)
    ax.scatter(*START, s=14, color=INK, depthshade=False); ax.scatter(*GOAL, s=24, marker="*", color=INK,
                                                                    depthshade=False)
    ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_zlim(0, 5)
    ax.set_box_aspect((20, 20, 8))
    ax.view_init(elev=32, azim=-62)
    ax.set_xlabel("x [m]", labelpad=-6); ax.set_ylabel("y [m]", labelpad=-6); ax.set_zlabel("z [m]", labelpad=-8)
    ax.tick_params(pad=-3, labelsize=6)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.92))
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


def figure_xyz(gen, fname):
    traj = gen.trajectory
    t, P = traj.sample(0.01)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.4 * CM, 7.6 * CM))
    tw, W, orig = gen.waypoint_times, gen.waypoints, gen.is_original
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        ax.plot(t, P[:, k], color=C_TRAJ, lw=1.3, label="trajectory")
        ax.plot(tw[~orig], W[~orig, k], "o", ms=2.5, color=INK2, label="subdivision points")
        ax.plot(tw[orig], W[orig, k], "o", ms=4.5, mfc=C_OURS, mec="white", mew=0.6, label="path points")
        ax.set_ylabel(f"{name} [m]"); ax.grid(True)
    axes[0].legend(loc="upper left", ncol=1, handlelength=1.6)
    axes[-1].set_xlabel("t [s]")
    axes[-1].set_xlim(0, traj.total_time)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


def figure_profiles(gen_snap, gen_jerk, v_max, a_max, fname):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.4 * CM, 5.6 * CM))
    for g, c, ls, lab in ((gen_snap, C_TRAJ, "-", "minimum snap"), (gen_jerk, C_JERK, (0, (4, 1.5)), "minimum jerk")):
        t, _ = g.trajectory.sample(0.01)
        v = np.linalg.norm(g.trajectory.evaluate(t, 1), axis=1)
        a = np.linalg.norm(g.trajectory.evaluate(t, 2), axis=1)
        axes[0].plot(t, v, color=c, ls=ls, label=lab)
        axes[1].plot(t, a, color=c, ls=ls, label=lab)
    axes[0].axhline(v_max, color=INK2, ls=":", lw=0.8); axes[1].axhline(a_max, color=INK2, ls=":", lw=0.8)
    axes[0].text(4.0, v_max + 0.04, r"$v_{\max}$", fontsize=7, color=INK2, va="bottom")
    axes[1].text(4.0, a_max + 0.04, r"$a_{\max}$", fontsize=7, color=INK2, va="bottom")
    axes[0].set_ylabel(r"$\|\dot{\mathbf{p}}\|$ [m/s]"); axes[1].set_ylabel(r"$\|\ddot{\mathbf{p}}\|$ [m/s$^2$]")
    axes[1].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(True)
    axes[0].set_ylim(0, v_max * 1.2); axes[1].set_ylim(0, a_max * 1.25)
    axes[0].legend(loc="lower center", ncol=2)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


def figure_tradeoff(study, fname):
    fig, ax = plt.subplots(figsize=(8.0 * CM, 4.8 * CM))
    for res, c, mk in ((0.2, C_OURS, "o"), (0.3, C_ASTAR, "s")):
        rows = sorted([v for v in study.values() if abs(v["res"] - res) < 1e-9], key=lambda r: r["w"])
        x = [r["time_ms"] for r in rows]; y = [r["excess"] for r in rows]
        ax.plot(x, y, "-", marker=mk, color=c, ms=4, mec="white", mew=0.6, label=f"resolution {res} m")
        for r in rows:
            ax.annotate(f"{r['w']:g}", (r["time_ms"], r["excess"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.margins(x=0.08, y=0.12)
    ax.set_xlim(left=min(8.0, ax.get_xlim()[0]))
    ax.set_xlabel("mean planning time [ms] (log scale)")
    ax.set_ylabel("length above lower bound [%]")
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


def representative_figures(seed, n_obs):
    env = make_env(seed, n_obs)
    ours = PathPlanner(env, START, GOAL, heuristic_weight=1.1)
    p_ours = ours.plan()
    a = AStarPlanner(env, START, GOAL); p_a = a.plan()
    r = RRTStarPlanner(env, START, GOAL, seed=seed); p_r = r.plan()
    lt = PathPlanner(env, START, GOAL, heuristic_weight=1.1, refine=False); p_lt = lt.plan()
    figure_topview(env, {"A*": p_a, "RRT*": p_r, "Lazy Theta*": p_lt, "Ours": p_ours}, "fig_topview.pdf")

    g = TrajectoryGenerator(p_ours, env, order=4); g.generate()
    gj = TrajectoryGenerator(p_ours, env, order=3); gj.generate()
    _, pts = g.trajectory.sample(0.02)
    figure_3d(env, p_ours, pts, "fig_3d.png")
    figure_xyz(g, "fig_xyz.pdf")
    figure_profiles(g, gj, g.v_max, g.a_max, "fig_profiles.pdf")
    info = dict(seed=seed, n_obs=n_obs, path=p_ours.tolist(), length=path_length(p_ours),
                plan_stats={k: float(v) for k, v in ours.stats.items()},
                traj_stats={k: float(v) for k, v in g.stats.items()},
                jerk_stats={k: float(v) for k, v in gj.stats.items()},
                lengths={"A*": path_length(p_a), "RRT*": path_length(p_r), "LazyTheta*": path_length(p_lt),
                         "Ours": path_length(p_ours)})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--scene-seed", type=int, default=4)
    ap.add_argument("--scene-obstacles", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n_seeds = 20 if args.quick else args.seeds

    cases = []
    for n_obs in (50, 100):
        for seed in range(n_seeds):
            env = make_env(seed, n_obs)
            if env is not None:
                cases.append((seed, n_obs, env))
    print(f"{len(cases)} feasible random environments")

    results = {"n_cases": len(cases), "lower_bound": LOWER_BOUND}
    print("[1/4] representative scene figures")
    results["scene"] = representative_figures(args.scene_seed, args.scene_obstacles)
    print("[2/4] planner comparison")
    results["planners"] = {}
    for n_obs in (50, 100):
        sub = [c for c in cases if c[1] == n_obs]
        results["planners"][n_obs] = run_planner_comparison(sub)
    print("[3/4] heuristic weight / resolution study")
    results["weights"] = run_weight_study(cases)
    figure_tradeoff(results["weights"], "fig_tradeoff.pdf")
    print("[4/4] trajectory study")
    results["trajectory"] = run_trajectory_study(cases)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    # LaTeX rows for the report tables
    lines = []
    for n_obs, summ in results["planners"].items():
        lines.append(f"% N = {n_obs}")
        for m, s in summ.items():
            lines.append(f"{m} & {100 * s['success']:.0f} & {s['len_mean']:.3f}$\\pm${s['len_std']:.3f} & "
                         f"{s['excess']:.2f} & {s['turn']:.1f} & {s['vertices']:.1f} & {s['time_ms']:.1f} \\\\")
    lines.append("% trajectory")
    for name, s in results["trajectory"].items():
        if name == "same_time":
            lines.append("% same time allocation: " + json.dumps(s))
            continue
        lines.append(f"{name} & {s['total_time']:.2f} & {s['J_jerk']:.1f} & {s['J_snap']:.1f} & "
                     f"{s['a_peak']:.2f} & {s['inserted']:.2f} & {s['collision_free']:.1f} & {s['time_ms']:.1f} \\\\")
    with open(os.path.join(OUT, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

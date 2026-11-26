"""
Smooth trajectory generation through the planned path points.

Implemented method (written from scratch for this assignment):

    * Each axis is a piecewise polynomial of degree 2r-1 (r = 4 -> 7th order,
      minimum snap; r = 3 -> 5th order, minimum jerk).  Every segment uses a
      normalised time tau = (t - t_i) / T_i in [0, 1] for numerical conditioning.
    * The coefficients minimise  J = sum_i  integral_0^{T_i} ||p^{(r)}(t)||^2 dt
      subject to linear equality constraints:
          - the trajectory passes exactly through every path point,
          - it starts and ends at rest (derivatives 1 .. r-1 are zero),
          - derivatives 1 .. r-1 are continuous at interior path points.
      This equality-constrained QP is solved in closed form through its KKT
      linear system (one factorisation shared by the x, y and z axes).
    * Time allocation: a trapezoidal velocity profile (v_max, a_max) is laid
      along the cumulative arc length of the path; the time at which the profile
      reaches each path point becomes that point's time stamp.  Afterwards all
      durations are uniformly stretched if the optimised trajectory exceeds
      v_max or a_max (dynamic feasibility by time scaling).
    * Safety: the trajectory is sampled densely and checked against the
      obstacles.  If a polynomial segment leaves the free space, the midpoint of
      the corresponding (collision-free) straight path segment is inserted as an
      extra waypoint and the problem is re-solved (Richter et al. style).

Only numpy / math / matplotlib are used.
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from path_planner import CollisionChecker


def _derivative_coeffs(n, d):
    """Factors k!/(k-d)! for the d-th derivative of tau^k, k = 0..n (zero for k < d)."""
    out = np.zeros(n + 1)
    for k in range(d, n + 1):
        out[k] = math.factorial(k) / math.factorial(k - d)
    return out


def _basis(n, d, tau):
    """Row vector of the d-th derivative of [1, tau, ..., tau^n] w.r.t. tau."""
    c = _derivative_coeffs(n, d)
    row = np.zeros(n + 1)
    for k in range(d, n + 1):
        row[k] = c[k] * tau ** (k - d)
    return row


def _cost_matrix(n, r):
    """Q_hat with  integral_0^1 (d^r p / d tau^r)^2 d tau = c^T Q_hat c."""
    Q = np.zeros((n + 1, n + 1))
    c = _derivative_coeffs(n, r)
    for i in range(r, n + 1):
        for j in range(r, n + 1):
            Q[i, j] = c[i] * c[j] / (i + j - 2 * r + 1)
    return Q


def trapezoidal_time_allocation(waypoints, v_max, a_max, t_min=0.0):
    """Segment durations from a trapezoidal speed profile laid along the path length.

    The time stamp of a waypoint is the time at which the profile reaches its arc
    length.  Each duration is bounded below by t_min so that a very short path
    segment (e.g. a few millimetres next to an obstacle) does not get an almost
    zero duration, which would cause excessive derivatives.
    """
    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L * a_max >= v_max ** 2:                 # cruise phase exists
        t_acc, s_acc, v_peak = v_max / a_max, 0.5 * v_max ** 2 / a_max, v_max
    else:                                        # triangular profile
        t_acc = math.sqrt(L / a_max)
        s_acc, v_peak = 0.5 * L, a_max * t_acc
    t_total = 2 * t_acc + (L - 2 * s_acc) / v_peak
    times = np.empty_like(s)
    for i, si in enumerate(s):
        if si <= s_acc:
            times[i] = math.sqrt(2 * si / a_max)
        elif si <= L - s_acc:
            times[i] = t_acc + (si - s_acc) / v_peak
        else:
            times[i] = t_total - math.sqrt(max(2 * (L - si) / a_max, 0.0))
    return np.maximum(np.diff(times), max(t_min, 1e-6))


class PolynomialTrajectory:
    """Piecewise polynomial in normalised time; evaluation of position and derivatives."""

    def __init__(self, coeffs, durations):
        self.coeffs = coeffs                    # (M, n+1, 3)
        self.durations = np.asarray(durations, dtype=float)
        self.knots = np.concatenate([[0.0], np.cumsum(self.durations)])
        self.n = coeffs.shape[1] - 1

    @property
    def total_time(self):
        return float(self.knots[-1])

    def evaluate(self, t, d=0):
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, self.total_time)
        idx = np.clip(np.searchsorted(self.knots, t, side="right") - 1, 0, len(self.durations) - 1)
        T = self.durations[idx]
        tau = (t - self.knots[idx]) / T
        fac = _derivative_coeffs(self.n, d)
        powers = np.zeros((len(t), self.n + 1))
        for k in range(d, self.n + 1):
            powers[:, k] = fac[k] * tau ** (k - d)
        vals = np.einsum("tk,tkc->tc", powers, self.coeffs[idx])
        return vals / (T ** d)[:, None]

    def sample(self, dt=0.01):
        t = np.arange(0.0, self.total_time + 1e-12, dt)
        if t[-1] < self.total_time:
            t = np.append(t, self.total_time)
        return t, self.evaluate(t)

    def cost(self, r):
        """Integral of the squared r-th derivative norm (analytic)."""
        Q = _cost_matrix(self.n, r)
        J = 0.0
        for c, T in zip(self.coeffs, self.durations):
            J += np.trace(c.T @ Q @ c) * T ** (1 - 2 * r)
        return float(J)


class TrajectoryGenerator:
    """Minimum-snap (or minimum-jerk) trajectory through the path points.

    Usage:
        gen = TrajectoryGenerator(path, env)
        traj = gen.generate()            # PolynomialTrajectory
        t, pos = traj.sample(0.01)
    """

    def __init__(self, path, env=None, v_max=2.0, a_max=2.0, order=4, max_seg_len=2.0, t_min=0.0,
                 local_time_iter=8, check_collision=True, dt_check=0.01, max_insertions=30,
                 epsilon=0.2):
        self.path = np.asarray(path, dtype=float)
        self.env = env
        self.v_max = v_max
        self.a_max = a_max
        self.max_seg_len = max_seg_len
        self.t_min = t_min
        self.local_time_iter = local_time_iter
        self.r = order                       # 4: minimum snap, 3: minimum jerk
        self.n = 2 * order - 1               # polynomial degree
        self.dt_check = dt_check
        self.max_insertions = max_insertions
        self.checker = CollisionChecker(env, epsilon=epsilon) if (env is not None and check_collision) else None
        self.stats = {}

    # ----------------------------------------------------------------- public
    def generate(self):
        t0 = time.perf_counter()
        waypoints, original = self._subdivide(self.path)
        n_inserted = 0
        while True:
            durations = trapezoidal_time_allocation(waypoints, self.v_max, self.a_max, self.t_min)
            traj = self._solve(waypoints, durations)
            traj = self._local_time_adjustment(traj, waypoints)
            traj, scale = self._time_scaling(traj)
            bad = self._colliding_segments(traj)
            if len(bad) == 0 or n_inserted >= self.max_insertions:
                break
            # insert midpoints of the offending straight path segments (back to front)
            for i in sorted(bad, reverse=True):
                mid = 0.5 * (waypoints[i] + waypoints[i + 1])
                waypoints = np.insert(waypoints, i + 1, mid, axis=0)
                original = np.insert(original, i + 1, False)
                n_inserted += 1

        self.waypoints = waypoints
        self.is_original = original
        self.waypoint_times = traj.knots.copy()
        self.trajectory = traj
        t_s, _ = traj.sample(self.dt_check)
        v = np.linalg.norm(traj.evaluate(t_s, 1), axis=1)
        a = np.linalg.norm(traj.evaluate(t_s, 2), axis=1)
        self.stats = dict(time=time.perf_counter() - t0, inserted=n_inserted,
                          subdivided=int(np.sum(~original)) - n_inserted, segments=len(durations),
                          total_time=traj.total_time, time_scale=scale, v_peak=float(v.max()),
                          a_peak=float(a.max()), collision_free=len(bad) == 0,
                          J_jerk=traj.cost(3), J_snap=traj.cost(4))
        return traj

    def _subdivide(self, path):
        """Split straight path segments longer than max_seg_len into equal pieces.

        A single polynomial piece cannot hold a constant cruise speed over a long
        straight segment; the extra points lie on the collision-free segment.
        """
        pts, orig = [path[0]], [True]
        for a, b in zip(path[:-1], path[1:]):
            m = max(int(math.ceil(np.linalg.norm(b - a) / self.max_seg_len - 1e-9)), 1) \
                if np.isfinite(self.max_seg_len) else 1
            for k in range(1, m):
                pts.append(a + (b - a) * k / m)
                orig.append(False)
            pts.append(b)
            orig.append(True)
        return np.array(pts), np.array(orig)

    # ------------------------------------------------------------ QP via KKT
    def _solve(self, waypoints, durations):
        n, r = self.n, self.r
        M = len(durations)
        nc = n + 1
        nv = M * nc
        Qh = _cost_matrix(n, r)
        scale = np.mean(durations) ** (2 * r - 1)          # cost normalisation (does not move the optimum)
        H = np.zeros((nv, nv))
        for i, T in enumerate(durations):
            H[i * nc:(i + 1) * nc, i * nc:(i + 1) * nc] = Qh * (scale * T ** (1 - 2 * r))

        rows, rhs = [], []

        def add(seg, d, tau, coef=1.0, row=None):
            if row is None:
                row = np.zeros(nv)
            row[seg * nc:(seg + 1) * nc] += coef * _basis(n, d, tau) / durations[seg] ** d
            return row

        zero = np.zeros(3)
        # boundary conditions: start/end position, rest for derivatives 1..r-1
        for d in range(r):
            rows.append(add(0, d, 0.0)); rhs.append(waypoints[0] if d == 0 else zero)
            rows.append(add(M - 1, d, 1.0)); rhs.append(waypoints[-1] if d == 0 else zero)
        # interior points: pass through + continuity of derivatives 1..r-1
        for j in range(1, M):
            rows.append(add(j - 1, 0, 1.0)); rhs.append(waypoints[j])
            rows.append(add(j, 0, 0.0)); rhs.append(waypoints[j])
            for d in range(1, r):
                row = add(j - 1, d, 1.0)
                row = add(j, d, 0.0, coef=-1.0, row=row)
                rows.append(row); rhs.append(zero)
        A = np.vstack(rows)
        b = np.vstack(rhs)
        m = A.shape[0]
        K = np.zeros((nv + m, nv + m))
        K[:nv, :nv] = 2.0 * H
        K[:nv, nv:] = A.T
        K[nv:, :nv] = A
        rhs_full = np.vstack([np.zeros((nv, 3)), b])
        sol = np.linalg.solve(K, rhs_full)
        coeffs = sol[:nv].reshape(M, nc, 3)
        return PolynomialTrajectory(coeffs, durations)

    # -------------------------------------------------------- feasibility
    def _segment_ratios(self, traj):
        """Per segment: max(|v|/v_max, sqrt(|a|/a_max)) over dense samples."""
        t, _ = traj.sample(self.dt_check)
        v = np.linalg.norm(traj.evaluate(t, 1), axis=1) / self.v_max
        a = np.sqrt(np.linalg.norm(traj.evaluate(t, 2), axis=1) / self.a_max)
        seg = np.clip(np.searchsorted(traj.knots, t, side="right") - 1, 0, len(traj.durations) - 1)
        ratio = np.zeros(len(traj.durations))
        np.maximum.at(ratio, seg, np.maximum(v, a))
        return ratio

    def _local_time_adjustment(self, traj, waypoints):
        """Stretch only the segments that violate the limits and re-solve.

        A uniform time scaling is dictated by the single worst segment; stretching
        the violating segments first keeps the rest of the trajectory fast.
        """
        durations = traj.durations.copy()
        for _ in range(self.local_time_iter):
            ratio = self._segment_ratios(traj)
            viol = ratio > 1.0 + 1e-3
            if not np.any(viol):
                break
            durations[viol] *= np.minimum(ratio[viol], 1.5)
            traj = self._solve(waypoints, durations)
        return traj

    def _time_scaling(self, traj):
        """Uniformly stretch time so that |v| <= v_max and |a| <= a_max (guarantee)."""
        k = max(1.0, float(np.max(self._segment_ratios(traj))))
        k = k * 1.002 if k > 1.0 else k          # small allowance for the discrete sampling
        if k > 1.0 + 1e-6:
            # p(t) -> p(t/k): same normalised coefficients, durations multiplied by k
            traj = PolynomialTrajectory(traj.coeffs, traj.durations * k)
        return traj, k

    def _colliding_segments(self, traj):
        if self.checker is None:
            return []
        t, pts = traj.sample(self.dt_check)
        pts = np.where(np.abs(pts) < 1e-9, 0.0, pts)           # numerical noise at z = 0
        bad = ~self.checker.points_free(pts)
        if not np.any(bad):
            return []
        seg = np.clip(np.searchsorted(traj.knots, t[bad], side="right") - 1, 0, len(traj.durations) - 1)
        return sorted(set(seg.tolist()))


# ---------------------------------------------------------------------- plot
def plot_trajectory(generator, dt=0.01, show_inserted=True, fig=None):
    """Three stacked subplots x(t), y(t), z(t) with the path points marked at their time stamps."""
    traj = generator.trajectory
    t, P = traj.sample(dt)
    if fig is None:
        fig = plt.figure(figsize=(8, 7))
    axes = fig.subplots(3, 1, sharex=True)
    orig = generator.is_original
    tw = generator.waypoint_times
    W = generator.waypoints
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        ax.plot(t, P[:, k], color="tab:blue", lw=1.8, label="trajectory")
        ax.plot(tw[orig], W[orig, k], "o", ms=6, mfc="tab:red", mec="k", label="path points")
        if show_inserted and np.any(~orig):
            ax.plot(tw[~orig], W[~orig, k], "x", ms=6, color="k", label="inserted points")
        ax.set_ylabel(f"{name} [m]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title("Minimum-snap trajectory and planned path points")
    fig.tight_layout()
    return fig, axes

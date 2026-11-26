"""
Path planning for the cylinder-cluttered flight environment.

Implemented method (written from scratch for this assignment):

    1. Voxel occupancy map: the environment is discretised into a regular 3-D grid.
       A node is occupied if it is outside the workspace (`env.is_outside`) or
       lies inside an obstacle inflated by the safety distance used in
       `env.is_collide` plus a small extra margin.  The map is built with
       vectorised numpy operations instead of calling `is_collide` per node.
    2. Lazy Theta* search: an any-angle variant of A* on the 26-connected grid.
       A node may take the parent of its predecessor as its own parent whenever
       the straight segment between them is collision free, so the search
       directly returns a sparse polyline whose headings are not restricted to
       the 26 grid directions.  Line-of-sight (LOS) is checked analytically
       against the cylinders (continuous, not sampled).
    3. Post-optimisation: redundant vertices are removed (LOS shortcutting) and
       the remaining vertices are iteratively "pulled taut" toward the chord of
       their neighbours while the path stays collision free.  The length is
       non-increasing at every step and converges to a locally shortest path in
       the homotopy class found by the global search.

Only numpy / math / heapq from the Python standard scientific stack are used.
"""

import heapq
import math
import time

import numpy as np


class CollisionChecker:
    """Vectorised, analytic collision queries against vertical cylinders.

    The collision model is identical to `FlightEnvironment.is_collide`: a point
    collides with cylinder (cx, cy, h, r) if 0 <= z <= h and its horizontal
    distance to (cx, cy) is <= r + epsilon.  An additional `margin` can be added
    to keep the path slightly away from the exact safety boundary.
    """

    def __init__(self, env, epsilon=0.2, margin=0.0):
        cyl = np.asarray(env.cylinders, dtype=float)
        self.cx = cyl[:, 0].copy()
        self.cy = cyl[:, 1].copy()
        self.h = cyl[:, 2].copy()
        self.R = cyl[:, 3] + epsilon + margin          # inflated radius
        self.R2 = self.R ** 2
        self.bounds = np.array([env.env_width, env.env_length, env.env_height], dtype=float)
        self.n_los = 0                                  # statistics: number of LOS queries

    # ------------------------------------------------------------------ points
    def inside_bounds(self, p):
        return bool(np.all(p >= 0.0) and np.all(p <= self.bounds))

    def points_free(self, pts):
        """Boolean mask, True where a point is inside the workspace and collision free."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        inside = np.all((pts >= 0.0) & (pts <= self.bounds), axis=1)
        dx = pts[:, 0:1] - self.cx[None, :]
        dy = pts[:, 1:2] - self.cy[None, :]
        z = pts[:, 2:3]
        hit = ((dx * dx + dy * dy) <= self.R2[None, :]) & (z >= 0.0) & (z <= self.h[None, :])
        return inside & ~np.any(hit, axis=1)

    # ---------------------------------------------------------------- segments
    def segment_free(self, p, q):
        """Exact test whether the straight segment p->q is collision free.

        For every cylinder we first clip the segment parameter t in [0, 1] to the
        interval in which 0 <= z(t) <= h.  On that interval the squared horizontal
        distance to the axis is a convex quadratic in t, so its minimum is found by
        clamping the unconstrained minimiser.  The workspace is a box (convex), so
        a segment is inside it iff both end points are.
        """
        self.n_los += 1
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        qx, qy, qz = float(q[0]), float(q[1]), float(q[2])
        W, L, H = self.bounds
        if not (0.0 <= px <= W and 0.0 <= py <= L and 0.0 <= pz <= H and
                0.0 <= qx <= W and 0.0 <= qy <= L and 0.0 <= qz <= H):
            return False
        dx, dy, dz = qx - px, qy - py, qz - pz
        # --- admissible t-interval from the height constraint 0 <= z <= h
        if abs(dz) < 1e-12:
            lo = 0.0
            hi = np.where(pz <= self.h, 1.0, -1.0)
        else:
            ta = -pz / dz
            tb = (self.h - pz) / dz
            if dz > 0:
                lo = max(ta, 0.0)
                hi = np.minimum(tb, 1.0)
            else:
                lo = np.maximum(tb, 0.0)
                hi = min(ta, 1.0)
        # --- closest approach in the xy-plane on [lo, hi]
        ox = px - self.cx
        oy = py - self.cy
        dxy2 = dx * dx + dy * dy
        if dxy2 < 1e-12:
            t = lo
        else:
            t = (ox * dx + oy * dy) * (-1.0 / dxy2)
            t = np.minimum(np.maximum(t, lo), hi)
        ex = ox + t * dx
        ey = oy + t * dy
        hit = ((ex * ex + ey * ey) <= self.R2) & (lo <= hi)
        return not hit.any()

    def path_free(self, path):
        return all(self.segment_free(path[i], path[i + 1]) for i in range(len(path) - 1))


def path_length(path):
    path = np.asarray(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


class OccupancyGrid:
    """Padded 3-D voxel grid stored as a flat array (one blocked layer on every side).

    Node (i, j, k) of the workspace has coordinates (i*res, j*res, k*res).  The
    padding removes all bounds checks from the inner loop of the graph search:
    every neighbour index `s + offset` is valid and border nodes are blocked.
    """

    def __init__(self, env, checker, resolution):
        self.res = float(resolution)
        W, L, H = env.env_width, env.env_length, env.env_height
        self.n = np.array([int(math.floor(W / self.res + 1e-9)) + 1,
                           int(math.floor(L / self.res + 1e-9)) + 1,
                           int(math.floor(H / self.res + 1e-9)) + 1])
        nx, ny, nz = self.n
        self.shape = (nx + 2, ny + 2, nz + 2)
        self.sy = self.shape[2]
        self.sx = self.shape[1] * self.shape[2]

        free = np.zeros(self.shape, dtype=bool)
        free[1:-1, 1:-1, 1:-1] = True
        xs = np.arange(nx) * self.res
        ys = np.arange(ny) * self.res
        zs = np.arange(nz) * self.res
        # Extra inflation so that the straight edge between two adjacent free nodes
        # is always collision free (bounded by the sagitta of a chord of length
        # res*sqrt(2) in the xy-plane, and one layer of nodes above the cylinder top).
        for cx, cy, h, R in zip(checker.cx, checker.cy, checker.h, checker.R):
            Rg = R + (R - math.sqrt(max(R * R - 0.5 * self.res ** 2, 0.0))) + 1e-6
            i0, i1 = max(int(math.floor((cx - Rg) / self.res)), 0), min(int(math.ceil((cx + Rg) / self.res)), nx - 1)
            j0, j1 = max(int(math.floor((cy - Rg) / self.res)), 0), min(int(math.ceil((cy + Rg) / self.res)), ny - 1)
            k1 = min(int(math.floor((h + self.res) / self.res + 1e-9)), nz - 1)
            if i0 > i1 or j0 > j1 or k1 < 0:
                continue
            gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
            disk = (gx - cx) ** 2 + (gy - cy) ** 2 <= Rg * Rg
            sub = free[1 + i0:2 + i1, 1 + j0:2 + j1, 1:2 + k1]
            sub &= ~disk[:, :, None]
        self.free = free.ravel()

        # 26-connected neighbourhood as flat-index offsets with metric edge costs
        self.neighbours = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == dj == dk == 0:
                        continue
                    off = di * self.sx + dj * self.sy + dk
                    self.neighbours.append((off, self.res * math.sqrt(di * di + dj * dj + dk * dk)))

    def to_index(self, p):
        i, j, k = (int(round(c / self.res)) for c in p)
        return (i + 1) * self.sx + (j + 1) * self.sy + (k + 1)

    def to_point(self, s):
        i, rem = divmod(s, self.sx)
        j, k = divmod(rem, self.sy)
        return np.array([(i - 1) * self.res, (j - 1) * self.res, (k - 1) * self.res])

    def nearest_free_visible(self, p, checker, max_radius=3):
        """Closest free node that can be connected to p by a free segment."""
        base = np.round(np.asarray(p) / self.res).astype(int)
        best, best_d = None, np.inf
        for rad in range(0, max_radius + 1):
            rng = range(-rad, rad + 1)
            for di in rng:
                for dj in rng:
                    for dk in rng:
                        if max(abs(di), abs(dj), abs(dk)) != rad:
                            continue
                        idx = base + np.array([di, dj, dk])
                        if np.any(idx < 0) or np.any(idx >= self.n):
                            continue
                        s = (idx[0] + 1) * self.sx + (idx[1] + 1) * self.sy + (idx[2] + 1)
                        if not self.free[s]:
                            continue
                        q = idx * self.res
                        d = float(np.linalg.norm(q - p))
                        if d < best_d and checker.segment_free(p, q):
                            best, best_d = s, d
            if best is not None:
                return best
        return None


class PathPlanner:
    """Lazy Theta* on a 3-D occupancy grid followed by path shortening.

    Usage:
        planner = PathPlanner(env, start, goal)
        path = planner.plan()          # N x 3 numpy array, start ... goal
    """

    def __init__(self, env, start, goal, resolution=0.2, margin=0.05, epsilon=0.2,
                 heuristic_weight=1.0, refine=True, max_refine_iter=100):
        self.env = env
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.resolution = resolution
        self.heuristic_weight = heuristic_weight
        self.refine = refine
        self.max_refine_iter = max_refine_iter
        # The extra margin must not make a valid start/goal infeasible.  For every
        # obstacle the margin is reduced to half of the clearance of the end points,
        # so a start lying close to one cylinder keeps some room to leave it while
        # all other cylinders keep the full margin.
        base = CollisionChecker(env, epsilon=epsilon, margin=0.0)
        clearance = np.minimum(self._clearance(base, self.start), self._clearance(base, self.goal))
        self.margin = np.clip(np.minimum(margin, 0.5 * clearance), 0.0, None)
        self.checker = CollisionChecker(env, epsilon=epsilon, margin=self.margin)
        self.stats = {}

    @staticmethod
    def _clearance(checker, p):
        """Per cylinder: horizontal distance from p to its inflated surface (inf if p is above it)."""
        d = np.hypot(p[0] - checker.cx, p[1] - checker.cy) - checker.R
        return np.where((p[2] >= 0.0) & (p[2] <= checker.h), d, np.inf)

    # ----------------------------------------------------------------- public
    def plan(self):
        t0 = time.perf_counter()
        for name, p in (("start", self.start), ("goal", self.goal)):
            if self.env.is_outside(p) or self.env.is_collide(p) or not self.checker.points_free(p)[0]:
                raise ValueError(f"The {name} point {tuple(p)} is outside the map or in collision.")

        # Trivial case: direct line of sight
        if self.checker.segment_free(self.start, self.goal):
            path = np.vstack([self.start, self.goal])
            self.stats.update(t_grid=0.0, t_search=0.0, t_refine=0.0, expanded=0,
                              raw_length=path_length(path), raw_vertices=2,
                              t_total=time.perf_counter() - t0, length=path_length(path),
                              vertices=2, los_checks=self.checker.n_los)
            return path

        t1 = time.perf_counter()
        self.grid = OccupancyGrid(self.env, self.checker, self.resolution)
        t2 = time.perf_counter()

        s = self.grid.nearest_free_visible(self.start, self.checker)
        g = self.grid.nearest_free_visible(self.goal, self.checker)
        if s is None or g is None:
            raise RuntimeError("Start or goal cannot be connected to the search grid.")
        nodes = self._lazy_theta_star(s, g)
        if nodes is None:
            raise RuntimeError("No collision-free path exists between start and goal.")
        # exact start/goal + the grid vertices (the snapped end nodes are visible from
        # start/goal by construction; redundant vertices are removed by the pruning)
        path = [self.start.copy()] + [self.grid.to_point(n) for n in nodes] + [self.goal.copy()]
        path = [p for i, p in enumerate(path) if i == 0 or np.linalg.norm(p - path[i - 1]) > 1e-12]
        t3 = time.perf_counter()
        self.stats["raw_length"] = path_length(path)
        self.stats["raw_vertices"] = len(path)

        path = self._prune(path)
        if self.refine:
            path = self._pull_taut(path)
            path = self._merge_close_vertices(path)
        t4 = time.perf_counter()

        path = np.vstack(path)
        self.stats.update(t_grid=t2 - t1, t_search=t3 - t2, t_refine=t4 - t3,
                          t_total=t4 - t0, length=path_length(path), vertices=len(path),
                          los_checks=self.checker.n_los)
        return path

    # ------------------------------------------------------- Lazy Theta* core
    def _lazy_theta_star(self, s_start, s_goal):
        grid, checker = self.grid, self.checker
        free = grid.free.tolist()
        nbrs = grid.neighbours
        sx, sy, res = grid.sx, grid.sy, grid.res
        segment_free = checker.segment_free
        sqrt = math.sqrt
        heappush, heappop = heapq.heappush, heapq.heappop

        def xyz(n):                       # flat index -> metric coordinates
            i, rem = divmod(n, sx)
            j, k = divmod(rem, sy)
            return (i - 1) * res, (j - 1) * res, (k - 1) * res

        gx, gy, gz = xyz(s_goal)
        w = self.heuristic_weight
        inf = math.inf
        N = len(free)
        g = [inf] * N
        parent = [-1] * N
        closed = bytearray(N)

        x0, y0, z0 = xyz(s_start)
        g[s_start] = 0.0
        parent[s_start] = s_start
        heap = [(sqrt((x0 - gx) ** 2 + (y0 - gy) ** 2 + (z0 - gz) ** 2), 0.0, s_start)]
        expanded = 0

        while heap:
            _, _, s = heappop(heap)
            if closed[s]:
                continue
            # --- SetVertex: verify the lazily assumed line of sight to the parent
            p = parent[s]
            if p != s and not segment_free(xyz(p), xyz(s)):
                best, best_p = inf, -1
                for off, c in nbrs:
                    n = s + off
                    if closed[n]:
                        v = g[n] + c
                        if v < best:
                            best, best_p = v, n
                if best_p < 0:          # cannot happen with the inflated grid, kept for safety
                    continue
                g[s] = best
                parent[s] = best_p
            closed[s] = 1
            expanded += 1
            if s == s_goal:
                break
            # --- expand: neighbours inherit the parent of s (path 2 of Theta*)
            ps = parent[s]
            gps = g[ps]
            px, py, pz = xyz(ps)
            for off, _c in nbrs:
                n = s + off
                if not free[n] or closed[n]:
                    continue
                i, rem = divmod(n, sx)
                j, k = divmod(rem, sy)
                x, y, z = (i - 1) * res, (j - 1) * res, (k - 1) * res
                new_g = gps + sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
                if new_g < g[n]:
                    g[n] = new_g
                    parent[n] = ps
                    heappush(heap, (new_g + w * sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2),
                                    -new_g, n))
        else:
            return None

        self.stats["expanded"] = expanded
        nodes = [s_goal]
        while nodes[-1] != s_start:
            nodes.append(parent[nodes[-1]])
        return nodes[::-1]

    # -------------------------------------------------------- post-processing
    def _prune(self, path):
        """Greedy line-of-sight shortcutting: keep only necessary vertices."""
        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not self.checker.segment_free(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _pull_taut(self, path, tol=1e-4, bisect_steps=8):
        """Iteratively move each vertex toward the chord of its neighbours.

        Moving vertex b on the straight line toward the closest point of chord a-c
        never increases |a-b| + |b-c| (convexity), so the path length decreases
        monotonically.  The step is the largest fraction (found by bisection) that
        keeps both adjacent segments collision free.
        """
        P = [np.asarray(p, dtype=float) for p in path]
        seg_free = self.checker.segment_free
        for _ in range(self.max_refine_iter):
            gain = 0.0
            i = 1
            while i < len(P) - 1:
                a, b, c = P[i - 1], P[i], P[i + 1]
                if seg_free(a, c):              # vertex became redundant
                    gain += np.linalg.norm(a - b) + np.linalg.norm(b - c) - np.linalg.norm(a - c)
                    del P[i]
                    continue
                ac = c - a
                t = float(np.clip(np.dot(b - a, ac) / max(np.dot(ac, ac), 1e-12), 0.0, 1.0))
                target = a + t * ac
                step = target - b
                if np.dot(step, step) > 1e-10:
                    lo, hi = 0.0, 1.0
                    for _ in range(bisect_steps):
                        mid = 0.5 * (lo + hi)
                        cand = b + mid * step
                        if seg_free(a, cand) and seg_free(cand, c):
                            lo = mid
                        else:
                            hi = mid
                    if lo > 0.0:
                        nb = b + lo * step
                        gain += (np.linalg.norm(a - b) + np.linalg.norm(b - c)
                                 - np.linalg.norm(a - nb) - np.linalg.norm(nb - c))
                        P[i] = nb
                # height-only relaxation: |a-b| + |b-c| is convex in b_z with its
                # minimum where z varies linearly with horizontal arc length.
                b = P[i]
                h1, h2 = np.linalg.norm((b - a)[:2]), np.linalg.norm((c - b)[:2])
                if h1 + h2 > 1e-9:
                    z_star = a[2] + (c[2] - a[2]) * h1 / (h1 + h2)
                    dz = z_star - b[2]
                    if abs(dz) > 1e-6:
                        for frac in (1.0, 0.5, 0.25):
                            nb = b.copy()
                            nb[2] += frac * dz
                            if seg_free(a, nb) and seg_free(nb, c):
                                gain += (np.linalg.norm(a - b) + np.linalg.norm(b - c)
                                         - np.linalg.norm(a - nb) - np.linalg.norm(nb - c))
                                P[i] = nb
                                break
                i += 1
            if gain < tol:
                break
        return P

    def _merge_close_vertices(self, P, d_min=0.3):
        """Replace pairs of nearly coincident vertices by a single vertex.

        Pulling taut around one obstacle can leave two vertices a few millimetres
        apart.  They do not change the length but would force a very short
        trajectory segment.  The pair (b, c) between a and d is replaced by the
        first collision-free candidate among: intersection of the extended edges
        a->b and d->c, the midpoint of b and c, b, c.
        """
        P = list(P)
        seg_free = self.checker.segment_free
        i = 1
        while i < len(P) - 2:
            a, b, c, d = P[i - 1], P[i], P[i + 1], P[i + 2]
            if np.linalg.norm(c - b) >= d_min:
                i += 1
                continue
            cands = []
            u, v = b - a, c - d                          # edge directions toward the pair
            A = np.array([[u @ u, -u @ v], [u @ v, -v @ v]])
            if abs(np.linalg.det(A)) > 1e-9:
                s, w = np.linalg.solve(A, np.array([(d - a) @ u, (d - a) @ v]))
                if s > 0 and w > 0:
                    x = 0.5 * ((a + s * u) + (d + w * v))
                    if np.linalg.norm(x - 0.5 * (b + c)) < d_min:
                        cands.append(x)
            cands += [0.5 * (b + c), b, c]
            for x in cands:
                if seg_free(a, x) and seg_free(x, d):
                    P[i:i + 2] = [x]
                    break
            else:
                i += 1
        # End segments: the start/goal cannot move.  When the start lies close to
        # an obstacle the shortest path first slides a few millimetres tangentially;
        # push that vertex further along the same (tangential) direction instead.
        for first in (True, False):
            if len(P) < 3:
                break
            e, v, nxt = (P[0], P[1], P[2]) if first else (P[-1], P[-2], P[-3])
            L = np.linalg.norm(v - e)
            if L >= d_min or L < 1e-12:
                continue
            u = (v - e) / L
            for k in np.linspace(d_min, L, 6):
                x = e + k * u
                if seg_free(e, x) and seg_free(x, nxt):
                    if first:
                        P[1] = x
                    else:
                        P[-2] = x
                    break
        return P


def validate_path(env, path, step=0.05):
    """Densely re-check a polyline with the environment's own is_outside / is_collide."""
    path = np.asarray(path, dtype=float)
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            p = a + t * (b - a)
            if env.is_outside(p) or env.is_collide(p):
                return False
    return True

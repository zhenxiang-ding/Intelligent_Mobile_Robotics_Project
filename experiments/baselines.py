"""
Baseline planners used only for the comparison experiments in the report
(written from scratch, sharing the collision model of path_planner.py).

    * AStarPlanner   : classic A* on the same 26-connected occupancy grid,
                       exact 3-D octile heuristic.
    * RRTStarPlanner : RRT* in continuous space (Karaman & Frazzoli, 2011)
                       with goal biasing, choose-parent and rewiring.
"""

import heapq
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_planner import CollisionChecker, OccupancyGrid, PathPlanner, path_length  # noqa: E402


class AStarPlanner(PathPlanner):
    """26-connected grid A*; optional line-of-sight pruning of the grid path."""

    def __init__(self, env, start, goal, resolution=0.2, margin=0.05, prune=False):
        super().__init__(env, start, goal, resolution=resolution, margin=margin, refine=False)
        self.prune = prune

    def plan(self):
        t0 = time.perf_counter()
        self.grid = OccupancyGrid(self.env, self.checker, self.resolution)
        s = self.grid.nearest_free_visible(self.start, self.checker)
        g = self.grid.nearest_free_visible(self.goal, self.checker)
        nodes = self._astar(s, g)
        if nodes is None:
            raise RuntimeError("A*: no path")
        path = [self.start] + [self.grid.to_point(n) for n in nodes] + [self.goal]
        # drop duplicated end points (start/goal usually coincide with grid nodes)
        out = [path[0]]
        for p in path[1:]:
            if np.linalg.norm(p - out[-1]) > 1e-9:
                out.append(p)
        if self.prune:
            out = self._prune(out)
        path = np.vstack(out)
        self.stats.update(t_total=time.perf_counter() - t0, length=path_length(path), vertices=len(path))
        return path

    def _astar(self, s_start, s_goal):
        grid = self.grid
        free = grid.free.tolist()
        nbrs = grid.neighbours
        sx, sy, res = grid.sx, grid.sy, grid.res
        r2, r3 = math.sqrt(2.0) - 1.0, math.sqrt(3.0) - math.sqrt(2.0)

        def ijk(n):
            i, rem = divmod(n, sx)
            j, k = divmod(rem, sy)
            return i, j, k

        gi, gj, gk = ijk(s_goal)

        def h(n):                                 # exact octile distance in 3-D
            i, j, k = ijk(n)
            d = sorted((abs(i - gi), abs(j - gj), abs(k - gk)))
            return res * (d[2] + r2 * d[1] + r3 * d[0])

        N = len(free)
        g = [math.inf] * N
        parent = [-1] * N
        closed = bytearray(N)
        g[s_start] = 0.0
        heap = [(h(s_start), 0.0, s_start)]
        expanded = 0
        while heap:
            _, _, s = heapq.heappop(heap)
            if closed[s]:
                continue
            closed[s] = 1
            expanded += 1
            if s == s_goal:
                break
            gs = g[s]
            for off, c in nbrs:
                n = s + off
                if not free[n] or closed[n]:
                    continue
                ng = gs + c
                if ng < g[n]:
                    g[n] = ng
                    parent[n] = s
                    heapq.heappush(heap, (ng + h(n), -ng, n))
        else:
            return None
        self.stats["expanded"] = expanded
        nodes = [s_goal]
        while nodes[-1] != s_start:
            nodes.append(parent[nodes[-1]])
        return nodes[::-1]


class RRTStarPlanner:
    """RRT* with goal bias, near-neighbour choose-parent and rewiring (with cost propagation)."""

    def __init__(self, env, start, goal, max_iter=3000, step=1.5, goal_bias=0.05,
                 gamma=12.0, margin=0.05, seed=0):
        self.env = env
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.max_iter = max_iter
        self.step = step
        self.goal_bias = goal_bias
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        base = PathPlanner(env, start, goal, margin=margin)     # reuse the adaptive margin
        self.checker = base.checker
        self.stats = {}

    def plan(self):
        t0 = time.perf_counter()
        ck = self.checker
        hi = ck.bounds
        cap = self.max_iter + 1
        X = np.empty((cap, 3))
        cost = np.empty(cap)
        parent = np.full(cap, -1, dtype=int)
        children = [[] for _ in range(cap)]
        X[0], cost[0], n = self.start, 0.0, 1
        best_cost, best_node = math.inf, -1

        for _ in range(self.max_iter):
            x_rand = self.goal if self.rng.random() < self.goal_bias else self.rng.uniform(0.0, hi)
            d = np.linalg.norm(X[:n] - x_rand, axis=1)
            i_near = int(np.argmin(d))
            if d[i_near] < 1e-9:
                continue
            x_new = X[i_near] + (x_rand - X[i_near]) * min(1.0, self.step / d[i_near])
            if not ck.points_free(x_new)[0]:
                continue
            radius = min(self.gamma * (math.log(n + 1) / (n + 1)) ** (1.0 / 3.0), 2.0 * self.step)
            dn = np.linalg.norm(X[:n] - x_new, axis=1)
            near = np.flatnonzero(dn <= radius)
            if i_near not in near:
                near = np.append(near, i_near)
            # choose parent
            order = near[np.argsort(cost[near] + dn[near])]
            p_best = -1
            for j in order:
                if ck.segment_free(X[j], x_new):
                    p_best = j
                    break
            if p_best < 0:
                continue
            k = n
            X[k], parent[k], cost[k] = x_new, p_best, cost[p_best] + dn[p_best]
            children[p_best].append(k)
            n += 1
            # rewire
            for j in near:
                if j == p_best:
                    continue
                c_new = cost[k] + dn[j]
                if c_new + 1e-9 < cost[j] and ck.segment_free(x_new, X[j]):
                    children[parent[j]].remove(j)
                    parent[j] = k
                    children[k].append(j)
                    delta = cost[j] - c_new
                    stack = [j]
                    while stack:                      # propagate the cost decrease
                        q = stack.pop()
                        cost[q] -= delta
                        stack.extend(children[q])
            # goal connection
            dg = np.linalg.norm(self.goal - x_new)
            if dg <= self.step and ck.segment_free(x_new, self.goal) and cost[k] + dg < best_cost:
                best_cost, best_node = cost[k] + dg, k
        if best_node < 0:
            self.stats.update(t_total=time.perf_counter() - t0, success=False)
            return None
        # re-evaluate the best goal connection after rewiring
        cands = np.flatnonzero(np.linalg.norm(X[:n] - self.goal, axis=1) <= self.step)
        for k in cands:
            c = cost[k] + np.linalg.norm(self.goal - X[k])
            if c < best_cost and ck.segment_free(X[k], self.goal):
                best_cost, best_node = c, k
        path = [self.goal]
        k = best_node
        while k >= 0:
            path.append(X[k].copy())
            k = parent[k]
        path = np.vstack(path[::-1])
        self.stats.update(t_total=time.perf_counter() - t0, length=path_length(path),
                          vertices=len(path), success=True, tree_size=n)
        return path

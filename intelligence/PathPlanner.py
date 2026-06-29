"""
PathPlanner.py
==============
Global Path Planner for autonomous drone navigation.

This module implements a 3D A* (A-Star) pathfinding algorithm on a voxel grid.
It is intentionally decoupled from any simulation framework (AirSim, ROS, etc.)
and any reinforcement-learning library. It is a pure algorithmic unit that
accepts a voxel occupancy map and two 3D integer coordinates, then returns an
ordered list of 3D waypoints from start to goal.

Design Principles
-----------------
- Clean Architecture: zero imports of AirSim, gymnasium, stable-baselines3, or
  any RL / simulator dependency.
- Single Responsibility: path planning only.
- Dependency-Inversion ready: callers depend on the ``PathPlanner`` abstraction,
  not on a concrete grid representation.

Usage Example
-------------
>>> import numpy as np
>>> grid = np.zeros((50, 50, 20), dtype=np.uint8)  # 0 = free, 1 = occupied
>>> grid[10:15, 10:15, 3:8] = 1                    # add a building obstacle
>>> planner = PathPlanner(grid, voxel_size=1.0)
>>> waypoints = planner.plan((0, 0, 5), (40, 40, 5))
>>> print(waypoints)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Coordinate3D = Tuple[int, int, int]   # (x_vox, y_vox, z_vox) integer voxel index
Waypoints     = List[Tuple[float, float, float]]   # world-space (m) waypoints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal node used by the priority queue
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _Node:
    """Priority-queue node for A* open list.

    The ``f_score`` is placed first so Python's ``heapq`` (min-heap) orders
    nodes by total estimated cost.
    """

    f_score:  float
    g_score:  float = field(compare=False)
    position: Coordinate3D = field(compare=False)
    parent:   Optional["_Node"] = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# PathPlanner
# ---------------------------------------------------------------------------
class PathPlanner:
    """3D A* path planner operating on a voxel occupancy grid.

    Parameters
    ----------
    voxel_grid:
        A 3-D NumPy array of shape ``(X, Y, Z)`` where a value of ``0``
        indicates a free voxel and any non-zero value indicates an occupied
        voxel (obstacle / building / no-fly zone).
    voxel_size:
        Edge length of each cubic voxel in metres.  Used to convert voxel
        indices back to world-space coordinates in the returned waypoints.
    origin:
        World-space (x, y, z) coordinate in metres that corresponds to voxel
        index ``(0, 0, 0)``.  Defaults to the origin.
    allow_diagonal:
        When ``True`` the planner considers all 26 neighbours (face, edge, and
        corner adjacency).  When ``False`` only the 6 face-adjacent neighbours
        are considered, which is cheaper but may produce longer paths.
    safety_margin:
        Number of extra voxels to inflate around each obstacle cell before
        planning.  A value of ``1`` means any voxel within 1 voxel of an
        obstacle is treated as occupied.  Increases safety at the cost of
        navigable space.

    Raises
    ------
    ValueError
        If ``voxel_grid`` is not 3-D, or if start / goal coordinates are out
        of bounds or lie inside an obstacle.
    RuntimeError
        If no path exists between start and goal.
    """

    # 26-connectivity neighbour offsets (all combinations of -1, 0, +1 except (0,0,0))
    _NEIGHBOURS_26: List[Coordinate3D] = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    # 6-connectivity neighbour offsets (face-adjacent only)
    _NEIGHBOURS_6: List[Coordinate3D] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    def __init__(
        self,
        voxel_grid: np.ndarray,
        voxel_size: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        allow_diagonal: bool = True,
        safety_margin: int = 1,
    ) -> None:
        if voxel_grid.ndim != 3:
            raise ValueError(
                f"voxel_grid must be 3-D, got shape {voxel_grid.shape}"
            )

        self._raw_grid   = voxel_grid.astype(np.uint8)
        self._grid       = self._inflate_obstacles(self._raw_grid, safety_margin)
        self._shape      = self._grid.shape          # (X, Y, Z)
        self.voxel_size  = float(voxel_size)
        self.origin      = tuple(float(v) for v in origin)
        self._neighbours = (
            self._NEIGHBOURS_26 if allow_diagonal else self._NEIGHBOURS_6
        )
        logger.info(
            "PathPlanner initialised | grid=%s | voxel_size=%.2fm | "
            "safety_margin=%d | connectivity=%d",
            self._shape, voxel_size, safety_margin,
            len(self._neighbours),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Coordinate3D,
        goal: Coordinate3D,
    ) -> Waypoints:
        """Run A* from *start* to *goal* and return world-space waypoints.

        The returned list begins with the start position and ends with the
        goal position.  Intermediate waypoints are post-processed with a
        line-of-sight (string-pulling) pass to remove collinear nodes and
        produce cleaner, more flyable paths.

        Parameters
        ----------
        start:
            Voxel index ``(ix, iy, iz)`` of the starting position.
        goal:
            Voxel index ``(ix, iy, iz)`` of the goal position.

        Returns
        -------
        Waypoints
            Ordered list of ``(x, y, z)`` world-space coordinates in metres.

        Raises
        ------
        ValueError
            If start or goal is out of bounds or inside an obstacle.
        RuntimeError
            If no collision-free path exists.
        """
        self._validate_coord(start, label="start")
        self._validate_coord(goal,  label="goal")

        logger.info("Planning path from %s to %s …", start, goal)

        voxel_path = self._astar(start, goal)
        pruned     = self._string_pull(voxel_path)
        waypoints  = [self._voxel_to_world(v) for v in pruned]

        logger.info(
            "Path found | raw_nodes=%d | pruned_nodes=%d",
            len(voxel_path), len(pruned),
        )
        return waypoints

    def is_free(self, coord: Coordinate3D) -> bool:
        """Return ``True`` if *coord* is within bounds and not occupied."""
        ix, iy, iz = coord
        X, Y, Z = self._shape
        if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z):
            return False
        # Cast explicitly to Python bool so callers can safely use `is True/False`
        return bool(self._grid[ix, iy, iz] == 0)

    # ------------------------------------------------------------------
    # Private: A* core
    # ------------------------------------------------------------------

    def _astar(self, start: Coordinate3D, goal: Coordinate3D) -> List[Coordinate3D]:
        """Pure A* search returning a list of voxel indices from start to goal.

        Uses an Euclidean-distance heuristic, which is admissible for 26-connected
        3-D grids with uniform edge weights.
        """
        open_heap:  List[_Node]              = []
        g_scores:   Dict[Coordinate3D, float] = {start: 0.0}
        closed_set: set[Coordinate3D]        = set()

        start_node = _Node(
            f_score  = self._heuristic(start, goal),
            g_score  = 0.0,
            position = start,
        )
        heapq.heappush(open_heap, start_node)

        while open_heap:
            current = heapq.heappop(open_heap)
            pos     = current.position

            if pos in closed_set:
                continue

            # Goal check
            if pos == goal:
                return self._reconstruct_path(current)

            closed_set.add(pos)

            for neighbour_pos in self._expand(pos):
                if neighbour_pos in closed_set:
                    continue

                # Edge weight: Euclidean distance between voxel centres
                step_cost = self._edge_cost(pos, neighbour_pos)
                tentative_g = current.g_score + step_cost

                if tentative_g < g_scores.get(neighbour_pos, float("inf")):
                    g_scores[neighbour_pos] = tentative_g
                    f = tentative_g + self._heuristic(neighbour_pos, goal)
                    neighbour_node = _Node(
                        f_score  = f,
                        g_score  = tentative_g,
                        position = neighbour_pos,
                        parent   = current,
                    )
                    heapq.heappush(open_heap, neighbour_node)

        raise RuntimeError(
            f"A* found no path from {start} to {goal}. "
            "Check that start and goal are reachable and that the voxel grid "
            "is not fully obstructed."
        )

    def _expand(self, pos: Coordinate3D) -> List[Coordinate3D]:
        """Return all free, in-bounds neighbours of *pos*."""
        X, Y, Z = self._shape
        ix, iy, iz = pos
        neighbours: List[Coordinate3D] = []
        for dx, dy, dz in self._neighbours:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                if self._grid[nx, ny, nz] == 0:
                    neighbours.append((nx, ny, nz))
        return neighbours

    # ------------------------------------------------------------------
    # Private: heuristic and edge cost
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(a: Coordinate3D, b: Coordinate3D) -> float:
        """Euclidean distance heuristic (admissible for uniform-cost edges)."""
        ax, ay, az = a
        bx, by, bz = b
        return float(np.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2))

    @staticmethod
    def _edge_cost(a: Coordinate3D, b: Coordinate3D) -> float:
        """Cost of moving between two adjacent voxels (Euclidean distance)."""
        ax, ay, az = a
        bx, by, bz = b
        return float(np.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2))

    # ------------------------------------------------------------------
    # Private: path reconstruction and post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_path(node: _Node) -> List[Coordinate3D]:
        """Walk parent pointers from *node* back to the start."""
        path: List[Coordinate3D] = []
        current: Optional[_Node] = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

    def _string_pull(self, path: List[Coordinate3D]) -> List[Coordinate3D]:
        """Remove redundant collinear waypoints using a greedy line-of-sight test.

        Iterates over the path and skips any intermediate node that is visible
        (collision-free straight line) from the previous kept node.  This is
        the "Theta*"-style post-processing step, reducing the waypoint count
        while preserving safety.

        Parameters
        ----------
        path:
            Raw A* voxel path (may contain hundreds of nodes).

        Returns
        -------
        List[Coordinate3D]
            Pruned path with only the geometrically necessary waypoints.
        """
        if len(path) < 3:
            return path

        pruned = [path[0]]
        anchor = 0

        for i in range(2, len(path)):
            if not self._line_of_sight(path[anchor], path[i]):
                # The node just before i (i-1) is the last visible one
                pruned.append(path[i - 1])
                anchor = i - 1

        pruned.append(path[-1])
        return pruned

    def _line_of_sight(self, a: Coordinate3D, b: Coordinate3D) -> bool:
        """3-D Bresenham line-of-sight check between voxels *a* and *b*.

        Returns ``True`` if every voxel along the integer raster line from
        *a* to *b* is free of obstacles.
        """
        ax, ay, az = a
        bx, by, bz = b
        dx, dy, dz = bx - ax, by - ay, bz - az

        n_steps = max(abs(dx), abs(dy), abs(dz))
        if n_steps == 0:
            return True

        for t in range(n_steps + 1):
            frac = t / n_steps
            vx = round(ax + frac * dx)
            vy = round(ay + frac * dy)
            vz = round(az + frac * dz)
            if self._grid[vx, vy, vz] != 0:
                return False
        return True

    # ------------------------------------------------------------------
    # Private: coordinate helpers
    # ------------------------------------------------------------------

    def _voxel_to_world(self, voxel: Coordinate3D) -> Tuple[float, float, float]:
        """Convert a voxel index to a world-space coordinate (voxel centre)."""
        ix, iy, iz = voxel
        ox, oy, oz = self.origin
        x = ox + (ix + 0.5) * self.voxel_size
        y = oy + (iy + 0.5) * self.voxel_size
        z = oz + (iz + 0.5) * self.voxel_size
        return (x, y, z)

    def _validate_coord(self, coord: Coordinate3D, label: str) -> None:
        """Raise ``ValueError`` if *coord* is out of bounds or occupied."""
        X, Y, Z = self._shape
        ix, iy, iz = coord
        if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z):
            raise ValueError(
                f"{label} coordinate {coord} is out of grid bounds {self._shape}."
            )
        if self._grid[ix, iy, iz] != 0:
            raise ValueError(
                f"{label} coordinate {coord} lies inside an obstacle voxel "
                f"(inflated grid value = {self._grid[ix, iy, iz]})."
            )

    # ------------------------------------------------------------------
    # Private: safety inflation
    # ------------------------------------------------------------------

    @staticmethod
    def _inflate_obstacles(grid: np.ndarray, margin: int) -> np.ndarray:
        """Return a copy of *grid* with obstacles dilated by *margin* voxels.

        Uses a 3-D binary dilation so that the planner keeps the drone at least
        *margin* voxels away from the surface of any building or obstacle.

        Parameters
        ----------
        grid:
            Raw uint8 occupancy grid (0 = free, 1 = occupied).
        margin:
            Dilation radius in voxels.  ``0`` disables inflation.
        """
        if margin <= 0:
            return grid.copy()

        from scipy.ndimage import binary_dilation  # local import — stdlib-only dep

        struct = np.ones(
            (2 * margin + 1, 2 * margin + 1, 2 * margin + 1), dtype=bool
        )
        inflated = binary_dilation(grid.astype(bool), structure=struct)
        return inflated.astype(np.uint8)

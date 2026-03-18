"""
pathfinding.py - Advanced pathfinding for CopperHead Bot v2

Key improvements over v1:
- True simultaneous-BFS Voronoi (no expansion bias)
- Articulation-point detection to find board-splitting chokepoints
- Optimized flood fill with early-exit and component awareness
- Parent-map BFS/A* (no path copies in queue — O(n) reconstruct)
- Deeper & faster minimax lookahead with alpha-beta style pruning
- Multi-opponent danger zone calculation
- Connected-component split detection after a move
"""

from __future__ import annotations

import logging
from collections import deque
import heapq
from typing import List, Optional, Set, Tuple, Dict

from utils import (
    get_neighbors,
    manhattan_distance,
    is_in_bounds,
    get_new_position,
    ALL_DIRECTIONS,
    OPPOSITES,
    DIRECTIONS,
)

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = frozenset(ALL_DIRECTIONS)


# ---------------------------------------------------------------------------
# BFS distance / path (parent-map based — no path copies in queue)
# ---------------------------------------------------------------------------

def bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> int:
    """Shortest-path distance via BFS. Returns -1 if unreachable."""
    if start == goal:
        return 0
    if goal in obstacles:
        return -1

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        pos, dist = queue.popleft()
        for nb in get_neighbors(pos[0], pos[1], width, height):
            if nb == goal:
                return dist + 1
            if nb not in visited and nb not in obstacles:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return -1


def bfs_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Shortest path via BFS using a parent map (memory-efficient)."""
    if start == goal:
        return [start]
    if goal in obstacles:
        return None

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        for nb in get_neighbors(pos[0], pos[1], width, height):
            if nb in parent or nb in obstacles:
                continue
            parent[nb] = pos
            if nb == goal:
                # reconstruct
                path = [nb]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nb)
    return None


# ---------------------------------------------------------------------------
# A* with parent map
# ---------------------------------------------------------------------------

def astar_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    danger_zones: Optional[Set[Tuple[int, int]]] = None,
    danger_penalty: int = 5,
) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    if goal in obstacles:
        return None

    danger_zones = danger_zones or set()
    counter = 0
    g_scores: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    open_set: list = [(manhattan_distance(start, goal), counter, start)]

    while open_set:
        _f, _, pos = heapq.heappop(open_set)
        if pos == goal:
            path = [pos]
            while pos in parent:
                pos = parent[pos]
                path.append(pos)
            path.reverse()
            return path

        cur_g = g_scores[pos]
        for nb in get_neighbors(pos[0], pos[1], width, height):
            if nb in obstacles:
                continue
            move_cost = 1 + (danger_penalty if nb in danger_zones else 0)
            new_g = cur_g + move_cost
            if nb not in g_scores or new_g < g_scores[nb]:
                g_scores[nb] = new_g
                parent[nb] = pos
                counter += 1
                heapq.heappush(open_set, (new_g + manhattan_distance(nb, goal), counter, nb))
    return None


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

def flood_fill_count(
    start: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    max_depth: Optional[int] = None,
) -> int:
    if start in obstacles:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for nb in get_neighbors(pos[0], pos[1], width, height):
            if nb not in visited and nb not in obstacles:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return len(visited)


def flood_fill_reachable(
    start: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    max_depth: Optional[int] = None,
) -> Set[Tuple[int, int]]:
    if start in obstacles:
        return set()
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for nb in get_neighbors(pos[0], pos[1], width, height):
            if nb not in visited and nb not in obstacles:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return visited


# ---------------------------------------------------------------------------
# Safe moves
# ---------------------------------------------------------------------------

def get_safe_moves(
    head: Tuple[int, int],
    current_direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> List[str]:
    safe = []
    opposite = OPPOSITES.get(current_direction, "")
    for d in ALL_DIRECTIONS:
        if d == opposite:
            continue
        nx, ny = get_new_position(head[0], head[1], d)
        if is_in_bounds(nx, ny, width, height) and (nx, ny) not in obstacles:
            safe.append(d)
    return safe


# ---------------------------------------------------------------------------
# Corridor / chokepoint scoring
# ---------------------------------------------------------------------------

def get_corridor_score(
    pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> int:
    """Number of safe neighbours around *pos* (0-4)."""
    return sum(1 for nb in get_neighbors(pos[0], pos[1], width, height) if nb not in obstacles)


def is_dead_end(
    pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    min_space: int = 5,
) -> bool:
    return flood_fill_count(pos, width, height, obstacles) < min_space


# ---------------------------------------------------------------------------
# Articulation-point detection (Tarjan's on the grid)
# ---------------------------------------------------------------------------

def find_articulation_points(
    region: Set[Tuple[int, int]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """
    Find articulation points in *region* — cells whose removal splits
    the reachable area into disconnected components.

    This is critical for detecting chokepoints; moving *through* an
    articulation point may be fine, but moving to block one can trap
    the opponent (or yourself).
    """
    if not region:
        return set()

    # Build adjacency inside region (non-obstacle, in-bounds)
    free = region - obstacles

    disc: Dict[Tuple[int, int], int] = {}
    low: Dict[Tuple[int, int], int] = {}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    ap: Set[Tuple[int, int]] = set()
    timer = [0]

    def _dfs(u: Tuple[int, int]) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for nb in get_neighbors(u[0], u[1], width, height):
            if nb not in free:
                continue
            if nb not in disc:
                children += 1
                parent[nb] = u
                _dfs(nb)
                low[u] = min(low[u], low[nb])
                # u is AP if it's root with 2+ children, or non-root with low[nb] >= disc[u]
                if parent[u] is None and children > 1:
                    ap.add(u)
                if parent[u] is not None and low[nb] >= disc[u]:
                    ap.add(u)
            elif nb != parent.get(u):
                low[u] = min(low[u], disc[nb])

    # Limit region size to avoid spending too long on huge boards
    if len(free) > 300:
        return set()

    for node in free:
        if node not in disc:
            parent[node] = None
            _dfs(node)

    return ap


# ---------------------------------------------------------------------------
# True simultaneous-BFS Voronoi (no expansion bias)
# ---------------------------------------------------------------------------

def calculate_voronoi_control(
    my_head: Tuple[int, int],
    opponent_head: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Tuple[int, int, Set[Tuple[int, int]]]:
    """
    Voronoi partitioning via true level-synchronous BFS.

    Both snakes expand one full BFS level at a time so neither
    side gets a head start.  Cells reached at the same depth
    are contested (counted for neither).

    Returns (my_cells, opp_cells, my_territory_set).
    """
    my_dist: Dict[Tuple[int, int], int] = {my_head: 0}
    opp_dist: Dict[Tuple[int, int], int] = {opponent_head: 0}
    my_frontier: List[Tuple[int, int]] = [my_head]
    opp_frontier: List[Tuple[int, int]] = [opponent_head]
    depth = 0

    while my_frontier or opp_frontier:
        depth += 1
        next_my: List[Tuple[int, int]] = []
        for pos in my_frontier:
            for nb in get_neighbors(pos[0], pos[1], width, height):
                if nb in obstacles or nb in my_dist:
                    continue
                my_dist[nb] = depth
                next_my.append(nb)

        next_opp: List[Tuple[int, int]] = []
        for pos in opp_frontier:
            for nb in get_neighbors(pos[0], pos[1], width, height):
                if nb in obstacles or nb in opp_dist:
                    continue
                opp_dist[nb] = depth
                next_opp.append(nb)

        my_frontier = next_my
        opp_frontier = next_opp

    my_cells = 0
    opp_cells = 0
    my_territory: Set[Tuple[int, int]] = set()
    all_cells = set(my_dist.keys()) | set(opp_dist.keys())
    for cell in all_cells:
        md = my_dist.get(cell, float("inf"))
        od = opp_dist.get(cell, float("inf"))
        if md < od:
            my_cells += 1
            my_territory.add(cell)
        elif od < md:
            opp_cells += 1
    return my_cells, opp_cells, my_territory


def calculate_voronoi_multi(
    my_head: Tuple[int, int],
    opponent_heads: List[Tuple[int, int]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Tuple[int, int]:
    """Voronoi with multiple opponents. Returns (my_cells, total_opp_cells)."""
    my_dist: Dict[Tuple[int, int], int] = {my_head: 0}
    opp_dist: Dict[Tuple[int, int], int] = {}
    my_frontier: List[Tuple[int, int]] = [my_head]
    opp_frontier: List[Tuple[int, int]] = []
    for oh in opponent_heads:
        opp_dist[oh] = 0
        opp_frontier.append(oh)
    depth = 0
    while my_frontier or opp_frontier:
        depth += 1
        next_my: List[Tuple[int, int]] = []
        for pos in my_frontier:
            for nb in get_neighbors(pos[0], pos[1], width, height):
                if nb in obstacles or nb in my_dist:
                    continue
                my_dist[nb] = depth
                next_my.append(nb)
        next_opp: List[Tuple[int, int]] = []
        for pos in opp_frontier:
            for nb in get_neighbors(pos[0], pos[1], width, height):
                if nb in obstacles or nb in opp_dist:
                    continue
                opp_dist[nb] = depth
                next_opp.append(nb)
        my_frontier = next_my
        opp_frontier = next_opp

    my_cells = 0
    opp_cells = 0
    for cell in set(my_dist) | set(opp_dist):
        md = my_dist.get(cell, float("inf"))
        od = opp_dist.get(cell, float("inf"))
        if md < od:
            my_cells += 1
        elif od < md:
            opp_cells += 1
    return my_cells, opp_cells


# ---------------------------------------------------------------------------
# Space after move (with growing-snake awareness)
# ---------------------------------------------------------------------------

def calculate_space_after_move(
    head: Tuple[int, int],
    direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    my_tail: Optional[Tuple[int, int]] = None,
    tail_will_stay: bool = False,
) -> int:
    """
    Reachable cells after making a move.

    If *tail_will_stay* is True (we are about to eat), the tail does NOT
    free up a cell — critical edge case the v1 bot missed.
    """
    nx, ny = get_new_position(head[0], head[1], direction)
    if not is_in_bounds(nx, ny, width, height) or (nx, ny) in obstacles:
        return 0

    new_obstacles = set(obstacles)
    new_obstacles.add(head)
    if my_tail and not tail_will_stay and my_tail in new_obstacles:
        new_obstacles.discard(my_tail)

    return flood_fill_count((nx, ny), width, height, new_obstacles)


# ---------------------------------------------------------------------------
# Connected-component split detection
# ---------------------------------------------------------------------------

def move_splits_board(
    head: Tuple[int, int],
    direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    my_tail: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Does occupying the new cell split the remaining free space into
    separate components?  If yes, one of them may be a death trap.
    """
    nx, ny = get_new_position(head[0], head[1], direction)
    if not is_in_bounds(nx, ny, width, height):
        return True

    new_obs = set(obstacles)
    new_obs.add(head)
    new_obs.add((nx, ny))
    if my_tail and my_tail in new_obs:
        new_obs.discard(my_tail)

    # Check neighbours of (nx, ny); if they're in different components → split
    free_nbs = [nb for nb in get_neighbors(nx, ny, width, height) if nb not in new_obs]
    if len(free_nbs) <= 1:
        return False  # 0 or 1 free neighbour can't split

    # Flood from first free neighbour; see if all others are reached
    visited = flood_fill_reachable(free_nbs[0], width, height, new_obs, max_depth=40)
    for nb in free_nbs[1:]:
        if nb not in visited:
            return True
    return False


# ---------------------------------------------------------------------------
# Opponent prediction (improved multi-heuristic)
# ---------------------------------------------------------------------------

def predict_opponent_move(
    opponent_head: Tuple[int, int],
    opponent_direction: str,
    opponent_body: List[Tuple[int, int]],
    my_head: Tuple[int, int],
    foods: List[Tuple[int, int]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Tuple[int, int]:
    if not opponent_direction or opponent_direction not in _VALID_DIRECTIONS:
        opponent_direction = ""
    opposite = OPPOSITES.get(opponent_direction, "")
    best_pos = None
    best_score = float("-inf")
    opp_len = len(opponent_body)

    for d in ALL_DIRECTIONS:
        if d == opposite:
            continue
        nx, ny = get_new_position(opponent_head[0], opponent_head[1], d)
        if not is_in_bounds(nx, ny, width, height) or (nx, ny) in obstacles:
            continue
        new_pos = (nx, ny)
        score = 0.0

        # Food attraction (strongest signal for prediction)
        if foods:
            min_food_dist = min(manhattan_distance(new_pos, f) for f in foods)
            score += 120 / (min_food_dist + 1)
            if new_pos in foods:
                score += 600

        # Space (use limited depth for speed)
        temp_obs = obstacles | {opponent_head}
        space = flood_fill_count(new_pos, width, height, temp_obs, max_depth=10)
        score += space * 3

        # Prefer continuing straight
        if d == opponent_direction:
            score += 20

        # Opponent tries to avoid us if we're longer
        dist_to_us = manhattan_distance(new_pos, my_head)
        if dist_to_us <= 1:
            score -= 200
        elif dist_to_us <= 2:
            score -= 50

        if score > best_score:
            best_score = score
            best_pos = new_pos

    return best_pos if best_pos else opponent_head


# ---------------------------------------------------------------------------
# Opponent danger zone
# ---------------------------------------------------------------------------

def find_opponent_danger_zone(
    opponent_head: Tuple[int, int],
    opponent_direction: str,
    width: int,
    height: int,
    radius: int = 2,
) -> Set[Tuple[int, int]]:
    danger: Set[Tuple[int, int]] = set()
    queue = deque([(opponent_head, 0)])
    visited = {opponent_head}
    while queue:
        pos, depth = queue.popleft()
        if depth > radius:
            continue
        danger.add(pos)
        if depth < radius:
            for nb in get_neighbors(pos[0], pos[1], width, height):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
    return danger


# ---------------------------------------------------------------------------
# Tail chasing
# ---------------------------------------------------------------------------

def find_tail_chase_path(
    head: Tuple[int, int],
    tail: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Optional[str]:
    temp_obstacles = obstacles - {tail}
    path = bfs_path(head, tail, width, height, temp_obstacles)
    if path and len(path) >= 2:
        next_pos = path[1]
        for d in ALL_DIRECTIONS:
            nx, ny = get_new_position(head[0], head[1], d)
            if (nx, ny) == next_pos:
                return d
    return None


# ---------------------------------------------------------------------------
# Lookahead (minimax-style with pruning)
# ---------------------------------------------------------------------------

def lookahead_evaluate(
    head: Tuple[int, int],
    direction: str,
    body: List[Tuple[int, int]],
    opponent_head: Optional[Tuple[int, int]],
    opponent_body: Optional[List[Tuple[int, int]]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    depth: int = 3,
    foods: Optional[Set[Tuple[int, int]]] = None,
) -> float:
    """
    Deeper lookahead (default depth 3 vs v1's 2).
    Considers food pickups and growing.
    """
    if direction not in _VALID_DIRECTIONS:
        return float("-inf")

    nx, ny = get_new_position(head[0], head[1], direction)
    if not is_in_bounds(nx, ny, width, height) or (nx, ny) in obstacles:
        return float("-inf")

    new_pos = (nx, ny)
    foods = foods or set()
    ate = new_pos in foods

    if not body:
        return float(flood_fill_count(new_pos, width, height, obstacles | {head}, max_depth=20))

    # Simulate body movement
    new_body = [new_pos] + (body if ate else body[:-1])
    new_obstacles = set(obstacles)
    new_obstacles.add(head)
    if not ate and body[-1] in new_obstacles:
        new_obstacles.discard(body[-1])
    new_foods = foods - {new_pos}

    if depth <= 0:
        space = flood_fill_count(new_pos, width, height, new_obstacles, max_depth=12)
        bonus = 8 if ate else 0
        return float(space + bonus)

    # Evaluate future moves
    best_future = float("-inf")
    opposite = OPPOSITES.get(direction, "")
    for nd in ALL_DIRECTIONS:
        if nd == opposite:
            continue
        score = lookahead_evaluate(
            new_pos, nd, new_body,
            opponent_head, opponent_body,
            width, height, new_obstacles,
            depth - 1, new_foods,
        )
        if score > best_future:
            best_future = score

    return best_future if best_future != float("-inf") else 0.0


# ---------------------------------------------------------------------------
# Food race
# ---------------------------------------------------------------------------

def food_race_winner(
    my_head: Tuple[int, int],
    opponent_head: Optional[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> str:
    my_dist = bfs_distance(my_head, food_pos, width, height, obstacles)
    if opponent_head is None:
        return "me" if my_dist >= 0 else "unreachable"
    opp_dist = bfs_distance(opponent_head, food_pos, width, height, obstacles)
    if my_dist < 0 and opp_dist < 0:
        return "unreachable"
    if my_dist < 0:
        return "opponent"
    if opp_dist < 0:
        return "me"
    if my_dist < opp_dist:
        return "me"
    if opp_dist < my_dist:
        return "opponent"
    return "tie"


# ---------------------------------------------------------------------------
# Center distance
# ---------------------------------------------------------------------------

def center_distance(
    pos: Tuple[int, int], width: int, height: int
) -> float:
    return abs(pos[0] - width / 2) + abs(pos[1] - height / 2)


# ---------------------------------------------------------------------------
# Find best direction by space (fallback)
# ---------------------------------------------------------------------------

def find_best_direction_by_space(
    head: Tuple[int, int],
    current_direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    my_tail: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    safe = get_safe_moves(head, current_direction, width, height, obstacles)
    if not safe:
        return None
    best_d = None
    best_s = -1
    for d in safe:
        s = calculate_space_after_move(head, d, width, height, obstacles, my_tail)
        if s > best_s:
            best_s = s
            best_d = d
    return best_d

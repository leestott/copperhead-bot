"""
pathfinding.py - Pathfinding algorithms for CopperHead bot

Implements A*, BFS, flood-fill, Voronoi partitioning, and advanced space analysis.
Optimized for competitive tournament play.
"""

from typing import Tuple, List, Set, Optional
from collections import deque
import heapq

from utils import (
    get_neighbors, manhattan_distance, is_in_bounds,
    get_new_position, ALL_DIRECTIONS, OPPOSITES
)


def predict_opponent_move(
    opponent_head: Tuple[int, int],
    opponent_direction: str,
    opponent_body: List[Tuple[int, int]],
    my_head: Tuple[int, int],
    foods: List[Tuple[int, int]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Predict where opponent will move next tick.
    
    Uses heuristics: chase food if close, otherwise maximize space.
    
    Returns:
        Predicted next head position
    """
    opposite = OPPOSITES.get(opponent_direction, "")
    best_pos = None
    best_score = float('-inf')
    
    for direction in ALL_DIRECTIONS:
        if direction == opposite:
            continue
        
        new_x, new_y = get_new_position(opponent_head[0], opponent_head[1], direction)
        
        if not is_in_bounds(new_x, new_y, width, height):
            continue
        if (new_x, new_y) in obstacles:
            continue
        
        score = 0
        new_pos = (new_x, new_y)
        
        # Food attraction
        if foods:
            min_food_dist = min(manhattan_distance(new_pos, f) for f in foods)
            score += 100 / (min_food_dist + 1)
            # Immediate food capture
            if new_pos in foods:
                score += 500
        
        # Space consideration
        temp_obstacles = obstacles | {opponent_head}
        space = flood_fill_count(new_pos, width, height, temp_obstacles, max_depth=8)
        score += space * 2
        
        # Avoid us (basic self-preservation)
        dist_to_us = manhattan_distance(new_pos, my_head)
        if dist_to_us <= 1:
            score -= 200
        
        if score > best_score:
            best_score = score
            best_pos = new_pos
    
    return best_pos if best_pos else opponent_head


def calculate_voronoi_control(
    my_head: Tuple[int, int],
    opponent_head: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> Tuple[int, int, Set[Tuple[int, int]]]:
    """
    Calculate Voronoi space partitioning - cells closer to us vs opponent.
    
    Returns:
        (my_cells, opponent_cells, my_territory_set)
    """
    my_cells = 0
    opp_cells = 0
    my_territory = set()
    
    # BFS from both heads simultaneously
    my_dist = {my_head: 0}
    opp_dist = {opponent_head: 0}
    
    my_queue = deque([my_head])
    opp_queue = deque([opponent_head])
    
    # Expand both BFS in lockstep
    while my_queue or opp_queue:
        # Expand my frontier
        if my_queue:
            pos = my_queue.popleft()
            for neighbor in get_neighbors(pos[0], pos[1], width, height):
                if neighbor in obstacles:
                    continue
                if neighbor not in my_dist:
                    my_dist[neighbor] = my_dist[pos] + 1
                    my_queue.append(neighbor)
        
        # Expand opponent frontier
        if opp_queue:
            pos = opp_queue.popleft()
            for neighbor in get_neighbors(pos[0], pos[1], width, height):
                if neighbor in obstacles:
                    continue
                if neighbor not in opp_dist:
                    opp_dist[neighbor] = opp_dist[pos] + 1
                    opp_queue.append(neighbor)
    
    # Count territories
    all_cells = set(my_dist.keys()) | set(opp_dist.keys())
    for cell in all_cells:
        my_d = my_dist.get(cell, float('inf'))
        opp_d = opp_dist.get(cell, float('inf'))
        
        if my_d < opp_d:
            my_cells += 1
            my_territory.add(cell)
        elif opp_d < my_d:
            opp_cells += 1
        # Ties go to neither
    
    return my_cells, opp_cells, my_territory


def find_tail_chase_path(
    head: Tuple[int, int],
    tail: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> Optional[str]:
    """
    Find direction to chase our own tail (survival technique).
    
    Tail chasing keeps us alive when space is limited.
    
    Returns:
        Direction toward tail, or None if unreachable
    """
    # Tail will move, so it's actually reachable
    temp_obstacles = obstacles - {tail}
    
    path = bfs_path(head, tail, width, height, temp_obstacles)
    if path and len(path) >= 2:
        next_pos = path[1]
        for direction in ALL_DIRECTIONS:
            new_x, new_y = get_new_position(head[0], head[1], direction)
            if (new_x, new_y) == next_pos:
                return direction
    
    return None


def lookahead_evaluate(
    head: Tuple[int, int],
    direction: str,
    body: List[Tuple[int, int]],
    opponent_head: Optional[Tuple[int, int]],
    opponent_body: Optional[List[Tuple[int, int]]],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    depth: int = 2
) -> float:
    """
    Evaluate a move with N-move lookahead using minimax-style evaluation.
    
    Returns:
        Score for this move considering future positions
    """
    if depth == 0:
        # Base case: evaluate current position
        new_x, new_y = get_new_position(head[0], head[1], direction)
        if not is_in_bounds(new_x, new_y, width, height):
            return float('-inf')
        if (new_x, new_y) in obstacles:
            return float('-inf')
        
        space = flood_fill_count((new_x, new_y), width, height, obstacles | {head})
        return space
    
    new_x, new_y = get_new_position(head[0], head[1], direction)
    new_pos = (new_x, new_y)
    
    if not is_in_bounds(new_x, new_y, width, height):
        return float('-inf')
    if new_pos in obstacles:
        return float('-inf')
    
    # Simulate move
    new_body = [new_pos] + body[:-1]  # Move without eating
    new_obstacles = obstacles | {head}
    if body and body[-1] in new_obstacles:
        new_obstacles.discard(body[-1])  # Tail moved
    
    # Evaluate future moves
    best_future = float('-inf')
    opposite = OPPOSITES.get(direction, "")
    
    for next_dir in ALL_DIRECTIONS:
        if next_dir == opposite:
            continue
        
        future_score = lookahead_evaluate(
            new_pos, next_dir, new_body,
            opponent_head, opponent_body,
            width, height, new_obstacles,
            depth - 1
        )
        best_future = max(best_future, future_score)
    
    return best_future if best_future != float('-inf') else 0


def food_race_winner(
    my_head: Tuple[int, int],
    opponent_head: Optional[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> str:
    """
    Determine who would reach food first.
    
    Returns:
        "me", "opponent", "tie", or "unreachable"
    """
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
    elif opp_dist < my_dist:
        return "opponent"
    else:
        return "tie"


def center_distance(
    pos: Tuple[int, int],
    width: int,
    height: int
) -> float:
    """
    Calculate distance from grid center.
    
    Lower = closer to center = more options.
    """
    center_x = width / 2
    center_y = height / 2
    return abs(pos[0] - center_x) + abs(pos[1] - center_y)


def bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> int:
    """
    Calculate shortest path distance using BFS.
    
    Args:
        start: Starting position (x, y)
        goal: Target position (x, y)
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
    
    Returns:
        Shortest path length, or -1 if unreachable
    """
    if start == goal:
        return 0
    
    if goal in obstacles:
        return -1
    
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        pos, dist = queue.popleft()
        
        for neighbor in get_neighbors(pos[0], pos[1], width, height):
            if neighbor == goal:
                return dist + 1
            
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # Unreachable


def bfs_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> Optional[List[Tuple[int, int]]]:
    """
    Find shortest path using BFS.
    
    Args:
        start: Starting position (x, y)
        goal: Target position (x, y)
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
    
    Returns:
        List of positions from start to goal, or None if unreachable
    """
    if start == goal:
        return [start]
    
    if goal in obstacles:
        return None
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        pos, path = queue.popleft()
        
        for neighbor in get_neighbors(pos[0], pos[1], width, height):
            if neighbor == goal:
                return path + [neighbor]
            
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def astar_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    danger_zones: Optional[Set[Tuple[int, int]]] = None,
    danger_penalty: int = 5
) -> Optional[List[Tuple[int, int]]]:
    """
    Find path using A* algorithm with optional danger zone penalties.
    
    A* uses Manhattan distance as heuristic. Danger zones add extra cost
    to discourage paths through risky areas.
    
    Args:
        start: Starting position (x, y)
        goal: Target position (x, y)
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
        danger_zones: Optional set of positions to penalize
        danger_penalty: Extra cost for danger zone cells
    
    Returns:
        List of positions from start to goal, or None if unreachable
    """
    if start == goal:
        return [start]
    
    if goal in obstacles:
        return None
    
    danger_zones = danger_zones or set()
    
    # Priority queue: (f_score, counter, position, path)
    # Counter ensures consistent ordering for equal f_scores
    counter = 0
    open_set = [(manhattan_distance(start, goal), counter, start, [start])]
    g_scores = {start: 0}
    
    while open_set:
        f, _, pos, path = heapq.heappop(open_set)
        
        if pos == goal:
            return path
        
        current_g = g_scores[pos]
        
        for neighbor in get_neighbors(pos[0], pos[1], width, height):
            if neighbor in obstacles:
                continue
            
            # Base cost is 1, add penalty for danger zones
            move_cost = 1
            if neighbor in danger_zones:
                move_cost += danger_penalty
            
            new_g = current_g + move_cost
            
            if neighbor not in g_scores or new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)
                f_score = new_g + h
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))
    
    return None


def flood_fill_count(
    start: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    max_depth: Optional[int] = None
) -> int:
    """
    Count reachable cells using flood-fill from a starting position.
    
    This is crucial for avoiding self-trapping. A move that results in
    fewer reachable cells is likely a poor choice.
    
    Args:
        start: Starting position (x, y)
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
        max_depth: Optional maximum BFS depth (for performance)
    
    Returns:
        Number of reachable cells including start
    """
    if start in obstacles:
        return 0
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        pos, depth = queue.popleft()
        
        # Stop if we've reached max depth
        if max_depth is not None and depth >= max_depth:
            continue
        
        for neighbor in get_neighbors(pos[0], pos[1], width, height):
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return len(visited)


def flood_fill_reachable(
    start: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    max_depth: Optional[int] = None
) -> Set[Tuple[int, int]]:
    """
    Get all reachable cells using flood-fill.
    
    Args:
        start: Starting position (x, y)
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
        max_depth: Optional maximum BFS depth
    
    Returns:
        Set of all reachable positions
    """
    if start in obstacles:
        return set()
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        pos, depth = queue.popleft()
        
        if max_depth is not None and depth >= max_depth:
            continue
        
        for neighbor in get_neighbors(pos[0], pos[1], width, height):
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return visited


def get_safe_moves(
    head: Tuple[int, int],
    current_direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> List[str]:
    """
    Get all moves that don't result in immediate death.
    
    Args:
        head: Current head position
        current_direction: Currently facing direction
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
    
    Returns:
        List of safe direction strings
    """
    from utils import OPPOSITES
    
    safe = []
    opposite = OPPOSITES.get(current_direction, "")
    
    for direction in ALL_DIRECTIONS:
        # Can't reverse direction
        if direction == opposite:
            continue
        
        new_x, new_y = get_new_position(head[0], head[1], direction)
        
        # Check bounds
        if not is_in_bounds(new_x, new_y, width, height):
            continue
        
        # Check obstacles
        if (new_x, new_y) in obstacles:
            continue
        
        safe.append(direction)
    
    return safe


def get_corridor_score(
    pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]]
) -> int:
    """
    Calculate a corridor score - lower means more enclosed.
    
    A position with only 1 or 2 safe neighbors is in a corridor,
    which is dangerous as it limits escape options.
    
    Args:
        pos: Position to evaluate
        width: Grid width
        height: Grid height
        obstacles: Set of blocked positions
    
    Returns:
        Number of safe neighbors (0-4)
    """
    neighbors = get_neighbors(pos[0], pos[1], width, height)
    safe_count = sum(1 for n in neighbors if n not in obstacles)
    return safe_count


def find_opponent_danger_zone(
    opponent_head: Tuple[int, int],
    opponent_direction: str,
    width: int,
    height: int,
    radius: int = 2
) -> Set[Tuple[int, int]]:
    """
    Calculate positions where opponent could reach within N moves.
    
    Used to avoid head-on collisions and predict opponent movement.
    
    Args:
        opponent_head: Opponent's head position
        opponent_direction: Opponent's current direction
        width: Grid width
        height: Grid height
        radius: Number of moves to project
    
    Returns:
        Set of potentially dangerous positions
    """
    danger = set()
    queue = deque([(opponent_head, 0)])
    visited = {opponent_head}
    
    while queue:
        pos, depth = queue.popleft()
        
        if depth > radius:
            continue
        
        danger.add(pos)
        
        if depth < radius:
            for neighbor in get_neighbors(pos[0], pos[1], width, height):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    
    return danger


def calculate_space_after_move(
    head: Tuple[int, int],
    direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    my_tail: Optional[Tuple[int, int]] = None
) -> int:
    """
    Calculate reachable space after making a move.
    
    This is the key metric for avoiding self-trapping.
    
    Args:
        head: Current head position
        direction: Direction to move
        width: Grid width
        height: Grid height
        obstacles: Current obstacles
        my_tail: Our tail position (will free up after move)
    
    Returns:
        Number of reachable cells after move
    """
    new_x, new_y = get_new_position(head[0], head[1], direction)
    new_pos = (new_x, new_y)
    
    # Can't move there at all
    if not is_in_bounds(new_x, new_y, width, height):
        return 0
    if new_pos in obstacles:
        return 0
    
    # After moving, our head occupies new position
    # Tail will move away (unless we eat food), so we simulate that
    new_obstacles = obstacles.copy()
    new_obstacles.add(head)  # Old head position becomes part of body
    
    # Remove tail from obstacles if provided (it will move)
    if my_tail and my_tail in new_obstacles:
        new_obstacles.discard(my_tail)
    
    return flood_fill_count(new_pos, width, height, new_obstacles)


def is_dead_end(
    pos: Tuple[int, int],
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    min_space: int = 5
) -> bool:
    """
    Check if a position leads to a dead end (limited escape space).
    
    Args:
        pos: Position to check
        width: Grid width  
        height: Grid height
        obstacles: Set of blocked positions
        min_space: Minimum acceptable reachable space
    
    Returns:
        True if position is a dead end
    """
    reachable = flood_fill_count(pos, width, height, obstacles)
    return reachable < min_space


def find_best_direction_by_space(
    head: Tuple[int, int],
    current_direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
    my_tail: Optional[Tuple[int, int]] = None
) -> Optional[str]:
    """
    Find the direction that maximizes reachable space.
    
    This is a fallback strategy when no clear goal exists.
    
    Args:
        head: Current head position
        current_direction: Currently facing direction
        width: Grid width
        height: Grid height
        obstacles: Current obstacles
        my_tail: Our tail position
    
    Returns:
        Best direction by space, or None if no safe move
    """
    safe_moves = get_safe_moves(head, current_direction, width, height, obstacles)
    
    if not safe_moves:
        return None
    
    best_direction = None
    best_space = -1
    
    for direction in safe_moves:
        space = calculate_space_after_move(
            head, direction, width, height, obstacles, my_tail
        )
        if space > best_space:
            best_space = space
            best_direction = direction
    
    return best_direction

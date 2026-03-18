"""
utils.py - Enhanced utility functions for CopperHead Bot v2

Key improvements over v1:
- Multi-opponent support helpers
- Growing-snake awareness (detect when tail stays after eating)
- Cached neighbor generation
- Direction-aware utilities for wall proximity
"""

from typing import Tuple, List, Set, Dict, Optional

# Direction vectors: maps direction name to (dx, dy) movement
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Opposite directions: used to prevent illegal reversals
OPPOSITES: Dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# All possible directions as a list for iteration
ALL_DIRECTIONS: List[str] = ["up", "down", "left", "right"]

_DIRECTION_SET = frozenset(ALL_DIRECTIONS)


def get_new_position(x: int, y: int, direction: str) -> Tuple[int, int]:
    dx, dy = DIRECTIONS.get(direction, (0, 0))
    return x + dx, y + dy


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def get_neighbors(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    neighbors = []
    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    return neighbors


def parse_snake_body(body_list: List[List[int]]) -> List[Tuple[int, int]]:
    return [(seg[0], seg[1]) for seg in body_list]


def get_all_snake_positions(
    snakes: Dict,
    exclude_tail: bool = False,
    ate_food_ids: Optional[Set[str]] = None,
) -> Set[Tuple[int, int]]:
    """
    Get all positions occupied by any alive snake.

    Args:
        snakes: Snakes dict from game state.
        exclude_tail: If True, exclude tail positions (they will move).
        ate_food_ids: Set of snake IDs that just ate food – their tails stay.
    """
    ate_food_ids = ate_food_ids or set()
    positions: Set[Tuple[int, int]] = set()
    for sid, snake_data in snakes.items():
        if not snake_data.get("alive", True):
            continue
        body = snake_data.get("body", [])
        if not body:
            continue
        for i, segment in enumerate(body):
            # Skip tail only when it will actually move away (didn't eat)
            if exclude_tail and i == len(body) - 1 and sid not in ate_food_ids:
                continue
            positions.add((segment[0], segment[1]))
    return positions


def get_valid_directions(current_direction: str) -> List[str]:
    opposite = OPPOSITES.get(current_direction, "")
    return [d for d in ALL_DIRECTIONS if d != opposite]


def direction_from_positions(
    from_pos: Tuple[int, int], to_pos: Tuple[int, int]
) -> Optional[str]:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    for direction, (dir_dx, dir_dy) in DIRECTIONS.items():
        if dx == dir_dx and dy == dir_dy:
            return direction
    return None


def count_safe_neighbors(
    x: int, y: int, width: int, height: int, obstacles: Set[Tuple[int, int]]
) -> int:
    return sum(1 for nx, ny in get_neighbors(x, y, width, height) if (nx, ny) not in obstacles)


def is_opposite_direction(dir1: str, dir2: str) -> bool:
    return OPPOSITES.get(dir1) == dir2


def get_head_position(snake_data: Dict) -> Optional[Tuple[int, int]]:
    body = snake_data.get("body", [])
    if not body:
        return None
    return (body[0][0], body[0][1])


def get_snake_length(snake_data: Dict) -> int:
    return len(snake_data.get("body", []))


def find_closest_food(
    head: Tuple[int, int], foods: List[Dict]
) -> Optional[Tuple[int, int]]:
    if not foods:
        return None
    closest = None
    min_dist = float("inf")
    for food in foods:
        pos = (food["x"], food["y"])
        dist = manhattan_distance(head, pos)
        if dist < min_dist:
            min_dist = dist
            closest = pos
    return closest


def get_food_positions(foods: List[Dict]) -> List[Tuple[int, int]]:
    return [(food["x"], food["y"]) for food in foods]


def wall_proximity(pos: Tuple[int, int], width: int, height: int) -> int:
    """Return the minimum distance from *pos* to any wall (0 = on edge)."""
    return min(pos[0], pos[1], width - 1 - pos[0], height - 1 - pos[1])


def get_all_opponents(
    snakes: Dict, player_id: int
) -> List[Dict]:
    """Return a list of dicts with parsed info for every alive opponent."""
    opponents = []
    pid = str(player_id)
    for sid, sdata in snakes.items():
        if sid == pid:
            continue
        if not sdata.get("alive", True):
            continue
        body_raw = sdata.get("body", [])
        if not body_raw:
            continue
        body = parse_snake_body(body_raw)
        opponents.append({
            "id": sid,
            "body": body,
            "head": body[0],
            "tail": body[-1] if len(body) > 1 else body[0],
            "direction": sdata.get("direction", "right"),
            "length": len(body),
            "buff": sdata.get("buff", "default"),
        })
    return opponents


def opponent_reachable_in_one(
    opp_head: Tuple[int, int],
    opp_direction: str,
    width: int,
    height: int,
    obstacles: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """All squares an opponent could occupy on the very next tick."""
    opposite = OPPOSITES.get(opp_direction, "")
    reachable: Set[Tuple[int, int]] = set()
    for d in ALL_DIRECTIONS:
        if d == opposite:
            continue
        nx, ny = get_new_position(opp_head[0], opp_head[1], d)
        if is_in_bounds(nx, ny, width, height) and (nx, ny) not in obstacles:
            reachable.add((nx, ny))
    return reachable

"""
utils.py - Utility functions for CopperHead bot

Provides coordinate manipulation, direction helpers, and grid utilities.
"""

from typing import Tuple, List, Set, Dict, Optional

# Direction vectors: maps direction name to (dx, dy) movement
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

# Opposite directions: used to prevent illegal reversals
OPPOSITES: Dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left"
}

# All possible directions as a list for iteration
ALL_DIRECTIONS: List[str] = ["up", "down", "left", "right"]


def get_new_position(x: int, y: int, direction: str) -> Tuple[int, int]:
    """
    Calculate the new position after moving in a given direction.
    
    Args:
        x: Current x coordinate
        y: Current y coordinate
        direction: One of "up", "down", "left", "right"
    
    Returns:
        Tuple of (new_x, new_y)
    """
    dx, dy = DIRECTIONS.get(direction, (0, 0))
    return x + dx, y + dy


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
    
    Returns:
        Manhattan distance (sum of absolute differences)
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """
    Check if a position is within the game grid.
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Grid width
        height: Grid height
    
    Returns:
        True if position is within bounds
    """
    return 0 <= x < width and 0 <= y < height


def get_neighbors(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    """
    Get all valid neighboring positions within bounds.
    
    Args:
        x: Current x coordinate
        y: Current y coordinate
        width: Grid width
        height: Grid height
    
    Returns:
        List of valid neighboring positions
    """
    neighbors = []
    for direction in ALL_DIRECTIONS:
        nx, ny = get_new_position(x, y, direction)
        if is_in_bounds(nx, ny, width, height):
            neighbors.append((nx, ny))
    return neighbors


def parse_snake_body(body_list: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Convert server snake body format to list of tuples.
    
    Args:
        body_list: List of [x, y] from server
    
    Returns:
        List of (x, y) tuples
    """
    return [(segment[0], segment[1]) for segment in body_list]


def get_all_snake_positions(snakes: Dict, exclude_tail: bool = False) -> Set[Tuple[int, int]]:
    """
    Get all positions occupied by any snake.
    
    Args:
        snakes: Snakes dict from game state
        exclude_tail: If True, exclude tail positions (they'll move away)
    
    Returns:
        Set of all occupied positions
    """
    positions = set()
    for snake_data in snakes.values():
        if not snake_data.get("alive", True):
            continue
        body = snake_data.get("body", [])
        if not body:
            continue
        
        # Convert to tuples and add to set
        for i, segment in enumerate(body):
            # Skip tail if requested (useful for predicting where we can move)
            if exclude_tail and i == len(body) - 1:
                continue
            positions.add((segment[0], segment[1]))
    
    return positions


def get_valid_directions(current_direction: str) -> List[str]:
    """
    Get all valid directions (excluding reverse of current).
    
    Args:
        current_direction: Currently facing direction
    
    Returns:
        List of valid directions to move
    """
    opposite = OPPOSITES.get(current_direction, "")
    return [d for d in ALL_DIRECTIONS if d != opposite]


def direction_from_positions(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Optional[str]:
    """
    Determine which direction to move from one position to an adjacent one.
    
    Args:
        from_pos: Starting position
        to_pos: Target position (must be adjacent)
    
    Returns:
        Direction string, or None if not adjacent
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    for direction, (dir_dx, dir_dy) in DIRECTIONS.items():
        if dx == dir_dx and dy == dir_dy:
            return direction
    
    return None


def count_safe_neighbors(
    x: int, 
    y: int, 
    width: int, 
    height: int, 
    obstacles: Set[Tuple[int, int]]
) -> int:
    """
    Count how many adjacent squares are safe (not wall or obstacle).
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Grid width
        height: Grid height
        obstacles: Set of occupied positions
    
    Returns:
        Number of safe adjacent squares (0-4)
    """
    count = 0
    for nx, ny in get_neighbors(x, y, width, height):
        if (nx, ny) not in obstacles:
            count += 1
    return count


def is_opposite_direction(dir1: str, dir2: str) -> bool:
    """
    Check if two directions are opposite.
    
    Args:
        dir1: First direction
        dir2: Second direction
    
    Returns:
        True if directions are opposite
    """
    return OPPOSITES.get(dir1) == dir2


def get_head_position(snake_data: Dict) -> Optional[Tuple[int, int]]:
    """
    Extract head position from snake data.
    
    Args:
        snake_data: Snake object from game state
    
    Returns:
        (x, y) tuple of head position, or None
    """
    body = snake_data.get("body", [])
    if not body:
        return None
    return (body[0][0], body[0][1])


def get_snake_length(snake_data: Dict) -> int:
    """
    Get the length of a snake.
    
    Args:
        snake_data: Snake object from game state
    
    Returns:
        Length of snake body
    """
    return len(snake_data.get("body", []))


def find_closest_food(
    head: Tuple[int, int], 
    foods: List[Dict]
) -> Optional[Tuple[int, int]]:
    """
    Find the closest food item by Manhattan distance.
    
    Args:
        head: Current head position
        foods: List of food objects from game state
    
    Returns:
        Position of closest food, or None if no food
    """
    if not foods:
        return None
    
    closest = None
    min_dist = float('inf')
    
    for food in foods:
        pos = (food["x"], food["y"])
        dist = manhattan_distance(head, pos)
        if dist < min_dist:
            min_dist = dist
            closest = pos
    
    return closest


def get_food_positions(foods: List[Dict]) -> List[Tuple[int, int]]:
    """
    Extract all food positions from game state.
    
    Args:
        foods: List of food objects from game state
    
    Returns:
        List of (x, y) food positions
    """
    return [(food["x"], food["y"]) for food in foods]

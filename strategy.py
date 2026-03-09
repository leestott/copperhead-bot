"""
strategy.py - Multi-layer strategy engine for CopperHead bot

Implements a sophisticated decision-making system with:
1. Safety-first survival
2. Short-horizon pathfinding with lookahead
3. Area control and Voronoi space partitioning
4. Adaptive aggression with head-collision tactics
5. Food racing with opponent prediction
6. Tail-chasing survival technique
7. Tournament awareness
"""

from typing import Tuple, List, Set, Dict, Optional
from dataclasses import dataclass

from utils import (
    DIRECTIONS, OPPOSITES, ALL_DIRECTIONS,
    get_new_position, manhattan_distance, is_in_bounds,
    get_all_snake_positions, parse_snake_body, get_valid_directions,
    direction_from_positions, count_safe_neighbors, get_head_position,
    get_snake_length, find_closest_food, get_food_positions
)
from pathfinding import (
    bfs_distance, bfs_path, astar_path, flood_fill_count,
    get_safe_moves, get_corridor_score, find_opponent_danger_zone,
    calculate_space_after_move, is_dead_end, find_best_direction_by_space,
    predict_opponent_move, calculate_voronoi_control, find_tail_chase_path,
    lookahead_evaluate, food_race_winner, center_distance
)


@dataclass
class MoveScore:
    """Score breakdown for a potential move."""
    direction: str
    total: float
    safety: float
    food: float
    space: float
    aggression: float
    corridor: float
    voronoi: float = 0.0
    center: float = 0.0
    lookahead: float = 0.0
    
    def __lt__(self, other):
        return self.total < other.total


class StrategyEngine:
    """
    Multi-layer strategy engine for competitive Snake AI.
    
    Evaluates moves based on multiple weighted criteria and selects
    the optimal direction considering safety, food, space, opponent prediction,
    Voronoi control, and multi-move lookahead.
    """
    
    # Weight constants - TUNED FOR TOURNAMENT WINNING
    WEIGHT_SAFETY = 10000      # Safety is paramount
    WEIGHT_FOOD = 150          # Increased food priority
    WEIGHT_SPACE = 80          # Increased space priority
    WEIGHT_AGGRESSION = 50     # More aggressive when winning
    WEIGHT_CORRIDOR = 30       # Avoid corridors
    WEIGHT_VORONOI = 40        # Territory control
    WEIGHT_CENTER = 15         # Prefer center positions
    WEIGHT_LOOKAHEAD = 60      # Future move evaluation
    WEIGHT_GRAPES = 300        # Grapes are extremely valuable (shrink opponent)
    
    # Minimum acceptable reachable space (avoids self-trap)
    MIN_SAFE_SPACE = 10        # Increased from 8
    
    def __init__(self, difficulty: int = 10):
        """
        Initialize strategy engine.
        
        Args:
            difficulty: 1-10, affects decision quality (10 = optimal play)
        """
        self.difficulty = difficulty
        self.grid_width = 30
        self.grid_height = 20
        
        # Game tracking
        self.my_length = 3
        self.opponent_length = 3
        self.games_played = 0
        self.wins = 0
        
        # Cache for performance
        self._last_opponent_head = None
        self._predicted_opponent_pos = None
    
    def update_grid_size(self, width: int, height: int):
        """Update grid dimensions from game state."""
        self.grid_width = width
        self.grid_height = height
    
    def reset_game_state(self):
        """Reset state for new game (called on gameover)."""
        self.my_length = 3
        self.opponent_length = 3
        self._last_opponent_head = None
        self._predicted_opponent_pos = None
    
    def record_game_result(self, won: bool):
        """Track game results for adaptive strategy."""
        self.games_played += 1
        if won:
            self.wins += 1
    
    def calculate_move(self, game_state: Dict, player_id: int) -> Optional[str]:
        """
        Main decision function - calculates the best move.
        
        Enhanced with opponent prediction, Voronoi control, and lookahead.
        """
        if not game_state:
            return None
        
        # Extract game info
        snakes = game_state.get("snakes", {})
        foods = game_state.get("foods", [])
        grid = game_state.get("grid", {})
        
        # Update grid dimensions
        self.grid_width = grid.get("width", 30)
        self.grid_height = grid.get("height", 20)
        
        # Get our snake
        my_snake = snakes.get(str(player_id))
        if not my_snake or not my_snake.get("body"):
            return None
        
        if not my_snake.get("alive", True):
            return None
        
        # Extract our info
        my_body = parse_snake_body(my_snake["body"])
        my_head = my_body[0]
        my_tail = my_body[-1] if len(my_body) > 1 else my_head
        my_direction = my_snake.get("direction", "right")
        my_buff = my_snake.get("buff", "default")
        self.my_length = len(my_body)
        
        # Find opponent snake
        opponent_snake = None
        opponent_head = None
        opponent_body = None
        opponent_direction = None
        for sid, snake_data in snakes.items():
            if sid != str(player_id) and snake_data.get("alive", True):
                opponent_snake = snake_data
                opp_body_raw = snake_data.get("body", [])
                if opp_body_raw:
                    opponent_body = parse_snake_body(opp_body_raw)
                    opponent_head = opponent_body[0]
                    opponent_direction = snake_data.get("direction", "right")
                    self.opponent_length = len(opponent_body)
                break
        
        # Build obstacle set
        obstacles = get_all_snake_positions(snakes, exclude_tail=True)
        for segment in my_body[1:]:
            obstacles.add(segment)
        
        # CRITICAL: Predict opponent's next move
        food_positions = get_food_positions(foods)
        if opponent_head and opponent_body and opponent_direction:
            self._predicted_opponent_pos = predict_opponent_move(
                opponent_head, opponent_direction, opponent_body,
                my_head, food_positions,
                self.grid_width, self.grid_height, obstacles
            )
        else:
            self._predicted_opponent_pos = None
        
        # Get all safe moves
        safe_moves = get_safe_moves(
            my_head, my_direction, 
            self.grid_width, self.grid_height, 
            obstacles
        )
        
        if not safe_moves:
            # Try tail-chasing as survival tactic
            tail_dir = find_tail_chase_path(
                my_head, my_tail,
                self.grid_width, self.grid_height, obstacles
            )
            if tail_dir:
                return tail_dir
            return self._desperation_move(my_head, my_direction, obstacles)
        
        if len(safe_moves) == 1:
            return safe_moves[0]
        
        # Score each safe move with all criteria
        scores = []
        for direction in safe_moves:
            score = self._score_move(
                direction, my_head, my_body, my_tail, my_direction,
                opponent_head, opponent_body, opponent_direction,
                foods, obstacles
            )
            scores.append(score)
        
        # Sort by total score (highest first)
        scores.sort(key=lambda s: s.total, reverse=True)
        
        return scores[0].direction
    
    def _score_move(
        self,
        direction: str,
        my_head: Tuple[int, int],
        my_body: List[Tuple[int, int]],
        my_tail: Tuple[int, int],
        my_direction: str,
        opponent_head: Optional[Tuple[int, int]],
        opponent_body: Optional[List[Tuple[int, int]]],
        opponent_direction: Optional[str],
        foods: List[Dict],
        obstacles: Set[Tuple[int, int]]
    ) -> MoveScore:
        """
        Calculate comprehensive score for a potential move.
        
        Enhanced with Voronoi, lookahead, food racing, and head collision tactics.
        """
        new_x, new_y = get_new_position(my_head[0], my_head[1], direction)
        new_pos = (new_x, new_y)
        
        # Initialize score components
        safety_score = self.WEIGHT_SAFETY
        food_score = 0.0
        space_score = 0.0
        aggression_score = 0.0
        corridor_score = 0.0
        voronoi_score = 0.0
        center_score = 0.0
        lookahead_score = 0.0
        
        # === SAFETY LAYER WITH HEAD COLLISION TACTICS ===
        if opponent_head:
            dist_to_opp = manhattan_distance(new_pos, opponent_head)
            
            # Head-to-head collision detection
            if dist_to_opp == 1:
                if self.my_length > self.opponent_length:
                    # WE WIN HEAD COLLISION - this is GOOD
                    safety_score += self.WEIGHT_SAFETY * 0.5
                    aggression_score += self.WEIGHT_AGGRESSION * 3
                elif self.my_length == self.opponent_length:
                    # Tie = bad, avoid
                    safety_score -= self.WEIGHT_SAFETY * 0.9
                else:
                    # We lose - very bad
                    safety_score -= self.WEIGHT_SAFETY * 0.95
            
            # Check if we'd collide with predicted opponent position
            if self._predicted_opponent_pos and new_pos == self._predicted_opponent_pos:
                if self.my_length > self.opponent_length:
                    aggression_score += self.WEIGHT_AGGRESSION * 2
                else:
                    safety_score -= self.WEIGHT_SAFETY * 0.7
        
        # === SPACE LAYER ===
        space_after = calculate_space_after_move(
            my_head, direction, 
            self.grid_width, self.grid_height,
            obstacles, my_tail
        )
        
        total_cells = self.grid_width * self.grid_height
        space_ratio = space_after / total_cells
        space_score = space_ratio * self.WEIGHT_SPACE * 10
        
        # Heavy penalty for trapping moves
        if space_after < self.MIN_SAFE_SPACE:
            safety_score -= self.WEIGHT_SAFETY * 0.6
            space_score -= self.WEIGHT_SPACE * 10
        elif space_after < self.my_length + 3:
            # Not enough space to fit ourselves
            safety_score -= self.WEIGHT_SAFETY * 0.3
        
        # === VORONOI TERRITORY CONTROL ===
        if opponent_head:
            my_cells, opp_cells, _ = calculate_voronoi_control(
                new_pos, opponent_head,
                self.grid_width, self.grid_height,
                obstacles | {my_head}
            )
            
            territory_advantage = my_cells - opp_cells
            voronoi_score = territory_advantage * self.WEIGHT_VORONOI / 50
            
            # Big bonus for dominating territory
            if my_cells > opp_cells * 1.5:
                voronoi_score += self.WEIGHT_VORONOI
        
        # === FOOD LAYER WITH RACING ===
        if foods:
            # Find grapes (super valuable - shrink opponent)
            grapes = [f for f in foods if f.get("type") == "grapes"]
            regular_food = [f for f in foods if f.get("type") != "grapes"]
            
            # Prioritize grapes heavily
            if grapes:
                grape_pos = (grapes[0]["x"], grapes[0]["y"])
                if new_pos == grape_pos:
                    food_score = self.WEIGHT_GRAPES
                else:
                    grape_dist = bfs_distance(
                        new_pos, grape_pos,
                        self.grid_width, self.grid_height, obstacles
                    )
                    if grape_dist > 0:
                        race_result = food_race_winner(
                            new_pos, opponent_head, grape_pos,
                            self.grid_width, self.grid_height, obstacles
                        )
                        if race_result == "me":
                            food_score = self.WEIGHT_GRAPES * (1.0 / (grape_dist + 1))
                        elif race_result == "tie":
                            food_score = self.WEIGHT_GRAPES * 0.3 / (grape_dist + 1)
            
            # Regular food
            closest_food = find_closest_food(new_pos, foods)
            if closest_food:
                if new_pos == closest_food:
                    food_score = max(food_score, self.WEIGHT_FOOD * 10)
                else:
                    path_dist = bfs_distance(
                        new_pos, closest_food,
                        self.grid_width, self.grid_height, obstacles
                    )
                    if path_dist > 0:
                        # Check if we can win the race
                        race = food_race_winner(
                            new_pos, opponent_head, closest_food,
                            self.grid_width, self.grid_height, obstacles
                        )
                        
                        base_food = self.WEIGHT_FOOD * (1.0 / (path_dist + 1))
                        
                        if race == "me":
                            food_score = max(food_score, base_food * 2)
                        elif race == "tie":
                            # Tie goes to longer snake, so worth if longer
                            if self.my_length >= self.opponent_length:
                                food_score = max(food_score, base_food * 1.2)
                            else:
                                food_score = max(food_score, base_food * 0.5)
                        else:
                            # Opponent wins - still go if nothing better
                            food_score = max(food_score, base_food * 0.3)
                
                # Urgency adjustment
                if self.my_length < self.opponent_length:
                    food_score *= 1.8
                elif self.my_length > self.opponent_length + 5:
                    food_score *= 0.5
        
        # === CORRIDOR LAYER ===
        corridor_value = get_corridor_score(
            new_pos, self.grid_width, self.grid_height, obstacles
        )
        if corridor_value <= 1:
            corridor_score = -self.WEIGHT_CORRIDOR * 5
        elif corridor_value == 2:
            corridor_score = -self.WEIGHT_CORRIDOR * 2
        else:
            corridor_score = self.WEIGHT_CORRIDOR * (corridor_value - 2)
        
        # === CENTER CONTROL ===
        center_dist = center_distance(new_pos, self.grid_width, self.grid_height)
        max_center_dist = (self.grid_width + self.grid_height) / 2
        center_score = self.WEIGHT_CENTER * (1 - center_dist / max_center_dist)
        
        # === LOOKAHEAD EVALUATION ===
        lookahead_value = lookahead_evaluate(
            my_head, direction, my_body,
            opponent_head, opponent_body,
            self.grid_width, self.grid_height,
            obstacles, depth=2
        )
        if lookahead_value > 0:
            lookahead_score = min(lookahead_value, 100) * self.WEIGHT_LOOKAHEAD / 100
        elif lookahead_value == float('-inf'):
            # This leads to death within lookahead - severely penalize
            safety_score -= self.WEIGHT_SAFETY * 0.4
        
        # === AGGRESSION LAYER ===
        if opponent_head and opponent_direction:
            opp_dist = manhattan_distance(new_pos, opponent_head)
            
            if self.my_length > self.opponent_length:
                # We're winning - apply pressure, cut off escape routes
                if 2 <= opp_dist <= 5:
                    aggression_score += self.WEIGHT_AGGRESSION
                
                # Block opponent's likely path
                if self._predicted_opponent_pos:
                    if manhattan_distance(new_pos, self._predicted_opponent_pos) <= 2:
                        aggression_score += self.WEIGHT_AGGRESSION * 0.5
                
                # Extra bonus for cornering opponent
                opp_space = flood_fill_count(
                    opponent_head, self.grid_width, self.grid_height,
                    obstacles, max_depth=10
                )
                if opp_space < 15:
                    aggression_score += self.WEIGHT_AGGRESSION * 1.5
            else:
                # We're behind - play safe, avoid confrontation
                if opp_dist < 3:
                    aggression_score -= self.WEIGHT_AGGRESSION
                
                opp_danger = find_opponent_danger_zone(
                    opponent_head, opponent_direction,
                    self.grid_width, self.grid_height, radius=2
                )
                if new_pos in opp_danger:
                    aggression_score -= self.WEIGHT_AGGRESSION * 1.5
        
        # Calculate total score
        total = (safety_score + food_score + space_score + aggression_score + 
                corridor_score + voronoi_score + center_score + lookahead_score)
        
        return MoveScore(
            direction=direction,
            total=total,
            safety=safety_score,
            food=food_score,
            space=space_score,
            aggression=aggression_score,
            corridor=corridor_score,
            voronoi=voronoi_score,
            center=center_score,
            lookahead=lookahead_score
        )
    
    def _desperation_move(
        self,
        head: Tuple[int, int],
        current_direction: str,
        obstacles: Set[Tuple[int, int]]
    ) -> Optional[str]:
        """
        Find least-bad move when all seem unsafe.
        
        Enhanced with tail-chase consideration.
        """
        valid_dirs = get_valid_directions(current_direction)
        
        best_dir = None
        best_space = -1
        
        for direction in valid_dirs:
            new_x, new_y = get_new_position(head[0], head[1], direction)
            
            if not is_in_bounds(new_x, new_y, self.grid_width, self.grid_height):
                continue
            
            test_obstacles = obstacles.copy()
            space = flood_fill_count(
                (new_x, new_y),
                self.grid_width, self.grid_height,
                test_obstacles,
                max_depth=10
            )
            
            if space > best_space:
                best_space = space
                best_dir = direction
        
        return best_dir if best_dir else current_direction
    
    def get_debug_info(self, game_state: Dict, player_id: int) -> Dict:
        """Get debug information for current state."""
        snakes = game_state.get("snakes", {})
        my_snake = snakes.get(str(player_id), {})
        
        return {
            "grid_size": f"{self.grid_width}x{self.grid_height}",
            "my_length": self.my_length,
            "opponent_length": self.opponent_length,
            "length_advantage": self.my_length - self.opponent_length,
            "games_played": self.games_played,
            "wins": self.wins,
            "my_buff": my_snake.get("buff", "none"),
            "predicted_opponent": self._predicted_opponent_pos,
        }

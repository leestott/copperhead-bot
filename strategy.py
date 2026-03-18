"""
strategy.py - Tournament-winning strategy engine for CopperHead Bot v2

Key improvements over v1:
1.  Phase-aware weights (opening / mid-game / endgame)
2.  Multi-opponent support
3.  Growing-snake awareness (tail stays after eating)
4.  Full opponent-reachable-in-1 head-collision check
5.  Articulation-point awareness — avoid creating or entering chokepoints
6.  Board-split detection — never pick moves that split us into the smaller component
7.  Deeper lookahead (depth 3)
8.  Better corridor scoring via actual flood fill
9.  Smarter food valuation — risk-adjusted, not just distance
10. Adaptive aggression based on length delta AND board state
11. Wall-proximity penalty to avoid getting pinned against edges
12. Opponent-ate-food tracking (detect when opponent tail didn't move)
"""

import json
import os
from pathlib import Path
from typing import Tuple, List, Set, Dict, Optional
from dataclasses import dataclass

from utils import (
    DIRECTIONS, OPPOSITES, ALL_DIRECTIONS,
    get_new_position, manhattan_distance, is_in_bounds,
    get_all_snake_positions, parse_snake_body, get_valid_directions,
    direction_from_positions, count_safe_neighbors, get_head_position,
    get_snake_length, find_closest_food, get_food_positions,
    wall_proximity, get_all_opponents, opponent_reachable_in_one,
)
from pathfinding import (
    bfs_distance, bfs_path, astar_path, flood_fill_count,
    get_safe_moves, get_corridor_score, find_opponent_danger_zone,
    calculate_space_after_move, is_dead_end, find_best_direction_by_space,
    predict_opponent_move, calculate_voronoi_control, calculate_voronoi_multi,
    find_tail_chase_path, lookahead_evaluate, food_race_winner,
    center_distance, move_splits_board, find_articulation_points,
    flood_fill_reachable,
)


@dataclass
class MoveScore:
    """Score breakdown for a potential move."""
    direction: str
    total: float = 0.0
    safety: float = 0.0
    food: float = 0.0
    space: float = 0.0
    aggression: float = 0.0
    corridor: float = 0.0
    voronoi: float = 0.0
    center: float = 0.0
    lookahead: float = 0.0
    wall: float = 0.0
    split: float = 0.0

    def __lt__(self, other: "MoveScore") -> bool:
        return self.total < other.total


@dataclass
class AdaptiveProfile:
    """Session-scoped weight multipliers learned from round outcomes."""

    food_mult: float = 1.0
    space_mult: float = 1.0
    aggression_mult: float = 1.0
    wall_mult: float = 1.0
    lookahead_mult: float = 1.0
    center_mult: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "food_mult": self.food_mult,
            "space_mult": self.space_mult,
            "aggression_mult": self.aggression_mult,
            "wall_mult": self.wall_mult,
            "lookahead_mult": self.lookahead_mult,
            "center_mult": self.center_mult,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "AdaptiveProfile":
        if not isinstance(data, dict):
            return cls()
        return cls(
            food_mult=float(data.get("food_mult", 1.0)),
            space_mult=float(data.get("space_mult", 1.0)),
            aggression_mult=float(data.get("aggression_mult", 1.0)),
            wall_mult=float(data.get("wall_mult", 1.0)),
            lookahead_mult=float(data.get("lookahead_mult", 1.0)),
            center_mult=float(data.get("center_mult", 1.0)),
        )

    def copy(self) -> "AdaptiveProfile":
        return AdaptiveProfile.from_dict(self.to_dict())

    def clamp(self) -> None:
        self.food_mult = min(1.35, max(0.70, self.food_mult))
        self.space_mult = min(1.50, max(0.85, self.space_mult))
        self.aggression_mult = min(1.35, max(0.45, self.aggression_mult))
        self.wall_mult = min(1.60, max(0.85, self.wall_mult))
        self.lookahead_mult = min(1.45, max(0.85, self.lookahead_mult))
        self.center_mult = min(1.35, max(0.85, self.center_mult))


# ---------------------------------------------------------------------------
# Game-phase enum
# ---------------------------------------------------------------------------
_PHASE_OPENING = "opening"
_PHASE_MID = "mid"
_PHASE_ENDGAME = "endgame"


class StrategyEngine:
    """
    Multi-layer tournament strategy engine – v2.

    Evaluates each candidate direction across ~10 weighted criteria.
    Weights shift dynamically based on game phase and length advantage.
    """

    # === Base weights (tournament-tuned v3) ===
    W_SAFETY     = 10_000
    W_FOOD       = 250           # grow faster — length wins games
    W_SPACE      = 180           # survival is king
    W_AGGRESSION = 40            # conservative aggression
    W_CORRIDOR   = 60            # strong corridor avoidance
    W_VORONOI    = 50
    W_CENTER     = 25            # center is safer
    W_LOOKAHEAD  = 90
    W_GRAPES     = 500           # grapes are game-changing
    W_WALL       = 45            # walls are death traps
    W_SPLIT      = 8_000         # huge penalty for board-splitting moves
    W_TAIL_REACH = 6_000         # must be able to reach tail
    W_STRAIGHT   = 12            # reduce wobbling
    W_FOOD_TRAP  = 3_000         # don't eat into dead ends
    W_FOOD_DENY  = 110           # deny high-value food races when we can't arrive first
    W_INTERCEPT  = 95            # cut off opponent's next growth when behind
    W_RECOVERY   = 140           # recover food tempo when behind ladder bots

    MIN_SAFE_SPACE = 12          # balanced — not too conservative

    def __init__(self, difficulty: int = 10):
        self.difficulty = max(1, min(10, difficulty))
        self.grid_width = 30
        self.grid_height = 20
        self.current_opponent_name = ""

        # Per-game tracking
        self.my_length = 3
        self.opponent_lengths: Dict[str, int] = {}
        self.games_played = 0
        self.wins = 0
        self.tick = 0

        # Opponent-ate tracking
        self._prev_opponent_tails: Dict[str, Tuple[int, int]] = {}
        self._opponent_ate: Set[str] = set()

        # Prediction cache
        self._predicted_positions: Dict[str, Tuple[int, int]] = {}
        self._last_direction_by_snake: Dict[str, str] = {}
        self._direction_change_tick: Dict[str, int] = {}

        # Session adaptation learned from recent rounds
        self.profile = AdaptiveProfile()
        self._global_profile = AdaptiveProfile()
        self._opponent_profiles: Dict[str, AdaptiveProfile] = {}
        self._cohort_profiles: Dict[str, AdaptiveProfile] = {}
        self._active_profile_key = ""
        self._active_cohort_keys: List[str] = []
        self._learning_stats: Dict[str, Dict[str, int | List[str]]] = {}
        self._recent_loss_reasons: List[str] = []
        self._last_decision_trace = ""
        self._opening_anchor: Optional[Tuple[int, int]] = None
        self._recent_heads: List[Tuple[int, int]] = []
        self.learning_state_path = Path(
            os.environ.get(
                "COPPERHEAD_LEARNING_FILE",
                str(Path(__file__).resolve().parent / ".copperhead-learning.json"),
            )
        )
        self._load_learning_state()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def update_grid_size(self, width: int, height: int) -> None:
        self.grid_width = width
        self.grid_height = height

    def reset_game_state(self) -> None:
        self.my_length = 3
        self.opponent_lengths.clear()
        self.tick = 0
        self._prev_opponent_tails.clear()
        self._opponent_ate.clear()
        self._predicted_positions.clear()
        self._last_direction_by_snake.clear()
        self._direction_change_tick.clear()
        self._last_decision_trace = ""
        self._opening_anchor = None
        self._recent_heads.clear()

    def _remember_head(self, head: Tuple[int, int]) -> None:
        if not self._recent_heads or self._recent_heads[-1] != head:
            self._recent_heads.append(head)
            if len(self._recent_heads) > 18:
                self._recent_heads = self._recent_heads[-18:]

    def set_current_opponent(self, opponent_name: str) -> None:
        self.current_opponent_name = (opponent_name or "").strip()
        self._active_profile_key = self._profile_key_for_name(self.current_opponent_name)
        self._active_cohort_keys = self._cohort_keys_for_name(self.current_opponent_name)
        self._activate_profile_for_current_opponent()

    def _profile_key_for_name(self, opponent_name: str) -> str:
        name = (opponent_name or "").strip().lower()
        if not name:
            return ""
        if "copperbot l" in name:
            level = self._opening_level()
            if level is not None:
                return f"copperbot-l{level}"
        return name

    def _cohort_keys_for_name(self, opponent_name: str) -> List[str]:
        name = (opponent_name or "").strip().lower()
        keys: List[str] = []
        if "copperbot l" in name:
            level = self._opening_level()
            keys.append("copperbot-all-ladder")
            if level is not None:
                if level <= 4:
                    keys.append("copperbot-band-l1-l4")
                elif level <= 7:
                    keys.append("copperbot-band-l5-l7")
                else:
                    keys.append("copperbot-band-l8-l10")
        return keys

    def _blend_profiles(
        self,
        weighted_profiles: List[tuple[AdaptiveProfile, float]],
    ) -> AdaptiveProfile:
        if not weighted_profiles:
            return AdaptiveProfile()
        total_weight = sum(weight for _, weight in weighted_profiles)
        if total_weight <= 0:
            return AdaptiveProfile()
        blended = AdaptiveProfile(
            food_mult=sum(profile.food_mult * weight for profile, weight in weighted_profiles) / total_weight,
            space_mult=sum(profile.space_mult * weight for profile, weight in weighted_profiles) / total_weight,
            aggression_mult=sum(profile.aggression_mult * weight for profile, weight in weighted_profiles) / total_weight,
            wall_mult=sum(profile.wall_mult * weight for profile, weight in weighted_profiles) / total_weight,
            lookahead_mult=sum(profile.lookahead_mult * weight for profile, weight in weighted_profiles) / total_weight,
            center_mult=sum(profile.center_mult * weight for profile, weight in weighted_profiles) / total_weight,
        )
        blended.clamp()
        return blended

    def _activate_profile_for_current_opponent(self) -> None:
        weighted_profiles: List[tuple[AdaptiveProfile, float]] = [(self._global_profile, 0.25)]
        for cohort_key in self._active_cohort_keys:
            cohort_profile = self._cohort_profiles.get(cohort_key)
            if cohort_profile is None:
                continue
            cohort_weight = 0.2 if cohort_key == "copperbot-all-ladder" else 0.25
            weighted_profiles.append((cohort_profile, cohort_weight))
        opponent_profile = self._opponent_profiles.get(self._active_profile_key)
        if opponent_profile is not None:
            weighted_profiles.append((opponent_profile, 0.35))
        self.profile = self._blend_profiles(weighted_profiles)

    def _load_learning_state(self) -> None:
        try:
            if not self.learning_state_path.exists():
                return
            payload = json.loads(self.learning_state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        self._global_profile = AdaptiveProfile.from_dict(payload.get("global_profile"))
        profiles = payload.get("opponent_profiles", {})
        if isinstance(profiles, dict):
            self._opponent_profiles = {
                str(key): AdaptiveProfile.from_dict(value)
                for key, value in profiles.items()
                if isinstance(key, str)
            }
        cohort_profiles = payload.get("cohort_profiles", {})
        if isinstance(cohort_profiles, dict):
            self._cohort_profiles = {
                str(key): AdaptiveProfile.from_dict(value)
                for key, value in cohort_profiles.items()
                if isinstance(key, str)
            }
        stats = payload.get("learning_stats", {})
        if isinstance(stats, dict):
            self._learning_stats = {
                str(key): value for key, value in stats.items() if isinstance(value, dict)
            }
        self._activate_profile_for_current_opponent()

    def _save_learning_state(self) -> None:
        payload = {
            "global_profile": self._global_profile.to_dict(),
            "opponent_profiles": {
                key: profile.to_dict() for key, profile in self._opponent_profiles.items()
            },
            "cohort_profiles": {
                key: profile.to_dict() for key, profile in self._cohort_profiles.items()
            },
            "learning_stats": self._learning_stats,
        }
        try:
            self.learning_state_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _persist_learning_snapshot(self) -> None:
        active_key = self._active_profile_key
        if active_key:
            self._opponent_profiles[active_key] = self.profile.copy()
            stats = self._learning_stats.setdefault(active_key, {"rounds": 0, "losses": 0, "wins": 0, "recent_loss_reasons": []})
            stats["recent_loss_reasons"] = list(self._recent_loss_reasons)

        for cohort_key in self._active_cohort_keys:
            existing = self._cohort_profiles.get(cohort_key)
            if existing is None:
                self._cohort_profiles[cohort_key] = self.profile.copy()
                continue
            blended = AdaptiveProfile(
                food_mult=(existing.food_mult * 0.78) + (self.profile.food_mult * 0.22),
                space_mult=(existing.space_mult * 0.78) + (self.profile.space_mult * 0.22),
                aggression_mult=(existing.aggression_mult * 0.78) + (self.profile.aggression_mult * 0.22),
                wall_mult=(existing.wall_mult * 0.78) + (self.profile.wall_mult * 0.22),
                lookahead_mult=(existing.lookahead_mult * 0.78) + (self.profile.lookahead_mult * 0.22),
                center_mult=(existing.center_mult * 0.78) + (self.profile.center_mult * 0.22),
            )
            blended.clamp()
            self._cohort_profiles[cohort_key] = blended

        self._global_profile.food_mult = (self._global_profile.food_mult * 0.7) + (self.profile.food_mult * 0.3)
        self._global_profile.space_mult = (self._global_profile.space_mult * 0.7) + (self.profile.space_mult * 0.3)
        self._global_profile.aggression_mult = (self._global_profile.aggression_mult * 0.7) + (self.profile.aggression_mult * 0.3)
        self._global_profile.wall_mult = (self._global_profile.wall_mult * 0.7) + (self.profile.wall_mult * 0.3)
        self._global_profile.lookahead_mult = (self._global_profile.lookahead_mult * 0.7) + (self.profile.lookahead_mult * 0.3)
        self._global_profile.center_mult = (self._global_profile.center_mult * 0.7) + (self.profile.center_mult * 0.3)
        self._global_profile.clamp()
        self._save_learning_state()

    def is_ladder_opponent(self) -> bool:
        return self._is_ladder_bot()

    def _opening_level(self) -> Optional[int]:
        name = self.current_opponent_name.lower()
        marker = "copperbot l"
        start = name.find(marker)
        if start < 0:
            return None
        index = start + len(marker)
        digits: List[str] = []
        while index < len(name) and name[index].isdigit():
            digits.append(name[index])
            index += 1
        if not digits:
            return None
        return int("".join(digits))

    def _is_ladder_bot(self) -> bool:
        return self._opening_level() is not None

    def _opening_template(self) -> str:
        name = self.current_opponent_name.lower()
        if "copperbot l10" in name:
            return "l10"
        if "copperbot l9" in name:
            return "l9"
        if "copperheadrecycle" in name or "recycle" in name:
            return "recycle"
        return "default"

    def _should_trace_opening(self) -> bool:
        return (self._is_ladder_bot() or self._opening_template() == "recycle") and self.tick <= 12

    def _should_use_opening_anchor(self) -> bool:
        if self._opening_template() == "recycle":
            return True
        level = self._opening_level()
        return level is not None and level >= 7

    def _reset_decision_trace(self) -> None:
        self._last_decision_trace = ""

    def _append_trace(self, message: str) -> None:
        if not self._should_trace_opening():
            return
        if self._last_decision_trace:
            self._last_decision_trace += " | "
        self._last_decision_trace += message

    def get_last_decision_trace(self) -> str:
        return self._last_decision_trace

    def record_game_result(self, won: bool) -> None:
        self.games_played += 1
        if won:
            self.wins += 1

    def learn_from_round(self, summary: Dict) -> None:
        """
        Adapt session-level weights from the most recent round summary.

        The goal is not deep learning; it is pragmatic weight nudging so the bot
        becomes less likely to repeat obvious mistakes during a tournament run.
        """
        if not summary:
            return

        won = bool(summary.get("won", False))
        last_state = summary.get("last_state") or {}
        if not isinstance(last_state, dict):
            last_state = {}

        reasons: List[str] = []
        wall_distance = int(last_state.get("wall_distance", 99))
        free_ratio = float(last_state.get("free_ratio", 1.0))
        opponent_distance = int(last_state.get("nearest_opponent_distance", 99))
        my_length = int(last_state.get("my_length", 0))
        opponent_length = int(last_state.get("nearest_opponent_length", my_length))
        ate_recently = bool(last_state.get("ate_recently", False))
        aggressive_posture = bool(last_state.get("aggressive_posture", False))

        active_key = self._active_profile_key
        if active_key:
            stats = self._learning_stats.setdefault(active_key, {"rounds": 0, "losses": 0, "wins": 0, "recent_loss_reasons": []})
            stats["rounds"] = int(stats.get("rounds", 0)) + 1
            if won:
                stats["wins"] = int(stats.get("wins", 0)) + 1
            else:
                stats["losses"] = int(stats.get("losses", 0)) + 1

        if won:
            # Small decay toward baseline prevents overfitting to one good round.
            self.profile.food_mult = self.profile.food_mult * 0.985 + 0.015
            self.profile.space_mult = self.profile.space_mult * 0.985 + 0.015
            self.profile.wall_mult = self.profile.wall_mult * 0.985 + 0.015
            self.profile.lookahead_mult = self.profile.lookahead_mult * 0.985 + 0.015
            self.profile.center_mult = self.profile.center_mult * 0.985 + 0.015
            if my_length > opponent_length:
                self.profile.aggression_mult += 0.02
            else:
                self.profile.aggression_mult = self.profile.aggression_mult * 0.99 + 0.01
            self.profile.clamp()
            self._persist_learning_snapshot()
            return

        if wall_distance <= 1:
            reasons.append("wall_pressure")
            self.profile.wall_mult += 0.14
            self.profile.center_mult += 0.09
            self.profile.aggression_mult -= 0.04

        if free_ratio < 0.22:
            reasons.append("space_starvation")
            self.profile.space_mult += 0.10
            self.profile.lookahead_mult += 0.06
            self.profile.food_mult -= 0.04

        if opponent_distance <= 2 and my_length <= opponent_length:
            reasons.append("head_pressure")
            self.profile.aggression_mult -= 0.08
            self.profile.space_mult += 0.04
            self.profile.lookahead_mult += 0.04

        if ate_recently and free_ratio < 0.30:
            reasons.append("food_trap")
            self.profile.food_mult -= 0.07
            self.profile.space_mult += 0.06

        if my_length + 1 < opponent_length and free_ratio > 0.35 and not aggressive_posture:
            reasons.append("lost_length_race")
            self.profile.food_mult += 0.08

        if last_state.get("tick", 999) <= 25 and wall_distance <= 1:
            reasons.append("opening_wall_loss")
            self.profile.food_mult -= 0.06
            self.profile.space_mult += 0.06
            self.profile.center_mult += 0.05

        if not reasons:
            reasons.append("generic_loss")
            self.profile.space_mult += 0.03
            self.profile.lookahead_mult += 0.02
            self.profile.aggression_mult -= 0.02

        self._recent_loss_reasons = (self._recent_loss_reasons + reasons)[-8:]
        self.profile.clamp()
        self._persist_learning_snapshot()

    def get_adaptation_summary(self) -> Dict[str, float | List[str]]:
        return {
            "food_mult": round(self.profile.food_mult, 3),
            "space_mult": round(self.profile.space_mult, 3),
            "aggression_mult": round(self.profile.aggression_mult, 3),
            "wall_mult": round(self.profile.wall_mult, 3),
            "lookahead_mult": round(self.profile.lookahead_mult, 3),
            "center_mult": round(self.profile.center_mult, 3),
            "recent_loss_reasons": list(self._recent_loss_reasons),
            "active_profile": self._active_profile_key,
            "active_cohorts": list(self._active_cohort_keys),
        }

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------

    def _detect_phase(self, total_snake_cells: int) -> str:
        total = self.grid_width * self.grid_height
        fill = total_snake_cells / total
        if self.tick < 25 and fill < 0.08:
            return _PHASE_OPENING
        if fill > 0.30:
            return _PHASE_ENDGAME
        return _PHASE_MID

    def _phase_weights(self, phase: str, length_adv: int):
        """Return (w_food, w_space, w_aggression, w_voronoi, w_lookahead) tuned per phase."""
        if phase == _PHASE_OPENING:
            # Early: grow quickly, but not by suiciding into edge food.
            return (self.W_FOOD * 2.2, self.W_SPACE * 1.3, self.W_AGGRESSION * 0.0,
                    self.W_VORONOI * 0.5, self.W_LOOKAHEAD * 0.8)
        if phase == _PHASE_ENDGAME:
            # Late: survival > everything, huge space weight, minimal food
            return (self.W_FOOD * 0.3, self.W_SPACE * 3.0, self.W_AGGRESSION * 0.4,
                    self.W_VORONOI * 1.5, self.W_LOOKAHEAD * 1.8)
        # Mid-game: balanced; aggression only when clearly ahead
        agg_mult = 0.5 + max(0, length_adv - 2) * 0.15
        return (self.W_FOOD * 1.2, self.W_SPACE * 1.4, self.W_AGGRESSION * agg_mult,
                self.W_VORONOI, self.W_LOOKAHEAD)

    def _apply_profile(self, weights: tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
        w_food, w_space, w_aggression, w_voronoi, w_lookahead = weights
        return (
            w_food * self.profile.food_mult,
            w_space * self.profile.space_mult,
            w_aggression * self.profile.aggression_mult,
            w_voronoi,
            w_lookahead * self.profile.lookahead_mult,
        )

    def _update_direction_history(self, snakes: Dict) -> None:
        for sid, snake_data in snakes.items():
            if not snake_data.get("alive", True):
                continue
            if not snake_data.get("body"):
                continue
            current_direction = snake_data.get("direction", "right")
            previous_direction = self._last_direction_by_snake.get(sid)
            if previous_direction is None:
                self._direction_change_tick.setdefault(sid, -10_000)
            elif previous_direction != current_direction:
                self._direction_change_tick[sid] = self.tick
            self._last_direction_by_snake[sid] = current_direction

    def _compare_turn_priority(self, my_id: str, opp_id: str) -> int:
        """
        Compare simultaneous-crash tie-break priority.

        Returns 1 if we likely win an equal-length simultaneous crash,
        -1 if we likely lose it, 0 if unknown/even.
        """
        my_tick = self._direction_change_tick.get(my_id, -10_000)
        opp_tick = self._direction_change_tick.get(opp_id, -10_000)
        if my_tick < opp_tick:
            return 1
        if my_tick > opp_tick:
            return -1
        return 0

    def _opening_center_move(
        self,
        my_head: Tuple[int, int],
        my_tail: Tuple[int, int],
        my_direction: str,
        safe_moves: List[str],
        obstacles: Set[Tuple[int, int]],
        opponents: List[Dict],
    ) -> Optional[str]:
        """Prefer central, spacious, low-collision moves in the opening."""
        best_dir: Optional[str] = None
        best_score = float("-inf")
        board_center = (self.grid_width / 2, self.grid_height / 2)
        template = self._opening_template()
        opening_level = self._opening_level()
        center_scale = 10.0
        wall_scale = 34.0
        risk_scale = 20.0
        if template in {"l9", "l10"}:
            center_scale = 12.0
            wall_scale = 42.0
            risk_scale = 28.0
        elif opening_level is not None:
            center_scale = 11.0 + min(2.0, opening_level * 0.12)
            wall_scale = 36.0 + min(10.0, opening_level * 0.8)
            risk_scale = 22.0 + min(8.0, opening_level * 0.6)
        elif template == "recycle":
            center_scale = 11.0
            wall_scale = 38.0
            risk_scale = 24.0

        for direction in safe_moves:
            nx, ny = get_new_position(my_head[0], my_head[1], direction)
            next_pos = (nx, ny)
            space = calculate_space_after_move(
                my_head, direction, self.grid_width, self.grid_height, obstacles, my_tail,
            )
            center_bias = -manhattan_distance(next_pos, (int(board_center[0]), int(board_center[1])))
            wall_bias = wall_proximity(next_pos, self.grid_width, self.grid_height)
            risk_bias = 0.0
            for opp in opponents:
                if next_pos in opponent_reachable_in_one(
                    opp["head"], opp["direction"], self.grid_width, self.grid_height, obstacles,
                ):
                    risk_bias -= 1000.0
                risk_bias -= max(0, 4 - manhattan_distance(next_pos, opp["head"])) * risk_scale
            score = space * 8.0 + center_bias * center_scale + wall_bias * wall_scale + risk_bias
            if self.my_length <= 2 and wall_bias <= 1:
                score -= 160.0
                if template in {"l9", "l10", "recycle"}:
                    score -= 80.0
            self._append_trace(
                f"center:{direction}:score={score:.0f},space={space},wall={wall_bias},center={-center_bias}"
            )
            if direction == my_direction:
                score += 6.0
            if score > best_score:
                best_score = score
                best_dir = direction
        if best_dir is not None:
            self._append_trace(f"center_pick:{best_dir}:{best_score:.0f}")
        return best_dir

    def _select_opening_anchor(
        self,
        my_head: Tuple[int, int],
        opponents: List[Dict],
    ) -> Tuple[int, int]:
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        template = self._opening_template()
        opening_level = self._opening_level()
        if opening_level is not None:
            # CopperBot ladder bots heavily reward mirrored openings, so bias toward
            # a same-side flank box and increase the offset on higher levels.
            x_sign = 1 if my_head[0] >= center_x else -1
            y_sign = -1 if my_head[0] >= center_x else 1
            primary_x_offset = 2 + (1 if opening_level >= 8 else 0)
            secondary_x_offset = 1 + (1 if opening_level >= 10 else 0)
            primary_y_offset = 1 + (1 if opening_level >= 9 else 0)
            primary_lane_x = max(2, min(self.grid_width - 3, center_x + x_sign * primary_x_offset))
            secondary_lane_x = max(2, min(self.grid_width - 3, center_x + x_sign * secondary_x_offset))
            primary_lane_y = max(2, min(self.grid_height - 3, center_y + y_sign * primary_y_offset))
            secondary_lane_y = max(2, min(self.grid_height - 3, center_y + y_sign))
            anchors = [
                (primary_lane_x, primary_lane_y),
                (secondary_lane_x, primary_lane_y),
                (primary_lane_x, secondary_lane_y),
                (secondary_lane_x, secondary_lane_y),
            ]
        elif template == "recycle":
            x_sign = 1 if my_head[0] >= center_x else -1
            anchors = [
                (max(2, min(self.grid_width - 3, center_x + x_sign * 2)), center_y - 1),
                (max(2, min(self.grid_width - 3, center_x + x_sign * 2)), center_y),
                (max(2, min(self.grid_width - 3, center_x + x_sign)), center_y - 1),
                (max(2, min(self.grid_width - 3, center_x + x_sign)), center_y),
            ]
        else:
            anchors = [
                (center_x - 1, center_y - 1),
                (center_x, center_y - 1),
                (center_x - 1, center_y),
                (center_x, center_y),
            ]
        opp_head = opponents[0]["head"] if opponents else None

        def anchor_score(anchor: Tuple[int, int]) -> tuple[int, int]:
            my_dist = manhattan_distance(my_head, anchor)
            opp_dist = manhattan_distance(opp_head, anchor) if opp_head else 99
            return (opp_dist - my_dist, -my_dist)

        return max(anchors, key=anchor_score)

    def _opening_anchor_move(
        self,
        my_head: Tuple[int, int],
        my_tail: Tuple[int, int],
        my_direction: str,
        safe_moves: List[str],
        obstacles: Set[Tuple[int, int]],
        opponents: List[Dict],
    ) -> Optional[str]:
        opening_level = self._opening_level()
        if self._opening_anchor is None:
            self._opening_anchor = self._select_opening_anchor(my_head, opponents)
            self._append_trace(f"anchor:set={self._opening_anchor}")

        anchor = self._opening_anchor
        current_anchor_dist = manhattan_distance(my_head, anchor)
        min_box_x = max(1, anchor[0] - 1)
        max_box_x = min(self.grid_width - 2, anchor[0] + 1)
        min_box_y = max(1, anchor[1] - 1)
        max_box_y = min(self.grid_height - 2, anchor[1] + 1)
        best_dir: Optional[str] = None
        best_score = float("-inf")
        anchor_space_weight = 7.5
        anchor_distance_weight = 70.0
        anchor_wall_weight = 30.0
        anchor_box_bonus = 170.0
        anchor_box_escape_penalty = 240.0
        if opening_level is not None:
            anchor_space_weight += min(1.5, opening_level * 0.12)
            anchor_distance_weight += min(24.0, opening_level * 1.8)
            anchor_wall_weight += min(10.0, opening_level * 0.8)
            anchor_box_bonus += min(70.0, opening_level * 5.0)
            anchor_box_escape_penalty += min(120.0, opening_level * 8.0)
        for direction in safe_moves:
            nx, ny = get_new_position(my_head[0], my_head[1], direction)
            next_pos = (nx, ny)
            space = calculate_space_after_move(
                my_head, direction, self.grid_width, self.grid_height, obstacles, my_tail,
            )
            anchor_dist = manhattan_distance(next_pos, anchor)
            wall_bias = wall_proximity(next_pos, self.grid_width, self.grid_height)
            in_anchor_box = (
                min_box_x <= next_pos[0] <= max_box_x and
                min_box_y <= next_pos[1] <= max_box_y
            )
            opp_penalty = 0.0
            for opp in opponents:
                if next_pos in opponent_reachable_in_one(
                    opp["head"], opp["direction"], self.grid_width, self.grid_height, obstacles,
                ):
                    opp_penalty -= 1000.0
                opp_penalty -= max(0, 4 - manhattan_distance(next_pos, opp["head"])) * 22.0

            if current_anchor_dist <= 1 and in_anchor_box:
                opp_penalty = max(opp_penalty, -220.0)

            score = space * anchor_space_weight - anchor_dist * anchor_distance_weight + wall_bias * anchor_wall_weight + opp_penalty
            if next_pos == anchor:
                score += 120.0
            if current_anchor_dist <= 1:
                if in_anchor_box:
                    score += anchor_box_bonus
                else:
                    score -= anchor_box_escape_penalty
            if direction == my_direction:
                score += 8.0
            if self.my_length <= 2 and wall_bias <= 1:
                score -= 180.0
            self._append_trace(
                f"anchor:{direction}:score={score:.0f},dist={anchor_dist},box={int(in_anchor_box)},space={space},wall={wall_bias}"
            )
            if score > best_score:
                best_score = score
                best_dir = direction
        if best_dir is not None:
            self._append_trace(f"anchor_pick:{best_dir}:{best_score:.0f}")
        return best_dir

    def _prefer_inner_moves(
        self,
        my_head: Tuple[int, int],
        my_tail: Tuple[int, int],
        candidate_moves: List[str],
        obstacles: Set[Tuple[int, int]],
        food_set: Set[Tuple[int, int]],
    ) -> List[str]:
        """For short snakes, prefer moves that keep a two-cell wall buffer when possible."""
        if self.my_length > 2 or len(candidate_moves) <= 1:
            return candidate_moves

        inner_moves: List[str] = []
        buffered_moves: List[str] = []
        for direction in candidate_moves:
            nx, ny = get_new_position(my_head[0], my_head[1], direction)
            next_pos = (nx, ny)
            wp = wall_proximity(next_pos, self.grid_width, self.grid_height)
            if next_pos in food_set:
                buffered_moves.append(direction)
                continue
            if wp >= 2:
                inner_moves.append(direction)
            if wp >= 1:
                buffered_moves.append(direction)

        if inner_moves:
            self._append_trace(f"inner_filter:strict={','.join(inner_moves)}")
            return inner_moves
        if buffered_moves:
            self._append_trace(f"inner_filter:buffered={','.join(buffered_moves)}")
            return buffered_moves
        self._append_trace("inner_filter:none")
        return candidate_moves

    def _best_opening_food_move(
        self,
        my_id: str,
        my_head: Tuple[int, int],
        my_tail: Tuple[int, int],
        my_direction: str,
        safe_moves: List[str],
        obstacles: Set[Tuple[int, int]],
        opponents: List[Dict],
        food_positions: List[Tuple[int, int]],
    ) -> Optional[str]:
        """Pick an opening food line only when the race and board geometry are favorable."""
        best_dir: Optional[str] = None
        best_score = float("-inf")
        template = self._opening_template()
        opening_level = self._opening_level()
        max_distance = 5
        concede_penalty = 220.0
        tie_penalty = 120.0
        if template == "l10":
            max_distance = 3
            concede_penalty = 320.0
            tie_penalty = 180.0
        elif template == "l9":
            max_distance = 4
            concede_penalty = 280.0
            tie_penalty = 150.0
        elif opening_level is not None:
            max_distance = 4 if opening_level >= 6 else 5
            concede_penalty = 220.0 + min(70.0, opening_level * 10.0)
            tie_penalty = 120.0 + min(40.0, opening_level * 5.0)
        elif template == "recycle":
            max_distance = 4
            concede_penalty = 260.0
            tie_penalty = 150.0

        for food_pos in food_positions:
            path = bfs_path(my_head, food_pos, self.grid_width, self.grid_height, obstacles)
            if not path or len(path) < 2:
                continue

            next_cell = path[1]
            food_dir = direction_from_positions(my_head, next_cell)
            if not food_dir or food_dir not in safe_moves:
                continue

            distance = len(path) - 1
            if distance > max_distance:
                continue
            step_space = calculate_space_after_move(
                my_head, food_dir, self.grid_width, self.grid_height, obstacles, my_tail,
            )
            target_wall = wall_proximity(food_pos, self.grid_width, self.grid_height)
            step_wall = wall_proximity(next_cell, self.grid_width, self.grid_height)

            closest_opp = None
            closest_food_dist = float("inf")
            for opp in opponents:
                opp_food_dist = manhattan_distance(opp["head"], food_pos)
                if opp_food_dist < closest_food_dist:
                    closest_food_dist = opp_food_dist
                    closest_opp = opp

            race = "me"
            if closest_opp is not None:
                race = food_race_winner(
                    my_head, closest_opp["head"], food_pos,
                    self.grid_width, self.grid_height, obstacles,
                )

            score = 220.0 / (distance + 1)
            if distance == 1:
                score += 120.0
            if step_space < max(self.my_length + 6, 10):
                score -= 300.0
            elif step_space > max(self.my_length * 4, 12):
                score += 45.0

            if self.tick <= 12 and distance > 3:
                score -= 140.0
            if self.my_length <= 2 and distance > 4:
                score -= 220.0

            if target_wall == 0:
                score -= 90.0 if distance > 2 else 25.0
            elif target_wall == 1:
                score -= 25.0
            if step_wall == 0:
                score -= 110.0

            if race == "me":
                score += 120.0
            elif race == "tie":
                if closest_opp is not None and (
                    self.my_length > closest_opp["length"] or
                    self._compare_turn_priority(my_id, closest_opp["id"]) > 0
                ):
                    score += 30.0
                else:
                    score -= tie_penalty
            else:
                score -= concede_penalty

            if self.my_length <= 2 and race != "me" and distance > 1:
                score -= 180.0
            if self.my_length <= 2 and target_wall <= 1 and distance > 2:
                score -= 120.0

            if food_dir == my_direction:
                score += 10.0

            if score > best_score:
                best_score = score
                best_dir = food_dir
            self._append_trace(
                f"food:{food_dir}@{food_pos}:score={score:.0f},dist={distance},race={race},space={step_space}"
            )

        if best_dir and best_score > 0:
            self._append_trace(f"food_pick:{best_dir}:{best_score:.0f}")
            return best_dir
        self._append_trace("food_pick:none")
        return None

    def _food_candidate_score(
        self,
        new_pos: Tuple[int, int],
        food_pos: Tuple[int, int],
        lifetime: Optional[int],
        opponents: List[Dict],
        obstacles: Set[Tuple[int, int]],
        w_food: float,
    ) -> float:
        pd = bfs_distance(new_pos, food_pos, self.grid_width, self.grid_height, obstacles)
        if pd < 0:
            return float("-inf")
        if lifetime is not None and pd >= lifetime:
            return float("-inf")

        if new_pos == food_pos:
            score = w_food * 15
        else:
            best_opp = None
            best_opp_dist = float("inf")
            for opp in opponents:
                opp_dist = bfs_distance(opp["head"], food_pos, self.grid_width, self.grid_height, obstacles)
                if opp_dist < 0:
                    opp_dist = manhattan_distance(opp["head"], food_pos) + 2
                if opp_dist < best_opp_dist:
                    best_opp_dist = opp_dist
                    best_opp = opp

            race = food_race_winner(
                new_pos,
                best_opp["head"] if best_opp else None,
                food_pos,
                self.grid_width,
                self.grid_height,
                obstacles,
            )
            base = w_food * (1.0 / (pd + 1))
            score = 0.0
            if race == "me":
                score = base * 2.7
            elif race == "tie":
                if best_opp and self.my_length >= best_opp["length"]:
                    score = base * 1.5
                else:
                    score = base * 0.45
            else:
                score = base * 0.08
                if best_opp is not None and pd <= best_opp_dist + 1:
                    # Stay near strategically relevant food instead of wandering to dead targets.
                    score += self.W_FOOD_DENY / (pd + 1)

            if best_opp is not None:
                length_deficit = max(0, best_opp["length"] - self.my_length)
                if length_deficit > 0:
                    score *= 1.0 + min(0.22 * length_deficit, 0.9)

        target_wall = wall_proximity(food_pos, self.grid_width, self.grid_height)
        if self.my_length <= 2 and target_wall == 0 and pd > 2:
            score *= 0.55
        elif self.my_length <= 2 and target_wall == 1 and pd > 3:
            score *= 0.8

        center_bonus = max(0.0, 6.0 - center_distance(food_pos, self.grid_width, self.grid_height))
        score += center_bonus * 4.0
        return score

    def _nearest_food_for_head(
        self,
        head: Tuple[int, int],
        foods: List[Dict],
        obstacles: Set[Tuple[int, int]],
    ) -> tuple[Optional[Tuple[int, int]], int]:
        best_food: Optional[Tuple[int, int]] = None
        best_dist = 10_000
        for food in foods:
            food_pos = (food["x"], food["y"])
            dist = bfs_distance(head, food_pos, self.grid_width, self.grid_height, obstacles)
            if dist < 0:
                dist = manhattan_distance(head, food_pos) + 2
            if dist < best_dist:
                best_food = food_pos
                best_dist = dist
        return best_food, best_dist if best_food is not None else -1

    def _recovery_mode_score(
        self,
        my_head: Tuple[int, int],
        new_pos: Tuple[int, int],
        foods: List[Dict],
        opponents: List[Dict],
        obstacles: Set[Tuple[int, int]],
        phase: str,
    ) -> float:
        opening_level = self._opening_level()
        if phase == _PHASE_ENDGAME:
            return 0.0
        if not opponents or not foods:
            return 0.0

        longest_opp = max(opp["length"] for opp in opponents)
        length_deficit = longest_opp - self.my_length
        high_ladder = opening_level is not None and opening_level >= 8
        generic_recovery = length_deficit >= 4
        if not high_ladder and not generic_recovery:
            return 0.0
        if length_deficit < 2:
            return 0.0

        best_food, current_food_dist = self._nearest_food_for_head(my_head, foods, obstacles)
        if best_food is None or current_food_dist < 0:
            return 0.0

        next_food_dist = bfs_distance(new_pos, best_food, self.grid_width, self.grid_height, obstacles)
        if next_food_dist < 0:
            next_food_dist = manhattan_distance(new_pos, best_food) + 2

        score = 0.0
        dist_delta = current_food_dist - next_food_dist
        if dist_delta > 0:
            score += dist_delta * self.W_RECOVERY
        elif dist_delta < 0:
            score += dist_delta * (self.W_RECOVERY * 0.75)

        current_wall = wall_proximity(my_head, self.grid_width, self.grid_height)
        next_wall = wall_proximity(new_pos, self.grid_width, self.grid_height)
        nearest_food_dist_now = current_food_dist
        if current_wall <= 1 and next_wall > current_wall:
            score += 160.0
        if next_wall <= 1 and next_food_dist > 2:
            score -= 150.0 + (20.0 * min(3, length_deficit))
        elif next_wall == 2 and next_food_dist > 4:
            score -= 55.0

        if self.my_length <= 4 and current_wall <= 1 and next_wall <= current_wall and dist_delta <= 0:
            score -= 220.0
        elif self.my_length <= 4 and current_wall <= 1 and next_wall > current_wall:
            score += 90.0

        if center_distance(new_pos, self.grid_width, self.grid_height) < center_distance(my_head, self.grid_width, self.grid_height):
            score += 35.0

        for opp in opponents:
            opp_food_dist = bfs_distance(opp["head"], best_food, self.grid_width, self.grid_height, obstacles)
            if opp_food_dist < 0:
                opp_food_dist = manhattan_distance(opp["head"], best_food) + 2
            if next_food_dist <= opp_food_dist:
                score += 90.0
            elif next_food_dist <= opp_food_dist + 1:
                score += 35.0

        return score

    def _opening_steal_or_concede_move(
        self,
        my_id: str,
        my_head: Tuple[int, int],
        my_tail: Tuple[int, int],
        my_direction: str,
        safe_moves: List[str],
        obstacles: Set[Tuple[int, int]],
        opponents: List[Dict],
        food_positions: List[Tuple[int, int]],
    ) -> Optional[str]:
        """In the first few ticks, either take a clearly favorable food race or concede it immediately."""
        if self.tick > 10 or self.my_length > 2:
            return None

        template = self._opening_template()
        opening_level = self._opening_level()
        max_distance = 5
        if template == "l10":
            max_distance = 3
        elif template in {"l9", "recycle"}:
            max_distance = 4
        elif opening_level is not None and opening_level >= 6:
            max_distance = 4

        best_dir: Optional[str] = None
        best_score = float("-inf")
        decisive_found = False
        opponent_fast_food = False

        for food_pos in food_positions:
            path = bfs_path(my_head, food_pos, self.grid_width, self.grid_height, obstacles)
            if not path or len(path) < 2:
                continue

            distance = len(path) - 1
            if distance > max_distance:
                continue

            next_cell = path[1]
            food_dir = direction_from_positions(my_head, next_cell)
            if not food_dir or food_dir not in safe_moves:
                continue

            step_space = calculate_space_after_move(
                my_head, food_dir, self.grid_width, self.grid_height, obstacles, my_tail,
            )
            if step_space < max(self.my_length + 6, 10):
                continue

            closest_opp = None
            closest_opp_dist = 10_000
            for opp in opponents:
                opp_dist = bfs_distance(opp["head"], food_pos, self.grid_width, self.grid_height, obstacles)
                if opp_dist < 0:
                    opp_dist = manhattan_distance(opp["head"], food_pos) + 2
                if opp_dist < closest_opp_dist:
                    closest_opp_dist = opp_dist
                    closest_opp = opp

            if closest_opp_dist <= 2:
                opponent_fast_food = True

            race = "me"
            favorable_tie = False
            if closest_opp is not None:
                race = food_race_winner(
                    my_head, closest_opp["head"], food_pos,
                    self.grid_width, self.grid_height, obstacles,
                )
                favorable_tie = (
                    race == "tie" and (
                        self.my_length > closest_opp["length"] or
                        self._compare_turn_priority(my_id, closest_opp["id"]) > 0
                    )
                )

            if race == "me" or favorable_tie:
                decisive_found = True
                score = 320.0 / (distance + 1)
                if distance <= 2:
                    score += 120.0
                if food_dir == my_direction:
                    score += 15.0
                score += step_space * 1.2
                if score > best_score:
                    best_score = score
                    best_dir = food_dir
            self._append_trace(
                f"steal:{food_dir}@{food_pos}:dist={distance},opp={closest_opp_dist},race={race},tie={int(favorable_tie)},space={step_space}"
            )

        if decisive_found and best_dir is not None:
            self._append_trace(f"steal_pick:{best_dir}:{best_score:.0f}")
            return best_dir

        if opponent_fast_food:
            if self._should_use_opening_anchor():
                self._append_trace("steal_pick:concede_to_anchor")
                return self._opening_anchor_move(
                    my_head, my_tail, my_direction, safe_moves, obstacles, opponents,
                )
            self._append_trace("steal_pick:concede_to_center")
            return self._opening_center_move(
                my_head, my_tail, my_direction, safe_moves, obstacles, opponents,
            )

        self._append_trace("steal_pick:none")
        return None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def calculate_move(self, game_state: Dict, player_id: int) -> Optional[str]:
        if not game_state:
            return None

        self._reset_decision_trace()
        self.tick += 1
        snakes = game_state.get("snakes", {})
        foods = game_state.get("foods", [])
        grid = game_state.get("grid", {})
        self.grid_width = grid.get("width", 30)
        self.grid_height = grid.get("height", 20)
        self._update_direction_history(snakes)

        my_snake = snakes.get(str(player_id))
        if not my_snake or not my_snake.get("body") or not my_snake.get("alive", True):
            return None

        my_body = parse_snake_body(my_snake["body"])
        my_head = my_body[0]
        my_tail = my_body[-1] if len(my_body) > 1 else my_head
        my_direction = my_snake.get("direction", "right")
        self.my_length = len(my_body)
        self._remember_head(my_head)

        # ---- opponents ----
        opponents = get_all_opponents(snakes, player_id)
        for opp in opponents:
            self.opponent_lengths[opp["id"]] = opp["length"]

        # Detect which opponents just ate (tail didn't move)
        self._opponent_ate.clear()
        for opp in opponents:
            prev_tail = self._prev_opponent_tails.get(opp["id"])
            if prev_tail is not None and opp["tail"] == prev_tail:
                self._opponent_ate.add(opp["id"])
            self._prev_opponent_tails[opp["id"]] = opp["tail"]

        # ---- obstacles ----
        # get_all_snake_positions with exclude_tail=True already includes
        # all body segments except tails that will move.  Do NOT re-add
        # our own tail — it must remain free so tail-chasing works.
        obstacles = get_all_snake_positions(snakes, exclude_tail=True, ate_food_ids=self._opponent_ate)

        food_positions = get_food_positions(foods)
        food_set = set(food_positions)

        # Will WE eat on the NEXT move? (for tail-stays-awareness)
        # We don't know yet which direction we'll pick, so we check per-move inside _score_move.

        # ---- predict opponents ----
        self._predicted_positions.clear()
        for opp in opponents:
            pred = predict_opponent_move(
                opp["head"], opp["direction"], opp["body"],
                my_head, food_positions,
                self.grid_width, self.grid_height, obstacles,
            )
            self._predicted_positions[opp["id"]] = pred

        # ---- safe moves ----
        safe_moves = get_safe_moves(my_head, my_direction, self.grid_width, self.grid_height, obstacles)

        if not safe_moves:
            # Last resort: try chasing our own tail
            td = find_tail_chase_path(my_head, my_tail, self.grid_width, self.grid_height, obstacles)
            if td:
                return td
            return self._desperation_move(my_head, my_body, my_direction, obstacles)

        if len(safe_moves) == 1:
            # Only one option — but verify it doesn't trap us
            only_dir = safe_moves[0]
            space = calculate_space_after_move(
                my_head, only_dir, self.grid_width, self.grid_height,
                obstacles, my_tail,
            )
            # If trapped with the only option, try tail chasing instead
            if space < self.my_length and space > 0:
                td = find_tail_chase_path(my_head, my_tail, self.grid_width, self.grid_height, obstacles)
                if td and td != only_dir:
                    # Check if tail-chase direction is actually safe
                    tnx, tny = get_new_position(my_head[0], my_head[1], td)
                    if is_in_bounds(tnx, tny, self.grid_width, self.grid_height) and (tnx, tny) not in obstacles:
                        ts = calculate_space_after_move(
                            my_head, td, self.grid_width, self.grid_height,
                            obstacles, my_tail,
                        )
                        if ts > space:
                            return td
            return only_dir

        # ---- phase & weights ----
        total_snake_cells = sum(opp["length"] for opp in opponents) + self.my_length
        phase = self._detect_phase(total_snake_cells)
        primary_opp = opponents[0] if opponents else None
        length_adv = self.my_length - (primary_opp["length"] if primary_opp else self.my_length)
        pw = self._apply_profile(self._phase_weights(phase, length_adv))

        # ---- Dynamic minimum safe space ----
        min_safe = max(self.my_length + 2, self.MIN_SAFE_SPACE)

        # ---- SMALL-SNAKE FOOD RUSH ----
        # When small, the #1 priority is EATING. BFS directly to food.
        if self.my_length <= 3 and food_positions:
            opening_decision = self._opening_steal_or_concede_move(
                str(player_id), my_head, my_tail, my_direction,
                safe_moves, obstacles, opponents, food_positions,
            )
            if opening_decision:
                self._append_trace(f"opening_choice:{opening_decision}:steal_or_concede")
                return opening_decision

            best_food_dir = self._best_opening_food_move(
                str(player_id), my_head, my_tail, my_direction,
                safe_moves, obstacles, opponents, food_positions,
            )

            if best_food_dir:
                self._append_trace(f"opening_choice:{best_food_dir}:food")
                return best_food_dir
            if phase == _PHASE_OPENING:
                if self._should_use_opening_anchor():
                    anchor_dir = self._opening_anchor_move(
                        my_head, my_tail, my_direction, safe_moves, obstacles, opponents,
                    )
                    if anchor_dir:
                        self._append_trace(f"opening_choice:{anchor_dir}:anchor")
                        return anchor_dir
                center_dir = self._opening_center_move(
                    my_head, my_tail, my_direction, safe_moves, obstacles, opponents,
                )
                if center_dir:
                    self._append_trace(f"opening_choice:{center_dir}:center")
                    return center_dir
            # Fall through to normal scoring if no safe food path found

        # ---- pre-filter lethal moves ----
        # Eliminate moves with less space than our body length (guaranteed death)
        move_spaces = {}
        for d in safe_moves:
            sp = calculate_space_after_move(
                my_head, d, self.grid_width, self.grid_height,
                obstacles, my_tail,
            )
            move_spaces[d] = sp

        viable_moves = [d for d in safe_moves if move_spaces[d] >= self.my_length]
        if not viable_moves:
            # All moves are tight — pick the one with most space (pure survival)
            best_d = max(safe_moves, key=lambda d: move_spaces[d])
            return best_d

        viable_moves = self._prefer_inner_moves(
            my_head, my_tail, viable_moves, obstacles, food_set,
        )

        # ---- Hard-filter head-collision risk ----
        # Remove moves where opponent head could reach same square
        # ONLY when we can't win the collision (equal or shorter)
        if len(viable_moves) > 1 and opponents:
            non_collision = []
            for d in viable_moves:
                nx2, ny2 = get_new_position(my_head[0], my_head[1], d)
                pos2 = (nx2, ny2)
                fatal = False
                for opp in opponents:
                    if self.my_length > opp["length"]:
                        continue  # we'd win this collision
                    opp_reach = opponent_reachable_in_one(
                        opp["head"], opp["direction"],
                        self.grid_width, self.grid_height, obstacles,
                    )
                    if pos2 in opp_reach:
                        if self.my_length == opp["length"] and self._compare_turn_priority(str(player_id), opp["id"]) > 0:
                            continue
                        fatal = True
                        break
                if not fatal:
                    non_collision.append(d)
            if non_collision:
                viable_moves = non_collision

        # If in extreme survival situation, just pick max space
        # but ALSO check for head collisions
        if all(move_spaces[d] < self.my_length * 2 for d in viable_moves):
            td = find_tail_chase_path(my_head, my_tail, self.grid_width, self.grid_height, obstacles)
            if td and td in viable_moves:
                return td
            return max(viable_moves, key=lambda d: move_spaces[d])

        # ---- score each viable move ----
        scores: List[MoveScore] = []
        for d in viable_moves:
            ms = self._score_move(
                d, my_head, my_body, my_tail, my_direction,
                str(player_id), opponents, foods, food_set, obstacles, phase, pw,
            )
            scores.append(ms)

        scores.sort(key=lambda s: s.total, reverse=True)

        # Log decision for debugging
        import logging
        _log = logging.getLogger("strategy")
        if _log.isEnabledFor(logging.DEBUG):
            for s in scores:
                _log.debug(
                    "T%d %s: total=%.0f safe=%.0f food=%.0f space=%.0f agg=%.0f wall=%.0f split=%.0f",
                    self.tick, s.direction, s.total, s.safety, s.food,
                    s.space, s.aggression, s.wall, s.split,
                )

        return scores[0].direction

    # ------------------------------------------------------------------
    # Per-move scoring
    # ------------------------------------------------------------------

    def _score_move(
        self,
        direction: str,
        my_head: Tuple[int, int],
        my_body: List[Tuple[int, int]],
        my_tail: Tuple[int, int],
        my_direction: str,
        my_id: str,
        opponents: List[Dict],
        foods: List[Dict],
        food_set: Set[Tuple[int, int]],
        obstacles: Set[Tuple[int, int]],
        phase: str,
        pw: tuple,
    ) -> MoveScore:
        w_food, w_space, w_aggression, w_voronoi, w_lookahead = pw

        nx, ny = get_new_position(my_head[0], my_head[1], direction)
        new_pos = (nx, ny)

        safety    = float(self.W_SAFETY)
        food_s    = 0.0
        space_s   = 0.0
        agg_s     = 0.0
        corr_s    = 0.0
        vor_s     = 0.0
        center_s  = 0.0
        look_s    = 0.0
        wall_s    = 0.0
        split_s   = 0.0
        straight_s = 0.0
        reachable_foods: List[Dict] = [
            food for food in foods
            if food.get("type") != "grapes" and (food.get("lifetime") is None or food["lifetime"] > 1)
        ]
        loop_s = 0.0

        adapted_wall_weight = self.W_WALL * self.profile.wall_mult
        adapted_center_weight = self.W_CENTER * self.profile.center_mult

        will_eat = new_pos in food_set

        # === STRAIGHT-LINE BONUS (reduces erratic movement) ===
        if direction == my_direction:
            straight_s = self.W_STRAIGHT

        # === SAFETY: head-collision with EVERY opponent ===
        for opp in opponents:
            opp_head = opp["head"]
            dist = manhattan_distance(new_pos, opp_head)

            # Check all squares opponent can reach in 1 tick
            opp_reach = opponent_reachable_in_one(
                opp_head, opp["direction"],
                self.grid_width, self.grid_height, obstacles,
            )
            if new_pos in opp_reach:
                if self.my_length > opp["length"]:
                    # Official rules: longest snake wins simultaneous crash.
                    safety += self.W_SAFETY * 0.35
                    agg_s += w_aggression * 1.8
                elif self.my_length == opp["length"]:
                    turn_edge = self._compare_turn_priority(my_id, opp["id"])
                    if turn_edge > 0:
                        safety += self.W_SAFETY * 0.08
                        agg_s += w_aggression * 0.6
                    elif turn_edge < 0:
                        safety -= self.W_SAFETY * 2.2
                    else:
                        safety -= self.W_SAFETY * 1.4
                else:
                    # We lose = guaranteed death
                    safety -= self.W_SAFETY * 2.5

            # General proximity avoidance for equal/shorter opponents
            if self.my_length <= opp["length"] and dist <= 2:
                safety -= self.W_SAFETY * 0.5
            elif self.my_length <= opp["length"] and dist <= 3:
                safety -= self.W_SAFETY * 0.15

            # Predicted position collision
            pred = self._predicted_positions.get(opp["id"])
            if pred and new_pos == pred:
                if self.my_length > opp["length"]:
                    agg_s += w_aggression * 1.0
                elif self.my_length <= opp["length"]:
                    safety -= self.W_SAFETY * 1.5

        # === SPACE (with growing awareness) ===
        space_after = calculate_space_after_move(
            my_head, direction,
            self.grid_width, self.grid_height,
            obstacles, my_tail,
            tail_will_stay=will_eat,
        )
        total_cells = self.grid_width * self.grid_height
        space_ratio = space_after / total_cells
        space_s = space_ratio * w_space * 10

        # Dynamic min safe space based on snake length
        dynamic_min = max(self.my_length + 2, self.MIN_SAFE_SPACE)

        if space_after < self.my_length:
            # Fatal: less space than our body length
            safety -= self.W_SAFETY * 0.85
            space_s -= w_space * 20
        elif space_after < dynamic_min:
            safety -= self.W_SAFETY * 0.5
            space_s -= w_space * 10
        elif space_after < self.my_length + 5:
            safety -= self.W_SAFETY * 0.2
            space_s -= w_space * 2
        elif space_after < self.my_length * 2:
            space_s -= w_space * 1.0

        # === TAIL REACHABILITY CHECK ===
        # After this move, can we still reach our own tail?
        # If not, we risk trapping ourselves.
        if len(my_body) > 3 and space_after < total_cells * 0.5:
            new_obs_for_tail = set(obstacles)
            new_obs_for_tail.add(my_head)  # head becomes body
            if not will_eat and my_tail in new_obs_for_tail:
                new_obs_for_tail.discard(my_tail)
            tail_target = my_body[-2] if will_eat else my_tail
            tail_dist = bfs_distance(
                new_pos, tail_target,
                self.grid_width, self.grid_height, new_obs_for_tail,
            )
            if tail_dist < 0:
                # Can't reach tail — big danger of trapping ourselves
                safety -= self.W_TAIL_REACH

        # === FOOD-TRAP DETECTION ===
        # If eating food here gives us less space than our length, skip it
        if will_eat:
            space_after_eating = calculate_space_after_move(
                my_head, direction,
                self.grid_width, self.grid_height,
                obstacles, my_tail,
                tail_will_stay=True,
            )
            if space_after_eating < self.my_length + 3:
                food_s -= self.W_FOOD_TRAP

        # === BOARD-SPLIT DETECTION ===
        if move_splits_board(my_head, direction, self.grid_width, self.grid_height, obstacles, my_tail):
            # We end up in the smaller component — very dangerous
            free_cells = total_cells - len(obstacles)
            if space_after < free_cells * 0.4:
                split_s -= self.W_SPLIT
            elif space_after < free_cells * 0.6:
                split_s -= self.W_SPLIT * 0.3

        # === VORONOI (skip in opening for speed) ===
        if opponents and phase != _PHASE_OPENING:
            if len(opponents) == 1:
                my_cells, opp_cells, _ = calculate_voronoi_control(
                    new_pos, opponents[0]["head"],
                    self.grid_width, self.grid_height,
                    obstacles | {my_head},
                )
            else:
                opp_heads = [o["head"] for o in opponents]
                my_cells, opp_cells = calculate_voronoi_multi(
                    new_pos, opp_heads,
                    self.grid_width, self.grid_height,
                    obstacles | {my_head},
                )

            territory_adv = my_cells - opp_cells
            vor_s = territory_adv * w_voronoi / 50
            if my_cells > opp_cells * 1.5:
                vor_s += w_voronoi

        # === FOOD ===
        if foods:
            grapes = [f for f in foods if f.get("type") == "grapes"]

            # Grapes scoring
            if grapes:
                grape = grapes[0]
                gp = (grape["x"], grape["y"])
                gl = grape.get("lifetime")
                closest_opp_head = None
                closest_opp_dist = float("inf")
                for opp in opponents:
                    d = manhattan_distance(opp["head"], gp)
                    if d < closest_opp_dist:
                        closest_opp_dist = d
                        closest_opp_head = opp["head"]
                if new_pos == gp:
                    food_s = self.W_GRAPES
                else:
                    gd = bfs_distance(new_pos, gp, self.grid_width, self.grid_height, obstacles)
                    reachable = gd > 0 and (gl is None or gd < gl)
                    if reachable:
                        race = food_race_winner(
                            new_pos, closest_opp_head, gp,
                            self.grid_width, self.grid_height, obstacles,
                        )
                        if race == "me":
                            food_s = self.W_GRAPES * (1.0 / (gd + 1))
                        elif race == "tie":
                            food_s = self.W_GRAPES * 0.3 / (gd + 1)

            # Regular food
            reachable_foods = [f for f in foods if f.get("lifetime") is None or f["lifetime"] > 1]
            food_candidates: List[Tuple[int, Optional[int], Tuple[int, int]]] = []
            for food in reachable_foods:
                food_pos = (food["x"], food["y"])
                dist = bfs_distance(new_pos, food_pos, self.grid_width, self.grid_height, obstacles)
                if dist < 0:
                    continue
                food_candidates.append((dist, food.get("lifetime"), food_pos))

            food_candidates.sort(key=lambda item: item[0])
            for _, lifetime, food_pos in food_candidates[:4]:
                candidate_score = self._food_candidate_score(
                    new_pos, food_pos, lifetime, opponents, obstacles, w_food,
                )
                if candidate_score > food_s:
                    food_s = candidate_score

                # Length urgency
                if opponents:
                    min_opp_len = min(o["length"] for o in opponents)
                    if self.my_length < min_opp_len:
                        deficit = min_opp_len - self.my_length
                        food_s *= 1.4 + min(deficit * 0.25, 1.0)
                    elif self.my_length > min_opp_len + 5:
                        food_s *= 0.45

        anti_snowball = bool(opponents) and self.my_length <= 2 and min(o["length"] for o in opponents) > self.my_length
        recovery_s = self._recovery_mode_score(
            my_head, new_pos, reachable_foods, opponents, obstacles, phase,
        )

        # === CORRIDOR ===
        corr_val = get_corridor_score(new_pos, self.grid_width, self.grid_height, obstacles)
        if corr_val <= 1:
            corr_s = -self.W_CORRIDOR * 5
        elif corr_val == 2:
            corr_s = -self.W_CORRIDOR * 2
        else:
            corr_s = self.W_CORRIDOR * (corr_val - 2)

        # === WALL PROXIMITY ===
        wp = wall_proximity(new_pos, self.grid_width, self.grid_height)
        if wp == 0:
            wall_s = -adapted_wall_weight * 5
            # Extra penalty for corners (two walls at once)
            if (new_pos[0] == 0 or new_pos[0] == self.grid_width - 1) and \
               (new_pos[1] == 0 or new_pos[1] == self.grid_height - 1):
                wall_s -= adapted_wall_weight * 5
        elif wp == 1:
            wall_s = -adapted_wall_weight * 2
        elif wp == 2:
            wall_s = -adapted_wall_weight * 0.5
        else:
            wall_s = adapted_wall_weight * 0.4

        if self.my_length <= 2 and not will_eat:
            if wp == 0:
                wall_s -= adapted_wall_weight * 4
            elif wp == 1:
                wall_s -= adapted_wall_weight * 2.5
            elif wp == 2:
                wall_s -= adapted_wall_weight * 0.8

        if reachable_foods:
            nearest_food_now = self._nearest_food_for_head(my_head, reachable_foods, obstacles)[1]
            _, nearest_food_next = self._nearest_food_for_head(new_pos, reachable_foods, obstacles)
            current_wall_state = wall_proximity(my_head, self.grid_width, self.grid_height)
            longest_opp_len = max((opp["length"] for opp in opponents), default=self.my_length)
            if self.my_length <= 5 and phase != _PHASE_OPENING:
                if wp <= 1 and nearest_food_next > 2:
                    wall_s -= adapted_wall_weight * 1.6
                if current_wall_state <= 1 and wp <= current_wall_state and nearest_food_next >= nearest_food_now:
                    wall_s -= adapted_wall_weight * 2.1
                    center_s -= adapted_center_weight * 0.8
                if current_wall_state <= 1 and wp > current_wall_state and nearest_food_next <= nearest_food_now + 1:
                    center_s += adapted_center_weight * 1.2
                    wall_s += adapted_wall_weight * 0.5
            if phase != _PHASE_OPENING:
                if wp <= 1 and nearest_food_next > 5:
                    wall_s -= adapted_wall_weight * 1.1
                if longest_opp_len >= self.my_length and wp <= 1 and nearest_food_next >= nearest_food_now:
                    wall_s -= adapted_wall_weight * 1.4
                if longest_opp_len > self.my_length and current_wall_state <= 1 and wp <= current_wall_state:
                    center_s -= adapted_center_weight * 0.9
                if longest_opp_len > self.my_length and current_wall_state <= 1 and wp > current_wall_state:
                    center_s += adapted_center_weight * 0.8

        # === CENTER ===
        cd = center_distance(new_pos, self.grid_width, self.grid_height)
        max_cd = (self.grid_width + self.grid_height) / 2
        center_s = adapted_center_weight * (1 - cd / max_cd)

        if self.my_length <= 2:
            center_s += adapted_center_weight * 0.9 * (1 - cd / max_cd)

        if anti_snowball:
            center_s += adapted_center_weight * 0.8 * (1 - cd / max_cd)
            if wp <= 1:
                wall_s -= adapted_wall_weight * 1.2
            if space_after > self.my_length * 3:
                space_s += w_space * 0.4

        if recovery_s:
            food_s += recovery_s * 0.65
            center_s += recovery_s * 0.12
            wall_s += recovery_s * 0.08

        # === ANTI-LOOP ===
        # Prefer fresh territory when multiple safe options exist; repeated cycles in open
        # space are a common way to lose initiative and food races.
        recent_window = self._recent_heads[-10:]
        visit_count = recent_window.count(new_pos)
        if visit_count > 0:
            loop_s -= 70.0 * visit_count
        if len(self._recent_heads) >= 3 and new_pos == self._recent_heads[-2] and not will_eat:
            loop_s -= 160.0
        if len(self._recent_heads) >= 5 and new_pos == self._recent_heads[-4] and not will_eat:
            loop_s -= 110.0
        if len(self._recent_heads) >= 9 and new_pos in self._recent_heads[-8:-2] and not will_eat:
            loop_s -= 55.0
        if self.my_length <= 5 and visit_count > 0 and wp <= 1:
            loop_s -= 95.0 * visit_count
        if len(recent_window) >= 8 and not will_eat:
            xs = [pos[0] for pos in recent_window]
            ys = [pos[1] for pos in recent_window]
            recent_box_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
            repeated_cells = len(recent_window) - len(set(recent_window))
            nearest_food_from_new = -1
            if reachable_foods:
                _, nearest_food_from_new = self._nearest_food_for_head(new_pos, reachable_foods, obstacles)
            if phase != _PHASE_OPENING and recent_box_area <= 25 and repeated_cells >= 2:
                loop_s -= 140.0 + repeated_cells * 35.0
                if nearest_food_from_new >= 5:
                    loop_s -= 120.0
                if new_pos in recent_window[-6:]:
                    loop_s -= 80.0
        if space_after <= self.my_length * 2:
            loop_s *= 0.35
        elif will_eat:
            loop_s *= 0.2

        # === LOOKAHEAD ===
        # Depth 2 for speed; only go deeper in truly cramped situations
        la_depth = 2
        if space_after < self.my_length * 3:
            la_depth = 3
        la = lookahead_evaluate(
            my_head, direction, my_body,
            opponents[0]["head"] if opponents else None,
            opponents[0]["body"] if opponents else None,
            self.grid_width, self.grid_height,
            obstacles, depth=la_depth, foods=food_set,
        )
        if la > 0:
            look_s = min(la, 200) * w_lookahead / 200
        elif la == float("-inf"):
            safety -= self.W_SAFETY * 0.5

        # === AGGRESSION ===
        for opp in opponents:
            oh = opp["head"]
            od = opp["direction"]
            dist = manhattan_distance(new_pos, oh)

            if self.my_length > opp["length"] + 2:
                # Clearly dominant — apply pressure intelligently
                if 3 <= dist <= 5:
                    agg_s += w_aggression * 0.8
                # Corner opponent only if we have plenty of safe space
                if space_after > self.my_length * 4:
                    opp_space = flood_fill_count(
                        oh, self.grid_width, self.grid_height,
                        obstacles, max_depth=10,
                    )
                    if opp_space < 10:
                        agg_s += w_aggression * 1.5
            elif self.my_length > opp["length"]:
                # Slightly longer — light pressure only from safe distance
                if 4 <= dist <= 6:
                    agg_s += w_aggression * 0.3
            else:
                # Equal or shorter — STAY AWAY (critical for survival)
                if dist < 4:
                    agg_s -= w_aggression * 2.0
                if dist < 6:
                    danger = find_opponent_danger_zone(
                        oh, od, self.grid_width, self.grid_height, radius=3,
                    )
                    if new_pos in danger:
                        agg_s -= w_aggression * 2.5

            # When they are only slightly ahead, contest their next food instead of only chasing ours.
            length_gap = opp["length"] - self.my_length
            if 1 <= length_gap <= 2 and reachable_foods:
                opp_food, opp_food_dist = self._nearest_food_for_head(oh, reachable_foods, obstacles)
                if opp_food is not None and opp_food_dist >= 0:
                    my_food_dist = bfs_distance(new_pos, opp_food, self.grid_width, self.grid_height, obstacles)
                    midpoint = ((oh[0] + opp_food[0]) // 2, (oh[1] + opp_food[1]) // 2)
                    mid_dist = manhattan_distance(new_pos, midpoint)
                    current_mid_dist = manhattan_distance(my_head, midpoint)

                    if my_food_dist >= 0 and space_after > self.my_length + 4:
                        if my_food_dist <= opp_food_dist:
                            agg_s += self.W_INTERCEPT * 1.2 / (my_food_dist + 1)
                        elif my_food_dist <= opp_food_dist + 1:
                            agg_s += self.W_INTERCEPT * 0.7 / (my_food_dist + 1)
                        elif my_food_dist <= opp_food_dist + 3:
                            agg_s += self.W_INTERCEPT * 0.25 / (my_food_dist + 1)

                    if mid_dist < current_mid_dist and mid_dist <= 4 and wp > 0:
                        agg_s += self.W_INTERCEPT * 0.45 / (mid_dist + 1)

        # === TOTAL ===
        total = (safety + food_s + space_s + agg_s + corr_s +
             vor_s + center_s + look_s + wall_s + split_s + straight_s + loop_s)

        return MoveScore(
            direction=direction, total=total,
            safety=safety, food=food_s, space=space_s,
            aggression=agg_s, corridor=corr_s, voronoi=vor_s,
            center=center_s, lookahead=look_s, wall=wall_s, split=split_s,
        )

    # ------------------------------------------------------------------
    # Desperation fallback
    # ------------------------------------------------------------------

    def _desperation_move(
        self,
        head: Tuple[int, int],
        body: List[Tuple[int, int]],
        current_direction: str,
        obstacles: Set[Tuple[int, int]],
    ) -> Optional[str]:
        """Last-resort move: try every direction including reverse, pick most space."""
        best_dir = None
        best_space = -1
        # In desperation, even try the reverse direction (server may allow
        # it for snakes of length 1, and it's better than guaranteed death)
        for d in ALL_DIRECTIONS:
            nx, ny = get_new_position(head[0], head[1], d)
            if not is_in_bounds(nx, ny, self.grid_width, self.grid_height):
                continue
            if (nx, ny) in obstacles:
                # Even occupied cells might free up (opponent tail)
                continue
            space = flood_fill_count(
                (nx, ny), self.grid_width, self.grid_height,
                obstacles, max_depth=20,
            )
            if space > best_space:
                best_space = space
                best_dir = d
        return best_dir if best_dir else current_direction

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def get_debug_info(self, game_state: Dict, player_id: int) -> Dict:
        snakes = game_state.get("snakes", {})
        my_snake = snakes.get(str(player_id), {})
        return {
            "grid_size": f"{self.grid_width}x{self.grid_height}",
            "my_length": self.my_length,
            "opponent_lengths": dict(self.opponent_lengths),
            "games_played": self.games_played,
            "wins": self.wins,
            "tick": self.tick,
            "my_buff": my_snake.get("buff", "none"),
            "opponents_ate": list(self._opponent_ate),
            "adaptive_profile": self.get_adaptation_summary(),
        }

import math
import numpy as np
from typing import Any, List
from rlgym_compat import common_values
from rlgym_compat import PlayerData, GameState, PhysicsObject


class ObservationState:
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        pass
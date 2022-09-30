from bot import Agent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
from agent import Agent
from training.observation import ObservationState
from rlgym_compat import GameState


class Bot(Agent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        # TODO Write custom Obs
        self.obs_builder = ObservationState()
        
        self.agent = Agent()
        # Adjust the tickskip if your agent was trained with a different value
        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.ticks = 0
        self.prev_time = 0
        self.observed = False
        self.acted = False
        self.expected_teammates = 0
        self.expected_opponents = 1
        self.current_obs = None
        print(f'{self.name} Ready - Index:', index)


    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # Take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.tick_multi = 120


    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
        
if __name__ == "__main__":
    pass
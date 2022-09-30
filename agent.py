import numpy as np
from stable_baselines3 import PPO
import pathlib
from parsers.continuous_act import ContinuousAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = PPO.load(str(_path) + '/exit_save.zip', device='cpu', custom_objects=custom_objects)
        self.parser = ContinuousAction()


    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]

if __name__ == "__main__":
    pass
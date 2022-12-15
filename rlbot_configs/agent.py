from stable_baselines3 import PPO
import pathlib
from rlgym.utils.action_parsers.discrete_act import DiscreteAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.00005,
            "clip_range": .02,
            "n_envs": 1,
            "device": "cuda"
        }

        model_to_load = '[insert_model]'
        
        self.actor = PPO.load(str(_path) + '/' + model_to_load + '.zip', device='cuda', custom_objects=custom_objects)
        self.parser = DiscreteAction()


    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]

if __name__ == "__main__":
    pass
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from training.observation import ObservationState
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv


from training.reward import RewardFunction

from training.discrete_act import DiscreteAction

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Discount

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds)) # Discount reward factor
    
    target_steps = 100_000 ## how many training steps per epoch of training
    agents_per_match = 2 ## Will vary based off of what gamemode the agent is trained for
    num_instances = 1
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps



    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  
            tick_skip=frame_skip,
            reward_function=RewardFunction(),  ## TODO Add new reward functions  
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=ObservationState(),  # TODO Write custom Obs
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Number of training instances
    env = VecCheckNan(env)                                # Check for NaNs
    env = VecMonitor(env)                                 # Log mean reward and episode length to tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Normalize rewards

    try:
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device = "auto"
        )
    except:
        from torch.nn import Tanh ## Activation function used in PPO
        policy_kwargs = dict(
            activation_fn = Tanh,
            net_arch=[512,512, dict(pi=[256,256,256], vf=[256,256,256])],
        )
    # Hyperparameters
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,
            learning_rate=5e-5,
            ent_coef=0.01,
            vf_coef=1.,
            gamma=gamma,
            verbose=3,
            batch_size=batch_size,
            n_steps=steps,
            tensorboard_log="logs", 
            device="auto"                # Default to GPU
        )


    # Saving agent data
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")
    
    while True:
        model.learn(25_000_000, callback=callback, reset_num_timesteps=False)  # Use reset_num_timesteps=False to keep going with same logger/checkpoints
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")
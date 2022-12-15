import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions.combined_reward import CombinedReward

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from training.rewards import OswaldRewardFunction, PlayerToBallRewardFunction, HitSpeedRewardFunction, AirdribbleRewardFunction, BallToGoalRewardFunction, PlayerVelocityReward
from training.state_setter import TrainingStateSetter
from training.observations import OswaldObservations


if __name__ == "__main__":
    logging_directory_name = "test"

    frame_skip = 8     # Number of ticks to repeat an action
    half_life_seconds = 5   # Discount reward factor

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds)) # Discount reward factor
    STEPS = 1_000_000 ## how many training steps per epoch of training
    agents_per_match = 1 # 1v1 Bot

    # For running multiple instances
    num_instances = 1
    batch_size = 100_000

    model_path = "models5"

    # Exit save directly to the rlbot_configs directory and current models directory
    def exit_save(model):
        exit_model_path = "rlbot_configs/exit_save"
        model.save(f"{exit_model_path}")
        model.save(f"models/{model_path}/exit_save")
    
        

    # Instantiate rewards, observations, state setters, and action parser
    def get_match():
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function= CombinedReward(
                (
                    PlayerToBallRewardFunction(),
                    HitSpeedRewardFunction(reward_weight=5),
                    AirdribbleRewardFunction(reward_weight=3),
                    BallToGoalRewardFunction(reward_weight=5),
                    PlayerVelocityReward(),
                    OswaldRewardFunction(
                        goal_weight=10,
                        concede_weight=-10,
                        touch_weight=1,
                        shot_weight=5,
                        save_weight=5,
                        boost_pickup_weight=0.1
                        ),
                ),
            (1, 1, 1, 1, 0.005, 1)),
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(10000), NoTouchTimeoutCondition(2500), GoalScoredCondition()],
            obs_builder=OswaldObservations(),  
            state_setter=TrainingStateSetter(),
            action_parser=DiscreteAction()
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances) # Starts Rocket League instance and waits before opening next one
    env = VecCheckNan(env) # Checks for nans in tensor
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Normalize rewards
    env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard

    # Hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        n_epochs=10,
        learning_rate=5e-5,
        ent_coef=0.01,
        vf_coef=1.,
        gamma=gamma,
        clip_range= 0.2,
        verbose=1,
        batch_size=batch_size,
        n_steps=STEPS,
        tensorboard_log="logs", 
        device="cuda" 
    )
    
    # Model to be loaded. 
    model_to_load = "rl_model_30045336_steps"

    try:
        model = PPO.load(
            f"models/models4/{model_to_load}.zip",
            env,
            device='cuda',
            custom_objects={"n_envs": env.num_envs}
        )
        print(f"Loaded {model_to_load}.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[dict(pi=[512, 512, 512], vf=[400, 400, 400])],
        )

    # Evaluation callback to periodically check reward
    eval_callback = EvalCallback(env, best_model_save_path="./best_model/",
                            log_path="./logs/", eval_freq=max(100_000 // num_instances, 1),
                            deterministic=True, render=False)
    
    # Checkpoint callback to periodically save the model
    checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="models/models4/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=False,
    )

    # Training loop
    try:
        for i in range(1000):
            model.learn(total_timesteps=10_000_000, reset_num_timesteps=False, tb_log_name=f"{logging_directory_name}", callback=[eval_callback, checkpoint_callback], progress_bar=True)
    except:
        print("exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
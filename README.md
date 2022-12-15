# RL in RL
The repository for Oswald, the deep reinforcement learning agent, being trained to play the game Rocket League.
To run Oswald, or to create your own agent, follow the instruction below.

## Requirements
- Rocket League (Epic Games or Steam Version)
- BakkesMod
- RLBotGUI
- Python Version betwen 3.7 and 3.9
- RLGym
- Stable-baselines3
- RLGym-tools


## Installation

Rocket League can be downloaded via the Steam or Epic Games. It is important to note that to run multiple instances of the game for training, the Epic Games version of Rocket League must be used.

BakkesMod can be downloaded from bakkesmod.com.

The RLBotGUI can be downloaded from RLBot.org.

The version of Python you are running must be between 3.7 and 3.9 -- as RLGym is not compatible with any versions outside of this range. 

The RLGym plugin for BakkesMod can be downloaded by pip installing it via a terminal/command prompt: 

`pip install rlgym`

StableBaselines3 can also be pip installed via:

 `pip install stable-baselines3[extra]`

Finally RLGym-tools can be pip installed via:

`pip install rlgym-tools`



## Running the Code (Training)

1. Install Rocket League and BakkesMod.
2. Ensure that the version of Python you are running is between 3.7 and 3.9.
3. Install the RLGym plugin by running pip install rlgym in a terminal/command prompt.
4. Install StableBaselines3 by running pip install stable-baselines3[extra] in a terminal/command prompt.
5. Install RLGym-tools by running pip install rlgym-tools in a terminal/command prompt.
6. Open the main.py file.
7. Check the number of instances of Rocket League being run and make sure it isn't too high for your computer to handle.
8. Ensure that BakkesMod is running with administrative privileges and that the RLGym plugin is turned on within the BakkesMod interface, under the "plugins" tab.
9. Change the variable "model_to_load" to the model you desire to train and the model path in the f string that leads to this model. Otherwise, a new model will be created.
10. When the in-game timer counts down, the model will begin training.

### Uploading Model to RLBot for Evaluation

To upload a trained model to RLBot for evaluation:

Before anything, be sure to install RLBotGUI which can be found from https://rlbot.org/

1. Insert your trained model's <model_name>.zip file into the RLBot Config directory.
2. Open the agent class in the code and find the variable "model_to_load"
3. Assign the model_to_load variable to a string that includes the name of the model that comes before the ".zip".
4. Ensure that the [your_agent].zip file is in the rlbot_config directory. There are examples of models already inside.
5. In the RLBotGUI, delete any bots currently under the blue team and orange team.
6. Hit the "add" button near the top left to navigate to the RLBot Config directory and upload the bot.cfg file.
7. A new bot should appear in the list of all other bots, named "Oswald".
8. Setup the teams as desired and launch Rocket League and start the match.
9. Make sure that any instances of Rocket League are closed before launching the match, or else you will encounter an error.

## Code Architecture

├── Logs  
│   └── <subdirectories of tensorboard log outputs>  
├── Models  
│   └── <consists of subdirectories containing trained models with different rewards, hyperparameters, and network architectures>  
├── rlbot_configs  
│   ├── bot.py  
│   └── agent.py  
├── Training  
│   ├── custom_observations.py  
│   ├── custom_rewards.py  
│   └── custom_state_setters.py  
└── main.py  

### Logs
The logs directory contains the tensorboard log outputs from training.

### Models
The models directory contains sub-directories with different models that were trained on different rewards, hyperparameter configurations, and network architectures. The models directory is not included because there are too many files; however, it will be created automatically when training is ran.

### RLBot Configurations
The rlbot_configs directory contains the bot.py and agent.py files that must be reconfigured to upload your own agent into RLBotGUI for evaluation or to play against other bots. It also includes agent.zip files which must be present when adding an agent to be loaded in the agent.py
This directory was cloned from: https://github.com/RLGym/RLGymExampleBot, which provides a template of how to alter the bot.py and agent.py to ensure the agent is able to be uploaded to RLBotGUI properly. The main changes in the agent.py were the custom_objects, which contains information about some of the hyperparameters your agent was trained on and the "model_to_load" variable which was assigned a string of the model.zip to be used in RLBotGUI. The main change to bot.py was the path to the observations.py was added and the OswaldObservations class was instantiated to be used as the observations for the agent to be uploaded to RLBotGUI. Additionally, if the bot seems to be exploiting a certain reward and repeatedly does the same behavior, under the "act" method, try changing "deterministic" equal to False. This will have the agent sample from the distribution of actions in its policy as opposed to what it deems to be the "best" action at every state. This can be beneficial for observing and analysing other behaviors that could be reward shaped and exploited to help the agent learn.

## Training
The Training directory contains all of the Python files that relate to training. This includes custom_observations, custom_rewards, and custom_state_setters.

### Custom Observations

The observations.py file defines a custom observation class for use in training the agent. This custom observation class extends the basic observation class provided by RLGym, and adds additional information that the agent can use to make decisions.

The custom observation class normalizes distances using standard deviation values for position and angle. This allows the observations to be represented using comparable units, which can improve the performance of the reinforcement learning algorithm.

The custom observation class also adds information about the positions, velocities, and other relevant data for the ball, player, and other cars in the game. This information is calculated relative to the ball, and relative to the goals that each team is attacking or defending. This allows the agent to make decisions based on the positions of all relevant objects in the game.

In the build_obs function, the custom observation class takes in the current game state, the player data, and the previous action taken by the agent. It then calculates the relevant data for each object in the game, and adds it to the observation. This includes the ball, the player's car, and the cars of other players. The observations are normalized and concatenated into a single numpy array, which is returned by the function. How often the agent receives the information from these observations is determined by the "frame" or "tick" skip variables found in main.py and bot.py

### Custom Rewards
The first custom reward class is the OswaldRewardFunction. This reward function calculates the reward for an agent based on a set of adjustable weights for different events that can occur in a Rocket League game, such as scoring goals, conceding goals, touching the ball, taking shots, making saves, and picking up boost. The weights for these events can be specified as input parameters to the class, allowing the user to adjust the importance of each event in the reward calculation. The reward function accumulates the rewards for these events over the course of a match, rather than calculating the reward based on individual events.

The second custom reward class is the PlayerToBallRewardFunction. This reward function calculates the reward for a player based on the distance between the player and the ball, the direction of the player's movement, and the speed of the ball. The reward function takes into account the direction and speed of the ball, and rewards the player for moving towards the ball and increasing the ball's speed. The reward function also takes into account the position and speed of the player, and rewards the player for moving towards the ball and maintaining a high speed.

The AirdribbleRewardFunction class calculates the reward for a player based on the distance between the player and the ball. If the distance between the player and the ball is below a specified minimum threshold, then a reward is returned. This reward is weighted by the reward_weight parameter that is passed to the class when it is initialized.

The BallToGoalRewardFunction class calculates the reward for a player based on the distance between the ball and the goal that the player's team is attacking. If the distance between the ball and the goal is below a specified minimum threshold and the ball is moving towards the goal, then a reward is returned. This reward is weighted by the reward_weight parameter that is passed to the class when it is initialized.

The PlayerVelocityReward class calculates the reward for a player based on the magnitude of the player's velocity. This reward is returned as is, with weighting applied in main.py.

## Main
The main.py file is the entry point for training the agent. It imports and uses classes and functions from the other files in the repository.
In the main.py file, the user can specify the PPO hyperparameters for training, such as the learning rate, discount factor, and number of training epochs (just to name a few). The user can also specify the path for saving logs, the path for saving models, and the network architecture to be used by the actor-critic model.

The main.py file also imports the custom rewards, custom observations, and custom state-setters from the Training directory. These files define the rewards, observations, and state information that will be used to train the agent.

The main.py file creates the model and passes in the specified hyperparameters, paths, and network architecture.

Once the agent is trained, the user can use the bot.py and agent.py files from the rlbot_configs directory to upload the trained agent to RLBotGUI for evaluation or to play against other bots. The user will need to reconfigure these files to specify the path to the trained model's zip file, as well as any other necessary information.

After the agent is uploaded to RLBotGUI, the user can launch Rocket League and start a match to see the trained agent in action.

import numpy as np
import math
from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgym.utils.gamestates import GameState, PlayerData


class OswaldRewardFunction(RewardFunction):
    """
    This class calculates the reward for an agent based on the events that occur in a Rocket League, such as scoring goals,
    conceding goals, touching the ball, taking shots, making saves, and picking up boost. The rewards for these events
    are weighted according to a set of adjustable weights that can be specified as input parameters to the class. Based off of cumulation
    of events as opposed to events per episode

    """

    def __init__(self, goal_weight=0, concede_weight=0., touch_weight=0., shot_weight=0., save_weight=0., 
                 demo_weight=0., boost_pickup_weight=0.):

        super().__init__()
        self.weights = np.array([goal_weight, concede_weight, touch_weight, shot_weight, save_weight, 
                                 demo_weight, boost_pickup_weight])

        # Track changes when event occurs
        self.last_stored_values = {}


    def _extract_player_values(self, player: PlayerData, state: GameState):
        # Determine which team reward is being calculated for
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, opponent, player.ball_touched, player.match_shots,
                        player.match_saves, player.match_demolishes, player.boost_amount])

    def _calculate_value_differences(self, old_values, new_values):
        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # Increasing values
        return diff_values


    def reset(self, initial_state: GameState, optional_data=None):
        self.cumulative_values = {}
        for player in initial_state.players:
            self.cumulative_values[player.car_id] = self._extract_player_values(player, initial_state)
    
    def _get_event_reward(self, diff_values):
        return np.dot(self.weights, diff_values)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        reward = 0

        # Reward for events (goal, save, ball touch, etc.)
        cumulative_values = self.cumulative_values[player.car_id]
        new_values = self._extract_player_values(player, state)
        cumulative_values += new_values

        reward = reward + self._get_event_reward(cumulative_values)
        self.cumulative_values[player.car_id] = cumulative_values
        return float(reward)



class PlayerToBallRewardFunction(RewardFunction):
    """
    Calculates the reward for a player based on the distance between the player and the ball, 
    the direction of the player's movement, and the speed of the ball.
    """

    def __init__(self, ball_speed_factor=0.1, max_reward=10):
        super().__init__()
        self.ball_speed_factor = ball_speed_factor
        self.max_reward = max_reward
    
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        ball_position = state.ball.position
        car_position = player.car_data.position
        ball_speed = state.ball.linear_velocity
        if np.linalg.norm(state.ball.linear_velocity) == 0:
            ball_direction = 0
        else:
            ball_direction = state.ball.linear_velocity / np.linalg.norm(state.ball.linear_velocity)


        # Check if the ball is moving
        if math.vecmag(ball_speed) == 0:
            # If the ball is not moving, return a small constant reward
            return -0.1

        # Calculate the distance from the player to the ball
        distance = np.linalg.norm(ball_position - car_position)

        # Initialize car direction so not None
        if player.car_data.linear_velocity.size == 0:
            car_direction = 0
        else:
            car_direction = player.car_data.linear_velocity / np.linalg.norm(player.car_data.linear_velocity)

        # Calculate the angle between the ball's direction and the player's direction
        angle = np.arccos(np.dot(ball_direction, car_direction))

        # Calculate the reward based on the distance and ball speed
        reward = (1 - distance / (BACK_WALL_Y - BALL_RADIUS)) * (1 + self.ball_speed_factor * ball_speed / BALL_MAX_SPEED)

        # Adjust the reward based on the angle between the ball's direction and the player's direction
        if angle < np.pi / 2:
            # If the player is moving towards the ball, increase the reward
            reward *= 1 + angle / np.pi
        else:
            # If the player is moving away from the ball, decrease the reward
            reward *= 1 - (angle - np.pi / 2) / np.pi

        # convert reward from ndarray to scalar
        return float(reward.item(0))



class HitSpeedRewardFunction(RewardFunction):
    """
    Calculates the reward for the agent based on the speed of the ball after it is hit.
    """
    def __init__(self, reward_weight, min_speed_threshold=100):
        super().__init__()
        self.reward_weight = reward_weight
        self.min_speed_threshold = min_speed_threshold

    def reset(self, initial_state: GameState, optional_data=None):
        # Initialize optional_data as an empty dictionary if it is None
        if optional_data is None:
            optional_data = {}

        ball_linear_velocity = math.vecmag(initial_state.ball.linear_velocity)
        optional_data['previous_ball_speed'] = ball_linear_velocity
        return optional_data

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        ball_linear_velocity = math.vecmag(state.ball.linear_velocity) 
        if optional_data is None:
            optional_data = {}

        # Retrieve the previous ball speed from the optional_data dictionary
        previous_ball_speed = optional_data.get('previous_ball_speed', 0)
      
        # Calculate the reward based on the change in ball speed
        reward = 0
        if ball_linear_velocity != None:
            if ball_linear_velocity > previous_ball_speed:
                reward = self.reward_weight * (ball_linear_velocity - previous_ball_speed) / BALL_MAX_SPEED
                # Update the previous ball speed for the next timestep
                optional_data['previous_ball_speed'] = ball_linear_velocity
        
        
        return float(reward)


class AirdribbleRewardFunction(RewardFunction):
    def __init__(self, reward_weight:float, min_distance_threshold=10):
        super().__init__()
        self.reward_weight = reward_weight
        self.min_distance_threshold = min_distance_threshold
    
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        # Calculate the distance between the bot and the ball
        distance = np.linalg.norm(player.car_data.position - state.ball.position)

        # Get the ball and car positions
        ball_position = state.ball.position
        car_position = player.car_data.position

        # Check if the ball and the car are both in the air
        if ball_position[2] > 0 and car_position[2] > 0:
            # If the distance between the bot and the ball is below the threshold, return a reward
            if distance < self.min_distance_threshold:
                return float(self.reward_weight)

        # Otherwise, return 0
        return 0

class BallToGoalRewardFunction(RewardFunction):
    def __init__(self, reward_weight:float, min_distance_threshold=10):
        super().__init__()
        self.reward_weight = reward_weight
        self.min_distance_threshold = min_distance_threshold
    
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        # Get the ball position and velocity
        ball_position = state.ball.position
        ball_velocity = state.ball.linear_velocity

        # Get the player's team
        player_team = player.team_num

        # Determine the position of the opposing team's goal
        if player_team == BLUE_TEAM:
            goal_position = ORANGE_GOAL_BACK
        else:
            goal_position = BLUE_GOAL_BACK

        # Calculate the distance between the ball and the goal
        distance = math.vecmag(ball_position - goal_position)

        # Check if the ball is moving towards the goal
        if np.dot(ball_velocity, goal_position - ball_position) > 0:
            # If the distance between the ball and the goal is below the threshold, return a reward
            if distance < self.min_distance_threshold:
                return float(self.reward_weight)

        # Otherwise, return very small punishment
        return -0.1
    
class PlayerVelocityReward(RewardFunction):
    def reset(self, initial_state: GameState, optional_data=None):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return float(math.vecmag(player.car_data.linear_velocity))
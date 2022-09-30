import numpy as np
from numpy import exp
from numpy.linalg import norm
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CEILING_Z, BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BALL_RADIUS, ORANGE_TEAM, GOAL_HEIGHT, CAR_MAX_ANG_VEL
from rlgym.utils.gamestates import GameState, PlayerData

#TODO Build custom reward functions
class OswaldRewardFunction(RewardFunction):
    pass
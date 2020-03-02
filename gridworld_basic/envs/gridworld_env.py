import gym
from gym import spaces
import numpy

def clamp(x,min,max):
    if x < min:
        return min
    if x > max:
        return max
    return x

class GridworldEnv(gym.Env):
    reward_range = (-1, 0)
    action_space = spaces.Discrete(4)
    # although there are 2 terminal squares in the grid
    # they are considered as 1 state
    # therefore observation is between 0 and 14
    observation_space = spaces.Discrete(15)

    def __init__(self):
        gridworld = numpy.arange(self.observation_space.n + 1).reshape((4, 4))
        gridworld[-1, -1] = 0
        # state transition matrix
        self.stateTransitionProb = numpy.zeros((self.action_space.n, self.observation_space.n, self.observation_space.n))
        # any action taken in terminal state has no effect
        self.stateTransitionProb[:, 0, 0] = 1
        flatiter = gridworld.flat[1:-1]
        for s in flatiter:
            row, col = numpy.argwhere(gridworld == s)[0]
            ziprange = zip(range(self.action_space.n),[(-1, 0), (0, 1), (1, 0), (0, -1)])
            for a, d in ziprange:
                next_row = clamp(row + d[0], 0, 3)
                next_col = clamp(col + d[1], 0, 3)
                s_prime = gridworld[next_row, next_col]
                self.stateTransitionProb[a, s, s_prime] = 1

        self.rewardMatrix = numpy.full((self.action_space.n, self.observation_space.n), -1)
        self.rewardMatrix[:, 0] = 0
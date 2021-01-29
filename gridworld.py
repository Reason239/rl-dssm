import gym
from gym import spaces
import numpy as np

class GridWorld(gym.Env):
    """
    Custom gridworld Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}
    action2name = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PRESS']
    action2delta = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int)

    def __init__(self, height=5, width=5, n_buttons=3):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete((height, width, n_buttons * 2 + 2))
        self.height = height
        self.width = width
        self.n_buttons = n_buttons

    def next_pos(self, pos, action):
        delta = self.action2delta[action]

        return

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
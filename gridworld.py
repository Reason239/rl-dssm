import gym
from gym import spaces
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np


def repeat_upsample(rgb_array, k=1, l=1):
    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def rgb_embed(bool_vect):
    rgb = np.array([0, 0, 0], dtype=np.uint8)
    n_buttons = len(bool_vect) // 2
    if bool_vect[0]:
        rgb[0] += 200
    ind_max = np.argmax(bool_vect[1:]) + 1
    ind_button = (ind_max - 1) // 2
    if bool_vect[ind_max]:
        if ind_max % 2 == 0:
            rgb[1] += 100 + 155 * (n_buttons - ind_button) // n_buttons
        else:
            rgb[2] += 100 + 155 * (n_buttons - ind_button) // n_buttons
    return rgb


def get_grid(state, pixels_per_tile=10):
    depth, height, width = state.shape
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            grid[h, w, :] = rgb_embed(state[:, h, w])
    grid = repeat_upsample(grid, pixels_per_tile, pixels_per_tile)
    return grid


def join_grids(grid1, grid2, pixels_between=10):
    sep = np.full((len(grid1), pixels_between, 3), 255, dtype=np.uint8)
    return np.concatenate((grid1, sep, grid2), axis=1)


class GridWorld(gym.Env):
    """
    Custom gridworld Environment that follows gym interface.
    Has height x width size and n_buttons of buttons
    Agent has to press them in ascending order
    """
    metadata = {'render.modes': ['human']}
    action2name = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PRESS']
    name2action = {k: v for v, k in enumerate(action2name)}
    action2delta = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int)

    def __init__(self, height=5, width=5, n_buttons=3, button_pos=None, pixels_per_tile=50, seed=None):
        """
        :param height: height of the world (in tiles)
        :param width: width of the world (in tiles)
        :param n_buttons: number of buttons
        :param button_pos: (optional) list of (2,) numpy arrays - positions of the buttons
        :param pixels_per_tile: height/width of a tile for rendering
        """
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * n_buttons + 1, height, width), dtype=np.int)
        self.height = height
        self.width = width
        self.n_buttons = n_buttons
        self.button_pos = button_pos
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        if self.button_pos is None:
            self.button_pos = []
            idx = np.random.choice(height * width, n_buttons, replace=False)
            for index in idx:
                self.button_pos.append(np.array([index // width, index % width], dtype=np.int))
        self.next_button = None
        self.pos = None
        self.viewer = SimpleImageViewer()
        self.pixels_per_tile = pixels_per_tile

    def next_pos(self, pos, action):
        delta = self.action2delta[action]
        res = pos + delta
        res[0] = np.clip(res[0], 0, self.height - 1)
        res[1] = np.clip(res[1], 0, self.width - 1)
        return res

    def get_observation(self):
        obs = np.zeros((2 * self.n_buttons + 1, self.height, self.width), dtype=np.int)
        h, w = self.pos
        obs[0, h, w] = 1
        for ind, b_pos in enumerate(self.button_pos):
            h, w = b_pos
            if ind < self.next_button:
                obs[2 * ind + 1, h, w] = 1
            else:
                obs[2 * ind + 2, h, w] = 1
        return obs

    def step(self, action):
        """
        :param action: int from [0; 5)
        :return: observation - np.array of shape ( 2 * n_buttons + 1, height, width). One channel for current position,
            two channels per button (unpressed/pressed)
            reward - float, 1 if the last button is pressed
            done - bool, True if the last button is pressed
            info - empty dict
        """
        info = {}
        if self.action2name[action] == 'PRESS':
            if np.all(self.pos == self.button_pos[self.next_button]):
                self.next_button += 1
        self.pos = self.next_pos(self.pos, action)
        obs = self.get_observation()
        done = (self.next_button == self.n_buttons)
        reward = float(done)
        return obs, reward, done, info

    def reset(self):
        self.next_button = 0
        self.pos = np.array([0, 0], dtype=np.int)
        return self.get_observation()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'human':
            grid = np.zeros((self.height, self.width, 3), dtype=np.int8)
            h, w = self.pos
            grid[h, w, 0] += 200
            for ind, b_pos in enumerate(self.button_pos):
                h, w = b_pos
                if ind >= self.next_button:
                    grid[h, w, 1] += 100 + 155 * (self.n_buttons - ind) // self.n_buttons
                else:
                    grid[h, w, 2] += 100 + 155 * (self.n_buttons - ind) // self.n_buttons
            grid = repeat_upsample(grid, self.pixels_per_tile, self.pixels_per_tile)
            self.viewer.imshow(grid)
            return self.viewer.isopen
        else:
            return

    def close(self):
        pass

    def get_expert_action(self, eps=0.05):
        if eps and np.random.rand() < eps:
            return int(np.random.randint(low=0, high=5))
        target = self.button_pos[self.next_button]
        if np.all(self.pos == target):
            return self.name2action['PRESS']
        vert = target[0] - self.pos[0]
        hor = target[1] - self.pos[1]
        if np.random.rand() < abs(vert) / (abs(vert) + abs(hor)):
            # go vertical
            if vert > 0:
                action = self.name2action['DOWN']
            else:
                action = self.name2action['UP']
        else:
            # go horisontal
            if hor > 0:
                action = self.name2action['RIGHT']
            else:
                action = self.name2action['LEFT']
        return action

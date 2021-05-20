import gym
from gym import spaces
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
from copy import copy


def repeat_upsample(rgb_array, k=1, l=1):
    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def stage_and_pos_from_state_data(height, width, n_buttons, button_pos, pixels_per_tile, pos, next_button):
    stage = 3 * next_button
    if next_button < n_buttons and np.all(pos == button_pos[next_button]):
        # standing on the next button
        stage += 1
    if next_button > 0 and np.all(pos == button_pos[next_button - 1]):
        # standing on a p(just) pressed button
        stage -= 1
    return stage, pos


def grid_from_state_data(height, width, n_buttons, button_pos, pixels_per_tile, pos, next_button):
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = pos
    grid[h, w, 0] += 200
    for ind, b_pos in enumerate(button_pos):
        h, w = b_pos
        if ind >= next_button:
            grid[h, w, 1] += 100 + 155 * (n_buttons - ind) // n_buttons
        else:
            grid[h, w, 2] += 100 + 155 * (n_buttons - ind) // n_buttons
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

    def __init__(self, height=5, width=5, n_buttons=3, button_pos=None, pixels_per_tile=10, seed=None,
                 obs_dtype='bool'):
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
        if self.button_pos is None:
            self.button_pos = []
            idx = np.random.choice(height * width, n_buttons, replace=False)
            for index in idx:
                self.button_pos.append(np.array([index // width, index % width], dtype=np.int))
            self.button_idx = tuple(idx)
        else:
            self.button_idx = tuple(a * width + b for (a, b) in button_pos)
        if obs_dtype not in ['bool', 'int']:
            raise ValueError
        self.obs_dtype = obs_dtype

        button_inds = []
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
        if self.obs_dtype == 'bool':
            obs = np.zeros((2 * self.n_buttons + 1, self.height, self.width), dtype=np.int)
            h, w = self.pos
            obs[0, h, w] = 1
            for ind, b_pos in enumerate(self.button_pos):
                h, w = b_pos
                if ind < self.next_button:
                    # pressed
                    obs[2 * ind + 1, h, w] = 1
                else:
                    # unpressed
                    obs[2 * ind + 2, h, w] = 1
            return obs
        if self.obs_dtype == 'int':
            obs = np.zeros((self.height, self.width), dtype=np.int)
            h, w = self.pos
            obs[h, w] = 2 * self.n_buttons + 1
            for ind, b_pos in enumerate(self.button_pos):
                h, w = b_pos
                if ind < self.next_button:
                    # pressed
                    obs[h, w] += 2 * ind + 1
                else:
                    # unpressed
                    obs[h, w] += 2 * ind + 2
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
        if self.action2name[action] == 'PRESS':
            if np.all(self.pos == self.button_pos[self.next_button]):
                self.next_button += 1
        self.pos = self.next_pos(self.pos, action)
        obs = self.get_observation()
        done = (self.next_button == self.n_buttons)
        reward = float(done)
        info = self.get_info()
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
            grid = grid_from_state_data(self.height, self.width, self.n_buttons, self.button_pos, self.pixels_per_tile,
                                        self.pos, self.next_button)
            self.viewer.imshow(grid)
            return self.viewer.isopen
        elif mode == 'get_grid':
            return grid_from_state_data(self.height, self.width, self.n_buttons, self.button_pos, self.pixels_per_tile,
                                        self.pos, self.next_button)
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

    def get_info(self):
        info = {'state_tuple': self.button_idx + (self.next_button,) + tuple(self.pos)}
        return info

    def to_random_state(self, seed=239):
        if seed:
            np.random.seed(seed)
        self.pos[0] = np.random.randint(0, self.height)
        self.pos[1] = np.random.randint(0, self.width)
        self.next_button = np.random.randint(0, self.n_buttons)
        return self.get_observation()

    def get_all_next_states_with_data(self):
        backup = self.pos.copy(), self.next_button
        states = []
        data = []
        for next_button in range(self.next_button, self.n_buttons):
            for pos_h in range(self.height):
                for pos_w in range(self.width):
                    self.pos[0] = pos_h
                    self.pos[1] = pos_w
                    self.next_button = next_button
                    states.append(self.get_observation())
                    data.append(self.get_state_data())
        self.next_button = self.n_buttons
        self.pos[:] = self.button_pos[-1]
        states.append(self.get_observation())
        data.append(self.get_state_data())

        self.pos, self.next_button = backup
        return states, data

    def get_state_data(self):
        return (
            copy(self.height), copy(self.width), copy(self.n_buttons), copy(self.button_pos),
            copy(self.pixels_per_tile), copy(self.pos), copy(self.next_button))

    def load_state_data(self, height, width, n_buttons, button_pos, pixels_per_tile, pos, next_button):
        self.height = height
        self.width = width
        self.n_buttons = n_buttons
        self.button_pos = button_pos
        self.pixels_per_tile = pixels_per_tile
        self.pos = pos
        self.next_button = next_button

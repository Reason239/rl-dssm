from gridworld import GridWorld
import numpy as np
from tqdm import tqdm
import pickle
from itertools import combinations
import pathlib


np.random.seed(42)
data_path = 'datasets/_1000/'
n_envs_train = 1000
n_envs_test = 200
# TODO try eps=0
eps = 0.05


def make_env_index_dataset(data_path, n_envs_train, n_envs_test, eps):
    train = []
    test = []
    button_positions = []
    seed = -1
    cur_num = 0
    idx_test = {}
    idx_train = {}
    for i in tqdm(range(n_envs_train + n_envs_test)):
        dataset = train if i < n_envs_train else test
        idx = idx_train if i < n_envs_train else idx_test
        # generate an env with NEW positions of the buttons
        repeats = True
        while repeats:
            seed += 1
            env = GridWorld(height=5, width=5, n_buttons=3, seed=seed)
            observation = env.reset()
            repeats = False
            for prev_button_pos in button_positions:
                if all(np.all(b1 == b2) for b1, b2 in zip(env.button_pos, prev_button_pos)):
                    repeats = True
                    print('WAOW')
                    break
        states = [observation.astype(np.bool_)]
        done = False
        while not done:
            observation, _, done, _ = env.step(env.get_expert_action(eps))
            states.append(observation.astype(np.bool_))
        dataset += list(combinations(states, 2))
        start = cur_num
        stop = cur_num + len(states) * (len(states) - 1) // 2
        idx[seed] = (start, stop) if i < n_envs_train else (start - len(train), stop - len(train))
        cur_num += len(states) * (len(states) - 1) // 2

    # save the data
    path = pathlib.Path(data_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(path / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(path / 'idx_train.pkl', 'wb') as f:
        pickle.dump(idx_train, f)
    with open(path / 'idx_test.pkl', 'wb') as f:
        pickle.dump(idx_test, f)



from gridworld import GridWorld
import numpy as np
from tqdm import tqdm
import pickle
from itertools import combinations
from collections import defaultdict
import pathlib


def make_env_index_dataset(data_path, n_envs_train, n_envs_test, eps, dtype, save_state_data=True,
                           min_states_interval=None, max_states_interval=None):
    if dtype == 'bool':
        np_dtype = np.bool_
    elif dtype == 'int':
        np_dtype = int
    else:
        raise ValueError
    if min_states_interval is None:
        min_states_interval = 1
    if max_states_interval is None:
        max_states_interval = 10000
    train = []
    test = []
    if save_state_data:
        train_state_data = []
        test_state_data = []
    button_idxs_used = set()
    seed = -1
    cur_num = 0
    idx_test = {}
    idx_train = {}
    for i in tqdm(range(n_envs_train + n_envs_test)):
        dataset = train if i < n_envs_train else test
        idx = idx_train if i < n_envs_train else idx_test
        if save_state_data:
            state_data = train_state_data if i < n_envs_train else test_state_data
        # generate an env with NEW positions of the buttons
        repeats = True
        while repeats:
            seed += 1
            env = GridWorld(height=5, width=5, n_buttons=3, seed=seed, obs_dtype=dtype)
            observation = env.reset()
            repeats = env.button_idx in button_idxs_used
        states = [observation.astype(np_dtype)]
        if save_state_data:
            state_datas = [env.get_state_data()]
        done = False
        while not done:
            observation, _, done, _ = env.step(env.get_expert_action(eps))
            states.append(observation.astype(np_dtype))
            if save_state_data:
                state_datas.append(env.get_state_data())
        prev_len = len(dataset)
        for i, s in enumerate(states):
            dataset += [(s, s_prime) for s_prime in states[i + min_states_interval: i + max_states_interval + 1]]
            if save_state_data:
                s_data = state_datas[i]
                state_data += [(s_data, s_prime_data) for s_prime_data in
                               state_datas[i + min_states_interval: i + max_states_interval + 1]]
        new_len = len(dataset)
        start = cur_num
        stop = cur_num + (new_len - prev_len)
        idx[seed] = (start, stop) if i < n_envs_train else (start - len(train), stop - len(train))
        cur_num += new_len - prev_len

    # save the data
    path = pathlib.Path(data_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(path / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)
    if save_state_data:
        with open(path / 'state_data_train.pkl', 'wb') as f:
            pickle.dump(train_state_data, f)
        with open(path / 'state_data_test.pkl', 'wb') as f:
            pickle.dump(test_state_data, f)
    with open(path / 'idx_train.pkl', 'wb') as f:
        pickle.dump(idx_train, f)
    with open(path / 'idx_test.pkl', 'wb') as f:
        pickle.dump(idx_test, f)


def make_states_dict_dataset(data_path, n_envs_train, n_envs_test, n_runs_per_env, eps):
    train = []
    test = []
    button_idxs_used = set()
    seed = -1
    cur_num = 0
    states_test = defaultdict(list)
    states_train = defaultdict(list)
    for i in tqdm(range(n_envs_train + n_envs_test)):
        dataset = train if i < n_envs_train else test
        states_dict = states_train if i < n_envs_train else states_test
        if i == n_envs_train:
            cur_num = 0
        # generate an env with NEW positions of the buttons
        repeats = True
        while repeats:
            seed += 1
            env = GridWorld(height=5, width=5, n_buttons=3, seed=seed)
            repeats = env.button_idx in button_idxs_used
        for j in range(n_runs_per_env):
            observation = env.reset()
            states = [observation.astype(np.bool_)]
            state_tuples = [env.get_info()['state_tuple']]
            done = False
            while not done:
                observation, _, done, info = env.step(env.get_expert_action(eps))
                states.append(observation.astype(np.bool_))
                state_tuples.append(info['state_tuple'])

            dataset += list(combinations(states, 2))
            for i in range(len(states) - 1):
                tup = state_tuples[i]
                start = cur_num + (2 * len(states) - i - 1) * i // 2
                stop = start + len(states) - i - 1
                states_dict[tup] += list(range(start, stop))
            cur_num += len(states) * (len(states) - 1) // 2

    # save the data
    path = pathlib.Path(data_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(path / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(path / 's_ind_train.pkl', 'wb') as f:
        pickle.dump(states_train, f)
    with open(path / 's_ind_test.pkl', 'wb') as f:
        pickle.dump(states_test, f)


if __name__ == '__main__':
    np.random.seed(42)
    data_path = 'datasets/all_1000_2_6/'
    n_envs_train = 1000
    n_envs_test = 200
    min_states_interval = 2
    max_states_interval = 6
    # TODO try eps=0
    eps = 0.05
    # eps = 0
    make_env_index_dataset(data_path, n_envs_train, n_envs_test, eps, 'bool', save_state_data=True,
                           min_states_interval=min_states_interval, max_states_interval=max_states_interval)

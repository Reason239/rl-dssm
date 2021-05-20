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
        for j, s in enumerate(states):
            dataset += [(s, s_prime) for s_prime in states[j + min_states_interval: j + max_states_interval + 1]]
            if save_state_data:
                s_data = state_datas[j]
                state_data += [(s_data, s_prime_data) for s_prime_data in
                               state_datas[j + min_states_interval: j + max_states_interval + 1]]
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


def make_validation_index_dataset(data_path, n_envs, n_negatives, expert_eps, fake_eps, dtype):
    if dtype == 'bool':
        np_dtype = np.bool_
    elif dtype == 'int':
        np_dtype = int
    else:
        raise ValueError
    dataset = []
    state_data = []
    button_idxs_used = set()
    seed = 10000
    cur_num = 0
    idx = {}
    for ind_env in tqdm(range(n_envs)):
        # generate an env with NEW positions of the buttons
        repeats = True
        while repeats:
            seed += 1
            env = GridWorld(height=5, width=5, n_buttons=3, seed=seed, obs_dtype=dtype)
            observation = env.reset()
            repeats = env.button_idx in button_idxs_used
        states = [observation.astype(np_dtype)]
        state_datas = [env.get_state_data()]
        done = False
        while not done:
            observation, _, done, _ = env.step(env.get_expert_action(eps=expert_eps))
            states.append(observation.astype(np_dtype))
            state_datas.append(env.get_state_data())
        n_states = len(states)
        s_ind, s_prime_ind = np.random.choice(np.arange(n_states), 2, replace=False)
        s_ind, s_prime_ind = min(s_ind, s_prime_ind), max(s_ind, s_prime_ind)
        s, s_prime = states[s_ind], states[s_prime_ind]
        s_data, s_prime_data = state_datas[s_ind], state_datas[s_prime_ind]
        distance = np.abs(s_data[5] - s_prime_data[5]).sum()
        negatives = []
        negatives_data = []
        for ind_neg in range(n_negatives):
            env2 = GridWorld(height=5, width=5, n_buttons=3, seed=ind_env * n_negatives + ind_neg, obs_dtype=dtype)
            env2.load_state_data(*s_data)
            # Random actions
            cur_distance = 0
            target_distance = max(1, distance + np.random.randint(-1, 2))
            while cur_distance != target_distance:
                negative, _, done, _ = env2.step(env2.get_expert_action(eps=fake_eps))
                if done:
                    break
                cur_distance = np.abs(s_data[5] - env2.pos).sum()
            negative_data = env2.get_state_data()
            negatives.append(negative)
            negatives_data.append(negative_data)
        dataset += [(s, s_prime)] + [(s, s_neg) for s_neg in negatives]
        state_data += [(s_data, s_prime_data)] + [(s_data, s_neg_data) for s_neg_data in negatives_data]
        start = cur_num
        stop = cur_num + (1 + n_negatives)
        idx[seed] = (start, stop)
        cur_num += 1 + n_negatives

    # save the data
    path = pathlib.Path(data_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'evaluate.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(path / 'state_data_evaluate.pkl', 'wb') as f:
        pickle.dump(state_data, f)
    with open(path / 'idx_evaluate.pkl', 'wb') as f:
        pickle.dump(idx, f)


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
    dtype = 'int'
    data_path = f'datasets/evaluate_{dtype}_1024_n4/'
    n_envs_train = 1000
    n_envs_test = 200
    min_states_interval = 2
    max_states_interval = 6
    # TODO try eps=0
    eps = 0.05
    # eps = 0
    # make_env_index_dataset(data_path, n_envs_train, n_envs_test, eps, 'int', save_state_data=True,
    #                        min_states_interval=min_states_interval, max_states_interval=max_states_interval)
    #
    n_envs = 1024
    n_negatives = 4
    expert_eps = 0.05
    fake_eps = 1.
    make_validation_index_dataset(data_path, n_envs, n_negatives, expert_eps, fake_eps, dtype)

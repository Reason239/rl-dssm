from gridworld import GridWorld
import numpy as np
from tqdm import tqdm
import pickle
from itertools import combinations

np.random.seed(42)
data_path = 'datasets/all_1000/'
n_envs_train = 1000
n_envs_test = 200
# TODO try eps=0
eps = 0.05
train = []
test = []
button_positions = []

for i in tqdm(range(n_envs_train + n_envs_test)):
    dataset = train if i < n_envs_train else test
    # generate an env with NEW positions of the buttons
    repeats = True
    while repeats:
        env = GridWorld(height=5, width=5, n_buttons=3)
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

# save the data
with open(data_path + 'train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open(data_path + 'test.pkl', 'wb') as f:
    pickle.dump(test, f)


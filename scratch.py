import numpy as np
import pickle

data_path_1 = 'datasets/int_1000/train.pkl'
data_path_2 = 'datasets/evaluate_int_1024_n4/evaluate.pkl'
with open(data_path_1, 'rb') as f:
    data = pickle.load(f)
# print(type(data[0][1]))

with open(data_path_2, 'rb') as f:
    data = pickle.load(f)
with open('datasets/evaluate_int_1024_n4/state_data_evaluate.pkl', 'rb') as f:
    state_data = pickle.load(f)
ind = 1
print('From')
print(state_data[ind * 4][0])
print(data[ind * 4][0])

for i in range(5):
    if not i:
        print('Expert')
    else:
        print('Fake')
    print(state_data[ind * 4 + i][1])
    print(data[ind * 4 + i][1])

# from gridworld import GridWorld
#
# best_i = 100000
# best_seed = None
# for seed in range(102):
#     env2 = GridWorld(height=5, width=5, n_buttons=3, seed=seed, obs_dtype='int')
#     # env2.load_state_data(*s_data)
#     env2.reset()
#     for i in range((100) ** 2):
#         negative, _, done, _ = env2.step(env2.get_expert_action(eps=1.))
#         if done:
#             if i < best_i:
#                 best_i = i
#                 best_seed = seed
#             break
#
# print(best_seed, best_i)
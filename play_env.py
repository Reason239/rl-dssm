from gridworld import GridWorld
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

np.random.seed(42)
env = GridWorld(5, 5, 3, seed=99, obs_dtype='int')
# rec = VideoRecorder(env, base_path='./videos/test1')
env.reset()
t = 1
eps=0.1
env.unwrapped.render()
# rec.capture_frame()
time.sleep(t)
done = False
while not done:
    # if action == 4:
    #     print(env.pos, env.button_pos[0], np.all(env.pos == env.button_pos[env.next_button]))
    action = env.get_expert_action(eps=1.)
    _, _, done, _ = env.step(action)
    # print(env.next_button)
    env.render()
    # rec.capture_frame()
    time.sleep(t)

# rec.close()
input()

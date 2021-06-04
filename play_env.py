"""Renders an expert trajectory in the Gridworld"""

from gridworld import GridWorld
import numpy as np
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

np.random.seed(42)
env = GridWorld(5, 5, 3, seed=0, obs_dtype='int', pixels_per_tile=50)
# rec = VideoRecorder(env, base_path='./videos/test1')
env.reset()
t = 1
eps=0.1
env.unwrapped.render()
# rec.capture_frame()
time.sleep(t)
done = False
while not done:
    action = env.get_expert_action(eps=1.)
    _, _, done, _ = env.step(action)
    env.render()
    # rec.capture_frame()
    time.sleep(t)

# rec.close()
input()

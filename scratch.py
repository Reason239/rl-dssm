from utils import plot_gridworld
import pathlib

save_path = pathlib.Path('experiments/test_bool.png')
plot_gridworld(5, 5, save_path=save_path, dtype='bool')

save_path = pathlib.Path('experiments/test_int.png')
plot_gridworld(5, 5, save_path=save_path, dtype='int')


from torch.utils.data import Dataset
import torch
import numpy as np
import pickle


class DatasetFromPickle(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(s.astype(np.float32)) for s in self.data[idx])

    def __len__(self):
        return len(self.data)

def format_time(seconds):
    time = round(seconds)
    hours = time // 3600
    minutes = (time % 3600) // 60
    secs = time % 60
    return f'{hours} hours {minutes} minutes {secs} seconds'
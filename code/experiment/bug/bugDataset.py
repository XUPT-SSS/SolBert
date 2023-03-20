import os
import pickle

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        output = {"label": sample['label'],
                  "file": sample['file_name'],
                  "value": sample['value']}
        return output

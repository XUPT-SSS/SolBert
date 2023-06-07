
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self,sentence):
        self.data = sentence
    def __len__(self):
        return len(self.data)
    def __getitem__(self,item):
        sample = self.data[item]
        return sample
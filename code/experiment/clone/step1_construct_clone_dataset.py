import os
import pickle
import sys,os
sys.path.append("..")
from utils import getParser
from config import Clone as cf
def pre_dataset():
    sample_set_1 = []
    sample_set_2 = []
    for root, dirs, files in os.walk(cf.colne_dataset_path):
        if len(files) == 0:
            continue
        try:
            path_1 = os.path.join(root, files[0])
            path_2 = os.path.join(root, files[1])
            lines1 = getParser(path_1)
            lines2 = getParser(path_2)
            sample1 = {'label': 1, 'value': lines1[0]}
            sample2 = {'label': 2, 'value': lines2[0]}
            sample_set_1.append(sample1)
            sample_set_2.append(sample2)
        except:
            print("[ERROR]", root)
    assert len(sample_set_1) == len(sample_set_2)
    length = int(len(sample_set_1) / 2)
    sample_set_2_random = sample_set_2[length:] + sample_set_2[:length]
    samples = []
    col1 = sample_set_1 * 2
    col2 = sample_set_2 + sample_set_2_random
    label = [1] * len(sample_set_1) + [0] * len(sample_set_1)
    for i in range(len(col1)):
        sample = {'label': label[i], 'value1': col1[i]['value'], 'value2': col2[i]['value']}
        samples.append(sample)
    with open(os.path.join(cf.clone_construct_data_path, "clone_data.pkl"), 'wb') as f:
        pickle.dump(samples, f)
    print("finish")
if __name__ == '__main__':
    pre_dataset()
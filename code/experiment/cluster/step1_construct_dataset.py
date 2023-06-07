import os
import pickle
import sys,os
sys.path.append("..")
from utils import getParser
from config import Cluster as cf
def prepare_my():
    label = 0
    data = []
    for root, dirs, files in os.walk(cf.cluster_dataset_path):
        if root == cf.cluster_dataset_path:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file1 = os.path.join(os.path.basename(root) + "/", file)
            lines = getParser(file_path)
            size = os.path.getsize(file_path)
            if size == 0:
                continue
            for sub in lines:
                sample = {'label': label, 'file_name': file1,  'value': sub}
                data.append(sample)
        label += 1
        print(label)
    print(label, len(data))
    label = []
    for sample in data:
        label.append(sample['label'])
    with open(os.path.join(cf.cluster_construct_data_path, "cluster.pkl"), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(cf.cluster_construct_data_path, "label.pkl"), 'wb') as f:
        pickle.dump(label, f)
    print("finish")

if __name__ == '__main__':
    # cluster_data = ClusterData(input_dir="E:\Project\python\data_set\cluster_normalize",
    #                            output_dir="E:\Project\python\data_set\other_paper\statement\cluster_construct1",
    #                            type_vocab_dir=None, value_vocab_dir=None,
    #                            type_vocab=None, value_vocab=None)
    prepare_my()

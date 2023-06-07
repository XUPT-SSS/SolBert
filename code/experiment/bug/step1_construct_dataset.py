import logging
import os
import pickle
import random

from tqdm import tqdm
import sys,os
sys.path.append("..")
from utils import  getParser
from config import Bug as cf
logger = logging.getLogger("sol")
def prepare_data(save_file=True,is_random=True):
    # buggy contract
    label_dict = {'A2': 1, 'A10': 2, 'A16': 3, 'B1': 4, 'B4': 5, 'B5': 6, 'B6': 7, 'B7': 8}
    faulty_data = []
    test_data = []
    database_size = 0
    buggy_size = 0
    for root, dirs, files in os.walk(os.path.join(cf.bug_dataset_path, "bug")):
        label = os.path.basename(root)
        if label in label_dict:
            count = 0
            test = []
            fault = []
            for file in tqdm(files):
                file_path = os.path.join(root, file)

                lines = getParser(file_path)
                if count <= len(files) // 2:
                    database_size += 1
                    for line in lines:
                        sample = {'label': label_dict[label], 'file_name': file_path, 'value': line}
                        fault.append(sample)
                else:
                    buggy_size += 1
                    for line in lines:
                        sample = {'label': label_dict[label], 'file_name': file_path, 'value': line}
                        test.append(sample)
                count += 1

            faulty_data += fault
            test_data += test

    if is_random:
        total = faulty_data + test_data
        n = len(total) // 2
        random.shuffle(total)
        faulty_data = total[:n]
        test_data = total[n:]

    validated_contracts = []
    valid_size = 0
    # add validated contracts
    flag = True
    for root, dirs, files in os.walk(os.path.join(cf.bug_dataset_path, "no_bug")):
        if not flag:
            break
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            file1 = os.path.join(os.path.basename(root) + "/", file)
            lines = getParser(file_path)
            valid_size += 1
            if valid_size == buggy_size:
                flag = False
                break
            for line in lines:
                sample = {'label': -1, 'file_name': file1, 'value': line}
                validated_contracts.append(sample)
    # buggy:test1 = 1:1
    validated_contracts = validated_contracts[:len(test_data)]
    test_data += validated_contracts
    random.shuffle(faulty_data)
    random.shuffle(test_data)

    logger.info(
        "Bug embedding matrix size: file number {}, contracts number {}".format(database_size, len(faulty_data)))
    logger.info("Test-Buggy: file number {}, contracts number {}".format(buggy_size, len(test_data)))
    logger.info("Test-Valid: file number {}, contracts number {}".format(valid_size, len(validated_contracts)))

    if save_file is True:
        data_path = os.path.join(cf.bug_construct_data_path, "fault_data1.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(faulty_data, f)
        test_path = os.path.join(cf.bug_construct_data_path, "test_data1.pkl")
        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)
if __name__ == '__main__':
    prepare_data()
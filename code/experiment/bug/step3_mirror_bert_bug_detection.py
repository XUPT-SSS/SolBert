import os
import pickle
import sys

import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np

from bugDataset import BaseDataset as MyDataset
import sys,os
sys.path.append("..")
from config import BertAndToken  as bt
from config import Bug as cf
class BugDetector:
    def __init__(self, output_dir, pooler_type):
        self.pooler_type = pooler_type
        self.gpu = 0
        self.output_dir = output_dir
        self.threshold = 0.95
        self.tokenizer = BertTokenizer.from_pretrained(bt.mirrot_bert)
        self.model = BertForMaskedLM.from_pretrained(bt.mirrot_bert)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.faulty_data = {}
        self.test_data = {}

    def get_line_embeddings(self, line):
        inputs = self.tokenizer(
            line,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.hidden_states[-1]
            embeddings = []
            if self.pooler_type == "avg":
                embeddings = (
                        (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            elif self.pooler_type == "avg_first_last":
                first_hidden = hidden_states[0]
                last_hidden = hidden_states[-1]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                    1) / attention_mask.sum(
                    -1).unsqueeze(-1)
                embeddings = pooled_result
            elif self.pooler_type == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                    1) / attention_mask.sum(-1).unsqueeze(-1)
                embeddings = pooled_result
            elif pooler_type == "cls":
                embeddings = last_hidden[:, 0]
            return embeddings

    def prepare_faulty_emb(self, data_loader):
        for data in tqdm(data_loader):
            # sent_vec = self.model.forward(data['type'], data['value'], test1=True)
            sent_vec = self.get_line_embeddings(data['value'])

            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.faulty_data:
                    self.faulty_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.faulty_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

    def prepare_test_emb(self, data_loader):
        for data in tqdm(data_loader):
            # data = {key: value.to(self.device) for key, value in batch_data.items()}
            sent_vec = self.get_line_embeddings(data['value'])
            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.test_data:
                    self.test_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.test_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

    def similarity_matrix(self, current_vector, embedding_matrix):
        numerator = cdist(current_vector, embedding_matrix)
        vec_norm = np.linalg.norm(embedding_matrix, axis=1)
        vec_tile = np.tile(vec_norm, (current_vector.shape[0], 1))
        emb_norm = np.linalg.norm(current_vector, axis=1)
        emb_tile = np.tile(emb_norm, (embedding_matrix.shape[0], 1)).transpose()
        denominator = np.add(vec_tile, emb_tile)
        similarity_matrix = 1 - np.divide(numerator, denominator)
        return similarity_matrix

    def predict_by_group_similarity(self):

        labels = []

        for idy, fault in self.faulty_data.items():
            labels.append(fault['label'])

        TP, FP, TN, FN = 0, 0, 0, 0
        for idx, test in self.test_data.items():
            all_sim_list = []
            for idy, fault in self.faulty_data.items():
                item_sim = []
                for sub_test in test['sub_contracts']:
                    sub_sim = []
                    for sub_fault in fault['sub_contracts']:
                        temp = torch.cosine_similarity(sub_test, sub_fault, dim=0)
                        sub_sim.append(temp)
                    avg_sim = torch.max(torch.stack(sub_sim))
                    item_sim.append(avg_sim)
                all_sim_list.append(torch.mean(torch.stack(item_sim)))
            max_sim = max(all_sim_list)
            predict_idx = all_sim_list.index(max(all_sim_list))
            predict_label = labels[predict_idx].item()
            true_label = test['label'].item()
            if max_sim > self.threshold:
                if true_label != -1:
                    TP += 1
                else:
                    FP += 1
            else:
                if true_label == -1:
                    TN += 1
                else:
                    FN += 1

        accuracy = TP + TN / (TP + TN + FP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print("tp:{},fp:{},fn:{},tn:{}".format(TP, FP, FN, TN))
        print("precision:{},recall:{},f1:{}".format(precision, recall, f1))

    def save_emb(self):
        fault_path = os.path.join(self.output_dir, "fault_emb.pkl")
        test_path = os.path.join(self.output_dir, "test_emb.pkl")

        with open(fault_path, "wb") as f:
            pickle.dump(self.faulty_data, f)

        with open(test_path, "wb") as f:
            pickle.dump(self.test_data, f)

    def load_emb(self):
        fault_path = os.path.join(self.output_dir, "fault_emb.pkl")
        test_path = os.path.join(self.output_dir, "test_emb.pkl")

        if os.path.exists(fault_path):
            with open(fault_path, 'rb') as f:
                self.faulty_data = pickle.load(f)

            with open(test_path, 'rb') as f:
                self.test_data = pickle.load(f)


if __name__ == '__main__':
    # todo
    test_data = MyDataset(data_path=os.path.join(cf.bug_construct_data_path,"test_data1.pkl"))
    test_data_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
    # todo
    fault_data = MyDataset(data_path=os.path.join(cf.bug_construct_data_path,"fault_data1.pkl"))
    fault_data_loader = DataLoader(dataset=fault_data, batch_size=64, shuffle=True)
    # todo
    args = sys.argv
    if len(args) == 1:
        pooler_type = "avg_first_last"  # avg_first_last
    else:
        pooler_type = args[1]
    detector = BugDetector(cf.bug_construct_data_path, pooler_type=pooler_type)

    detector.prepare_faulty_emb(data_loader=fault_data_loader)

    detector.prepare_test_emb(data_loader=test_data_loader)
    # detector.save_emb()
    detector.predict_by_group_similarity()

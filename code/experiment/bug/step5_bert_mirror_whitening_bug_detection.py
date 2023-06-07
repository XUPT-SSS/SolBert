import numpy as np
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import  BertForMaskedLM, BertTokenizer
sys.path.append("..")
from config import BertAndToken  as bt
from config import Bug as cf
import sys
from bugDataset import BaseDataset as MyDataset

class BugDetector:
    def __init__(self, pooler_typ):
        f = open(os.path.join(cf.bert_mirror_whitening_train_output,"kernel.pkl"), 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open(os.path.join(cf.bert_mirror_whitening_train_output,"bias.pkl"), 'rb')
        self.bias = pickle.load(f)
        f.close()
        self.pooler_type = pooler_typ
        self.gpu = 0
        self.output_dir = ""
        self.threshold = 0.95
        self.tokenizer = BertTokenizer.from_pretrained(bt.mirrot_bert)
        self.model = BertForMaskedLM.from_pretrained(bt.mirrot_bert)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.faulty_data = {}
        self.test_data = {}

    def compute_kernel_bias(self,vecs, n_components):
        """计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(s ** 0.5))
        W = np.linalg.inv(W.T)
        W = W[:, :n_components]
        return W, -mu

    def transform_and_normalize(self,vecs, kernel, bias):
        """应用变换，然后标准化
        """
        vecs = vecs.cpu().numpy()
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def get_line_embeddings(self, line,USE_WHITENING=True):
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
            # return embeddings
            # return output_hidden_state.cpu().numpy()[0]
            return embeddings
    def bert_whitening_embeddings(self,line):
            vecs = []
            embdeeings = self.get_line_embeddings(line)
            for embdeeing in embdeeings:
                body = self.transform_and_normalize(embdeeing, self.kernel, self.bias)
                # print(len(body))
                vecs.extend(body)
            return vecs
    def prepare_faulty_emb(self, data_loader):
        for data in tqdm(data_loader):
            # sent_vec = self.model.forward(data['type'], data['value'], test=True)
            # sent_vec = self.get_line_embeddings(data['value'])
            sent_vec = self.bert_whitening_embeddings(data['value'])
            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.faulty_data:
                    self.faulty_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.faulty_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

    def prepare_test_emb(self, data_loader):
        for data in tqdm(data_loader):
            # data = {key: value.to(self.device) for key, value in batch_data.items()}
            # sent_vec = self.get_line_embeddings(data['value'])
            sent_vec = self.bert_whitening_embeddings(data['value'])
            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.test_data:
                    self.test_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.test_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

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
                        sub_test1 = torch.from_numpy(sub_test)
                        sub_fault1 = torch.from_numpy(sub_fault)
                        # temp = torch.dist(sub_test1, sub_fault1, p=2)
                        temp = torch.cosine_similarity(sub_test1, sub_fault1, dim=0)
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

        print(accuracy, precision, recall, f1)
        print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP,FP,TN,FN))

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
    test_data = MyDataset(data_path=os.path.join(cf.bug_construct_data_path,"/test_data.pkl"))
    test_data_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
    # todo
    fault_data = MyDataset(data_path=os.path.join(cf.bug_construct_data_path,"/fault_data.pkl"))
    fault_data_loader = DataLoader(dataset=fault_data, batch_size=64, shuffle=True)
    # todo
    args = sys.argv
    if len(args) == 1:
        pooler_type = "avg_first_last" #ooler cls
    else:
        pooler_type = args[1]
    detector = BugDetector(pooler_typ=pooler_type)

    detector.prepare_faulty_emb(data_loader=fault_data_loader)

    detector.prepare_test_emb(data_loader=test_data_loader)
    # detector.save_emb()
    detector.predict_by_group_similarity()



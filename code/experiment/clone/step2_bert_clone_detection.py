import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel
from BaseDataset import BaseDataset

import sys,os
sys.path.append("..")
from config import BertAndToken  as bt
from config import Clone as cf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bt.tokenzier_model)
model = AutoModel.from_pretrained(bt.bert_train_output)
model.to(device)


class CloneDetector:
    def __init__(self, output_dir,):
        self.output_dir = output_dir
        self.threshold = 0.95

    def get_sentence_embedding(self,sentece, pooler_type):
        inputs = tokenizer(
            [sentece],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.hidden_states[-1]
            embeddings = []
            if pooler_type == "avg":
                embeddings = (
                            (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            elif pooler_type == "avg_first_last":
                first_hidden = hidden_states[0]
                last_hidden = hidden_states[-1]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                    1) / attention_mask.sum(
                    -1).unsqueeze(-1)
                embeddings = pooled_result
            elif pooler_type == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                    1) / attention_mask.sum(-1).unsqueeze(-1)
                embeddings = pooled_result
            elif pooler_type =="cls":
                embeddings = last_hidden[:, 0]
            return embeddings

    def get_vectors(self, data_loader,pooler_type):
        res = []
        for data in tqdm(data_loader):
            datas = []
            for i in range(len(data['value_scbert'])):
                datas.append(data['value_scbert'][i][0])
            datas =  " ".join(str(i) for i in datas)
            vectors = self.get_sentence_embedding(datas,pooler_type)
            res += vectors

        return res

    def laod_label(self,data_path):
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                return data

    def  clone_detection(self, data_loader_1, data_loader_2, label, threshold,pooler_type):

        vectors1 = self.get_vectors(data_loader_1,pooler_type)
        vectors2 = self.get_vectors(data_loader_2,pooler_type)
        length = len(vectors1)
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(length):
            dis = torch.cosine_similarity(vectors1[i], vectors2[i], dim=0)
            
            if dis >= threshold:
                if label[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if label[i] == 1:
                    FN += 1
                else:
                    TN += 1

        print(TP, FP, TN, FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print("P:", precision, "R:", recall, "F1-Score:", f1)
if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        pooler_type = "avg_top2"  # avg,avg_top2,avg_first_last
    else:
        pooler_type = args[1]
    # pooler_type = "cls"  # cls_before_pooler
    threshold = 0.95
    cloneDetector = CloneDetector(output_dir="")

    col1_path = os.path.join(cf.clone_construct_data_path,"col1.pkl")
    col2_path = os.path.join(cf.clone_construct_data_path,"col2.pkl")
    label = cloneDetector.laod_label(os.path.join(cf.clone_construct_data_path,"label.pkl"))
    col1 = BaseDataset(col1_path)
    col2 = BaseDataset(col2_path)
    loader_1 = DataLoader(dataset=col1, batch_size=1)
    loader_2 = DataLoader(dataset=col2, batch_size=1)
    label1 = label
    cloneDetector.clone_detection(loader_1, loader_2, label1, threshold=0.95,pooler_type=pooler_type)
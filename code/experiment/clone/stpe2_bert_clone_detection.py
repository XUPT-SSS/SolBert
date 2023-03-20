import os
import pickle
from tqdm import tqdm
import torch
from transformers import BertTokenizer, AutoModel
import sys,os
sys.path.append(os.getcwd())
from code.experiment.config import BertAndToken  as bt
from code.experiment.config import Clone as cf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bt.tokenzier_model)
model = AutoModel.from_pretrained(bt.bert_train_output)
model.to(device)
#add

def get_sentence_embedding(sentece, pooler_type):
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
            embeddings = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
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
        return embeddings


def get_clone_data(path):
    if os.path.exists(os.path.join(path, "data/clone_data.pkl")):
        with open(os.path.join(path, "data/clone_data.pkl"), 'rb') as f:
            samples = pickle.load(f)
    return samples


def get_embeddings(path,  pooler_type):
        clone_embeddings = []
        samples = get_clone_data(path)
        for sample in tqdm(samples):
            if sample['value1'] == [] or sample['value2'] == []:
                continue
            embedding1 = get_sentence_embedding(sample['value1'][0], pooler_type)
            embedding2 = get_sentence_embedding(sample['value2'][0], pooler_type)
            embeddings = {'label': sample['label'], 'embedding1': embedding1, 'embedding2': embedding2}
            clone_embeddings.append(embeddings)
        return clone_embeddings


def clone_detection(input_dir,  pooler_type, threshold):
    vectors1 = get_embeddings(input_dir,  pooler_type)
    print("calculate embeddings finish")
    print("*"*20)
    print("calculate similarity")
    length = len(vectors1)
    print(length)
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in tqdm(range(length)):
        dis = torch.cosine_similarity(vectors1[i]['embedding1'][0], vectors1[i]['embedding2'][0], dim=0)
        if dis >= threshold:
            if vectors1[i]['label'] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if vectors1[i]['label'] == 1:
                FN += 1
            else:
                TN += 1

    print(TP, FP, TN, FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("embedding: ",pooler_type)
    print("P:", precision, "R:", recall, "F1-Score:", f1)


if __name__ == '__main__':
    clone_path = cf.clone_construct_data_path
    # pooler_type = "cls_before_pooler"  #cls
    args = sys.argv
    if len(args) == 1:
        pooler_type = "avg_first_last"  # cls_before_pooler cls
    else:
        pooler_type = args[1]
    threshold = 0.95
    clone_detection(clone_path,  pooler_type, threshold)

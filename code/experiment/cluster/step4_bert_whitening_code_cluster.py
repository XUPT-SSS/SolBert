import os
import pickle

import numpy
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertForMaskedLM
import sys
import sys,os
sys.path.append("..")
from config import BertAndToken  as bt
from config import Cluster as cf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bt.tokenzier_model)
model = AutoModel.from_pretrained(bt.bert_train_output)
model.to(device)

f = open(os.path.join(cf.bert_whitening,"kernel.pkl"), 'rb')
kernel = pickle.load(f)
f.close()
f = open(os.path.join(cf.bert_whitening,"bias.pkl"), 'rb')
bias = pickle.load(f)
f.close()

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
        return embeddings.cpu().numpy()


def get_cluster_data(path):
    if os.path.exists(os.path.join(path, "cluster.pkl")):
        with open(os.path.join(path, "cluster.pkl"), 'rb') as f:
            lines = pickle.load(f)
    if os.path.exists(os.path.join(path, "label.pkl")):
        with open(os.path.join(path, "label.pkl"), 'rb') as f:
            labels = pickle.load(f)
    return lines, labels

def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    #vecs = vecs.cpu().numpy()
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def cluster_k_means(true_label, data,pooler_type):
    class_num = len(set(true_label))
    true_label = numpy.array(true_label)
    #data = data.squeeze(-2)
    #data = transform_and_normalize(data,kernel,bias)
    data = numpy.array(data)

    len_label = len(true_label)
    data = data.reshape((len_label,512))
    clf = KMeans(n_clusters=class_num, max_iter=100, init="k-means++", tol=1e-6)
    _ = clf.fit(data)
    # source = list(clf.predict(data))
    predict_label = clf.labels_

    ARI = metrics.adjusted_rand_score(true_label, predict_label)
    print("embedding_type:",pooler_type)
    print("adjusted_rand_score: ", ARI)


def get_embeddings(path, output_dir,pooler_type):
    if os.path.exists(os.path.join(output_dir, "cluster_embedding"+pooler_type+".pkl")):
        with open(os.path.join(path, "cluster_embedding"+pooler_type+".pkl"), 'rb') as f:
            cluster_embeddings = pickle.load(f)
        with open(os.path.join(path, "cluster_label"+pooler_type+".pkl"), 'rb') as f:
            labels = pickle.load(f)
        return cluster_embeddings, labels
    else:
        cluster_embeddings = []
        labels_new = []
        samples, labels = get_cluster_data(path)
        for sample in samples:
            embedding = get_sentence_embedding(sample['value'], pooler_type)
            embedding = transform_and_normalize(embedding, kernel, bias)
            cluster_embeddings.extend(embedding)
            labels_new.append(sample['label'])
        #with open(os.path.join(output_dir, "cluster_embedding"+pooler_type+".pkl"), 'wb') as f:
        #    pickle.dump(cluster_embeddings, f)
        #with open(os.path.join(output_dir, "cluster_label"+pooler_type+".pkl"), 'wb') as f:
        #    pickle.dump(labels_new, f)
        return cluster_embeddings, labels


if __name__ == '__main__':
    cluster_path = cf.cluster_construct_data_path
    output_dir = cluster_path
    args = sys.argv
    if len(args)==1:
        pooler_type = "avg_first_last"  # avg,avg_top2,avg_first_last
    else :
        pooler_type =args[1]
    embeddings, labels = get_embeddings(cluster_path, output_dir,pooler_type)
    # print(labels)
    cluster_k_means(labels, embeddings,pooler_type)

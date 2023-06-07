import os
import pickle

import numpy
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel
import sys,os
sys.path.append("..")
from config import BertAndToken  as bt
from config import Cluster as cf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bt.mirrot_bert)
model = AutoModel.from_pretrained(bt.mirrot_bert)
model.to(device)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        elif pooler_type=="cls":
            embeddings = last_hidden[:, 0]
        return embeddings.cpu().numpy()


def get_cluster_data(path):
    if os.path.exists(os.path.join(path, "cluster.pkl")):
        with open(os.path.join(path, "cluster.pkl"), 'rb') as f:
            lines = pickle.load(f)
    if os.path.exists(os.path.join(path, "label.pkl")):
        with open(os.path.join(path, "label.pkl"), 'rb') as f:
            labels = pickle.load(f)
    return lines, labels

def fun(vector,corpus):
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    embedd = tsne.fit_transform(vector)
    plt.figure(figsize=(14, 7))
    plt.rcParams['savefig.dpi'] = 2000  # 图片像素
    plt.rcParams['figure.dpi'] = 2000
    for i, label in enumerate(corpus):
        x, y = embedd[i, :]
        plt.scatter(x, y)
    for i in range(len(corpus)):
        x = embedd[i][0]
        y = embedd[i][1]
        plt.text(x, y, corpus[i].replace("%%%%", ""))
    # plt.savefig(FIGURE_PATH_NAME, format=FIGURE_FORMAT)
    plt.show()

def cluster_k_means(true_label, data):
    class_num = len(set(true_label))
    true_label = numpy.array(true_label)
    data = numpy.array(data)
    len_label = len(true_label)
    data = data.reshape((len_label,768))
    clf = KMeans(n_clusters=class_num, max_iter=100, init="k-means++", tol=1e-6)
    _ = clf.fit(data)
    # source = list(clf.predict(data))
    predict_label = clf.labels_

    ARI = metrics.adjusted_rand_score(true_label, predict_label)
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
        for sample in tqdm(samples):
            embedding = get_sentence_embedding(sample['value'], pooler_type)
            cluster_embeddings.append(embedding)
            labels_new.append(sample['label'])
        with open(os.path.join(output_dir, "cluster_embedding"+pooler_type+".pkl"), 'wb') as f:
            pickle.dump(cluster_embeddings, f)
        with open(os.path.join(output_dir, "cluster_label"+pooler_type+".pkl"), 'wb') as f:
            pickle.dump(labels_new, f)
        return cluster_embeddings, labels


if __name__ == '__main__':
    cluster_path = cf.cluster_construct_data_path
    output_dir = cluster_path
    pooler_type = "avg_first_last"  # avg,avg_top2,avg_first_last
    embeddings, labels = get_embeddings(cluster_path, output_dir,pooler_type)
    cluster_k_means(labels, embeddings)

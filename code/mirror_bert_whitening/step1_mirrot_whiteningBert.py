import torch
import numpy as np
from tqdm import tqdm
from transformers import  BertForMaskedLM, BertTokenizer
import os
import  sys
sys.path.append("..")
from whiteningDataset import BaseDataset as MyDataset
from torch.utils.data import DataLoader
import config as cf
POOLING = 'avg_first_last'
# POOLING = 'last_avg':
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 512
MAX_LENGTH = 512

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model():
    tokenizer = BertTokenizer.from_pretrained(cf.mirrot_bert)
    model = BertForMaskedLM.from_pretrained(cf.mirrot_bert)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []

    with torch.no_grad():
        # for sent in sents:
        sents_data = MyDataset(sents)
        fault_data_loader = DataLoader(dataset=sents_data, batch_size=200, shuffle=True)
        # for sent in tqdm(sents):
        for sent in tqdm(fault_data_loader):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'avg_first_last':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'avg_top2':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            elif POOLING == 'cls':
                output_hidden_state = hidden_states[-1][:, 0]
            else:
                raise Exception("unknown pooling {}".format(POOLING))
            # output_hidden_state [batch_size, hidden_size]
            vec = output_hidden_state.cpu().numpy()
            vecs.extend(vec)
    # assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    print(vecs.shape)
    return vecs


def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def read_train_dataset(file_path):
    with open(file_path, encoding="utf-8") as f:
        return f.read().splitlines()

def main():
    tokenizer, model = build_model()
    print("Building {} tokenizer and model successfuly.".format("solidit bert"))
    code_list = read_train_dataset(cf.target_data_path)
    print(len(code_list))
    print("Transfer sentences to BERT vectors.")
    vecs_func_body = sents_to_vecs(code_list, tokenizer, model) # [code_list_size, 768]
    kernel = []
    bias = []
    if USE_WHITENING:
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            vecs_func_body
        ], n_components=N_COMPONENTS)
        vecs_func_body = transform_and_normalize(vecs_func_body, kernel, bias) # [code_list_size, dim]
    else:
        vecs_func_body = normalize(vecs_func_body)# [code_list_size, 768]
    print(vecs_func_body.shape)
    import pickle
    f = open(os.path.join(cf.bert_mirror_whitening_train_output, '/kernel.pkl'), 'wb')
    pickle.dump(kernel, f)
    f.close()
    f = open(os.path.join(cf.bert_mirror_whitening_train_output, "/bias.pkl"), 'wb')
    pickle.dump(bias, f)
    f.close()


if __name__ == "__main__":
    main()

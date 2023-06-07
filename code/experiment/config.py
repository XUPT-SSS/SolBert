class BertAndToken():
    tokenzier_model="/code/solidity_tokenize"
    bert_train_output = "/code/solidity_model"
    mirrot_bert="/code/mirror-bert/tmp/mirror_bert_mean"
    bert_whitening="/code/bert_whitening/data/"
    bert_mirror_whitening_train_output = "/code/mirror_bert_whitening/data/"
#clone path
class Clone():
    #original clone data
    colne_dataset_path="/dataset/experiment/clone//"
    # construct clone detection data
    clone_construct_data_path="/code/experiment//clone//data//"
    clone_construct_clone_embedding_data_path="/code/experiment//clone//data//"
class Cluster():
    #original clone data
    cluster_dataset_path="/code/experiment//cluster/data"
    # construct clone detection data
    cluster_construct_data_path="/code/experiment//cluster//data//"
class Bug():
    #original clone data
    bug_dataset_path="/code/experiment//bug//data//"
    # construct clone detection data
    bug_construct_data_path="/code//experiment//bug//data//"

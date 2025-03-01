import os

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import BertForTokenClassification

from code.utils.TTPObjective import BertForTokenTypePrediction
from utils.TTPObjective import BertTTPHead

import config as config
from utils import data_prepare

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_MODE"] = "dryrun"
CUDA_VISIBLE_DEVICES=0
def get_bert_config():
    config = BertConfig(
        vocab_size=32000,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=8,
        type_vocab_size=1)  # 指代 token_type_ids 的类别
    return config

def get_bert_model_ttp():
    bert_config = get_bert_config()
    bert_tokenizer = BertTokenizer.from_pretrained(config.tokenzier_model)
    bert_model = BertForTokenTypePrediction.from_pretrained(config=bert_config, num_labels=2)

    dataset = data_prepare.TokenDataset(tokenizer=bert_tokenizer,
                                    file_path=config.target_data_path)

    trainArgs = TrainingArguments(
        output_dir=config.bert_train_output,  # 输出路径
        overwrite_output_dir=True,  # 可以覆盖之前的输出
        do_train=True,  # 训练
        num_train_epochs=config.train_epochs,
        per_device_train_batch_size=config.batch_size,
        save_steps=500,
        local_rank=-1,
        save_total_limit=2)
    trainer = Trainer(
        model=bert_model,  # 模型对象
        args=trainArgs,  # 训练参数
        train_dataset=dataset,# 数据集
    )
    return trainer

def get_bert_model_mlm():
    bert_config = get_bert_config()
    bert_tokenizer = BertTokenizer.from_pretrained(config.tokenzier_model)
    bert_model = BertForMaskedLM(config=bert_config)
    # dataset = MyIterableDataset(tokenizer=bert_tokenizer, file_path=config.train_data_path)
    dataset = LineByLineTextDataset(tokenizer=bert_tokenizer,  # 分词器
                                    file_path=config.target_data_path,  # 文本数据
                                    block_size=config.batch_size)# 每批读取128行
    data_collector = DataCollatorForLanguageModeling(tokenizer=bert_tokenizer,  # 分词器
                                                     mlm=True,  # mlm模型
                                                     mlm_probability=0.15)
    trainArgs = TrainingArguments(
        output_dir=config.bert_train_output,  # 输出路径
        overwrite_output_dir=True,  # 可以覆盖之前的输出
        do_train=True,  # 训练
        num_train_epochs=config.train_epochs,
        per_device_train_batch_size=config.batch_size,
        save_steps=500,
        local_rank=-1,
        save_total_limit=2)
    trainer = Trainer(
        model=bert_model,  # 模型对象
        args=trainArgs,  # 训练参数
        data_collator=data_collector,  # collector
        train_dataset=dataset,# 数据集

    )
    return trainer


if __name__ == '__main__':
    for epoch in range(config.total_epochs):
        print(f"Epoch {epoch + 1}")
        trainer_mlm = get_bert_model_mlm()
        print("Training MLM...")
        trainer_mlm.train()
        trainer_mlm.save_model(config.bert_train_output)
        print("Training TTP...")
        trainer_ttp = get_bert_model_ttp()
        trainer_ttp.train()
        trainer_ttp.save_model(config.bert_train_output)

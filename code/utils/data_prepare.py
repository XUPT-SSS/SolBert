import os

import torch
import random

from transformers import PreTrainedTokenizer


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        with open(file_path, encoding="utf-8") as f:
            texts = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.input_ids, self.attention_mask, self.labels = prepare_data(texts, tokenizer, mask_prob=0.1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def prepare_data(texts, tokenizer, mask_prob=0.1):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 初始化标签
    labels = torch.full_like(input_ids, -100)  # 初始化为-100，表示不预测的位置

    for i in range(input_ids.size(0)):
        num_tokens = attention_mask[i].sum().item()
        num_to_mask = max(1, int(num_tokens * mask_prob))
        mask_indices = random.sample(range(num_tokens), num_to_mask)

        token_types = [1 if is_code_token(token) else 0 for token in tokenizer.convert_ids_to_tokens(input_ids[i])]

        for idx in mask_indices:
            labels[i, idx] = token_types[idx]

    return input_ids, attention_mask, labels

from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
import config as config

if __name__ == '__main__':
    tokenize = BertWordPieceTokenizer()
    train_data_path = config.target_data_path
    tokenzier_model_path = config.tokenzier_model

    tokenize.train(
        files=[train_data_path],
        vocab_size=32000,
        min_frequency=2,
        limit_alphabet=100,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenize.save_model(config.tokenzier_model)
    tokenize.post_processor = BertProcessing(
        ("[CLS]", tokenize.token_to_id("[CLS]")),
        ("[SEP]", tokenize.token_to_id("[SEP]")))
    tokenize.enable_truncation(max_length=512)
    tokenize.save(tokenzier_model_path + "/config.json")

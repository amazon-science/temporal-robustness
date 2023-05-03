import os
from transformers import AutoModel, AutoTokenizer

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# CKPT_DIR = os.path.join(BASE_DIR, 'pretrained_models')
# RES_DIR = os.path.join(BASE_DIR, 'new_results')
# LOG_DIR = os.path.join(BASE_DIR, 'new_logs')
# CACHE_DIR = os.path.join(BASE_DIR, 'cached')

# available_datasets = {
#     "lama-conceptnet": {
#         "data_dir" : os.path.join(DATA_DIR, "lama", "ConceptNet"),
#         "temporal": None
#     },
#     "lama-google-re": {
#         "data_dir": os.path.join(DATA_DIR, "lama", "Google_RE"),
#         "temporal": None
#     },
#     "lama-squad": {
#         "data_dir": os.path.join(DATA_DIR, "lama", "Squad"),
#         "temporal": None
#     },
#     "lama-trex": {
#         "data_dir": os.path.join(DATA_DIR, "lama", "TREx"),
#         "temporal": None
#     },
#     "templama": {
#         "data_dir": os.path.join(DATA_DIR, "templama"),
#         "temporal": ['2010', '2011','2012','2013','2014','2015','2016','2017', '2018', '2019', '2020']
#     }
# }

LMs_names = [
    #############################
    # BERT
    #############################
    "bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased",
    #############################
    # RoBERTa
    #############################
    "roberta-base", "roberta-large",
    #############################
    # Twitter BERT
    #############################
    "bertweet",
    #############################
    # Twitter RoBERTa
    #############################
    "cardiffnlp/twitter-roberta-base", "cardiffnlp/twitter-roberta-base-2021-124m",
    #############################
    # TimeLMs (Twitter RoBERTa)
    #############################
    # 2019                                   
    "cardiffnlp/twitter-roberta-base-2019-90m",
    # 2020
    "cardiffnlp/twitter-roberta-base-mar2020", "cardiffnlp/twitter-roberta-base-jun2020", "cardiffnlp/twitter-roberta-base-sep2020", "cardiffnlp/twitter-roberta-base-dec2020",
    # 2021                                   
    "cardiffnlp/twitter-roberta-base-mar2021", "cardiffnlp/twitter-roberta-base-jun2021", "cardiffnlp/twitter-roberta-base-sep2021", "cardiffnlp/twitter-roberta-base-dec2021",
    # 2022                                   
    "cardiffnlp/twitter-roberta-base-mar2022", "cardiffnlp/twitter-roberta-base-jun2022",


]
LMs = {}

for lm in LMs_names:
    # print(lm)
    # tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)
    # vocab_size = tokenizer.vocab_size
    # ids2tokens = tokenizer.ids_to_tokens
    # vocab = tokenizer.vocab  # tokens2ids
    # mask = tokenizer.mask_token
    # mask_id = vocab[mask]
    if 'roberta' in lm:
        tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False, add_prefix_space=True)
        tokens2ids = tokenizer.encoder
        ids2tokens = tokenizer.decoder
        special_ids = tokenizer.all_special_ids
    elif 'bertweet' in lm:
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        tokens2ids = tokenizer.encoder
        ids2tokens = tokenizer.decoder
        special_ids = tokenizer.all_special_ids
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)
        tokens2ids = tokenizer.vocab
        ids2tokens = tokenizer.ids_to_tokens
        special_ids = tokenizer.all_special_ids
    LMs[lm] = {
        "tokenizer": tokenizer,
        "tokens2ids": tokens2ids,
        "ids2tokens": ids2tokens,
        "mask_token": tokenizer.mask_token,
        "vocab_size": len(tokens2ids),
        "max_seq_len": tokenizer.model_max_length,
        'special_ids': special_ids
    }
    # print()
    # print(LMs[lm]["vocab_size"])



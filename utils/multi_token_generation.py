"""
We follow the work "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model" to do
the multi-token generation

We also refer to https://github.com/nyu-dl/bert-gen for the implementation.
"""
import argparse
import os
import sys

import torch
from collections import Counter
from transformers import AutoModel, AutoTokenizer, pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd
import evaluate
from sklearn.metrics import f1_score

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True, num_masks=0):
    """ Generate a word from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if num_masks==1:  # if single-token then return the argmax
        idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx
    else:
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx

def tokenizer_return_id(tokenizer, text, filter_special_tokens=False):
    """
    Text to token ids for a string.
    """
    output = tokenizer(text)
    if filter_special_tokens:
        token_ids = [i for i in output['input_ids'] if i not in tokenizer.all_special_ids]
    else:
        token_ids = [i for i in output['input_ids'] ]
    return token_ids

# def tokenize_batch(tokenizer, batch):
#     """
#     Text to token ids for a list of strings.
#     """
# #     return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]
#     return [tokenizer_return_id(tokenizer, sent) for sent in batch]

def untokenize_id(tokenizer, ids):
    """
    Token ids to strings.
    """
#     return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]
    return [tokenizer.decode(_id) for _id in ids]

# def untokenize_batch(tokenizer, batch, special_ids, filter_special_tokens=False):
#     """
#     Token ids to strings for a list of ids.
#     """
# #     return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]
# #     print(label_id)
#     if filter_special_tokens:
#         _batch = []
#         for sent in batch:
#             _batch.append([x for x in sent if x not in special_ids])
#         batch = _batch
# #         return [tokenizer.decode(label_id) for label_id in batch if label_id not in special_ids]
# #     else:
#     return [tokenizer.decode(label_id) for label_id in batch]

def multi_token_evaluation(tokenizer, fill_mask_model, text_list, labels_list, labels_ids_list,
                           relation_list, num_answers_list,
                           save_dir,
                           temperature=1.0, N=100, M=5,
                           mask_id=None, quarter=None, model_name=None, split=None, seed=1210,
                           ):
    """
        Generate multiple tokens for a masked input with BERT-GEN.

    For example, given the input "Cristiano Ronaldo plays for <mask> <mask>."
    the function outputs a list of two tokens with their corresponding log probabilities.

    Args:
        tokenizer: tokenizer
        fill_mask_model: model checkpoint (fill-mask pipeline tye from HuggingFace)
        text_list: list of test examples (text)
        labels_list: list of labels (tokens)
        labels_ids_list: list of labels ids (token ids from vocabulary)
        relation_list: list of relation for each example (see templates.csv)
        num_answers_list: list of ints with number of correct answers per test example
        temperature: temperature to divide logits (set to 1.0 originally)
        N: number of 'shots' (batch_size)
        M: max number of masks
        mask_id: mask token id
        quarter: quarter (e.g. '2019-Q1')
        model_name: model checkpoint (e.g. 'cardiffnlp/twitter-roberta-base-jun2022')
        split: fine-grained split (e.g. 'unchanged')
        seed: seed to set for sampling

    Returns:

    """
    torch.manual_seed(seed)

    model_name = model_name.replace('cardiffnlp/', '')

    model = fill_mask_model.model
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    # N = batch_size  # number of "shots" for each test example
    # M = args.max_num_masks

    json_dict = {
        'model': model_name,
        'quarter': quarter,
        'split': split,
        'shots': N,
        'max_num_masks': M,
        'text': [],                 # list of size len(text_list)
        'gold_label': [],           # list of size len(text_list)
        'gold_num_masks': [],       # list of size len(text_list)
        'relation': [],             # list of size len(text_list)
        'num_answers': [],          # list of size len(text_list)
        'f1_micro': [],             # list of size len(text_list)
        'f1_macro': [],             # list of size len(text_list)
        'rouge': [],                # list of size len(text_list)
        'bleu': [],                 # list of size len(text_list)
        'bleu_uni_precision': [],   # list of size len(text_list)
        'bert_score': [],           # list of size len(text_list)
        'best_log_probs': [],       # list of size len(text_list)
        'best_pred_tokens': [],     # list of size len(text_list)
        'best_pred_strings': [],     # list of size len(text_list)
        'all_log_probs': [],  # dict
        'all_preds': [],      # dict

    }
    if cuda is None: cuda = torch.cuda.is_available()
    if mask_id is None: mask_id = tokenizer.mask_token_id

    # create all M combinations of masks [1,M]
    all_M_mask_combos = []
    for m in range(1, M + 1):
        all_M_mask_combos.append([mask_id for _ in range(m)])

    # all_log_probs_dict, all_pred_tokens_list = {}, {}

    for i in tqdm(range(0, len(text_list))):
    # for i in tqdm(range(0, 5)):  # DEBUGGING
        text_i = text_list[i]
        # text_i = ['Lionel M plays for <mask> <mask>.'][0]
        labels_i = labels_list[i]
        labels_ids_i = labels_ids_list[i]  # there is an 'extra' list that is why we put [0]
        relation_i = relation_list[i]
        num_answers_i = num_answers_list[i]  # this is not true anymore as we expand the test set
        # print('Example: {}, Labels: {}, Ids: {}'.format(text_i, labels_i, labels_ids_i))

        # tokenize text
        tokenized_sentence_orig = tokenizer_return_id(tokenizer, text_i)

        # mask indices to generate
        # mask_inds_orig = list(np.where(np.array(tokenized_sentence_orig) == mask_id)[0])
        mask_inds_orig = int(np.where(np.array(tokenized_sentence_orig) == mask_id)[0][0])

        # number of masks (tokens)
        # gold_num_masks = len(mask_inds_orig)

        # save to json dict
        json_dict['text'].append(text_i)
        json_dict['gold_label'].append(labels_i)
        # json_dict['gold_num_masks'].append(gold_num_masks)
        json_dict['gold_num_masks'].append(len(labels_ids_i))
        json_dict['relation'].append(relation_i)
        json_dict['num_answers'].append(num_answers_i)

        # find first and last masks
        # first_mask_idx = mask_inds_orig[0]
        first_mask_idx = mask_inds_orig
        # last_mask_idx = mask_inds_orig[-1]
        last_mask_idx = mask_inds_orig

        # split tokenized sentence to list of token ids before and after the masks
        before_mask_ids = tokenized_sentence_orig[:first_mask_idx]
        after_mask_ids = tokenized_sentence_orig[last_mask_idx + 1:]

        assert mask_id not in before_mask_ids
        assert mask_id not in after_mask_ids

        # add all M mask combos to list of token ids
        tokenized_sentence_list = []
        for mask_seq in all_M_mask_combos:
            tokenized_sentence_list.append(
                before_mask_ids + mask_seq + after_mask_ids)  # list of M lists (variable number of masks)

        # We do not know a priori the correct number of masks so we try all in range 1, ..., M (M=5)
        # tokenized_sentence_b_np, tokenized_sentence_b, logits_m, logits_m = [], [], [], []
        top_ranked_gen_token_seqs, top_ranked_log_probs = [], []
        all_ranked_gen_token_seqs, all_ranked_log_probs = [], []

        for num_mask_j in range(M):  # for each mask 1...M
            generated_seq_tokens = [[] for _ in range(N)]  # for each trial 1...N
            logits_list = [[] for _ in range(N)]           # for each trial 1...N

            # print('##' * 11)
            # print('We try with {}  mask(s)'.format(num_mask_j + 1))
            # print('##' * 11)
            tokenized_sentence = tokenized_sentence_list[num_mask_j]
            # print(tokenized_sentence)

            # list of indices of the mask tokens
            mask_inds = list(np.where(np.array(tokenized_sentence) == mask_id)[0])

            # create batch (same input, N times)
            tokenized_sentence_b = [tokenized_sentence for _ in range(N)]

            for m in mask_inds:
                with torch.no_grad():
                    # tensor
                    inp = torch.tensor(tokenized_sentence_b).cuda() if cuda else torch.tensor(tokenized_sentence_b)

                    # get logits
                    out = model(inp)
                    logits = out.logits.detach().cpu()  # batch_size x max_len x vocab

                    if len(mask_inds) == 1:  # single-token
                        # find logits
                        logits_m = logits[:, m]  # logits for the mask in position m

                        # add generated tokens and corresponding logits to lists
                        topk_logits, topk_inds = logits_m.topk(N, dim=-1)
                        idxs_b = topk_inds.numpy()[0].tolist()
                        logits_b = topk_logits.numpy()[0].tolist()

                        generated_seq_tokens = [[x] for x in idxs_b]
                        logits_list = [[x] for x in logits_b]

                    else:  # multi-token
                        # get new ids
                        idxs_b = generate_step(
                            logits, gen_idx=m, top_k=10, temperature=temperature,  # sample=(m < burnin)
                                               )

                        # replace mask with predicted token id
                        tokenized_sentence_b_np = np.array(tokenized_sentence_b)
                        tokenized_sentence_b_np[:, m] = np.array(idxs_b)
                        tokenized_sentence_b = tokenized_sentence_b_np.tolist()

                        # # the following code does not work *and I don't know why*
                        # # for jj in range(len(idxs_b)):
                        # #     print('before: {}, predicted token id: {}'.format(tokenized_sentence_b[jj][m],idxs_b[jj]))
                        # #     tokenized_sentence_b[jj][m] = idxs_b[jj]
                        #
                        assert sorted(idxs_b) == sorted([sent[m] for sent in tokenized_sentence_b])

                        # find logits
                        logits_m = logits[:, m]  # logits for the mask in position m (NxV size)
                        logits_b = logits_m[:, idxs_b].tolist()[0]  # logits for sampled tokens (list of N)

                        assert len(idxs_b) == len(logits_b)  # == N

                        # add generated tokens and corresponding logits to lists
                        for j in range(N):
                            generated_seq_tokens[j].append(idxs_b[j])
                            logits_list[j].append(logits_b[j])
                        # torch.cuda.empty_cache()

            # calculate sum of logits for each generated sequence of tokens
            sum_logits = np.array(logits_list).sum(axis=-1) / len(mask_inds)

            # finding ranking (return indices of the parallel lists for logits and generated tokens)
            ranked_inds_of_list = sum_logits.argsort()[::-1]

            # ranked logits sum in descending order
            ranked_logits = sum_logits[ranked_inds_of_list]

            # ranked generated tokens in descending order
            ranked_generated_tokens = np.array(generated_seq_tokens)[ranked_inds_of_list]

            # for no, p in enumerate(ranked_generated_tokens[:20]):
            #     print('{}: {}, {}'.format(no, "".join(untokenize_id(tokenizer, p)), ranked_logits[no]))

            # save the one with the highest log prob for each number of masks (argmax)
            top_ranked_gen_token_seqs.append(ranked_generated_tokens.tolist()[0])
            top_ranked_log_probs.append(ranked_logits.tolist()[0])

            # save all topk preds for each number of masks
            all_ranked_gen_token_seqs.append(ranked_generated_tokens.tolist())
            all_ranked_log_probs.append(ranked_logits.tolist())

        # Evaluation !
        f1_micro_list, f1_macro_list = [], []
        rouge_list, bleu_list, bleu_uni_list, bert_score_list = [], [], [], []

        gold_ids = labels_ids_i
        gold_tok = labels_i  # list of strings

        for m in range(len(top_ranked_log_probs)):
            pred_ids_m = top_ranked_gen_token_seqs[m]
            pred_tok_m = ["".join(untokenize_id(tokenizer, top_ranked_gen_token_seqs[m]))]  # list of strings

            # F1 score
            # F1_micro calculates metrics globally by counting the total true positives,
            # false negatives and false positives.
            if len(gold_ids) == len(pred_ids_m):
                f1_micro_list.append(f1_score(gold_ids, pred_ids_m, average='micro'))
                # F1_macro calculates metrics for each label, and finds their unweighted mean.
                # This does not take label imbalance into account.
                f1_macro_list.append(f1_score(gold_ids, pred_ids_m, average='macro'))
            else:
                f1_micro_list.append(0.0)
                f1_macro_list.append(0.0)

            # BLEU
            try:
                bleu_list.append(bleu.compute(references=gold_tok,
                                              predictions=pred_tok_m)['bleu'])
            except:
                print('something wrong happened when computing blue')
                bleu_list.append(0.0)

            # unigrams
            bleu_uni_list.append(bleu.compute(references=gold_tok,
                                              predictions=pred_tok_m)['precisions'][0])

            # ROUGE
            rouge_list.append(rouge.compute(references=gold_tok,
                                            predictions=pred_tok_m, use_aggregator=True,
                                            use_stemmer=True)['rouge1'])
            # BERT_SCORE
            bert_score_list.append(bertscore.compute(references=gold_tok,
                                                     predictions=pred_tok_m, lang="en")['f1'][0])

        pred_strings = ["".join(untokenize_id(tokenizer, pred)) for pred in top_ranked_gen_token_seqs]

        # print(pred_strings)
        # print('F1-micro: {}, F1-macro: {}, Bleu: {}, Rouge: {}, Bert-score: {}'.format(max(f1_micro_list),
        #                                                                                max(f1_macro_list),
        #                                                                                max(bleu_list),
        #                                                                                max(bleu_uni_list),
        #                                                                                max(rouge_list),
        #                                                                                max(bert_score_list)))
        # save to json dict
        json_dict['f1_micro'].append(f1_micro_list)
        json_dict['f1_macro'].append(f1_macro_list)
        json_dict['bleu'].append(bleu_list)
        json_dict['bleu_uni_precision'].append(bleu_uni_list)
        json_dict['rouge'].append(rouge_list)
        json_dict['bert_score'].append(bert_score_list)
        json_dict['best_log_probs'].append(top_ranked_log_probs)
        json_dict['best_pred_tokens'].append(top_ranked_gen_token_seqs)
        json_dict['best_pred_strings'].append(pred_strings)
        json_dict['all_log_probs'].append(all_ranked_log_probs)
        json_dict['all_preds'].append(all_ranked_gen_token_seqs)

    filename = 'full_results_{}_{}_{}_{}_{}_{}'.format(model_name, quarter, split, N, M, seed)
    torch.save(json_dict, os.path.join(save_dir, "{}.pt".format(filename)))
    return json_dict

if __name__ == "__main__":
    ##########################################################################
    # Setup args
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--lms",default=['cardiffnlp/twitter-roberta-base-jun2022'],nargs='+',required=False)
    ##########################################################################
    # Evaluation args
    ##########################################################################
    parser.add_argument("--topk", help="comma separated list of datasets (test sets)", default=100,required=False)
    parser.add_argument("--single_token",action="store_true", required=False)
    parser.add_argument("--max_num_masks", default=5,required=False)
    parser.add_argument("--batch_size", default=100,required=False)

    args = parser.parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CKPT_DIR = os.path.join(BASE_DIR, 'pretrained_models')
    RES_DIR = os.path.join(BASE_DIR, 'new_results')
    LOG_DIR = os.path.join(BASE_DIR, 'new_logs')
    CACHE_DIR = os.path.join(BASE_DIR, 'cached')

    lm = "cardiffnlp/twitter-roberta-base-mar2022"
    # dataset_filepath=CACHE_DIR+'/{}_dynamic-templama_multiple_masks.pt'.format(lm)
    dataset_filepath = CACHE_DIR + '/cardiffnlp-twitter-roberta-base-mar2022_dynamic-templama_multiple_masks.pt'

    data_dict_multi_token = torch.load(dataset_filepath)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False, add_prefix_space=True)

    CLS = tokenizer.cls_token
    PAD = tokenizer.pad_token
    SEP = tokenizer.sep_token
    MASK = tokenizer.mask_token

    mask_id = tokenizer.mask_token_id
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id

    special_ids = [mask_id, sep_id, cls_id, pad_id]
    # unchanged_t, new_t, updated_t, deleted_t, orig = split_dataset(data_dict_multi_token)

    # lm = "cardiffnlp/twitter-roberta-base-mar2022"
    fill_mask_model = pipeline(
        'fill-mask', model=lm, framework="pt",
        tokenizer=tokenizer, top_k=100
    )
    model = fill_mask_model.model
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    quarter = '2019-Q1'
    batch_size = 64
    N = batch_size  # number of shots
    temperature = 1.0
    burnin = 200
    text_list = data_dict_multi_token[quarter]['text']
    labels_list = data_dict_multi_token[quarter]['labels']
    labels_ids_list = data_dict_multi_token[quarter]['labels_ids']
    relation_list = data_dict_multi_token[quarter]['relation']
    num_answers_list = data_dict_multi_token[quarter]['num_answers']

    json_dict = multi_token_evaluation(tokenizer, fill_mask_model, text_list,labels_list,labels_ids_list,relation_list,
                                       num_answers_list, N=N, M=5, quarter=quarter,
                                       model_name=lm, split='unchanged', seed=123)
    print(json_dict)
    print('done!')
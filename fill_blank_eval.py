import argparse
import json
import os
import re
import sys
import boto3
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from utils.mlm_scoring import compute_mlm_scoring

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.multi_token_generation import multi_token_evaluation, bertscore
from utils.helpers import create_logdir_with_timestamp, init_logging, batchify, split_dataset, facts_over_time
from load_test_sets import TestSetLoader
from sys_config import LMs_names, LMs

# tagger = SequenceTagger.load("flair/pos-english")
comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')


# def find_ranking_position(results, labels, label_ids, input_text, mask_index, check_pos=False):
#     """
#
#     Args:
#         results:
#         labels:
#         label_ids:
#         input_text:
#         mask_index:
#         check_pos:
#
#     Returns:
#
#     """
#     if type(labels) is not list: labels = [labels]
#     if type(results) is not list: results = [results]
#     topk_tokens = [result['token_str'] for result in results]
#     topk_ids = [result['token'] for result in results]
#
#     # for the input
#     if check_pos:
#         input_tag_results = comprehend.detect_syntax(Text=input_text[0], LanguageCode='en')
#         input_tag = input_tag_results['SyntaxTokens'][mask_index[0]]['PartOfSpeech']['Tag']
#
#         topk_sentences = [result['sequence'] for result in results]
#         pred_tags = []
#         bs = 25
#         for i in [0, 25, 50, 75]:
#             pred_tag_list = comprehend.batch_detect_syntax(TextList=topk_sentences[i:i + bs], LanguageCode='en')
#             pred_tags += [res['SyntaxTokens'][mask_index[0]]['PartOfSpeech']['Tag'] for res in
#                           pred_tag_list['ResultList']]
#
#     # add similarity metric between gold label and topk predictions
#     ranking_position_per_example = []
#     for label, label_id in zip(labels, label_ids):
#         ranking_position = -1
#         if type(label) == list: label = label[0]
#         if type(label_id) == list: label_id = label_id[0]
#         if label in topk_tokens:
#             ranking_position = topk_tokens.index(label) + 1  # start from 1 not 0
#         elif label_id in topk_ids:
#             ranking_position = topk_ids.index(label_id) + 1
#         ranking_position_per_example.append(ranking_position)
#
#     best_ranking_position = min(ranking_position_per_example)
#
#     if check_pos:
#         percentage_same_tags = 1 - round(len([x for x in pred_tags if x != input_tag]) / len(pred_tags), 4)
#         return best_ranking_position, percentage_same_tags
#     else:
#         return best_ranking_position


def run_evaluation_single_token(args, model, samples_batches, labels_batches, labels_ids_batches,
                                relations_batches, logger, model_name, quarter, split, check_pos=False):
    """
    Single-token evaluation of dynamic TempLAMA probe.

    Args:
        args: standard experiments args see below
        model: fill-mask pipeline model checkopint (from HF)
        samples_batches: list of lists with batches of text (strings)
        labels_batches: list of lists with batches of labels (list of strings)
        labels_ids_batches: list of lists with batches of labels token ids (list of ints)
        logger: for logging
        model_name: model checkpoint name (e.g. 'cardiffnlp/twitter-roberta-base-jun2022')
        quarter: quarter split name (e.g. '2021-Q3')
        split: fine-grained split name (e.g. 'updated')

    Returns:

        json_dict: dict with all results & metrics (which is also saved as a .pt file)

    """

    json_dict = {
        'model': model_name,
        'quarter': quarter,
        'split': split,
        'text': [],  # list of size len(text_list)
        'gold_label': [],  # list of size len(text_list)
        'pred_label': [],  # list of size len(text_list)
        'relation': [],  # list of size len(text_list)
        'num_answers': [],  # list of size len(text_list)
        # metrics
        'ranking_position_list': [],  # list of size len(text_list)
        'p@1_list': [],  # list of size len(text_list)
        'p@10_list': [],  # list of size len(text_list)
        'p@20_list': [],  # list of size len(text_list)
        'p@50_list': [],  # list of size len(text_list)
        'p@100_list': [],  # list of size len(text_list)
        'mrr_list': [],  # list of size len(text_list)
        # similarity scores
        'bert_score_list': [],  # list of lists. size len(text_list) x topk
        'avg_bert_score_list': [],  # float
        'argmax_bert_score_list': [],  # float
        # pos scores
        'gold_pos_list': [],  # list of size len(text_list)
        'pred_pos_list': [],  # list of size len(text_list)
        'is_best_same_pos_as_gold_list': [],  # list of size len(text_list)
        'avg_pos_same_list': [],  # list of size len(text_list)
        # all predictions & probabilities
        'all_probs': [],  # list
        'all_preds': [],  # list

    }

    if split == 'facts_over_time':
        json_dict['model']=[]
        json_dict['quarter']=[]

    for i in tqdm(range(len(samples_batches))):  # for each batch i
        inputs_b = samples_batches[i]  # list of strings
        labels_b = labels_batches[i]  # list of list of strings (because multiple correct labels)
        labels_ids_b = labels_ids_batches[i]  # list of list of strings (because multiple correct labels)
        relations_b = relations_batches[i]  # list of list of strings (because multiple correct labels)

        # Pass input through model
        outputs_b = model(inputs_b)
        if len(inputs_b) == 1: outputs_b = [outputs_b]

        # For each example j in batch
        for j, output in enumerate(outputs_b):
            input = inputs_b[j]
            relation = relations_b[j]
            labels = labels_b[j]  # list of strings
            label_ids = labels_ids_b[j]  # list of token ids

            topk_tokens = [result['token_str'] for result in output]  # list of topk strings
            topk_ids = [result['token'] for result in output]
            topk_sentences = [result['sequence'] for result in output]
            topk_probs = [result['score'] for result in output]

            if split == 'facts_over_time':
                """
                Here we have a different format for this split, as we have one gold label for each quarter
                for a single test example (fact)
                """
                json_dict['all_probs'].append(topk_probs)
                json_dict['all_preds'].append(topk_tokens)
                # quarters = [list(q.keys())[0] for q in labels]

                # For correct answers in all quarters
                for label_quarter_dicts in zip(label_ids,labels):
                    label_ids_quarter_dict, labels_quarter_dict = label_quarter_dicts
                    quarter, _label_ids = list(label_ids_quarter_dict.items())[0]
                    _, _labels = list(labels_quarter_dict.items())[0]
                    # _label_ids is list of lists so we 'remove' the outer list
                    _label_ids = _label_ids [0]
                    _labels = _labels [0]

                    # For each correct answer
                    ranking_position_per_answer = []
                    for label_id in _label_ids:
                        ranking_position = -1
                        if label_id in topk_ids:
                            ranking_position = topk_ids.index(label_id) + 1  # start from 1 not 0
                        ranking_position_per_answer.append(ranking_position)

                    # we consider all possible labels as equally correct
                    best_ranking_position = min(ranking_position_per_answer)

                    # it will never return 0
                    if best_ranking_position == 0:
                        raise NotImplementedError

                    # save to json dict
                    json_dict['text'].append(input)
                    json_dict['model'].append(model_name)
                    json_dict['relation'].append(relation)
                    json_dict['quarter'].append(quarter)
                    json_dict['gold_label'].append(_labels)
                    json_dict['pred_label'].append(topk_tokens[0])
                    json_dict['num_answers'].append(len(_labels))

                    # Metrics
                    json_dict['ranking_position_list'].append(best_ranking_position)
                    json_dict['p@1_list'].append(1 if best_ranking_position == 1 else 0)
                    json_dict['p@10_list'].append(
                        1 if best_ranking_position >= 1 and best_ranking_position <= 10 else 0)
                    json_dict['p@20_list'].append(
                        1 if best_ranking_position >= 1 and best_ranking_position <= 20 else 0)
                    json_dict['p@50_list'].append(
                        1 if best_ranking_position >= 1 and best_ranking_position <= 50 else 0)
                    json_dict['p@100_list'].append(
                        1 if best_ranking_position >= 1 and best_ranking_position <= 100 else 0)
                    json_dict['mrr_list'].append(1 / best_ranking_position if best_ranking_position != -1 else 0)
            else:
                # save to json dict
                json_dict['text'].append(input)
                json_dict['gold_label'].append(labels)
                json_dict['relation'].append(relation)
                json_dict['num_answers'].append(len(label_ids))
                json_dict['all_probs'].append(topk_probs)
                json_dict['all_preds'].append(topk_tokens)

                check_pos = False
                # Compute POS tags for topk predictions
                if check_pos:
                    mask_token_index = re.split('\s|(?<!\d)[,.](?!\d)', input).index(args.mask_token)
                    pred_tokens_without_space = [pred_roberta_token.replace(' ', '') for pred_roberta_token in topk_tokens]

                    # comprehend.batch_detect_syntax has a limit of 25 batch size
                    bs = 25
                    pred_tag_list = []
                    for k in [0, 25, 50, 75]:
                        pred_tag_list += \
                        comprehend.batch_detect_syntax(TextList=topk_sentences[k:k + bs], LanguageCode='en')[
                            'ResultList']  # topk preds for full sequences

                    topk_pred_tags = []
                    # find pos tags for mask index for all predictions
                    for p, res in enumerate(pred_tag_list):
                        syntax_tokens = res['SyntaxTokens']
                        for token in syntax_tokens:
                            if token['Text'] == pred_tokens_without_space[p] and mask_token_index == token['TokenId'] - 1:
                                # print(token['PartOfSpeech'])
                                topk_pred_tags.append(token['PartOfSpeech']['Tag'])
                                continue
                            # sometimes the tokenization is different and the mask token is not correct
                            # in order to retrieve the correct pos tag....
                            elif token['Text'] == pred_tokens_without_space[p]:
                                topk_pred_tags.append(token['PartOfSpeech']['Tag'])  # ? wrong tokenization
                    """
                    FIX THIS!!!!!!!!
                    """
                    correct_tags = []
                    # print(labels)
                    # if type(labels[0]) is list: labels = [x[0] for x in labels]
                    correct_tokens_without_space = [label.replace(' ', '') for label in labels]
                    # find pos tags for mask index for gold label
                    for each_correct_label in correct_tokens_without_space:
                        orig_sentence = input.replace(args.mask_token, each_correct_label)
                        correct_label_pos = comprehend.detect_syntax(Text=orig_sentence, LanguageCode='en')['SyntaxTokens']
                        for token in correct_label_pos:
                            if token['Text'] == each_correct_label and mask_token_index == token['TokenId'] - 1:
                                # print(token['PartOfSpeech'])
                                correct_tags.append(token['PartOfSpeech']['Tag'])
                                continue
                            # sometimes the tokenization is different and the mask token is not correct
                            # in order to retrieve the correct pos tag....
                            elif token['Text'] == each_correct_label:
                                correct_tags.append(token['PartOfSpeech']['Tag'])  # ? wrong tokenization

                    # compute on average how many times the topk predictions have the same POS tag with the correct answers
                    percentage_same_tags = np.mean(
                        [1 - round(len([x for x in topk_pred_tags if x != correct_label_pos]) / len(topk_pred_tags), 4)
                         for correct_label_pos in correct_tags])

                    # save to json dict
                    json_dict['gold_pos_list'].append(correct_tags)
                    json_dict['pred_pos_list'].append(topk_pred_tags)
                    json_dict['is_best_same_pos_as_gold_list'].append(True if topk_pred_tags[0] in correct_tags
                                                                      else False)
                    json_dict['avg_pos_same_list'].append(percentage_same_tags)

                # BERT_SCORE between topk predictions and gold label
                all_bert_score_list = [bertscore.compute(references=[label] * len(topk_tokens),
                                                         predictions=topk_tokens, lang="en")['f1'] for label in labels]
                avg_bert_score = round(np.mean(all_bert_score_list), 4)  # avg in all topk predictions
                argmax_bert_score = round(np.mean([score[0] for score in all_bert_score_list]),
                                          4)  # avg across correct answers
                json_dict['bert_score_list'].append(all_bert_score_list)
                json_dict['avg_bert_score_list'].append(avg_bert_score)
                json_dict['argmax_bert_score_list'].append(argmax_bert_score)

                # For each correct answer
                ranking_position_per_answer = []
                for label_id in label_ids:
                    ranking_position = -1
                    if label_id in topk_ids:
                        ranking_position = topk_ids.index(label_id) + 1  # start from 1 not 0
                    ranking_position_per_answer.append(ranking_position)

                # we consider all possible labels as equally correct
                best_ranking_position = min(ranking_position_per_answer)

                # it will never return 0
                if best_ranking_position == 0:
                    raise NotImplementedError

                # Metrics
                json_dict['ranking_position_list'].append(best_ranking_position)
                json_dict['p@1_list'].append(1 if best_ranking_position == 1 else 0)
                json_dict['p@10_list'].append(1 if best_ranking_position >= 1 and best_ranking_position <= 10 else 0)
                json_dict['p@20_list'].append(1 if best_ranking_position >= 1 and best_ranking_position <= 20 else 0)
                json_dict['p@50_list'].append(1 if best_ranking_position >= 1 and best_ranking_position <= 50 else 0)
                json_dict['p@100_list'].append(1 if best_ranking_position >= 1 and best_ranking_position <= 100 else 0)
                json_dict['mrr_list'].append(1 / best_ranking_position if best_ranking_position != -1 else 0)

                logger.info('P@1 {}, P@10 {}, MRR {}, BS@1 {}, POS@1 {}'.format(json_dict['p@1_list'][-1],
                                                                                json_dict['p@10_list'][-1],
                                                                                json_dict['mrr_list'][-1],
                                                                                argmax_bert_score,
                                                                                None
                                                                                # json_dict['is_best_same_pos_as_gold_list'][
                                                                                #     -1]
                                                                                ))
    if split == 'facts_over_time':
        keys_to_keep = ['text', 'relation', 'gold_label', 'pred_label',
                        'model', 'quarter', 'mrr_list', 'p@1_list', 'ranking_position_list',
                        'p@10_list', 'p@20_list', 'p@50_list', 'p@100_list']
        filename = 'full_results_{}_{}_single_token'.format(model_name.split('-')[-1], split)
        if args.identifier is not None: filename += '_{}'.format(args.identifier)
        fot_path = os.path.join(args.SINGLE_TOKEN_RES_DIR, 'facts_over_time')
        if not os.path.exists(fot_path):
            os.makedirs(fot_path)
        torch.save(json_dict, os.path.join(fot_path, "{}.pt".format(filename)))
        dct_for_csv = { your_key: json_dict[your_key] for your_key in keys_to_keep }
        _df = pd.DataFrame(data=dct_for_csv)
        _df.to_csv(os.path.join(fot_path, "{}.csv".format(filename)),index=False)
    else:
        filename = 'full_results_{}_{}_{}_single_token'.format(model_name.split('-')[-1], quarter, split)
        torch.save(json_dict, os.path.join(args.SINGLE_TOKEN_RES_DIR, "{}.pt".format(filename)))
    return json_dict


def evaluate_model(args, model_name, test_name, test_dir, log_exp_string, temporal_string):
    """

    This function probes a masked language model (MLM) with a test set for the "fill-mask" (Cloze) task.

    :param args: arguments (see below)
    :param model_name: the name of the model checkpoint
    :param test_name: the name of the test set (the default is 'dynamic-templama')
    :param test_dir: the directory where the dataset is stored
    :param log_exp_string: string to differential between experiments (to be used for filenames)
    :param temporal_string: string in the format min_year-min_month-min-day_to_max-year_max-month_max-day_per_quarter

    :return:
    """
    ##########################################################################
    # Setup logging
    ##########################################################################
    if args.full_logdir is not None:
        args.full_logdir = os.path.join(args.LOG_DIR, args.full_logdir)
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.LOG_DIR, model_name)
        args.full_logdir = log_directory

    logger = init_logging(log_directory)
    args.logger = logger

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        _args = vars(args).copy()
        arguments_to_remove = ['tokens2ids', 'ids2tokens', 'tokenizer', 'logger']
        for a in arguments_to_remove:
            _args.pop(a, None)
        json.dump(_args, outfile)

    msg = "model name: {}\n".format(model_name)
    msg += "args: {}\n".format(_args)

    logger.info("\n" + msg + "\n")

    ##########################################################################
    # Load dataset (test set)
    ##########################################################################
    logger.info("Start loading {} test set.....".format(test_name))
    dataset_filename = 'timelms_{}_{}_single_token'.format(test_name,
                                                           temporal_string) if args.single_token else 'timelms_{}_{}_multi_token'.format(
        test_name, temporal_string)
    dataset_filepath = os.path.join(args.CACHE_DIR, "{}.pt".format(dataset_filename))

    if os.path.isfile(dataset_filepath):
        if 'lama-' in test_name:
            masked_sentences, labels, relation_types = torch.load(dataset_filepath)
        elif test_name in ['templama', 'dynamic-templama']:
            data_dict = torch.load(dataset_filepath)
    else:
        data_loader = TestSetLoader(args=args,
                                    test_name=test_name,
                                    test_dir=test_dir,
                                    logger=logger)
        if 'lama-' in test_name:
            masked_sentences, labels, relation_types = data_loader.get_test_set()
            torch.save([masked_sentences, labels, relation_types], dataset_filepath)
        elif test_name in ['templama', 'dynamic-templama']:
            data_dict = data_loader.get_test_set()
            torch.save(data_dict, dataset_filepath)

    ##########################################################################
    # Compute joint vocab across different models for fair comparison
    # -- TimeLMs have the same vocab
    ##########################################################################

    ##########################################################################
    # Define model / load pipeline
    ##########################################################################
    logger.info("Loading pipeline.....")
    fill_mask_model = pipeline(
        'fill-mask', model=model_name, top_k=args.N, framework="pt", batch_size=args.batch_size,
        tokenizer=args.tokenizer
    )

    print('Finished downloading the model!')
    ##########################################################################
    # Inference
    ##########################################################################
    # LAMA
    # if 'lama-' in test_name:
    #     # Create batches of data
    #     text_batches, labels_batches = batchify(test_name, text=masked_sentences, labels=labels,
    #                                             batch_size=args.batch_size)
    #     # Run evaluation
    #     p_at_1_list, p_at_k_list, mrr_list = run_evaluation_single_token(args, text_batches, labels_batches,
    #                                                                      fill_mask_model, logger)
    #     avg_p_at_1 = np.mean(p_at_1_list)
    #     avg_p_at_k = np.mean(p_at_k_list)
    #     avg_mrr = np.mean(mrr_list)
    #
    #     res_msg = "Model {}, Dataset {}, P@1 {}, P@k {} (k={}), MRR {}!".format(model_name,
    #                                                                             test_name,
    #                                                                             avg_p_at_1,
    #                                                                             avg_p_at_k,
    #                                                                             args.topk, avg_mrr)
    #     logger.info("\n" + "*****" + res_msg + "*****" + "\n")
    #     print(res_msg)
    #     return avg_p_at_1, avg_p_at_k, avg_mrr

    # TemLAMA & Ours
    if test_name in ['templama', 'dynamic-templama']:
        # Split dataset to fine-grained test sets (unchanged/new/updated/deleted)
        splits_filepath = os.path.join(args.CACHE_DIR, "{}_splits.pt".format(dataset_filename))
        if os.path.isfile(splits_filepath):
            unchanged_t, new_t, updated_t, deleted_t, orig = torch.load(splits_filepath)
        else:
            unchanged_t, new_t, updated_t, deleted_t, orig = split_dataset(data_dict)
            torch.save([unchanged_t, new_t, updated_t, deleted_t, orig], splits_filepath)

        # Find all facts that change over time (intersection of all datasets in order to be able to
        # compare the performance of a single model across different timesteps)
        fot_split_filepath = os.path.join(args.CACHE_DIR, "{}_facts_over_time_split.pt".format(dataset_filename))
        if os.path.isfile(fot_split_filepath):
            fot_dict = torch.load(fot_split_filepath)
        else:
            fot_dict = facts_over_time(data_dict)
            torch.save(fot_dict, fot_split_filepath)

        splits_dicts = {
            'unchanged': unchanged_t,
            'new': new_t,
            'updated': updated_t,
            'deleted': deleted_t,
            'facts_over_time': fot_dict
        }
        results_dict = {}

        # single-token metrics
        avg_p_at_1, avg_p_at_10, avg_p_at_20, avg_p_at_50, avg_p_at_100 = None, None, None, None, None
        avg_mrr, avg_per_same_tags = None, None

        # multi-token metrics
        avg_f1_micro, avg_f1_macro, avg_bleu, avg_bleu_uni = None, None, None, None
        avg_rouge, avg_bert_score = None, None

        # mlm scoring
        avg_pppl, median, all_pppl_scores = None, None, None

        for split in args.splits:
            _data_dict = splits_dicts[split]
            logger.info('\n' + "*" * 20 + split + "*" * 20 + '\n')
            ##############################################
            # Facts over time split
            ##############################################
            if split == 'facts_over_time':
                ############################################################################################
                # Single token evaluation -- one minibatch consists of multiple test examples
                ############################################################################################
                if args.single_token:
                    # Create batches of data
                    batches_dicts = batchify(test_name, data_dict=_data_dict, batch_size=args.batch_size)
                    text_batches, labels_batches, labels_ids_batches, relations_batches = batches_dicts
                    _ = run_evaluation_single_token(args, model=fill_mask_model,
                                                                            samples_batches=text_batches['text'],
                                                                            labels_batches=labels_batches[
                                                                                'labels'],
                                                                            labels_ids_batches=labels_ids_batches[
                                                                                'labels_ids'],
                                                                            relations_batches=relations_batches[
                                                                                'relation'],
                                                                            logger=logger,
                                                                            quarter=None,
                                                                            model_name=lm,
                                                                            split=split)
                ############################################################################################
                # Multi token evaluation -- one minibatch consists of a single test example
                ############################################################################################
                else:
                    raise NotImplementedError
            ##############################################
            # updated/new/deleted/unchanged splits
            ##############################################
            else:
                if args.single_token:
                    # Create batches of data
                    batches_dicts = batchify(test_name, data_dict=_data_dict, batch_size=args.batch_size)

                    text_batches, labels_batches, labels_ids_batches, relations_batches = batches_dicts

                # Run evaluation per year/quarter/month
                quarters_to_evaluate = _data_dict.keys()
                if args.quarter is not "all":
                    quarters_to_evaluate = [args.quarter]
                # for quarter in list(_data_dict.keys()):
                for quarter in quarters_to_evaluate:
                    logger.info('\n' + "*" * 20 + quarter + "*" * 20 + '\n')
                    ############################################################################################
                    # Single token evaluation -- one minibatch consists of multiple test examples
                    ############################################################################################
                    if args.single_token:  # this is more efficient for single-token prediction bcos it utilises the batch
                        single_token_results_dict = run_evaluation_single_token(args, model=fill_mask_model,
                                                                                samples_batches=text_batches[quarter],
                                                                                labels_batches=labels_batches[
                                                                                    quarter],
                                                                                labels_ids_batches=labels_ids_batches[
                                                                                    quarter],
                                                                                relations_batches=relations_batches[
                                                                                    quarter],
                                                                                logger=logger,
                                                                                quarter=quarter,
                                                                                model_name=lm,
                                                                                split=split)
                        # p_at_1, p_at_10, p_at_20, p_at_50, p_at_100 = p_at_lists
                        num_of_examples = sum([len(x) for x in text_batches[quarter]])

                        avg_p_at_1 = round(np.mean(single_token_results_dict['p@1_list']), 4)
                        avg_p_at_10 = round(np.mean(single_token_results_dict['p@10_list']), 4)
                        avg_p_at_20 = round(np.mean(single_token_results_dict['p@20_list']), 4)
                        avg_p_at_50 = round(np.mean(single_token_results_dict['p@50_list']), 4)
                        avg_p_at_100 = round(np.mean(single_token_results_dict['p@100_list']), 4)
                        avg_mrr = round(np.mean(single_token_results_dict['mrr_list']), 4)
                        avg_per_same_tags = round(np.mean(single_token_results_dict['avg_pos_same_list']), 4)
                        avg_bert_score = round(np.mean(single_token_results_dict['argmax_bert_score_list']), 4)

                    ############################################################################################
                    # MLM Scoring -- one minibatch consists of a single test example
                    # we compute pseudo perplexity for the multi-token gold label
                    ############################################################################################
                    elif args.mlm_scoring:
                        mlm_res_dict = compute_mlm_scoring(args.tokenizer, fill_mask_model, data_dict[quarter],
                                                        quarter=quarter, model_name=lm, split=split,
                                                       save_dir=args.MLM_SCORING_RES_DIR)
                        num_of_examples = len(_data_dict[quarter]['text'])

                        all_pppl_scores = mlm_res_dict["all_pppl_scores"]
                        avg_pppl = mlm_res_dict["avg_pppl"]
                        median_pppl = mlm_res_dict["median_pppl"]

                    ############################################################################################
                    # Multi token evaluation -- one minibatch consists of a single test example
                    ############################################################################################
                    else:
                        print('Start evaluating the model in {}!'.format(quarter))
                        multi_token_results_dict = multi_token_evaluation(
                            tokenizer=args.tokenizer, fill_mask_model=fill_mask_model,
                            text_list=_data_dict[quarter]['text'],
                            labels_list=_data_dict[quarter]['labels'],
                            labels_ids_list=_data_dict[quarter]['labels_ids'],
                            relation_list=_data_dict[quarter]['relation'],
                            num_answers_list=_data_dict[quarter]['num_answers'],
                            save_dir=args.MULTI_TOKEN_RES_DIR,
                            N=args.N, M=args.max_num_masks,
                            quarter=quarter, model_name=lm, split=split, seed=args.seed)

                        num_of_examples = len(_data_dict[quarter]['text'])
                        avg_f1_micro = round(np.mean([max(f1_list) for f1_list in multi_token_results_dict['f1_micro']]), 4)
                        avg_f1_macro = round(np.mean([max(f1_list) for f1_list in multi_token_results_dict['f1_macro']]), 4)
                        avg_rouge = round(np.mean([max(f1_list) for f1_list in multi_token_results_dict['rouge']]), 4)
                        avg_bleu = round(np.mean([max(f1_list) for f1_list in multi_token_results_dict['bleu']]), 4)
                        avg_bleu_uni = round(
                            np.mean([max(f1_list) for f1_list in multi_token_results_dict['bleu_uni_precision']]), 4)
                        avg_bert_score = round(
                            np.mean([max(f1_list) for f1_list in multi_token_results_dict['bert_score']]), 4)

                    res_msg = "Model {}, Dataset {}, Split {}, Quarter {}, Num of Examples {}, P@1 {}, P@10 {}, MRR {}, " \
                              "F1 macro {}, Rouge {}, Bert-score {}!".format(
                        model_name,
                        test_name,
                        split,
                        quarter,
                        num_of_examples,
                        avg_p_at_1, avg_p_at_10, avg_mrr,
                        avg_f1_macro, avg_rouge, avg_bert_score
                    )
                    logger.info("\n" + "*****" + res_msg + "*****" + "\n")
                    print(res_msg)

                    _res_dct = {
                        "size": num_of_examples,
                        "P@1": avg_p_at_1, "P@10": avg_p_at_10, "P@20": avg_p_at_20,
                        "P@50": avg_p_at_50, "P@100": avg_p_at_100,
                        "mrr": avg_mrr, 'same_pos': avg_per_same_tags,
                        "avg_f1_micro": avg_f1_micro, "avg_f1_macro": avg_f1_macro,
                        "avg_rouge": avg_rouge, "avg_bleu": avg_bleu, "avg_bleu_uni": avg_bleu_uni,
                        "avg_bert_score": avg_bert_score,
                        "avg_pppl": avg_pppl,
                        "median_pppl": median_pppl,
                        "all_pppl_scores":all_pppl_scores
                    }
                    if quarter in results_dict:
                        results_dict[quarter][split] = _res_dct
                    else:
                        results_dict[quarter] = {split: _res_dct
                                                 }
                    # pp(results_dict)
                # else:
                #     NotImplementedError
        return results_dict


def add_tokenizer_args(args, lm):
    args.tokenizer = LMs[lm]["tokenizer"]
    args.mask_token = LMs[lm]["mask_token"]
    args.tokens2ids = LMs[lm]["tokens2ids"]
    args.ids2tokens = LMs[lm]["ids2tokens"]
    args.vocab_size = LMs[lm]["vocab_size"]
    args.max_seq_len = LMs[lm]["max_seq_len"]
    args.special_ids = LMs[lm]["special_ids"]
    return args


if __name__ == "__main__":
    ##########################################################################
    # Setup args
    ##########################################################################
    parser = argparse.ArgumentParser()
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument(
        "--lms",
        help="comma separated list of language models. from {}".format(LMs_names),
        default=[
            'cardiffnlp/twitter-roberta-base-2019-90m',
            'cardiffnlp/twitter-roberta-base-mar2020',
            'cardiffnlp/twitter-roberta-base-jun2020',
            'cardiffnlp/twitter-roberta-base-sep2020',
            'cardiffnlp/twitter-roberta-base-dec2020',
            'cardiffnlp/twitter-roberta-base-mar2021',
            'cardiffnlp/twitter-roberta-base-jun2021',
            'cardiffnlp/twitter-roberta-base-sep2021',
            'cardiffnlp/twitter-roberta-base-dec2021',
            # 'cardiffnlp/twitter-roberta-base-2021-124m',
            'cardiffnlp/twitter-roberta-base-mar2022',
            'cardiffnlp/twitter-roberta-base-jun2022'
        ],
        nargs='+',
        required=False,
    )
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument(
        "--dataset",
        # help="dataset name (test sets) from {}".format(available_datasets),
        default="dynamic-templama",
        required=False,
    )
    parser.add_argument(
        "--splits",
        help="which splits to evaluate",
        default=["updated", "new", "deleted"],
        nargs='+',
        required=False,
    )
    parser.add_argument(
        "--quarter",
        help="which quarters to evaluate",
        default="all",
        # nargs='+',
        required=False,
    )
    ##########################################################################
    # Temporal args
    ##########################################################################
    parser.add_argument("--min_year", default=2019, help="minimum year to get facts", required=False)
    parser.add_argument("--min_month", default=1, help="minimum month to get facts", required=False)
    parser.add_argument("--min_day", default=1, help="minimum day to get facts", required=False)
    parser.add_argument("--max_year", default=2022, help="maximum year to get facts", required=False)
    parser.add_argument("--max_month", default=6, help="maximum month to get facts", required=False)
    parser.add_argument("--max_day", default=31, help="maximum day to get facts", required=False)
    parser.add_argument("--granularity", default="quarter", help="granularity to create test sets"
                                                                 "between [month, quarter,year]", required=False)
    ##########################################################################
    # Evaluation args
    ##########################################################################
    parser.add_argument(
        "--single_token",
        # action="store_true",
        default=False,
        type=bool,
        help="if True, we consider only single tokens as labels.",
        required=False,
    )
    parser.add_argument(
        "--mlm_scoring",
        # action="store_true",
        default=False,
        help="if True, we use mlm scoring.",
        required=False,
    )
    parser.add_argument(
        "--topk",
        help="When we sample for multi-token generation, sample from the topk predictions.",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--N",
        help="N: the number of 'shots' that we attempt (sampled sequences of tokens)",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--max_num_masks",
        # action="store_true",
        help="M: the maximum number of mask to try for multi-token generation in the range [1,M].",
        default=5,
        required=False,
    )
    parser.add_argument(
        "--seed",
        help="set the seed for sampling in multi-token generation",
        default=1210,  # my birthday
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size for single-token prediction (for multi-token te batch size is N)",
        default=128,  # my birthday
        required=False,
    )
    ##########################################################################
    # Miscellaneous
    ##########################################################################
    parser.add_argument(
        "--full_logdir",
        help="directory to save logs (relative to /logs/)",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--identifier",
        help="string to append to results filename",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--sagemaker",
        # action="store_true",
        default=None,
        help="if True, run code in SageMaker and change paths",
        required=False,
    )
    # ##############
    # parser.add_argument(
    #     "--spacy_model",
    #     "--sm",
    #     dest="spacy_model",
    #     default="en_core_web_sm",
    #     help="spacy model file path",
    # )

    args = parser.parse_args()

    print(args)
    print('Thanks Karthi')


    if args.sagemaker is not None:
        print('SAGEMAKER!')
        args.INPUT_DIR = "/opt/ml/input"
        args.OUT_DIR = "/opt/ml/output/data"
        args.DATA_DIR = os.path.join(args.INPUT_DIR, 'data')
        args.CACHE_DIR = os.path.join(args.INPUT_DIR, 'cached')
        args.RES_DIR = os.path.join(args.OUT_DIR, 'new_results')
        args.LOG_DIR = os.path.join(args.OUT_DIR, 'new_logs')
    else:
        print('NO SAGEMAKER!')
        args.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        args.DATA_DIR = os.path.join(args.BASE_DIR, 'data')
        args.RES_DIR = os.path.join(args.BASE_DIR, 'new_results')
        args.LOG_DIR = os.path.join(args.BASE_DIR, 'new_logs')
        args.CACHE_DIR = os.path.join(args.BASE_DIR, 'cached')

    args.SINGLE_TOKEN_RES_DIR = os.path.join(args.RES_DIR, 'single_token')
    args.MULTI_TOKEN_RES_DIR = os.path.join(args.RES_DIR, 'multi_token')
    args.MLM_SCORING_RES_DIR = os.path.join(args.RES_DIR, 'mlm_scoring')

    # try:
    #     print('/opt/ml: {}'.format(os.listdir('/opt/ml')))
    # except:
    #     print('EXCEPT')
    #
    # try:
    #     print('args.INPUT_DIR: {}'.format(os.listdir(args.INPUT_DIR)))
    # except:
    #     print('EXCEPT')
    # try:
    #     print('args.DATA_DIR: {}'.format(os.listdir(args.DATA_DIR)))
    # except:
    #     print('EXCEPT')

    for directory in [args.CACHE_DIR, args.RES_DIR, args.LOG_DIR,
                      args.SINGLE_TOKEN_RES_DIR, args.MULTI_TOKEN_RES_DIR, args.MLM_SCORING_RES_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    temporal_string = '{}-{}-{}_to_{}-{}-{}_per_{}'.format(args.min_year,
                                                           args.min_month,
                                                           args.min_day,
                                                           args.max_year,
                                                           args.max_month,
                                                           args.max_day,
                                                           args.granularity)

    if args.identifier is not None: temporal_string += '_{}'.format(args.identifier)

    test_dir = os.path.join(args.DATA_DIR, args.dataset, 'dataset_from_' + temporal_string)

    # Check if dataset exists
    if not os.path.isdir(test_dir):
        print(test_dir)
        raise "Dataset not found! Make sure to run `create_templates.py` first!"
        exit()

    if type(args.lms) is not list: args.lms = [args.lms]

    model_names_string = "timelms" if len(args.lms) == 11 else "_".join([x.split('-')[-1] for x in args.lms])
    if 'cardiffnlp' not in args.lms[0]:
        model_names_string = "_".join([x for x in args.lms])
    print(model_names_string)

    # dicts to save the results
    list_of_attributes = ['model', 'dataset', 'quarter', 'size', 'split',
                          # singe-token metrics
                          'P@1', 'P@10', 'P@20', 'P@50', 'P@100',
                          'same_pos', 'mrr',
                          # multi-token hard metrics
                          "avg_f1_micro", "avg_f1_macro",
                          # multi-token soft metrics
                          "avg_rouge", "avg_bleu", "avg_bleu_uni", "avg_bert_score",
                          # mlm scoring
                          "avg_pppl", "all_pppl_scores", "median_pppl"
                          ]
    results_dict = {key: [] for key in list_of_attributes}

    # Setup filename to save logs
    log_exp_string = "{}_{}_{}".format(model_names_string, args.dataset, temporal_string)
    if args.single_token:
        log_exp_string += "_single_token"
    elif args.mlm_scoring:
        log_exp_string += "_mlm_scoring"
    else:
        log_exp_string += "_multi_token_{}_{}_{}".format(args.seed, args.topk, args.N)


    splits_string = "_".join([x.split('-')[-1] for x in args.splits])
    log_exp_string += "_{}".format(splits_string)
    if args.quarter != "all":
        log_exp_string += "_{}".format(args.quarter)
    # if args.identifier is not None: log_exp_string += '_{}'.format(args.identifier)

    print(log_exp_string)

    # Evaluate each model in args.lms list
    for i, lm in enumerate(args.lms):
        args = add_tokenizer_args(args, lm)
        args.lowercase = True if 'uncased' in lm else False

        results = evaluate_model(args=args,
                                 model_name=lm,
                                 test_name=args.dataset,
                                 test_dir=test_dir,
                                 log_exp_string=log_exp_string,
                                 temporal_string=temporal_string)
        print(results.keys())
        for quarter in results.keys():
            print(results[quarter].keys())
            for split in results[quarter].keys():
                results_dict["model"].append(lm)
                results_dict["dataset"].append(args.dataset)
                results_dict["quarter"].append(quarter)
                results_dict["size"].append(results[quarter][split]['size'])
                results_dict["split"].append(split)
                # single-token
                results_dict["P@1"].append(results[quarter][split]['P@1'])
                results_dict["P@10"].append(results[quarter][split]['P@10'])
                results_dict["P@20"].append(results[quarter][split]['P@20'])
                results_dict["P@50"].append(results[quarter][split]['P@50'])
                results_dict["P@100"].append(results[quarter][split]['P@100'])
                results_dict["mrr"].append(results[quarter][split]['mrr'])
                results_dict["same_pos"].append(results[quarter][split]['same_pos'])
                # multi-token
                results_dict["avg_f1_micro"].append(results[quarter][split]['avg_f1_micro'])
                results_dict["avg_f1_macro"].append(results[quarter][split]['avg_f1_macro'])
                results_dict["avg_rouge"].append(results[quarter][split]['avg_rouge'])
                results_dict["avg_bleu"].append(results[quarter][split]['avg_bleu'])
                results_dict["avg_bleu_uni"].append(results[quarter][split]['avg_bleu_uni'])
                results_dict["avg_bert_score"].append(results[quarter][split]['avg_bert_score'])
                # mlm-scoring (pppl)
                results_dict["avg_pppl"].append(results[quarter][split]['avg_pppl'])
                results_dict["median_pppl"].append(results[quarter][split]['median_pppl'])
                results_dict["all_pppl_scores"].append(results[quarter][split]['all_pppl_scores'])
        print(results_dict)

        df_results = pd.DataFrame(results_dict)

        save_dir = args.MULTI_TOKEN_RES_DIR
        if args.single_token:
            save_dir = args.SINGLE_TOKEN_RES_DIR
        if args.mlm_scoring:
            save_dir = args.MLM_SCORING_RES_DIR
        df_results.to_csv(os.path.join(save_dir, log_exp_string + ".csv"))

        print('Filename {}'.format(os.path.join(save_dir, log_exp_string + ".csv")))

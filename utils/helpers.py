import logging
import os
import sys
import time

import pandas as pd
from transformers import AutoTokenizer

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def split_dataset(data):
    """
    Split temporal dataset Dt to D_unchanged, D_new and D_updated compared to D_(t-1) for all t.
    Specifically:
    - D_unchanged: data where text_t = text_(t-1) & label_t = label_(t-1)
    - D_updated: data where text_t = text_(t-1) & label_t != label_(t-1)
    - D_new: data where text_t not in D_(t-1)
    - D_deleted: data that exist in D_(t-1) but not in D_t

    Args:
        data: a dictionary with keys the time (year/quarter/month) and values dictionaries
        data = {
                '2019-Q1':
                    {
                    'text': [list of text],
                    'labels': [list of labels],
                    'labels_ids': [list of label token ids -- for a given model/tokenizer],
                    'relations' [list of Wikidata relations]
                    },
                '2019-Q2': {...}
                }

    Returns:
        D_unchanged, D_new, D_updated, D_deleted
    """
    unchanged_t, new_t, updated_t, deleted_t = {}, {}, {}, {}

    quarters = list(data.keys())
    t_0 = quarters[0]  # t=t0
    t_1 = quarters[0]  # t-1

    for t in quarters[1:]:
        print(t)
        if t in ['2022-Q3', '2022-Q4']:
            continue  # skip last two quarters of 2022
        data_t = data[t]  # D_t
        data_t_1 = data[t_1]  # D_(t-1)

        unchanged_t[t] = {key: [] for key in data_t.keys()}
        new_t[t] = {key: [] for key in data_t.keys()}
        updated_t[t] = {key: [] for key in data_t.keys()}
        deleted_t[t] = {key: [] for key in data_t.keys()}

        for i in range(0, len(data_t['text'])):  # for fact in D_t
            text_t = data_t['text'][i]  # string
            labels_ids_t = data_t['labels_ids'][i]  # list of lists
            if text_t in data_t_1['text']:
                t_1_index = data_t_1['text'].index(text_t)
                labels_inds_t_1 = data_t_1['labels_ids'][t_1_index]  # list of lists
                # because we have multiple correct answers (labels) we check each one separately
                """
                labels_ids_t: labels in timestep t
                labels_ids_t_1: labels in timestep t-1
                """
                for label_id, label_t in enumerate(labels_ids_t):
                    if label_t in labels_inds_t_1:
                        #######################
                        ###### UNCHANGED ######
                        #######################
                        # text_t = text_t-1 & label_t = label_t-1
                        # add to D_unchanged
                        for key in data_t.keys():
                            if key in ['labels', 'labels_ids', 'num_masks']:
                                unchanged_t[t][key].append(data_t[key][i][label_id])
                            else:
                                unchanged_t[t][key].append(data_t[key][i])
                    else:
                        #######################
                        ####### UPDATED #######
                        #######################
                        # text_t = text_(t-1) & label_t != label_(t-1)
                        # add to D_updated
                        for key in data_t.keys():
                            if key in ['labels', 'labels_ids', 'num_masks']:
                                updated_t[t][key].append(data_t[key][i][label_id])
                            else:
                                updated_t[t][key].append(data_t[key][i])
            else:
                #######################
                ######### NEW #########
                #######################
                # text_t not in D_(t-1) texts
                # add to D_new
                for key in data_t.keys():
                    for label_id, label_t in enumerate(labels_ids_t):
                        if key in ['labels', 'labels_ids', 'num_masks']:
                            new_t[t][key].append(data_t[key][i][label_id])
                        else:
                            new_t[t][key].append(data_t[key][i])

        for j in range(0, len(data_t_1['text'])):  # for fact in D_t-1
            text_t_1 = data_t_1['text'][j]
            labels_ids_t = data_t_1['labels_ids'][j]  # list of lists
            if text_t_1 not in data_t['text']:
                for label_id, label_t in enumerate(labels_ids_t):
                    #######################
                    ####### DELETED #######
                    #######################
                    # text_(t+1) not in D_t
                    # add to D_deleted
                    for key in data_t_1.keys():
                        if key in ['labels', 'labels_ids', 'num_masks']:
                            deleted_t[t][key].append(data_t_1[key][j][label_id])
                        else:
                            deleted_t[t][key].append(data_t_1[key][j])
                    # deleted_t[t][key].append(data_t_1[key][j])
        t_1 = t

        print(
            't={}: From total {} samples in D_t, {} are unchanged, {} are updated, {} are deleted and {} are new, compared to D_(t-1).'.format(
                t,
                len(data_t['text']),
                len(unchanged_t[t]['text']),
                len(updated_t[t]['text']),
                len(deleted_t[t]['text']),
                len(new_t[t]['text'])),
        )
    #         assert len(data_t['text']) == len(unchanged_t[t]['text']) + len(updated_t[t]['text']) + len(new_t[t]['text'])
    return unchanged_t, new_t, updated_t, deleted_t, data[t_0]


def facts_over_time(data):
    """
    This functions creates a test set with the intersection of all facts for which we have their objects (labels)
    for all timesteps (from 2019-Q1 until 2022-Q2). We do that because this way we create exactly the same test set
    for each quarter (same number of facts -- the only thing that might change is the label) and thus
    we are able to compare the performance of a *single model* across *different test sets*.
    If we didn't do this split, it would not be fair to compare the performance of a model in different test sets.
    Args:
        data: a dictionary with keys the time (year/quarter/month) and values dictionaries
        data = {
                '2019-Q1':
                    {
                    'text': [list of text],
                    'labels': [list of labels],
                    'labels_ids': [list of label token ids -- for a given model/tokenizer],
                    'relations' [list of Wikidata relations]
                    },
                '2019-Q2': {...}
                }

    Returns:
        facts_over_time: a dictionary
        data = {
                'facts': text with the fact, (e.g. )
                'relation': the relation of the fact (e.g. )
                'labels_[quarter]': list of labels for a specific quarter (e.g. quarter = '2019-Q1'),
                'labels_ids_[quarter]': list of corresponding token ids (based on the model's vocabulary/tokenization),
                ... (for all quarters in list(data.keys()))
            }
    """

    _quarters = list(data.keys())
    quarters = [q for q in _quarters if q not in ['2022-Q3', '2022-Q4']]

    # t=0
    orig_rel = data[quarters[0]]['relation']
    keys_for_dct = ['facts', 'relation'] + ['labels_{}'.format(q) for q in quarters] + ['labels_ids_{}'.format(q) for q
                                                                                 in quarters]

    # We create an initial dictionary with all facts in all timesteps/quarters and we fill it with None
    orig_facts = data[quarters[0]]['text']
    facts_over_time_dct = {k:[None]*len(orig_facts) for k in keys_for_dct}
    facts_over_time_dct['facts'] = orig_facts
    facts_over_time_dct['relation'] = orig_rel

    for fact_index, fact in enumerate(orig_facts):
        for t in quarters:
            facts_t = data[t]['text']
            labels_t = data[t]['labels']
            labels_ids_t = data[t]['labels_ids']

            # if intersection, we add the actual value to the dictionary
            if fact in facts_t:
                index_t = facts_t.index(fact)
                facts_over_time_dct['labels_{}'.format(t)][fact_index] = labels_t[index_t]
                facts_over_time_dct['labels_ids_{}'.format(t)][fact_index] = labels_ids_t[index_t]

    # We drop all None values to keep only those facts for which we have labels over time
    fot_df = pd.DataFrame(data=facts_over_time_dct).dropna()
    return fot_df.to_dict()



def batchify(test_name, data_dict={}, text=None, labels=None, batch_size=32):
    """
    Creates batches of input,output pairs to pass to the model
    :param test_name: the name of the test set
    :param data_dict: dictionary with "text", "labels", "labels_ids" -- for TempLAMA
    :param text: list of input text -- for LAMA
    :param labels: list of labels -- for LAMA
    :param batch_size: batch size
    :return:
    """
    # for TempLAMA
    text_batches_dict, labels_batches_dict, labels_ids_batches_dict, relations_batches_dict = {}, {}, {}, {}
    # for LAMA
    list_samples_batches, list_labels_batches = [], []
    current_samples_batch, current_labels_batches = [], []
    c = 0

    # LAMA
    if 'lama-' in test_name:
        data = list(zip(text, labels))
        # sort to group together sentences with similar length
        # for sample in sorted(
        #     data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
        # ):
        for sample in sorted(
                data, key=lambda k: len(" ".join(k[0]).split())
        ):
            masked_sentence, label = sample
            current_samples_batch.append(masked_sentence)
            current_labels_batches.append(label)
            c += 1
            if c >= batch_size:
                list_samples_batches.append(current_samples_batch)
                list_labels_batches.append(current_labels_batches)
                current_samples_batch = []
                current_labels_batches = []
                c = 0

        # last batch
        if current_samples_batch and len(current_samples_batch) > 0:
            list_samples_batches.append(current_samples_batch)
            list_labels_batches.append(current_labels_batches)

        return list_samples_batches, list_labels_batches
    # TempLAMA
    elif test_name in ['templama', 'dynamic-templama']:
        if 'facts' in data_dict.keys():  # facts over time dict / different format
            """
            data_dict = {
                'facts': {...},
                         'relation': [...],
                         'labels_2019-Q1': {...},
                         'labels_ids_2019-Q1': {...},
                         ...
                         }
            """
            text_list, labels_list, labels_ids_list, relations_list = [], [], [], []
            current_text_list, current_labels_list, current_labels_ids_list, current_relations_list = [], [], [], []
            unique_quarters = list(set([x.split('_')[-1] for x in data_dict.keys() if 'Q' in x.split('_')[-1]]))

            # fix minor format issue
            for key in data_dict:
                data_dict[key] = list(data_dict[key].values())

            for fact_index, fact in enumerate(data_dict['facts']):
                current_text_list.append(fact)
                current_relations_list.append(data_dict['relation'][fact_index])
                current_labels_list.append([{q: data_dict['labels_{}'.format(q)][fact_index]} for q in unique_quarters])
                current_labels_ids_list.append([{q: data_dict['labels_ids_{}'.format(q)][fact_index]} for q in unique_quarters])

                c += 1
                if c >= batch_size:
                    text_list.append(current_text_list)
                    labels_list.append(current_labels_list)
                    labels_ids_list.append(current_labels_ids_list)
                    relations_list.append(current_relations_list)
                    current_text_list, current_labels_list, current_labels_ids_list, current_relations_list = [], [], [], []
                    c = 0

            # last batch
            if current_text_list and len(current_text_list) > 0:
                text_list.append(current_text_list)
                labels_list.append(current_labels_list)
                labels_ids_list.append(current_labels_ids_list)
                relations_list.append(current_relations_list)

            text_batches_dict['text'] = text_list
            labels_batches_dict['labels'] = labels_list
            labels_ids_batches_dict['labels_ids'] = labels_ids_list
            relations_batches_dict['relation'] = relations_list
        else:
            # iterate per time period
            for year in data_dict.keys():
                text_list, labels_list, labels_ids_list, relations_list = [], [], [], []
                current_text_list, current_labels_list, current_labels_ids_list, current_relations_list = [], [], [], []
                data = list(zip(data_dict[year]["text"], data_dict[year]["labels"], data_dict[year]["labels_ids"],
                                data_dict[year]["relation"]))
                for sample in sorted(
                        data, key=lambda k: len(" ".join(k[0]).split())
                ):
                    masked_sentence, labels, labels_ids, relation = sample
                    current_text_list.append(masked_sentence)
                    current_labels_list.append(labels)
                    current_labels_ids_list.append(labels_ids)
                    current_relations_list.append(relation)
                    c += 1
                    if c >= batch_size:
                        text_list.append(current_text_list)
                        labels_list.append(current_labels_list)
                        labels_ids_list.append(current_labels_ids_list)
                        relations_list.append(current_relations_list)
                        current_text_list, current_labels_list, current_labels_ids_list, current_relations_list = [], [], [], []
                        c = 0

                # last batch
                if current_text_list and len(current_text_list) > 0:
                    text_list.append(current_text_list)
                    labels_list.append(current_labels_list)
                    labels_ids_list.append(current_labels_ids_list)
                    relations_list.append(current_relations_list)

                text_batches_dict[year] = text_list
                labels_batches_dict[year] = labels_list
                labels_ids_batches_dict[year] = labels_ids_list
                relations_batches_dict[year] = relations_list

    return [text_batches_dict, labels_batches_dict, labels_ids_batches_dict, relations_batches_dict]


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def init_logging(log_directory):
    logger = logging.getLogger("temporal_robustness_evaluation")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger

#
# def filter_logprobs(log_probs, indices):
#     new_log_probs = log_probs.index_select(dim=2, index=indices)
#     return new_log_probs
#
#
# def roberta_map_labels(label):
#     lm = "roberta-base"
#     roberta_tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False, add_prefix_space=True)
#
#     new_id_list = roberta_tokenizer(label)['input_ids']
#     new_id_list_no_special_tokens = [i for i in new_id_list if i not in roberta_tokenizer.all_special_ids]
#     if len(new_id_list_no_special_tokens) == 1:
#         return new_id_list_no_special_tokens[0]  # label_id !!!
#     else:
#         # initial word is now split in more than two token ids...
#         # e.g. Dreaming = 7419 (Dream) + 154 (ing) while dreaming = 26240
#         # check if we can change that by lowercasing
#         if label != label.lower():
#             roberta_map_labels(lm, label.lower())  # try again
#         else:
#             #  we cannot do anything more
#             return None

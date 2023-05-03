import math
import os

import numpy as np
import torch
from tqdm import tqdm

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

def compute_mlm_scoring(tokenizer, fill_mask_model, data_dict, model_name, quarter, split, save_dir):
    # tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False, add_prefix_space=True)
    # fill_mask_model = pipeline(
    #     'fill-mask', model=lm, framework="pt", batch_size=32,
    #     tokenizer=tokenizer, top_k=len(tokenizer)
    # )
    model = fill_mask_model.model
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    model_name = model_name.replace('cardiffnlp/', '')

    json_dict = {
        'model': model_name,
        'quarter': quarter,
        'split': split,
        # 'shots': N,
        # 'max_num_masks': M,
        'text': [],                 # list of size len(text_list)
        'gold_label': [],           # list of size len(text_list)
        'gold_num_masks': [],       # list of size len(text_list)
        'relation': [],             # list of size len(text_list)
        'num_answers': [],          # list of size len(text_list)
        # 'f1_micro': [],             # list of size len(text_list)
        # 'f1_macro': [],             # list of size len(text_list)
        # 'rouge': [],                # list of size len(text_list)
        # 'bleu': [],                 # list of size len(text_list)
        # 'bleu_uni_precision': [],   # list of size len(text_list)
        # 'bert_score': [],           # list of size len(text_list)
        # 'best_log_probs': [],       # list of size len(text_list)
        # 'best_pred_tokens': [],     # list of size len(text_list)
        # 'best_pred_strings': [],     # list of size len(text_list)
        'all_pppl_scores': [],  # dict
        'avg_pppl': None,      # dict
        'median_pppl': None

    }

    pppl_list = []

    for i in tqdm(range(len(data_dict['text']))):
    # for i in tqdm(range(10)):
        # load data
        text_i = data_dict['text'][i]  # contains only one <mask> independently of how many tokens the label has
        labels_i = data_dict['labels'][i]
        # labels_i_without_space = [label[1:] for label in labels_i if label[0] == ' ']
        labels_ids_i = data_dict['labels_ids'][i]  # multiple correct answers
        relation_i = data_dict['relation'][i]  # multiple correct answers

        # print(i, text_i)

        for l,label_ids_i in enumerate(labels_ids_i):
            # add extra masks if needed
            num_correct_masks = len(label_ids_i)
            new_masks = " ".join(["<mask>"] * num_correct_masks)
            orig_text_i = text_i.replace("<mask>", new_masks)

            # print(orig_text_i)
            # save to json dict
            json_dict['text'].append(text_i)
            json_dict['gold_label'].append(labels_i[l])
            # json_dict['gold_num_masks'].append(gold_num_masks)
            json_dict['gold_num_masks'].append(num_correct_masks)
            json_dict['relation'].append(relation_i)
            json_dict['num_answers'].append(len(labels_i))

            # tokenize orig sentence (with masks)
            tokenized_sentence_with_masks = tokenizer_return_id(tokenizer, orig_text_i)

            # where are the masks (indices)
            mask_inds = list(np.where(np.array(tokenized_sentence_with_masks) == tokenizer.mask_token_id)[0])

            # replace with correct labels
            tokenized_sentence_orig = [orig_tok if t not in mask_inds else label_ids_i[mask_inds.index(t)]
                                       for t, orig_tok in enumerate(tokenized_sentence_with_masks)
                                       ]
            sum_ppl = 0
            n_subtokens_total = num_correct_masks
            for mask_j in range(num_correct_masks):
                #         print(mask_j)
                # find index of current mask (in tokenized seuence)
                index_of_current_mask = mask_inds[mask_j]

                # change it with mask id
                tokenized_sentence = tokenized_sentence_orig.copy()
                tokenized_sentence[index_of_current_mask] = tokenizer.mask_token_id
                # print(tokenized_sentence)

                # correct label token id
                current_correct_token_id = label_ids_i[mask_j]

                with torch.no_grad():
                    # tensor
                    inp = torch.tensor([tokenized_sentence]).cuda() if cuda else torch.tensor([tokenized_sentence])

                    # get logits
                    out = model(inp)
                    logits = out.logits.detach().cpu()  # batch_size x max_len x vocab

                    # logit for the correct token
                    logits_for_mask = logits[:, index_of_current_mask][0]
                    log_softmax_for_mask = logits_for_mask.log_softmax(-1)

                    # log_softmax of correct label
                    score = log_softmax_for_mask[current_correct_token_id].item()
                    #             print(score)
                    sum_ppl += score

            pseudo_ppl = math.e ** (-1 * (sum_ppl / n_subtokens_total))
            # print('PPPL: {}'.format(pseudo_ppl))
            pppl_list.append(pseudo_ppl)

            json_dict['all_pppl_scores'].append(pseudo_ppl)
            # json_dict['all_preds'].append(all_ranked_gen_token_seqs)

    json_dict['avg_pppl'] = round(np.mean(pppl_list),4)
    json_dict['median_pppl'] = round(np.median(pppl_list),4)

    filename = 'full_results_{}_{}_{}'.format(model_name, quarter, split)
    torch.save(json_dict, os.path.join(save_dir, "{}.pt".format(filename)))
    return json_dict


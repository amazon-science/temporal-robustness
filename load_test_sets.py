import argparse
import sys
import os
import json
import pprint

import numpy as np
from tqdm import tqdm

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from fill_blank_eval import datasets
# from sys_config import DATA_DIR, available_datasets, LMs_names

class TestSetLoader():
    def __init__(self, args, test_name, test_dir, logger, use_negated_probes=False):
        """
        Loads a test set in the format required from the model under evaluation.
        BERT models use [MASK], while RoBERTa models use <mask>.
        """
        self.mask_token = args.mask_token
        self.tokenizer = args.tokenizer
        self.vocab_tokens = list(args.tokens2ids.keys())
        self.ids2tokens = args.ids2tokens
        self.tokens2ids = args.tokens2ids
        self.special_ids = args.special_ids
        self.max_seq_len = args.max_seq_len
        self.test_name = test_name
        self.test_dir = test_dir
        self.lowercase = args.lowercase
        self.use_negated_probes = use_negated_probes
        self.logger = logger
        self.single_mask = args.single_token
        self.max_num_masks = args.max_num_masks

    def get_test_set(self):
        """
        Returns list of masked_sentences and list of labels
        :return:
        """
        if self.test_name in ['lama-conceptnet', 'lama-google-re', 'lama-trex', 'lama-squad']:
            return self.load_lama()
        elif self.test_name == 'templama':
            return self.load_templama()
        elif self.test_name == 'dynamic-templama':
            return self.load_dynamic_templama()

    def load_dynamic_templama(self):
        """
        Our dynamically-created TempLAMA version
        :return:
        """
        test_filepath = os.path.join(self.test_dir, 'test.jsonl')
        test_data = self.load_file(test_filepath)
        quarters = sorted(list(set([d["date"] for d in test_data])))

        # split per quarter
        test_data_dict = {k: {"text": [], "labels": [], "labels_ids": [],
                              "relation": [], "num_answers":[], "num_masks":[]} for k in quarters}
        for d in tqdm(test_data):
            quarter = d["date"]  # string (e.g. '2019-Q1')
            relation = d['relation']  # string (e.g. 'P54')
            if type(d['answer']) is not list: d['answer'] = [d['answer']]
            labels_string_list = [label['name'] for label in d['answer']]  # list of strings with all the labels
            if self.lowercase:
                labels_string_list = [label_str.lower() for label_str in labels_string_list]
            labels_ids_list = [self.tokenizer_return_id(string) for string in labels_string_list]  # list of lists with ids
            num_masks_list = [len(tokens) for tokens in labels_ids_list]
            """
            If we want to consider more correct labels (e.g. synonyms, simplification etc) we have to change accepted labels
            """
            # Filter labels to keep only those that are maximum M tokens (for English we have 5 as default)
            accepted_labels_ids_index_list = [i for i,x in enumerate(labels_ids_list) if len(x) <= self.max_num_masks]
            if len(accepted_labels_ids_index_list)==0:
                continue
            else:
                accepted_labels_ids_list = np.array(labels_ids_list)[accepted_labels_ids_index_list].tolist()
                accepted_num_masks_list = np.array(num_masks_list)[accepted_labels_ids_index_list].tolist()

            # If we evaluate only single-token labels
            if self.single_mask:
                # check if there is answer with one mask
                if 1 in accepted_num_masks_list:
                    labels_ids_with_one_mask_index = [i for i, l in enumerate(accepted_num_masks_list) if
                                                      l == 1]  # list of ids where answer requires one mask/token
                    # accepted_labels = list(np.array(labels_string_list)[labels_ids_with_one_mask_index])
                    accepted_labels_ids = np.array(accepted_labels_ids_list)[labels_ids_with_one_mask_index].tolist()
                    accepted_labels = [[self.tokenizer.decode(label_id)] for label_id in accepted_labels_ids]
                    assert len(accepted_labels) == len(accepted_labels_ids)
                    assert len(accepted_labels[0]) == len(accepted_labels_ids[0])
                    text = d["query"].replace("_X_", self.mask_token)
                    num_answers = len(accepted_labels)
                    if self.lowercase:
                        text = d["query"].lower().replace("_x_", self.mask_token)
                else:
                    # skip this example
                    continue
            else:
                # check if needed to lowercase
                if self.lowercase:
                    accepted_labels_ids = accepted_labels_ids_list
                    accepted_labels = [[self.tokenizer.decode(label_id)] for label_id in accepted_labels_ids_list]
                    num_answers = len(accepted_labels_ids)
                    multiple_masks = [" ".join([self.mask_token for _ in range(0,num_masks)]) for num_masks in
                                      accepted_num_masks_list]
                    text = d["query"].lower().replace("_x_", self.mask_token)
                    # text = d["query"].lower().replace("_x_", multiple_masks)
                else:
                    # all_answers = [a["name"] for a in d["answer"]]
                    # accepted_labels = [[self.tokenizer.decode(label_id)] for label_id in accepted_labels_ids_list]
                    # num_answers = len(all_answers)
                    accepted_labels_ids = accepted_labels_ids_list
                    accepted_labels = [[self.tokenizer.decode(label_id)] for label_id in accepted_labels_ids_list]
                    num_answers = len(accepted_labels_ids)
                    multiple_masks = [" ".join([self.mask_token for _ in range(0,num_masks)]) for num_masks in
                                      accepted_num_masks_list]
                    # text = [d["query"].replace("_X_", mask_string) for mask_string in multiple_masks]
                    text = d["query"].replace("_X_", self.mask_token)

            test_data_dict[quarter]["text"].append(text)  # list of strings
            test_data_dict[quarter]["labels"].append(accepted_labels)  # list of list of strings
            test_data_dict[quarter]["labels_ids"].append(accepted_labels_ids)  # list of lists of ints
            test_data_dict[quarter]["relation"].append(relation) # string
            test_data_dict[quarter]["num_answers"].append(num_answers)  # int if num_answers > 1
            test_data_dict[quarter]["num_masks"].append(accepted_num_masks_list)

        return test_data_dict


    #
    def tokenizer_return_id(self, text):
        output = self.tokenizer(text)
        token_ids = [i for i in output['input_ids'] if i not in self.special_ids]
        return token_ids

    def load_file(self, filename):
        """
        :param filename:
        :return:
        """
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def change_mask_token(self, samples, use_negated_probes=False):
        """
        LAMA datasets are already filled with the [MASK] token, which is only for BER models.
        For RoBERTa and other models we should replace [MASK] with the correct mask token.
        :param samples:
        :return:
        """
        new_samples = []
        for sample in samples:
            new_masked_sentences = []

            if self.test_name == 'lama-trex':
                list_of_sentences = [x['masked_sentence'] for x in sample['evidences']]
            else:
                list_of_sentences = sample["masked_sentences"]
            for sentence in list_of_sentences:
                if '[MASK]' in sentence:
                    sentence = sentence.replace("[MASK]", self.mask_token)
                    new_masked_sentences.append(sentence)
            sample["masked_sentences"] = new_masked_sentences

            if "negated" in sample and use_negated_probes:
                for sentence in sample["negated"]:
                    if '[MASK]' in sentence:
                        sentence = sentence.lower()
                        sentence = sentence.replace("[MASK]", self.mask_token)
                        new_masked_sentences.append(sentence)
                sample["negated"] = new_masked_sentences

            new_samples.append(sample)
        return new_samples


if __name__ == "__main__":
    ##########################################################################
    # Setup args
    ##########################################################################
    parser = argparse.ArgumentParser()
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument(
        # "--language-models",
        "--lms",
        # dest="models",
        help="comma separated list of language models. from {}".format(LMs_names),
        # default=["bert-base-cased", "bert-base-uncased",
        #          "bert-large-uncased", "bert-large-cased"
        #                                "roberta-base", "roberta-large",
        #          "cardiffnlp/twitter-roberta-base", "cardiffnlp/twitter-roberta-base-2019-90m",
        #          "cardiffnlp/twitter-roberta-base-2021-124m"],
        default=[
            'cardiffnlp/twitter-roberta-base-2019-90m',
            # 'cardiffnlp/twitter-roberta-base-mar2020',
            # 'cardiffnlp/twitter-roberta-base-jun2020',
            # 'cardiffnlp/twitter-roberta-base-sep2020',
            # 'cardiffnlp/twitter-roberta-base-dec2020',
            # 'cardiffnlp/twitter-roberta-base-mar2021',
            # 'cardiffnlp/twitter-roberta-base-jun2021',
            # 'cardiffnlp/twitter-roberta-base-sep2021',
            # 'cardiffnlp/twitter-roberta-base-dec2021',
            # # 'cardiffnlp/twitter-roberta-base-2021-124m',
            # 'cardiffnlp/twitter-roberta-base-mar2022'
        ],
        nargs='+',
        required=False,
    )
    parser.add_argument(
        # "--language-models",
        "--vocab_subset",
        action="store_true",
        help="if added (True) then we compute a joint vocab from all the models we want to evaluate/compare (args.lms)",
        # default=["bert-base-cased"],
        required=False,
    )
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument(
        "--datasets",
        # "--lms",
        # dest="models",
        # help="comma separated list of datasets (test sets) from {}".format(available_datasets),
        # options=available_datasets,
        nargs='+',
        # default=["lama-google-re", "lama-squad", "lama-conceptnet", "lama-trex"],
        default=["dynamic-templama"],
        required=False,
    )
    # parser.add_argument(
    #     "--temporal",
    #     help="comma separated list of datasets (test sets) from {}".format(available_datasets),
    #     # options=available_datasets,
    #     nargs='+',
    #     # default=["lama-google-re", "lama-squad", "lama-conceptnet", "lama-trex"],
    #     required=False,
    # )
    parser.add_argument(
        # "--language-models",
        "--new",
        action="store_true",
        help="if added (True) use new data",
        # default=["bert-base-cased"],
        required=False,
    )
    ##########################################################################
    # Temporal args
    ##########################################################################
    parser.add_argument("--min_year", default=2018, help="minimum year to get facts", required=False)
    parser.add_argument("--min_month", default=1, help="minimum month to get facts", required=False)
    parser.add_argument("--min_day", default=1, help="minimum day to get facts", required=False)
    parser.add_argument("--max_year", default=2022, help="maximum year to get facts", required=False)
    parser.add_argument("--max_month", default=12, help="maximum month to get facts", required=False)
    parser.add_argument("--max_day", default=31, help="maximum day to get facts", required=False)
    parser.add_argument("--granularity", default="quarter", help="granularity to create test sets"
                                                                 "between [month, quarter,year]", required=False)
    ##########################################################################
    # Evaluation args
    ##########################################################################
    parser.add_argument(
        "--topk",
        # "--lms",
        # dest="models",
        help="comma separated list of datasets (test sets)",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--single_token",
        action="store_true",
        help="if True, we consider only single tokens as labels.",
        # default=False,
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
        "--batch_size",
        default=32,
        required=False,
    )
    parser.add_argument(
        "--threads",
        # "--lms",
        # dest="models",
        # help="directory to save logs (relative to /logs/)",
        default=0,
        required=False,
    )
    ##############
    parser.add_argument(
        "--spacy_model",
        "--sm",
        dest="spacy_model",
        default="en_core_web_sm",
        help="spacy model file path",
    )
    parser.add_argument(
        "--common-vocab-filename",
        "--cvf",
        dest="common_vocab_filename",
        help="common vocabulary filename",
    )
    parser.add_argument(
        "--interactive",
        "--i",
        dest="interactive",
        action="store_true",
        help="perform the evaluation interactively",
    )
    parser.add_argument(
        "--max-sentence-length",
        dest="max_sentence_length",
        type=int,
        default=100,
        help="max sentence lenght",
    )

    args = parser.parse_args()

    temporal_string = '{}-{}-{}_to_{}-{}-{}_per_{}'.format(args.min_year,
                                                           args.min_month,
                                                           args.min_day,
                                                           args.max_year,
                                                           args.max_month,
                                                           args.max_day,
                                                           args.granularity)

    test_dir = os.path.join(DATA_DIR, 'dynamic-templama', 'dataset_from_' + temporal_string)

    data_loader = TestSetLoader(args=args,
                                test_name="dynamic_templama",
                                test_dir=test_dir,
                                logger=None)

    # data_loader = TestSetLoader(model_type=model_type,
    #                             test_name=test_name,
    #                             test_dir=test_dir)
    test_set = data_loader.get_test_set()
    print()

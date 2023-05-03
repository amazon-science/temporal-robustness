import collections
import csv
import datetime
import json
import os
import pickle
import random
import tensorflow as tf
import torch
from tqdm import tqdm

from absl import app
from absl import logging
from absl import flags

FLAGS = flags.FLAGS

templama_docker_dir = os.path.dirname(os.path.realpath(__file__))
flags.DEFINE_string("out_dir",
                    'extracted_facts_from_2010',
                    # os.path.join(templama_docker_dir, 'extracted_facts'),
                    "Path to store constructed queries.")

flags.DEFINE_string("templates",
                    "my_templates.csv",
                    "Filename of csv templates file (e.g. templates.csv)")

flags.DEFINE_string("exp_identifier",
                    None,
                    "string to be appended in the end of the directory name where the dataset will be saved")

flags.DEFINE_integer(
    "min_year", 2019,
    "Starting year to construct queries from. Only facts which have a start / "
    "end date after this will be considered.")
flags.DEFINE_integer(
    "min_month", 1,
    "Starting month to construct queries from. Only facts which have a start / "
    "end date after this will be considered.")
flags.DEFINE_integer(
    "min_day", 1,
    "Starting day to construct queries from. Only facts which have a start / "
    "end date after this will be considered.")
flags.DEFINE_integer("max_year", 2022,
                     "Ending year to construct queries up till.")
flags.DEFINE_integer(
    "max_month", 12,
    "Ending month to construct queries up till.")
flags.DEFINE_integer(
    "max_day", 31,
    "Ending day to construct queries up till.")
flags.DEFINE_string("granularity",
                    "quarter",
                    "Granularity of created test sets between 'month', 'quarter','year'")
random.seed(42)
Y_TOK = "_X_"
WIKI_PRE = "/wp/en/"


def read_templates(csv_filename="my_templates.csv"):
    """Loads relation-specific templates from `templates.csv`.
    csv_filename: filename of the csv file with the templates, stored in the ame directory as this script
    Returns:
      a dict mapping relation IDs to string templates.
    """
    my_path = os.path.dirname(os.path.realpath(__file__))
    # template_file = os.path.join(my_path, "my_templates.csv")
    template_file = os.path.join(my_path, csv_filename)
    logging.info("Reading templates from %s", template_file)
    reader = csv.reader(tf.io.gfile.GFile(template_file))
    headers = next(reader, None)
    data = collections.defaultdict(list)
    for row in reader:
        for h, v in zip(headers, row):
            data[h].append(v)
    templates = dict(zip(data["Wikidata ID"], data["Template"]))
    logging.info("\n".join("%s: %s" % (k, v) for k, v in templates.items()))
    return templates


def _datetup2int(date):
    """Convert (year, month, day) to integer representation.

    Args:
      date: Tuple of (year, month, day).

    Returns:
      an int of year * 1e4 + month * 1e2 + day.
    """
    dint = date[0] * 1e4
    dint += date[1] * 1e2 if date[1] else 0
    dint += date[2] if date[2] else 0
    return dint


def date_in_interval(date, start, end):
    """Check if date is within start and end.

    Args:
      date: Tuple of (year, month, day).
      start: Start date (year, month, day).
      end: End date (year, month, day).

    Returns:
      a bool of whether start <= date <= end.
    """
    date_int = _datetup2int(date)
    start_int = _datetup2int(start) if start else 0
    end_int = _datetup2int(end) if end else 21000000
    return date_int >= start_int and date_int <= end_int


def parse_date(date_str):
    """Try to parse date from string.

    Args:
      date_str: String representation of the date.

    Returns:
      date: Tuple of (year, month, day).
    """
    date = None
    try:
        if len(date_str) == 4:
            date_obj = datetime.datetime.strptime(date_str, "%Y")
            date = (date_obj.year, None, None)
        elif len(date_str) == 6:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m")
            date = (date_obj.year, date_obj.month, None)
        elif len(date_str) == 8:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
            date = (date_obj.year, date_obj.month, date_obj.day)
    except ValueError:
        pass
    if date is not None and date[0] > 2100:
        # Likely an error
        date = None
    return date


def resolve_objects(facts):
    """Combine consecutive objects across years into one fact.

    Args:
      facts: A list of fact tuples.

    Returns:
      a list of fact tuples with consecutive facts with the same object merged.
    """

    def _datekey(fact):
        start = _datetup2int(fact[3]) if fact[3] else 0
        end = _datetup2int(fact[4]) if fact[4] else 21000000
        return (start, end)

    # First sort by start time and then by end time.
    sorted_facts = sorted(facts, key=_datekey)
    # Merge repeated objects into one.
    out_facts = [sorted_facts[0]]
    for fact in sorted_facts[1:]:
        if (fact[2] == out_facts[-1][2] and fact[3] != fact[4] and
                out_facts[-1][3] != out_facts[-1][4]):
            out_facts[-1][4] = fact[4]
        else:
            out_facts.append(fact)
    return out_facts


def _build_example(query):
    """Creates a tf.Example for prediction with T5 from the input query.

    Args:
      query: a dict mapping query features to their values.

    Returns:
      a tf.train.Example consisting of the query features.
    """
    # Inputs and targets.
    inp = query["query"].encode("utf-8")
    trg = query["answer"]["name"].encode("utf-8")
    # Metadata.
    id_ = query["id"].encode("utf-8")
    recent = query["most_recent_answer"]["name"].encode("utf-8")
    frequent = query["most_frequent_answer"]["name"].encode("utf-8")
    rel = query["relation"].encode("utf-8")
    # Construct TFRecord.
    feature = {
        "id":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_])),
        "date":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(query["date"])])),
        "relation":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[rel])),
        "query":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp])),
        "answer":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[trg])),
        "most_frequent_answer":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[frequent])),
        "most_recent_answer":
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[recent])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _map_years_to_objects(facts, qid_numfacts, min_year, max_year):
    """Map each year between min, max to the corresponding object in facts.

    Args:
      facts: a list of facts with the same subject and relation.
      qid_numfacts: a dict mapping wikidata QIDs to number of facts.
      min_year: an int, starting year to map.
      max_year: an int, ending year to map.

    Returns:
      year2obj: a dict mapping each year between (min_year, max_year) to the
        corresponding most 'popular' object for that year.
    """
    year2obj = {}
    numfacts = lambda x: qid_numfacts.get(x, 0)
    for f in facts:
        min_ = f[3][0] if f[3] is not None else min_year
        max_ = f[4][0] if f[4] is not None else max_year
        min_ = max(min_, min_year)
        max_ = min(max_, max_year)
        for yr in range(min_, max_ + 1):
            if yr in year2obj:
                # Keep the more popular object.
                if numfacts(year2obj[yr]) < numfacts(f[2]):
                    year2obj[yr] = f[2]
            else:
                year2obj[yr] = f[2]
    return year2obj


def _all_quarters_for_a_year(year):
    """

    Args:
        year (int): the year

    Returns:
    quarters (dict)
    """
    quarters = {
        "Q1": {'start_date': (year, 1, 1), 'end_date': (year, 3, 31)},
        "Q2": {'start_date': (year, 4, 1), 'end_date': (year, 6, 31)},
        "Q3": {'start_date': (year, 7, 1), 'end_date': (year, 9, 31)},
        "Q4": {'start_date': (year, 10, 1), 'end_date': (year, 12, 31)},
    }
    return quarters


def correct_date_tuple_form(current_tuple, day, month, year):
    """
    Get the date of a fact in the tuple format (Y,M,D) and returns a detailed format:
    - if the date is unknown (None) return the given (year,month,day)
    - if the year is unknown replace it with 'year'
    - if the month is unknown replace it with 'month'
    - if the day is unknown replace it with 'day'
    Args:
        current_tuple: a tuple of ints representing (Y,M,D). (could be `None` though!)
        day: an int, which day to map.
        month: an int, which month to map.
        year: an int, which year to map.

    Returns:

    """
    # correct_date = current_tuple
    # Create start/end dates for facts in the form (Y,M,D)
    if current_tuple is None:
        # If start date unknown, use the min_date
        correct_date = (year, month, day)
    else:
        current_year, current_month, current_day = current_tuple
        correct_year = year if current_year is None else current_year
        correct_month = month if current_month is None else current_month
        correct_day = day if current_day is None else current_day
        correct_date = (correct_year, correct_month, correct_day)
        # if current_year is None:
        #     # If start year unknown, use the min_year
        #     correct_date = (year, current_month, current_day)
        # if current_month is None:
        #     # If start month unknown, use the min_month
        #     correct_date = (current_year, month, current_day)
        # if current_day is None:
        #     # If start day unknown, use the min_day
        #     correct_date = (current_year, current_month, day)

    return correct_date


def _map_quarters_to_objects(facts, qid_numfacts,
                             min_year, min_month, min_day,
                             max_year, max_month, max_day):
    """Map each year between min, max to the corresponding object in facts.

    Args:
      facts: a list of facts with the same subject and relation.
      qid_numfacts: a dict mapping wikidata QIDs to number of facts.
      min_year: an int, starting year to map.
      max_year: an int, ending year to map.

    Returns:
      year2obj: a dict mapping each year between (min_year, max_year) to the
        corresponding most 'popular' object for that year.
    """
    quarter2obj = {}
    quarter2time = {}
    for year in range(min_year, max_year + 1):
        year_quarters = _all_quarters_for_a_year(year)
        for quarter in year_quarters:
            quarter2time['{}-{}'.format(year, quarter)] = year_quarters[quarter]
    all_quarters = list(quarter2time.keys())
    numfacts = lambda x: qid_numfacts.get(x, 0)
    for f in facts:
        start_date = correct_date_tuple_form(current_tuple=f[3],
                                             day=min_day, month=min_month, year=min_year)
        end_date = correct_date_tuple_form(current_tuple=f[4],
                                           day=max_day, month=max_month, year=max_year)

        # transform tuple (Y,M,D) to int for comparison
        start_date_int = _datetup2int(start_date)
        end_date_int = _datetup2int(end_date)

        for q in all_quarters:
            min_quarter_date_int = _datetup2int(quarter2time[q]['start_date'])
            max_quarter_date_int = _datetup2int(quarter2time[q]['end_date'])
            """
            Eligible facts for a Q are those that have:
              (1) start date earlier than max quarter date
              (2) end date later than min quarter date
            """
            if start_date_int <= max_quarter_date_int:  # eligible
                if end_date_int >= min_quarter_date_int:
                    # Add to bucket
                    if q in quarter2obj:
                        # Keep the more popular object.
                        # print(quarter2obj[q])
                        # if numfacts(quarter2obj[q]) < numfacts(f[2]):
                        quarter2obj[q].append(f[2])
                    else:
                        quarter2obj[q] = [f[2]]
    return quarter2obj


def create_queries(all_facts, templates, qid_names, qid_numfacts,
                   min_year, min_month, min_day,
                   max_year, max_month, max_day,
                   train_frac, val_frac,
                   granularity,
                   max_subject_per_relation,
                   exp_identifier=None):
    """Construct queries for most popular subjects for each relation.

    Args:
      out_dir: Path to store all queries as well as yearly slices.
      all_facts: a list of facts.
      templates: a dict mapping relation IDs to templates.
      qid_names: dict mapping wikidata QIDs to canonical names.
      qid_numfacts: dict mapping wikidata QIDs to number of facts.
      min_year: an int, starting year to map.
      max_year: an int, ending year to map.
      train_frac: a float, fraction of subjects to reserve for the train set.
      val_frac: a float, fraction of subjects to reserve for the val set.
      granularity: quarter/year/month
      max_subject_per_relation: number of subjects to keep per relation.
    """

    def _create_entity_obj(qid):
        return {"wikidata_id": qid, "name": qid_names[qid]}

    def _create_implicit_query(subj, tmpl):
        # change this if need to add more templates
        return tmpl.replace("<subject>", qid_names[subj]).replace("<object>", Y_TOK)

    def _most_frequent_answer(year2obj):
        counts = collections.defaultdict(int)
        for _, obj in year2obj.items():
            counts[obj] += 1
        return max(counts.items(), key=lambda x: x[1])[0]

    def _most_recent_answer(yr2obj):
        recent = max(yr2obj.keys())
        return yr2obj[recent]

    # Group by relation and by sort by subject
    logging.info("Keeping only facts with templates.")
    rel2subj = {}  # dict with keys the wikidata id (e.g. P286) and values dicts with subj ids + lists of facts
    for fact in tqdm(all_facts):
        if fact[0] not in templates:
            continue
        if fact[0] not in rel2subj:
            rel2subj[fact[0]] = {}
        if fact[1] not in rel2subj[fact[0]]:
            rel2subj[fact[0]][fact[1]] = []
        rel2subj[fact[0]][fact[1]].append(fact)
    logging.info('total facts ' + str([(x, len(rel2subj[x])) for x in rel2subj]))
    logging.info("Sorting subjects by 'popularity' resolving multiple objects.")
    sorted_rel2subj = {}
    for relation in rel2subj:
        sorted_subjs = sorted(
            rel2subj[relation].keys(),
            key=lambda x: qid_numfacts.get(x, 0),
            reverse=True)
        sorted_rel2subj[relation] = [
            (s, resolve_objects(rel2subj[relation][s])) for s in sorted_subjs
        ]

    logging.info("Keep only subjects with multiple objects.")
    total_facts = 0
    filt_rel2subj = {}
    for rel, subj2facts in sorted_rel2subj.items():
        filt_subj2facts = list(filter(lambda x: len(x[1]) > 1, subj2facts))
        if filt_subj2facts:
            filt_rel2subj[rel] = filt_subj2facts
            total_facts += sum([len(f) for _, f in filt_rel2subj[rel]])
    logging.info("# facts after filtering = %d", total_facts)
    logging.info('total facts ' + str([(x, len(rel2subj[x])) for x in rel2subj]))
    logging.info("Keep only %d subjects per relation, split into train/val/test",
                 max_subject_per_relation)
    train_queries, val_queries, test_queries = [], [], []
    tot_queries, tot_subj = 0, 0
    for relation, subj2facts in filt_rel2subj.items():
        num_subj = 0
        for subj, facts in subj2facts:
            if granularity == 'quarter':
                year2obj = _map_quarters_to_objects(facts, qid_numfacts, min_year, min_month, min_day,
                                                    max_year, max_month, max_day)
            elif granularity == 'month':
                NotImplementedError
            else:
                year2obj = _map_years_to_objects(facts, qid_numfacts, min_year, max_year)

            p = random.random()  # to decide which split this subject belongs to.
            for yr, obj_list in year2obj.items():
                query = {
                    "query":
                        _create_implicit_query(subj, templates[relation]),
                    "answer":
                        [_create_entity_obj(obj) for obj in obj_list],
                    "date":
                        str(yr),
                    "id":
                        subj + "_" + relation + "_" + str(yr),
                    # "most_frequent_answer":
                    #     _create_entity_obj(_most_frequent_answer(year2obj)),
                    # "most_recent_answer":
                    #     _create_entity_obj(_most_recent_answer(year2obj)),
                    "relation":
                        relation,
                }
                if p < train_frac:
                    train_queries.append(query)
                elif p < train_frac + val_frac:
                    val_queries.append(query)
                else:
                    test_queries.append(query)
                tot_queries += 1
            num_subj += 1
            if num_subj == max_subject_per_relation:
                break
        logging.info("%s: # subjects = %d # train = %d # val = %d # test = %d",
                     relation, len(subj2facts), len(train_queries),
                     len(val_queries), len(test_queries))
        tot_subj += num_subj

    save_dir = os.path.join(templama_docker_dir, 'dataset_from_{}-{}-{}_to_{}-{}-{}_per_{}'.format(min_year,
                                                                                                   min_month,
                                                                                                   min_day,
                                                                                                   max_year,
                                                                                                   max_month,
                                                                                                   max_day,
                                                                                                   granularity))

    if exp_identifier is not None:
        save_dir += '_{}'.format(exp_identifier)
    # Save all queries as a json.
    split2qrys = {
        "train": train_queries,
        "val": val_queries,
        "test": test_queries
    }
    tf.io.gfile.makedirs(save_dir)
    print("Saving all queries to %s", save_dir)
    for split in ["train", "val", "test"]:
        with tf.io.gfile.GFile(os.path.join(save_dir, f"{split}.jsonl"), "w") as f:
            for qry in split2qrys[split]:
                f.write(json.dumps(qry) + "\n")

    # # Make subdirectories and store each split.
    # for year in range(min_year, max_year + 1):
    #     subd = os.path.join(save_dir, "yearly", str(year))
    #     tf.io.gfile.makedirs(subd)
    #     logging.info("Saving queries for %d to %s", year, subd)
    #     counts = collections.defaultdict(int)
    #     for split in ["train", "val", "test"]:
    #         with tf.io.TFRecordWriter(os.path.join(subd, f"{split}.tf_record")) as f:
    #             for qry in split2qrys[split]:
    #                 if qry["date"] == str(year):
    #                     f.write(_build_example(qry).SerializeToString())
    #                     counts[split] += 1


def main(_):
    out_dir = os.path.join(templama_docker_dir, FLAGS.out_dir)
    qids_pt = os.path.join(out_dir, 'my_qids.pt')
    all_facts_pt = os.path.join(out_dir, 'my_all_facts.pt')

    # Load entity names, number of facts and wiki page titles from SLING.
    logging.info("Checking if qids_pt file exists...")
    if os.path.isfile(qids_pt):
        logging.info("Found! Loading from {}...".format(qids_pt))
        logging.info("This process usually takes up to 4 minutes.")
        qid_names, qid_mapping, qid_numfacts = torch.load(qids_pt)
    else:
        logging.info("Not found! Run get_facts.py first...")
        exit()

    # Load facts with qualifiers.
    logging.info("Checking if type all_facts_pt file exists...")
    if os.path.isfile(all_facts_pt):
        logging.info("Found! Loading from {}...".format(all_facts_pt))
        all_facts = torch.load(all_facts_pt)
    else:
        logging.info("Not found! Run get_facts.py first...")
        exit()

    # Load relation templates.
    logging.info("Read templates...")
    templates = read_templates(FLAGS.templates)

    logging.info("Start creating queries!...")
    create_queries(all_facts, templates, qid_names, qid_numfacts,
                   min_year=FLAGS.min_year, min_month=FLAGS.min_month, min_day=FLAGS.min_day,
                   max_year=FLAGS.max_year, max_month=FLAGS.max_month, max_day=FLAGS.max_day,
                   train_frac=0.0, val_frac=0.0,
                   granularity=FLAGS.granularity,
                   max_subject_per_relation=5000,
                   exp_identifier=FLAGS.exp_identifier)


if __name__ == "__main__":
    app.run(main)

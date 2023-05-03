# Dynamic Evaluation of Language Model Robustness on Temporal Concept Drift

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---
Please refer to our paper for more details:

[Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views Paper](https://arxiv.org/pdf/2302.12297.pdf) Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, Miguel Ballesteros. EACL 2023

This repository contains:
- the `Dynamic-TempLAMA` test set, an extension of the [TempLAMA](https://github.com/google-research/language/tree/master/language/templama) dataset that contains facts over time collected by [Wikidata](https://www.wikidata.org/wiki/Wikidata:List_of_properties);
- code to dynamically create new test sets and update the current dataset with new relations, more templates and in any time granularity (month/quarter/year);
- code to evaluate [TimeLMs](https://github.com/cardiffnlp/timelms), a series of RoBERTa-base Twitter models that are continuously trained on social media data each quarter (from `2019-Q4` until `2022-Q2`).


---
## Getting Started
After cloning this repository, you need to create a conda environment. Our implementation is based on Python 3.6+ and PyTorch. You can create a conda environment by running the following:
```bash
conda create -n robot -y python=3.7 && conda activate robot
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
---
## Download Data

### Dynamic-TempLAMA
The Dynamic-TempLAMA test sets are in the directory `/data/dynamic-templama`. You may find two variants of the dataset:
- `dataset_from_2019-1-1_to_2022-6-31_per_quarter`: In this folder, there is the file `test.jsonl` that contains all facts from 1/1/2019 until 31/6/2022 with the relations and the templates as defined in `my_templates.csv`.
- `dataset_from_2019-1-1_to_2022-6-31_per_quarter_improved` In this folder, there is also a `test.jsonl` file that contains the same facts but with different templates as defined in `my_improved_templates.csv`.

---

## Dynamically Update & Extend <our-dataset>
In the `dynamic_data_collection` folder, we provide the code that allows to dynamically create new test sets for our dataset, or to create customs test sets. 

We base our implementation to the [TempLAMA](https://github.com/google-research/language/tree/master/language/templama) repository, but we extend it in order to be able to download dynamically (i) any fact from Wikidata, (ii) select any time granularity among month, quarter and year, (iii) from any time period and (iv) multiple templates.
The initial script from the TempLAMA repository is `prepare_data.sh` and `templama.py`. But we split it in three scrips: `sling2facts.py`, `get_facts.py`, and `create_templates.py`.

[//]: # (### How to download new TempLAMA data with &#40;1&#41; more relations, &#40;2&#41; more templates, &#40;3&#41; different granularity &#40;not only yearly&#41; and &#40;4&#41; any period of time?)
[//]: # (We currently have the scripts in the `/templama_docker` directory due to the docker issue &#40;see below&#41;. )


The steps to re-create our test set or to create a custom one, are the following:
0. `templates.csv`: Use the existing file or update it accordingly in order to include all the facts/relations/templates that you want to download from [Wikidata](https://www.wikidata.org/wiki/Wikidata:List_of_properties).
1. `sling2facts.py`: Run this script to download the sling Wikidata KB.
2. `get_facts.py`: This script downloads all facts from sling KB and stores in the `out_dir` directory `torch` files with formatted facts required from the next script. It requires ~20' in total (~15' for the `qids` are required and ~5' for `all_facts`).
Example usage:
```bash
python get_facts.py --out_dir extracted_facts_from_2010 --min_year 2010) 
```
3. `create_templates.py`: This script finally creates the temporally aligned test sets. It requites ~5' to create all test sets.
Example usage:
```bash
python create_templates.py --min_year 2019 --min_month 1 --min_day 1 --max_year 2022 --max_month 6 --max_day 31 --granularity quarter --templates templates.csv 
```

---
## Models
We test the TimeLMs on our dynamically created test sets. We download the model checkpoints from [HuggingFace](https://huggingface.co/models) (all models can be found [here](https://huggingface.co/cardiffnlp)).
Our code currently support the following models:

TimeLMs (RoBERTa models trained in the Twitter domain)
- `cardiffnlp/twitter-roberta-base-2019-90m`
- `cardiffnlp/twitter-roberta-base-mar2020`
- `cardiffnlp/twitter-roberta-base-jun2020`
- `cardiffnlp/twitter-roberta-base-sep2020`
- `cardiffnlp/twitter-roberta-base-dec2020`
- `cardiffnlp/twitter-roberta-base-mar2021`
- `cardiffnlp/twitter-roberta-base-jun2021`
- `cardiffnlp/twitter-roberta-base-sep2021`
- `cardiffnlp/twitter-roberta-base-dec2021`
- `cardiffnlp/twitter-roberta-base-2021-124m`
- `cardiffnlp/twitter-roberta-base-mar2022`
- `cardiffnlp/twitter-roberta-base-jun2022`

BERT models:
- `bert-base-uncased`, `bert-base-cased`
- `bert-large-uncased`, `bert-large-cased`

RoBERTa models:
- `roberta-base`
- `roberta-large`
- `cardiffnlp/twitter-roberta-base`

---

# Evaluation
The main script for evaluating models for temporal robustness is `fill_blank_eval.py`. We provide three ways of evaluation:
- `single-token`: Following the original [LAMA probe](https://aclanthology.org/D19-1250.pdf) (and all related works for evaluating _masked_ language models) we consider only single-token objects as acceptable labels. This results in severe filtering of the original dataset, as most objects contain multiple tokens.
- `multi-token`: In order to consider objects with multiple tokens, we use the technique from [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://arxiv.org/abs/1902.04094).
- `mlm-scoring`: Based in [Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240/) paper & 
  implementation.

The following usage examples can be used to evaluate with the three aforementioned techniques, respectively:
```bash 
# single-token
python fill_blank_eval.py --lms cardiffnlp/twitter-roberta-base-mar2022 cardiffnlp/twitter-roberta-base-jun2022 --single-token
# multi-token
python fill_blank_eval.py --lms cardiffnlp/twitter-roberta-base-mar2022 cardiffnlp/twitter-roberta-base-jun2022 --seed 42
# mlm-scoring
```

---

# Acknowledgements

Our work Dynamic-TempLAMA extended TempLAMA. TempLAMA's repository:

https://github.com/google-research/language/tree/master/language/templama

containing the code as described in this paper:

Time-Aware Language Models as Temporal Knowledge Bases, Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, 
Daniel Gillick, Jacob Eisenstein, William W. Cohen, TACL 2022
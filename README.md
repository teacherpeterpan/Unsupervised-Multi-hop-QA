# Unsupervised-Multi-hop-QA

This repository contains code and models for the paper: [Unsupervised Multi-hop Question Answering by Question Generation (NAACL 2021)](https://arxiv.org/pdf/2010.12623.pdf). 

- We propose MQA-QG, an **unsupervised question answering** framework that can generate human-like multi-hop training pairs from both homogeneous and heterogeneous data sources. 

- We find that we can train a competent multi-hop QA model with only generated data. The F1 gap between the unsupervised and fully-supervised models is less than 20 in both the [HotpotQA](https://hotpotqa.github.io/) and the [HybridQA](https://hybridqa.github.io/) dataset.

- Pretraining a multi-hop QA model with our generated data would greatly reduce the demand for human-annotated training data for multi-hop QA. 

## Introduction

The model first defines a set of **basic operators** to
retrieve / generate relevant information from each
input source or to aggregate different information, as follows. 

<p align="center">
<img src=Resource/operators.png width=700/>
</p>

Afterwards, we define six **Reasoning Graphs**. Each corresponds to one type of multihop question and is formulated as a computation graph built upon the operators. We generate multihop question-answer pairs by executing the reasoning graph. 

<p align="center">
<img src=Resource/framework.png width=800/>
</p>

## Requirements

- Python 3.7.3
- torch 1.7.1
- tqdm 4.49.0
- transformers 4.3.3
- stanza 1.1.1
- nltk 3.5

## Data Preparation

Make the following data directories: 
```shell
mkdir -p ./Data
mkdir -p ./Data/HotpotQA
mkdir -p ./Data/HybridQA
```

### a) HotpotQA

First, download the raw dataset of hotpotQA. 

```shell
HOTPOT_HOME=./Data/HotpotQA
mkdir -p $HOTPOT_HOME/raw
mkdir -p $HOTPOT_HOME/dataset
cd $HOTPOT_HOME/raw
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

Then, run the following code to preprocess the raw dataset. 

```shell
python prep_data_hotpotQA.py \
  --train_dir $HOTPOT_HOME/raw/hotpot_train_v1.1.json \
  --dev_dir $HOTPOT_HOME/raw/hotpot_dev_distractor_v1.json \
  --output_dir $HOTPOT_HOME/dataset/
```

You would be able to get the following files in `./Data/HotpotQA/dataset/`
```
train.src.txt
train.qa.txt
dev.src.txt
dev.qa.txt
```

### b) HybridQA

Download all the tables and passages of HybridQA into your data folder. 

```shell
HYBRID_HOME=./Data/HybridQA
cd HYBRID_HOME
git clone https://github.com/wenhuchen/WikiTables-WithLinks
```

## Operators

Here are the codes that test our key operators: `QGwithAns` and `DescribeEnt`. 

### a) QGwithAns

QGwithAns generate a single-hop question Q with answer A from the input text D. We implement this module based on the pretrained QG model from [patil-suraj](https://github.com/patil-suraj/question_generation), a Google T5 model finetuned on the SQuAD 1.1 dataset. 

You could test this module by running the following python codes: 
```python
from MQA_QG.Operators import T5_QG

test_passage = '''Jenson Alexander Lyons Button (born 19 January 1980) is a British racing driver and former Formula One driver. He won the 2009 Formula One World Championship, driving for Brawn GP.'''

nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight")

print(nlp.qg_without_answer(test_passage))
print(nlp.qg_with_answer_text(test_passage, "19 January 1980"))
```

## Multi-hop Question Generation

Coming Soon...



## Unsupervised Multi-hop QA

Coming Soon...

## Reference
Please cite the paper in the following format if you use this dataset during your research.

```
@inproceedings{pan-etal-2021-MQA-QG,
  title={Unsupervised Multi-hop Question Answering by Question Generation},
  author={Liangming Pan, Wenhu Chen, Wenhan Xiong, Min-Yen Kan, William Yang Wang},
  booktitle = {Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  address = {Online},
  month = {June},
  year = {2021}
}
```

## Q&A
If you encounter any problem, please either directly contact the first author or leave an issue in the github repo.

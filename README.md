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
- dateparser 1.0.0

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

QGwithAns generate a single-hop question *Q* with answer *A* from the input text *D*. We implement this module based on the pretrained QG model from [patil-suraj](https://github.com/patil-suraj/question_generation), a Google T5 model finetuned on the SQuAD 1.1 dataset. 

You could test this module by running the following python codes: 
```python
from MQA_QG.Operators import T5_QG

test_passage = '''Jenson Alexander Lyons Button (born 19 January 1980) is a British racing driver and former Formula One driver. He won the 2009 Formula One World Championship, driving for Brawn GP.'''

nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight")

print(nlp.qg_without_answer(test_passage))
print(nlp.qg_with_answer_text(test_passage, "19 January 1980"))
```

### b) DescribeEnt

DescribeEnt generate a sentence *S* that describes the given entity *E* based on the information of the table *T*. We implement this using the [GPT-TabGen model](https://arxiv.org/pdf/2004.07347.pdf) (Chen et al., 2020a). The model first uses template to flatten the table *T* into a document *PT* and then feed *PT* to the pre-trained GPT-2 model to generate the output sentence *S*. The framework is as follows. 

<p align="center">
<img src=Resource/table2text.png width=600/>
</p>

We finetune the GPT2 model on the [ToTTo dataset](https://github.com/google-research-datasets/ToTTo) (Parikh et al., 2020), a large-scale dataset of controlled table-to-text generation. Our fine-tuned model can be downloaded [here](https://drive.google.com/file/d/1MREzgOdXcFEo-wmDmxLqW7YFCuMreRkQ/). After downloading the finetuned model, put it under the `Pretrained_Models` directory. Then you could test this module by running the following python codes: 
```python
from MQA_QG.Operators.Table_to_Text import get_GPT2_Predictor

predictor = get_GPT2_Predictor('./Pretrained_Models/table2text_GPT2_medium_ep9.pt', num_samples = 3)
flattened_table = '''The table title is Netherlands at the European Track Championships . The Medal is Bronze . The Championship is 2011 Apeldoorn . The Name is Kirsten Wild . The Event is Women's omnium . Start describing Kirsten Wild : '''
results = predictor.predict_output(flattened_table)
print(results)
```

## Multi-hop Question Generation

After data preparation and testing operators, you could generate different types of multi-hop questions from (table, passage) in HybridQA or passages in HotpotQA. You simply need to configure your experimental setting in `MQA_QG/config.py`, as follows: 

```python
###### Global Settings
EXPERIMENT = 'HybridQA' # The experiment you want to run, choose 'HotpotQA' or 'HybridQA'
QG_DEVICE = 5  # gpu device to run the QG module
BERT_DEVICE = 3 # gpu device to run the BERT module
TABLE2TEXT_DEVICE = 3 # gpu devide to run the Table2Text module
QUESTION_TYPE = 'table2text' # the type of question you want to generate
# for hybridQA, the options are: 'table2text', 'text2table', 'text_only', 'table_only'
# for hotpotQA, the options are: 'text2text', 'comparison'

###### User-specified data directory
DATA_PATH = '../Data/HybridQA/WikiTables-WithLinks/' # root data directory, '../Data/HybridQA/WikiTables-WithLinks/' for HybridQA; '../Data/HotpotQA/dataset/train.src.txt' for HotpotQA
OUTPUT_PATH = '../Outputs/train_table_to_text.json' # the json file to store the generated questions
DATA_RANGE = [0, 20] # for debug use: the range of the dataset you considered (use [0, -1] to use the full dataset)
Table2Text_Model_Path = '../Pretrained_Models/table2text_GPT2_medium_ep9.pt' # the path to the pretrained Table2Text model
```

Key parameters: 
- `EXPERIMENT`: the dataset you want to generate questions from, choose 'HotpotQA' or 'HybridQA'. 
- `QG_DEVICE`, `BERT_DEVICE`, `TABLE2TEXT_DEVICE`: the gpu device to run the QG module, BERT module, and Table2Text module. 
- `QUESTION_TYPE`: the type of question you want to generate. There are **6 different types of questions** you can generate. For hybridQA, the options are: 'table2text', 'text2table', 'text_only', 'table_only'. For hotpotQA, the options are: 'text2text', 'comparison'. 
- `DATA_PATH`: root data directory, the defaults are: '../Data/HybridQA/WikiTables-WithLinks/' for HybridQA; '../Data/HotpotQA/dataset/train.src.txt' for HotpotQA. 
- `OUTPUT_PATH`: the json file to store the generated questions
- `Table2Text_Model_Path`: the path to the pretrained Table2Text model. 

After configuration, run the following python code to generate multi-hop questions. 

```shell
cd MQA-QG
python run_multihop_generation
```

A sample of generated (question, answer) pair for **HybridQA** is: 
```json
{
  "table_id": "\"Weird_Al\"_Yankovic_0",
  "question": "Who wrote the episode that was directed by Yankovic in 2015 vs. Robin?",
  "answer-text": "Scott Snyder",
  "answer-node": [
    [
      "Batman vs. Robin",
      [
        12,
        1,
        0
      ],
      "/wiki/Batman_vs._Robin",
      "table"
    ]
  ],
  "question_id": "0",
  "where": "table",
  "question_postag": "WP VBD DT NN WDT VBD VBN IN NNP IN CD IN NNP ."
}
```

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

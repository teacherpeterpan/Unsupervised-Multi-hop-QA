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
<img src=Resource/framework.png width=700/>
</p>

## Requirements

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

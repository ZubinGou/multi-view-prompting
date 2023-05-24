# Multi-view Prompting (MvP)

Code for ACL 2023 paper "[MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction](https://arxiv.org/abs/2305.12627)".

## Introduction 

MvP is an element order-based prompt learning method:

- MvP unifies various tuple prediction tasks through the **combination of elements**:


<p align="center">
    <img src="./images/multi-task.png" width="1000">
</p>

- MvP aggregates multi-view results by considering **permutations of elements**:

<p align="center">
    <img src="./images/framework.png" width="1000">
</p>



## Results

MvP achieves **state-of-the-art** performance across 10 datasets encompassing 4 ABSA tasks with a **single** model:

<p align="center">
    <img src="./images/result_main.png" width="1000">
</p>

MvP with T5-base outperforms large language models ChatGPT (*gpt-3.5-turbo*) by a large margin, even in **few-shot** transfer settings:

<p align="center">
    <img src="./images/result_chatgpt.png" width="1000">
</p>

## Quick Start

The data and code will be cleaned and uploaded within a few days.


## Citation

If you find this repository helpful, please consider cite our paper:

```
@inproceedings{gou-etal-2023-mvp,
	title = {MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction},
	author = {Gou, Zhibin and Guo, Qingyan and Yang, Yujiu},
	booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
	year = {2023},
}
```
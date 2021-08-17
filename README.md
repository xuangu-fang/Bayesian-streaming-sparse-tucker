# Bayesian-streaming-sparse-tucker-decomposition
by Shikai Fang, Mike Kirby and Shandian Zhe


code of Bayesian streaming sparse tucker decomposition for UAI 2021

Links for [Paper](https://github.com/xuangu-fang/Bayesian-streaming-sparse-tucker/blob/main/paper/sparse_tucker.pdf) and [Supplementary](https://github.com/xuangu-fang/Bayesian-streaming-sparse-tucker/blob/main/paper/supplementary.pdf)

![model illustration](./figs/fig1.JPG)


## Requirements:
matlab >= 2016

## Instructions:
1. Clone this repository.
2. See model details in `bayes_sparse_tucker_streaming_seq.m` .  Run model with `acc_script.m`,`alog_script.m`

## Datasets & Baselines
Large dataset:acc, small dataset:alog

Baselines: cp_als, cp_wopt, cp_num (you may have to modify the data-loding path to run baselines)

Check infos for more datasets and baselines in our paper


## Citation
Please cite our work if you would like to use the code
```
@article{fangbayesian,
  title={Bayesian Streaming Sparse Tucker Decomposition},
  author={Fang, Shikai and Kirby, Robert M and Zhe, Shandian}
}


```

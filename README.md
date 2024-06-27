# Graph Condensation Benchmark (GC-Bench)
GC-Bench is an open and unified benchmark for Graph Condensation (GC) based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://www.pyg.org/). We embark on **12** state-of-the-art graph condensation algorithms in node-level and graph-level tasks and analyze their performance in **12** distinct graph datasets.
## Overview of GC-Bench

<p align="center">
<img src="figs/GC-bench.png" width="100%" class="center" alt="pipeline"/>
</p>


GC-Bench is a comprehensive Graph Condensation Benchmark designed to systematically analyze the performance of graph condensation methods in various scenarios. It examines the effectiveness, transferability, and complexity of graph condensation. We evaluate 12 state-of-the-art graph condensation algorithms on both node-level and graph-level tasks across 12 diverse graph datasets. Through benchmarking these GC algorithms, we make the following contributions:


- **Comprehensive Benchmark.** GC-Bench systematically integrates 12 representative and competitive GC methods on both node-level and graph-level tasks by unified condensation and evaluation, providing a thorough analysis in terms of effectiveness, transferability, and efficiency.
  
- **Multi-faceted Evaluation and Analysis.** We conduct a detailed evaluation of GC methods, examining their effectiveness, efficiency, and complexity. This comprehensive analysis uncovers the strengths and limitations of current GC algorithms, offering valuable insights for future research.

- **Open-sourced Benchmark Library.** GC-Bench is open-sourced and easy to extend with new methods and datasets. This facilitates further exploration and encourages reproducible research, helping to advance the field of graph condensation.


## Getting Started

To get started with GC-Bench, please follow the instructions below:

1. **Installation**
   ```bash
   git clone https://github.com/RingBDStack/GC-Bench.git
   cd GC-Bench
   pip install -r requirements.txt
   conda env create -f environment.yml

2. **Download Datasets**
   Download the datasets and store them in the `data` directory. The project structure should look like the following:
   ```bash
   GC-Bench
   ├── data
   │   ├── cora
   │   ├── citeseer
   │   └── ...
   └── DM
   └── ...

## Condense Graph Datasets

Different graph condensation methods (gradient-matching, distribution-matching, kernel ridge regression etc) can be used in corresponding directories.

For example, to run the Distribution Matching (DM) method, use the following command:
```
python DM/main.py --dataset=citeseer --epochs=2000 --gpu_id=0 --lr_adj=0.001 --lr_feat=0.01 --lr_model=0.1 --method=GCDM --nlayers=2 --outer=10 --reduction_rate=1 --save=1 --seed=1 --transductive=1
```

To run the Gradient Matching (GM) method for node classification, use the following command:
```
python GM/main_nc.py --dataset cora --transductive=1 --nlayers=2 --sgc=1 --lr_feat=1e-4 --lr_adj=1e-4 --r=0.5 --seed=1 --epoch=600 --save=1
```

To run the Gradient Matching (GM) method for graph classification, use the following command:
```
python GM/main_gc.py --dataset ogbg-molhiv --init real --nconvs=3 --dis=mse --lr_adj=0.01 --lr_feat=0.01 --epochs=1000 --eval_init=1 --net_norm=none --pool=mean --seed=1 --ipc=5 --save=1
```

Parameters can also be set in configuration files. To run experiments using a configuration file, use the following command:
```
python GM/main_nc.py --config config_DosCond --section DBLP-r0.250
```

This command will run the corresponding experiments with the parameters specified in the configuration file. The provided configuration files contain the parameters used to obtain the results presented in our benchmark.

## Evaluate condensed graphs
For evaluation on different architectures, you can simply run:
```
python baselines/test_nc.py --method ${method} --dataset cora --gpu_id=0 --r=0.5 --nruns=5
```
Replace ${method} with the specific condensation method you used.
For evaluation on different tasks, you can simply run:
```
python evaluator/test_other_tasks.py --method ${method} --dataset cora --gpu_id=0 --r=0.5 --seed=1 --nruns=5 --task=LP
```
Replace ${method} with the specific condensation method you used. The --task parameter can be set to LP for link prediction, AD for anomaly detection, etc.

## Algorithm References

Summary of Graph Condensation (GC) algorithms. We also provide public access to the official algorithm implementations. "KRR" is short for Kernel Ridge Regression and "CTC" is short for computation tree compression. "GNN" is short for Graph Neural Network, "GNTK" is short for Graph Neural Tangent Kernel, "SD" is short for Spectral Decomposition. "NC" is short for node classification, "LP" is short for link prediction, "AD" is short for anomaly detection, and "GC" is short for graph classification.

| **Method** | **Initialization**       | **Backbone Model** | **Downstream Task** | **Paper** | **Code**                                                     | **Venue**       |
| ---------- | ------------------------ | ------------------ | ------------------- | ------------------------------------------------------------ | --------------- | --------------- | 
| Random     | —                        | —                  | —                   | —                                                            | —               |
| Herding    | —                        | —                  | —                   |[Herding Dynamical Weights to Learn](http://machinelearning.org/archive/icml2009/papers/447.pdf)| [code](https://github.com/ozansener/active_learning_coreset) | ICML, 2009      |
| K-Center   | —                        | —                  | —   |[Active learning for convolutional neural networks: A core-set approach](https://arxiv.org/pdf/1708.00489)             | [code](https://github.com/ozansener/active_learning_coreset) | ICLR, 2018      |
| GCond      | Random Sample            | GNN                | NC                  |[Graph Condensation for Graph Neural Networks](https://arxiv.org/pdf/2110.07580)             |  [code](https://github.com/ChandlerBang/GCond)                | ICLR, 2021      |
| DosCond    | Random Sample            | GNN                | NC, GC              | [Condensing Graphs via One-Step Gradient Matching](https://arxiv.org/pdf/2206.07746)             |  [code](https://github.com/amazon-science/doscond)            | SIGKDD, 2022    |
| SGDD       | Random Sample            | GNN                | NC, LP, AD          | [Does Graph Distillation See Like Vision Dataset Counterpart?](https://arxiv.org/pdf/2310.09192)             |  [code](https://github.com/RingBDStack/SGDD)                  | NeurIPS, 2023   |
| GCDM       | Random Sample            | GNN                | NC                  | [Graph Condensation via Receptive Field Distribution Matching](https://arxiv.org/pdf/2206.13697)             | —                                                            | arXiv, 2022     |
| DM         | Random Sample            | GNN                | NC                  |[CaT: Balanced Continual Graph Learning with Graph Condensation](https://arxiv.org/pdf/2309.09455)             |  —                                                            | ICDM, 2023      |
| SFGC       | K-Center                 | GNN                | NC                  | [Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data](https://arxiv.org/pdf/2306.02664)             | [code](https://github.com/Amanda-Zheng/SFGC)                 | NeurIPS, 2023   |
| GEOM       | K-Center                 | GNN                | NC                  | [Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching](https://arxiv.org/pdf/2402.05011)             |  [code](https://github.com/NUS-HPC-AI-Lab/GEOM)               | ICML, 2024      |
| KiDD       | Random Sample            | GNTK               | GC                  | [Kernel Ridge Regression-Based Graph Dataset Distillation](https://dl.acm.org/doi/pdf/10.1145/3580305.3599398)             |[code](https://github.com/pricexu/KIDD)                      | SIGKDD, 2023    |
| Mirage     | —                        | GNN                | GC                  | [Mirage: Model-Agnostic Graph Distillation for Graph Classification](https://arxiv.org/pdf/2310.09486)             |[code](https://github.com/idea-iitd/Mirage)                  | ICLR, 2024      |

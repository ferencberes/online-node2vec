Online Node2Vec
===============

![build](https://github.com/ferencberes/online-node2vec/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/ferencberes/online-node2vec/branch/master/graph/badge.svg?token=H6RRUKXQRF)](https://codecov.io/gh/ferencberes/online-node2vec)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)

This repository contains the code related to the research of [Ferenc Béres](https://github.com/ferencberes), [Róbert Pálovics](https://github.com/rpalovics), [Domokos Miklós Kelen](https://github.com/proto-n) and [András A. Benczúr](https://mi.nemzetilabor.hu/people/andras-benczur).

# Introduction

We propose two online node embedding models (StreamWalk and online second order similarity) for temporally evolving networks. Two nodes are required to be mapped close in the vector space whenever they lie on short paths formed by recent edges in the first model, and whenever the set of their recent neighbors is similar in the second model.

Please cite our [paper](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0169-5) if you use our work:

```
@Article{Béres2019,
author="B{\'e}res, Ferenc
and Kelen, Domokos M.
and P{\'a}lovics, R{\'o}bert
and Bencz{\'u}r, Andr{\'a}s A.",
title="Node embeddings in dynamic graphs",
journal="Applied Network Science",
year="2019",
volume="4",
number="64",
pages="25",
}
```

I presented a former version of our work at the 7th International Conference on Complex Networks and Their Applications that is availabe on this [branch](https://github.com/ferencberes/online-node2vec/tree/complex_networks_2018).

# Data

**US Open 2017 (UO17)** and **Roland-Garros 2017 (RG17)** Twitter datasets were published in our [previous work](https://link.springer.com/article/10.1007/s41109-018-0080-5) for the first time. Please cite this article if you use our data sets in your research:

```
@Article{Béres2018,
author="B{\'e}res, Ferenc
and P{\'a}lovics, R{\'o}bert
and Ol{\'a}h, Anna
and Bencz{\'u}r, Andr{\'a}s A.",
title="Temporal walk based centrality metric for graph streams",
journal="Applied Network Science",
year="2018",
volume="3",
number="32",
pages="26",
}
```

These Twitter datasets are available on the [website](https://dms.sztaki.hu/~fberes/tennis/) of our research group. In order to process the data you need to install the [twittertennis](https://github.com/ferencberes/twittertennis) Python package. It will automatically [download and prepare](scripts/preprocess_data.py) the datasets for you.

# Install

```bash
python setup.py install
```

# Usage

After installing every requirement execute the following script to run both node representation learning and evaluation for the similarity search task.

```bash
cd scripts
bash run.sh
```

The major steps in our pipeline are:
   * [Download and preprocess](scripts/preprocess_data.py) data
   * Learning [StreamWalk](scripts/streamwalk_runner.py) representations
   * Learning [Second order similarity](scripts/second_order_runner.py) representations
   * [Evaluate](scripts/evaluate.py) node embeddings for the similarity search supervised experiment

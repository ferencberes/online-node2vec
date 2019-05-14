Online Node2Vec
========================

This repository contains the code related to the research of [Ferenc Béres](https://github.com/ferencberes), [Róbert Pálovics](https://github.com/rpalovics), [Domokos Miklós Kelen](https://github.com/proto-n) and András A. Benczúr.

## UPDATE:

*The implementation of our proposed node embedding algorithms (StreamWalk and second order similarity) have been recently updated (2019-05-14) to match the description in our paper that is currently under review.*

# Cite

I presented a former version of our work at the 7th International Conference on Complex Networks and Their Applications. The PDF [slides](docs/node_embeddings_in_dynamic_graphs_slides.pdf) and the submitted [extended abstract](docs/node_embeddings_in_dynamic_graphs_abstract.pdf) are also available in this repository. Please [cite](https://drive.google.com/file/d/1MJW9uuOPjclV0yA9OeKPIsHpj88DX8Mq/view) our work if you use this code or our [Twitter datasets](https://dms.sztaki.hu/hu/letoltes/temporal-katz-centrality-data-sets):

```
@conference{beres18on2v,
  author       = {Ferenc Béres, Róbert Pálovics, Domokos M. Kelen, Dávid Szabó and András A. Benczúr}, 
  title        = {Node Embeddings in Dynamic Graphs},
  booktitle    = {Book of Abstracts of the 7th International Conference on Complex Networks and Their Applications},
  pages        = {165--167},
  year         = {2018},
  isbn         = {978-2-9557050-2-5},
}
```

# Data

The **US Open 2017 (UO17)** and **Roland-Garros 2017 (RG17)** Twitter datasets are available on the [website](https://dms.sztaki.hu/hu/letoltes/temporal-katz-centrality-data-sets) of our research group or you can also download it with the following command
```bash
bash ./scripts/download_data.sh
```

These Twitter datasets were published in our [previous work](https://link.springer.com/article/10.1007/s41109-018-0080-5) for the first time.

# Requirements

   * UNIX environment
   * **Python 3.5** conda environment with pre-installed jupyter:

   ```bash
   conda create -n YOUR_CONDA_PY3_ENV python=3.5 jupyter
   source activate YOUR_CONDA_PY3_ENV
   ```
   * Install the following packages with *conda* or *pip*:
      * **data processing:** pandas, numpy
      * **scientific:** scipy, gensim, networkx, gmpy2
      * **general:** sys, os, time, random, collections

# Usage

After you have downloaded the UO17 and RG17 datasets you can run our online node embedding algorithms with the following scripts:

   * [StreamWalk](scripts/streamwalk_runner.py)
   * [Second order similarity](scripts/second_order_runner.py)

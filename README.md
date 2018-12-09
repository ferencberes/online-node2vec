Online Node2Vec
========================

This repository contains the code related to the research of [Ferenc Béres](https://github.com/ferencberes), [Róbert Pálovics](https://github.com/rpalovics), [Domokos Miklós Kelen](https://github.com/proto-n), Dávid Szabó and András A. Benczúr.

# Cite

I am going to present our work at the 7th International Conference on Complex Networks and Their Applications. Please cite our work if you use this code or our [Twitter datasets](https://dms.sztaki.hu/hu/letoltes/temporal-katz-centrality-data-sets):

```
@conference{beres18on2v,
  author       = {Ferenc Béres, Róbert Pálovics, Domokos M. Kelen, Dávid Szabó and András A. Benczúr}, 
  title        = {Node Embeddings in Dynamic Graphs},
  booktitle    = {Book of Abstracts of the 7th International Conference on Complex Networks and Their Applications},
  year         = {2018},
}
```

# Data

The **US Open 2017 (UO17)** and **Roland-Garros 2017 (RG17)** Twitter datasets are available on the [website](https://dms.sztaki.hu/hu/letoltes/temporal-katz-centrality-data-sets) of our research group.

You can also download all related data sets with the following command
```bash
bash ./scripts/download_data.sh
```

Please cite our previous [work](https://link.springer.com/article/10.1007/s41109-018-0080-5) ff you use these Twitter datasets. 

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
      * **general:** sys, os, time, random, functools, collections

# Usage

After you have downloaded the UO17 and RG17 datasets you can run our online graph embedding algorithms with the following [scripts](scripts/):

   * **Temporal Walk algorithm:** scripts/temp_walk_online_n2v_runner.py
   * **Temporal Neighbourhood algorithm:** scripts/second_order_sim_online_n2v_runner.py
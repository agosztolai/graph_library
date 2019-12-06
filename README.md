# Graph library

Use this script to quickly generate standard datasets and graphs for testing/benchmarking of your code.

### To install

```
pip install -e . 
```

### To generate a graph

```
from graph_library import graph_generator as gg
gg.whichgraph = 'barbell'
gg.file = 'graph_params'
G = gg.generate()
```

The .gpickle files will be saved in ~/whichgraph/

### Generate a dataset in the same way (Swiss-roll in this case)
```
gg.whichgraph = 'swiss-roll'
G = gg.generate()
```

### To also generate a graph by computing a similarity matrix (k-nearest neighbours in this case) use

```
G = gg.generate(similarity='knn')
```

### Default arguments

Examples of standard parameters are included in the graph_params.yaml file. The script automatically saves a figure of the graph by default. 

## Currently includes the following graphs and datasets

**barbell** : barbell graph\
**barbell_asy** : asymmetric barbell graph\
**celegans** :  neural network of neurons and synapses in C. elegans\
**grid** : rectangular grid\
**2grid**: 2 rectangular grids connected by a bottleneck\
**delaunay-grid** : Delaunay triangulation of uniformly spaced points\
**delauney-nonunif** : Delaunay triangulation of nonuniform points (2 Gaussians)\
**dolphin** : directed social network of bottlenose dolphins\
**email** : network of email data from a large European research institution\
**ER** : Erdos-Renyi graph\
**Fan** : Fan's benchmark graph\
**football** : \
**frucht** : \
**GN** : Girvan-Newman benchmark\
**gnr** : directed growing network\
**karate** : Zachary's karate club\
**LFR** : Lancichinetti-Fortunato-Radicchi benchmark\ 
**krackhardt** :\ 
**miserable** :\ 
**netscience**:\
**scalefree**: \   
**S** : S-curve\
**SM** : small-world network\
**SB**\
**SBM** : stochastic block model\
**swiss-roll** : Swiss-roll dataset\
**torus**\
**tree**\
**tutte**\




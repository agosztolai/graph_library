# Graph library

Use this script to quickly generate standard datasets and graphs for testing/benchmarking of your code.

### To install

```
pip install -e . 
```

### To generate a graph

```
from graph_library import generate

whichgraph = 'barbell'
G = generate(whichgraph, params=None, plot=True, save=True, outfolder='')
```
Select the parameter ```whichgraph``` from the currently implemented graphs found below. See also ```graph_library.py``` for the possible parameters to be passed via ```params```.

### Generate a dataset in the same way (Swiss-roll in this case) then compute the similarity matrix
```
whichgraph = 'swiss-roll'
G = generate(whichgraph, params=None, plot=False, save=True, outfolder='')
``` 

### To also generate a graph by computing a similarity matrix (k-nearest neighbours in this case) use

```
G = similarity_matrix(G, sim='knn', par=10, symmetric=True)
```

Look at ```similarity_matrix()``` in ```graph_library.py``` to see the implemented options and the meaning of the parameter par.

## Currently includes the following graphs and datasets (whichgraph)

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
**tutte**




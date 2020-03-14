"""example of how to generate a graph"""
import sys
import os
import yaml

import matplotlib.pyplot  as plt
from graph_library import generate

graph_name = sys.argv[-1]

outfolder = 'graphs'
if not os.path.exists(outfolder):
    os.mkdir(outfolder)

print('Generating graph', graph_name)
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
graph = generate(graph_name, params=graph_params, plot=True, save=True, outfolder=outfolder)

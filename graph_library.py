import numpy as np
import networkx as nx
from sklearn.utils import check_symmetric
import sklearn.datasets as skd
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import os
import sklearn.neighbors as skn
import yaml as yaml

'''
# =============================================================================
# Library of standard graphs
# =============================================================================
    
'''

class graph_generator(object): 

    def __init__(self, whichgraph='barbell', nsamples=1, paramsfile='graph_params.yaml',
                 outfolder='./', save=True):

        self.whichgraph = whichgraph
        self.color = []
        self.pos = None
        self.save = save
        self.params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)[whichgraph]
        self.nsamples = nsamples
        self.outfolder = outfolder
        
        print('\nGraph: ' + whichgraph)
        print('\nParameters:', self.params)


    def generate(self, similarity=None, symmetric=True):
        
        self.symmetric=symmetric
        
        #create a folder and move into it
        if self.save:
            if not os.path.isdir(self.outfolder + self.whichgraph):
                os.mkdir(self.outfolder + self.whichgraph)

            os.chdir(self.outfolder + self.whichgraph)
            
        for i in range(self.nsamples):
            self.params['counter'] = i
        
            #generate graph
            self.G, tpe = graphs(self.whichgraph, self.params)
        
            #compute similarity matrix if not assigned    
            if tpe == 'pointcloud':
                if similarity!=None:
                    self.similarity=similarity
                    similarity_matrix(self)
            
            #compute positions if not assigned    
            elif tpe =='graph':
                if 'pos' not in self.G.nodes[0]:
                    pos = nx.spring_layout(self.G)
                    for i in self.G:
                        self.G.nodes[i]['pos'] = pos[i]
#                    nx.set_node_attributes(self.G, pos)
                
            #set node colors if not assigned    
            if 'color' not in self.G.nodes[0]:
#                color = {i: 'k' for i in self.G.nodes}
                for i in self.G:
                    self.G.nodes[i]['color'] = 'k'
#                nx.set_node_attributes(self.G, color)   
            
            #this is for compatibility with PyGenStability
            if 'block' in self.G.nodes[0]:
#                old_label = {i: str(self.G.nodes[i]['block']) for i in self.G.nodes}
                for i in self.G:
                    self.G.nodes[i]['old_label'] = str(self.G.nodes[i]['block'])
#                nx.set_node_attributes(self.G, old_label) 
                self.G = nx.convert_node_labels_to_integers(self.G, label_attribute='old_label') 
             
            #check if graph is connected    
            assert nx.is_connected(self.G), 'Graph is disconnected!'
            
            #save
            nx.write_gpickle(self.G, self.whichgraph + '_' + str(self.params['counter']) + "_.gpickle")
            
            #plot 2D graph of 3D graph
            if self.save and self.G.graph.get('dim')==3:
                plot_graph_3D(self.G, params=self.params, save=True)   


# =============================================================================
# similarity matrix
# =============================================================================

def similarity_matrix(self):
    
    n = self.G.number_of_nodes()
    pos = nx.get_node_attributes(self.G,'pos')
    color = nx.get_node_attributes(self.G,'color')
    pos = np.reshape([pos[i] for i in range(n)],(n,len(pos[0])))
    color = [color[i] for i in range(n)]
    
    sim = self.params['similarity']
    if sim=='euclidean' or sim=='minkowski':
        A = squareform(pdist(pos, sim))
    
    elif sim=='knn':
        A = skn.kneighbors_graph(pos, self.params['k'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
        A = A.todense()
    
    elif sim=='radius':
        A = skn.radius_neighbors_graph(pos, self.params['radius'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
        A = A.todense()
    
    elif sim=='rbf':    
        self.gamma_ = (self.params['gamma']
                           if 'gamma' in self.params.keys() else 1.0 / pos.shape[1])
        A = rbf_kernel(pos, gamma=self.gamma_)

    if self.symmetric==True:
        A = check_symmetric(A)
    
    self.G = nx.from_numpy_matrix(A)  
    
    for i in self.G:
        self.G.nodes[i]['pos'] = pos[i]
        self.G.nodes[i]['color'] = color[i]

 
# =============================================================================
# graphs
# =============================================================================

def graphs(whichgraph, params):
    
    G = nx.Graph()
    
    if whichgraph == 'barbell':
        tpe = 'graph'
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
            
    elif whichgraph == 'barbell_noisy':
        tpe = 'graph'
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
        for i,j in G.edges():
            G[i][j]['weight'] = abs(np.random.normal(1,params['noise']))
                
    elif whichgraph == 'barbell_asy':
        tpe = 'graph'
        A = np.block([[np.ones([params['m1'], params['m1']]), np.zeros([params['m1'],params['m2']])],\
                       [np.zeros([params['m2'],params['m1']]), np.ones([params['m2'],params['m2']])]])
        A = A - np.eye(params['m1'] + params['m2'])
        A[params['m1']-1,params['m1']] = 1
        A[params['m1'],params['m1']-1] = 1
        G = nx.from_numpy_matrix(A)   
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
                
    elif whichgraph == 'celegans':
        tpe = 'graph'
        from skd.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../../datasets/celegans/')
        
        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']
            
    elif whichgraph == 'celegans_undirected':
        tpe = 'graph'
        from skd.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')
        
        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']
            
        G = G.to_undirected()        
        
    elif whichgraph == 'grid':
        tpe = 'graph'
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=False)
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])                  
            
    elif whichgraph == '2grid': 
        tpe = 'graph'          
        F = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        F = nx.convert_node_labels_to_integers(F, label_attribute='old_label')
        
        pos = {}
        for i in F:
            pos[i] = np.array(F.nodes[i]['old_label'])
            
        H = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        H = nx.convert_node_labels_to_integers(H, first_label=len(F), label_attribute='old_label')
        
        for i in H:
            pos[i] = np.array(H.nodes[i]['old_label']) + np.array([params['n']+5, 0])
            
        G = nx.compose(F, H)
        G.add_edge(int(params['n']**2-params['n']/2+1), int(params['n']**2 +params['n']/2+1))
        G.add_edge(int(params['n']**2-params['n']/2), int(params['n']**2 +params['n']/2))
        G.add_edge(int(params['n']**2-params['n']/2-1), int(params['n']**2 +params['n']/2-1))        
        
    elif whichgraph == 'delaunay-grid':
        tpe = 'graph'
        from scipy.spatial import Delaunay
        np.random.seed(0)
        x = np.linspace(0,1,params['n'])
        
        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j],x[i]])
        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])
            
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()        
        
    elif whichgraph == 'delaunay-nonunif':
        tpe = 'graph'
        from scipy.spatial import Delaunay
        np.random.seed(0)
        x = np.linspace(0,1,params['n'])
        
        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j],x[i]])
                
        points = np.array(points)
        
        gauss_pos = [.5, 0.5]
        gauss_pos2 = [0.7, 0.7]
        gauss_var = [.05,.05]
        new_points = np.random.normal(gauss_pos, gauss_var , [20,2])
        #new_points = np.concatenate( (new_points, np.random.normal( gauss_pos2, gauss_var, [50,2])) )
        
        for p in new_points:
            if p[0]>0 and p[0]<1. and p[1]>0 and p[1]<1:
                points = np.concatenate( (points, [p,]) )
                
        #points = np.concatenate( (points, np.random.normal(.5,.1, [200,2])) )
        
        tri = Delaunay(points)
        
        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])
            
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()
        
    elif whichgraph == 'dolphin':
        tpe = 'graph'
        G = nx.read_gml('../../datasets/dolphins.gml')
        G = nx.convert_node_labels_to_integers(G)     
        for i,j in G.edges:
            G[i][j]['weight']= 1.

    elif whichgraph == 'email':
        tpe = 'graph'
        edges = np.loadtxt('../../datasets/email-Eu-core.txt').astype(int)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        labels = np.loadtxt('../../datasets/email-Eu-core-department-labels.txt').astype(int)
        for i in G:
            G.nodes[i]['block'] = labels[i]
            
        G = nx.convert_node_labels_to_integers(G)
            
    elif whichgraph == 'ER':
        tpe = 'graph'
        G = nx.erdos_renyi_graph(params['n'], params['p'], seed=params['seed'])  
    
    elif whichgraph == 'Fan':
        tpe = 'graph'
        G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], params['seed'])   
        
        for i,j in G.edges:
            if G.nodes[i]['block'] == G.nodes[j]['block']:
                G[i][j]['weight'] = params['w_in']
        else:
            G[i][j]['weight'] = 2 - params['w_in']
            
        labels_gt = []
        for i in range(params['l']):
            labels_gt = np.append(labels_gt,i*np.ones(params['g']))
            
        for n in G.nodes:
            G.nodes[n]['block'] = labels_gt[n-1]   
            
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label') 
    
    elif whichgraph == 'football':
        tpe = 'graph'
        G = nx.read_gml('../datasets/football.gml')
        
    elif whichgraph == 'frucht':
        tpe = 'graph'
        G = nx.frucht_graph()    
      
    elif whichgraph == 'GN':
        tpe = 'graph'
        G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], seed=params['seed'])
            
        labels_gt = []
        for i in range(params['l']):
            labels_gt = np.append(labels_gt,i*np.ones(params['g']))
            
        for n in G.nodes:
            G.nodes[n]['block'] = labels_gt[n-1]    
    
    elif whichgraph == 'gnr':
        tpe = 'graph'
        #directed growing network
        G = nx.gnr_graph(params['n'], params['p'])

    elif whichgraph == 'karate':
        tpe = 'graph'
        G = nx.karate_club_graph()
        
        for i,j in G.edges:
            G[i][j]['weight']= 1.
    
        for i in G:
            G.nodes[i]['block'] =  str(i) + ' ' + G.nodes[i]['club']
    
    elif whichgraph == 'LFR':
        tpe = 'graph'        
        command = params['scriptfolder'] + \
        " -N " + str(params['n']) + \
        " -t1 " + str(params['tau1']) + \
        " -t2 " + str(params['tau2']) + \
        " -mut " + str(params['mu']) + \
        " -muw " + str(0.5) + \
        " -maxk " + str(params['n']) + \
        " -k " + str(params['k']) + \
        " -name data"
        
        os.system(command)
        G = nx.read_weighted_edgelist('data.nse', nodetype=int, encoding='utf-8')
        for e in G.edges:
            G.edges[e]['weight'] = 1
            
        labels = np.loadtxt('data.nmc',usecols=1,dtype=int)    
        for n in G.nodes:
            G.nodes[n]['block'] = labels[n-1]    
    
    elif whichgraph == 'krackhardt':
        tpe = 'graph'
        G = nx.Graph(nx.krackhardt_kite_graph())
        for i,j in G.edges:
            G[i][j]['weight']= 1.    
    
    elif whichgraph == 'miserable':
        tpe = 'graph'
        G = nx.read_gml('../datasets/lesmis.gml')
        
        for i,j in G.edges:
            G[i][j]['weight']= 1.
            
    elif whichgraph == 'netscience':
        tpe = 'graph'
        G_full = nx.read_gml('../../datasets/netscience.gml')
        G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='old_label')
        largest_cc = sorted(max(nx.connected_components(G_full), key=len))
        G = G_full.subgraph(largest_cc)
        G = nx.convert_node_labels_to_integers(G)   
        
    elif whichgraph == 'powerlaw':
        tpe = 'graph'
        G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    elif whichgraph == 'geometric':
        tpe = 'graph'
        G = nx.random_geometric_graph(params['n'], params['p'])
            
    elif whichgraph == 'powergrid':
        tpe = 'graph'
        edges    = np.genfromtxt('../datasets/UCTE_edges.txt')
        location = np.genfromtxt('../datasets/UCTE_nodes.txt')
        posx = location[:,1]
        posy = location[:,2]
        pos  = {}
        
        edges = np.array(edges,dtype=np.int32)
        G = nx.Graph() #empty graph
        G.add_edges_from(edges) #add edges        
        
        #create the position vector for plotting
        for i in G.nodes():
            pos[i] = [posx[G.nodes[i]['old_label']-1],posy[G.nodes[i]['old_label']-1]]
            #pos[i]= [posx[i-1],posy[i-1]]

    elif whichgraph == 'S':   
        tpe = 'pointcloud'
        G.graph['dim'] = 3
        pos, color = skd.samples_generator.make_s_curve(params['n'], random_state=params['seed'])
        for i, xyz in enumerate(pos):
            G.add_node(i, pos = xyz, color = color[i])

    elif whichgraph == 'scale-free':
        tpe = 'graph'
        G = nx.DiGraph(nx.scale_free_graph(params['n']))                      
        
    elif whichgraph == 'SBM' or whichgraph == 'SBM_2':
        tpe = 'graph'
        G = nx.stochastic_block_model(params['sizes'],np.array(params['probs'])/params['sizes'][0], seed=params['seed'])
        for i,j in G.edges:
            G[i][j]['weight'] = 1.
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='labels_orig')        

    elif whichgraph == 'SM':
        tpe = 'graph'
        G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'])    
        
    elif whichgraph == 'swiss-roll':
        tpe = 'pointcloud'
        G.graph['dim'] = 3
        pos, color = skd.make_swiss_roll(n_samples=params['n'], noise=params['noise'], random_state=None)    
        for i, xyz in enumerate(pos):
            G.add_node(i, pos = xyz, color = color[i])
            
    elif whichgraph == 'torus':
        tpe = 'graph'
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)

        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])

    elif whichgraph == 'tree':
        tpe = 'graph'
        G = nx.balanced_tree(params['r'], params['h'])      
        
    elif whichgraph == 'tutte':
        tpe = 'graph'
        G = nx.tutte_graph()  
    
    G.graph['name'] = whichgraph
    
    return G, tpe

# =============================================================================
# plot graph
# =============================================================================

def plot_graph_3D(G, node_colors=[], edge_colors=[], params=None, save=False):
 
    if params==None:
        params = {'elev': 10, 'azim':290}
    elif params!=None and 'elev' not in params.keys():
        params['elev'] = 10
        params['azim'] = 290
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if nx.get_node_attributes(G, 'pos') == {}:
        pos = nx.spring_layout(G, dim=3)
    else:
        pos = nx.get_node_attributes(G, 'pos')   
     
    xyz = []
    for i in range(len(pos)):
        xyz.append(pos[i])
        
    xyz = np.array(xyz)
        
    #node colors
    if node_colors=='degree':
        edge_max = max([G.degree(i) for i in range(n)])
        node_colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 
    elif nx.get_node_attributes(G, 'color')!={} and node_colors==[]:
        node_colors = nx.get_node_attributes(G, 'color')
        colors = []
        for i in range(n):
            colors.append(node_colors[i])
        node_colors = np.array(colors)    
    else:
        node_colors = 'k'
     
    #edge colors
    if edge_colors!=[]:
        edge_color = plt.cm.cool(edge_colors) 
        width = np.exp(-(edge_colors - np.min(np.min(edge_colors),0))) + 1
        norm = mpl.colors.Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
        cmap.set_array([])    
    else:
        edge_color = ['b' for x in range(m)]
        width = [1 for x in range(m)]
        
    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
                   
        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=node_colors, s=200, edgecolors='k', alpha=0.7)
           
        for i,j in enumerate(G.edges()): 
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
            ax.plot(x, y, z, c=edge_color[i], alpha=0.5, linewidth = width[i])
    
    if edge_colors!=[]:    
        fig.colorbar(cmap)        
    ax.view_init(elev = params['elev'], azim=params['azim'])

    ax.set_axis_off()
 
    if save is not False:
        if 'counter' in params.keys():
            fname = G.name + str(params['counter']) + '.svg'
        else:
            fname = G.name + '.svg'
        plt.savefig(fname)
        plt.close('all')       

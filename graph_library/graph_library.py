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
import sys

# =============================================================================
# Generate one graph    
# =============================================================================

_path = os.path.dirname(__file__)

def generate(whichgraph, params=None, plot=False, save=False, outfolder='./graphs'):    
    if params == None:
        G, pos =  getattr(sys.modules[__name__], "generate_%s" % whichgraph)()
    else:
        G, pos =  getattr(sys.modules[__name__], "generate_%s" % whichgraph)(params)
    
    #print('\nGraph: ' + whichgraph)
    #print('\nParameters:', params) 
    
    G.graph['name'] = whichgraph        
    
    #Generate similarity matrix
    if len(G.edges)==0:
        G = similarity_matrix(G, sim=params['similarity'], par=params['k'], symmetric=True)
        
    #check if graph is connected    
    if not nx.is_connected(G):
        print('Graph is disconnected!!')
        
    #Assign positional information to nodes    
    G = assign_graph_metadata(G, pos)  
    
    #Plot graph       
    if plot:
        if 'color' in G.nodes[1]:
            color = 'custom'
        else:
            color = 'k'
            
        plot_graph(G, node_colors=color, outfolder=outfolder) 
    
    #save graph
    if save:
        i = 0
        while os.path.exists(os.path.join(outfolder,f"{whichgraph}_{i}.gpickle")):
            i += 1
            
        print(os.path.join(outfolder,f"{whichgraph}_{i}.gpickle"))
        nx.write_gpickle(G, os.path.join(outfolder,f"{whichgraph}_{i}.gpickle"))
    
    return G


# =============================================================================
# Generate many graphs
# =============================================================================
def generate_graph_family(whichgraph, params={}, nsamples=2, seed=None, plot=True, save=False, outfolder='./graphs'):
    
    if seed is None:
        seed = 0
        
    #create a folder
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
        
    for i in range(nsamples):
        params['seed'] = seed + i
            
        G = generate(whichgraph, params=params, plot=plot, save=False, outfolder=outfolder)
            
        #save
        if save:
            i = 0
            while os.path.exists(os.path.join(outfolder,f"{whichgraph}_{i}.gpickle")):
                i += 1
                
            print(os.path.join(outfolder,f"{whichgraph}_{i}.gpickle"))
            nx.write_gpickle(G, os.path.join(outfolder,f"{whichgraph}_{i}.gpickle"))


# =============================================================================
# similarity matrix
# =============================================================================
def similarity_matrix(G, sim=None, par=None, symmetric=True):
        
    if sim == None:
        raise ValueError('Specify similarity measure!')
    if par == None:
        raise ValueError('Specify parameter(s) of similarity measure!')
    
    n = G.number_of_nodes()
    pos = nx.get_node_attributes(G,'pos')
    pos = np.reshape([pos[i] for i in range(n)],(n,len(pos[0])))

    if sim=='euclidean' or sim=='minkowski':
        A = squareform(pdist(pos, sim))
        
    elif sim=='knn':
        A = skn.kneighbors_graph(pos, par, mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
        A = A.todense()
        
    elif sim=='radius':
        A = skn.radius_neighbors_graph(pos, par, mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
        A = A.todense()
        
    elif sim=='rbf':    
        gamma_ = par
        A = rbf_kernel(pos, gamma=gamma_)

    if symmetric==True:
        A = check_symmetric(A)
        
    for i in range(n):
        for j in range(n):
            if np.abs(A[i,j])>0:
                G.add_edge(i,j, weight = A[i,j])

    return G
     

# =============================================================================
# Assign metadata to graph if not already there
# =============================================================================
def assign_graph_metadata(G, pos=None, color=None): 
    
    if pos is not None and 'pos' not in G.nodes[1]:
        for i in G:
            G.nodes[i]['pos'] = pos[i]
            
    if color is not None and 'color' not in G.nodes[1]:
        for i in G:
            G.nodes[i]['color'] = color[i]
            
    if pos is None and 'pos' not in G.nodes[1]:
        pos = nx.spring_layout(G, weight='weight')
        for i in G:
            G.nodes[i]['pos'] = pos[i]
        
    return G        


# =============================================================================
# plot graph
# =============================================================================
def plot_graph(G, node_colors='k', outfolder='', params=None):
    
    if outfolder is None:
        outfolder = '.'
        
    try:
        whichgraph = G.graph['name']
    except: 
        whichgraph='graph'
        pass       
        
    if len(G.nodes[1]['pos'])==3:
            fig = plot_graph_3D(G, node_colors=node_colors, params=params)  
            fig.savefig(outfolder + whichgraph + '.svg')
    elif len(G.nodes[1]['pos'])==2:
            fig = plot_graph_2D(G, node_colors)  
            fig.savefig(outfolder + whichgraph + '.svg')
    else:
        raise Exception('Need to specify positions to plot')
            

def plot_graph_3D(G, node_colors='custom', edge_colors=[], params=None):
    
    n = G.number_of_nodes()
    m = G.number_of_edges()

    pos = nx.get_node_attributes(G, 'pos')
    pos = np.array([pos[i] for i in range(n)])    
    node_colors = list(nx.get_node_attributes(G, 'color').values())
           
    #node colors
    if node_colors=='degree':
        edge_max = max([G.degree(i) for i in range(n)])
        node_colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 
    else:
        node_colors = 'k'
     
    #edge colors
    if edge_colors!=[]:
        edge_color = plt.cm.cool(edge_colors) 
        width = np.exp(-(edge_colors - np.min(np.min(edge_colors),0))) + 0.5
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
                   
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=node_colors, s=200, edgecolors='k', alpha=0.7)
           
        for i,j in enumerate(G.edges()): 
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
            ax.plot(x, y, z, c=edge_color[i], alpha=0.3, linewidth = width[i])
    
    if edge_colors!=[]:    
        fig.colorbar(cmap)   
        
    if params==None:
        params = {'elev': 10, 'azim':290}
    elif params!=None and 'elev' not in params.keys():
        params['elev'] = 10
        params['azim'] = 290    
    ax.view_init(elev = params['elev'], azim=params['azim'])

    ax.set_axis_off()
 
    return fig       


def plot_graph_2D(G, node_colors='cluster'):      

    pos = list(nx.get_node_attributes(G,'pos').values())

    if node_colors=='cluster' and 'block' in G.nodes[1]:
        _labels = list(nx.get_node_attributes(G,'block').values())
    else:
        _labels = [0] * G.number_of_nodes()

    fig = plt.figure(figsize = (5,4))
    nx.draw_networkx_nodes(G, pos=pos, node_size=20, node_color=_labels, cmap=plt.get_cmap("tab20"))
    nx.draw_networkx_edges(G, pos=pos, width=1)

    plt.axis('off')
    
    return fig


# =============================================================================
# Graph library
# =============================================================================
def generate_barbell(params = {'m1': 7, 'm2': 0}):
    
    G = nx.barbell_graph(params['m1'], params['m2'])
    
    for i in G:
        G.nodes[i]['block'] = np.mod(i, params['m1'])
        
    return G, None


def generate_celegans(params={'undirected': True}):
    
    from datasets.celegans.create_graph import create_celegans 
    
    G, pos, labels, neuron_type, colors = create_celegans(location = 'datasets/celegans/')
            
    for i in G:
        G.nodes[i]['block'] = G.nodes[i]['labels']
    
    if params['undirected']:
        G = G.to_undirected() 
        
    return G, pos    


def generate_clique(params = {'n': 5}):
    
    G = nx.complete_graph(params['n'])
    
    return G, None


def generate_email(params = {}):
    
    edges = np.loadtxt('../../datasets/email-Eu-core.txt').astype(int)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    labels = np.loadtxt('../../datasets/email-Eu-core-department-labels.txt').astype(int)
    for i in G:
        G.nodes[i]['block'] = labels[i]
            
    G = nx.convert_node_labels_to_integers(G)

    return G, None


def generate_karate(params = None):
    
    G = nx.karate_club_graph()
        
    for i,j in G.edges:
        G[i][j]['weight']= 1.
    
    for i in G:
        if G.nodes[i]['club'] == 'Mr. Hi':
            G.nodes[i]['block'] = 0
            G.nodes[i]['color'] = 0
        else:
            G.nodes[i]['block'] = 1
            G.nodes[i]['color'] = 1
    
    return G, None


def generate_barbell_noisy(params = {'m1': 7, 'm2': 0, 'noise': 0.5}):        

    if 'seed' in params.keys():
        np.random.seed(params['seed'])
        
    G = nx.barbell_graph(params['m1'], params['m2'])
    
    for i in G:
        G.nodes[i]['block'] = np.mod(i,params['m1'])
    for i,j in G.edges():
        G[i][j]['weight'] = abs(np.random.normal(1,params['noise']))
         
    return G, None


def generate_barbell_asy(params = {'m1': 7, 'm2': 5}):
    
    A = np.block([[np.ones([params['m1'], params['m1']]), np.zeros([params['m1'],params['m2']])],\
                   [np.zeros([params['m2'],params['m1']]), np.ones([params['m2'],params['m2']])]])
    A = A - np.eye(params['m1'] + params['m2'])
    A[params['m1']-1,params['m1']] = 1
    A[params['m1'],params['m1']-1] = 1
    G = nx.from_numpy_matrix(A)   
    
    for i in G:
        G.nodes[i]['block'] = np.mod(i,params['m1'])   
        
    return G, None


def generate_dolphin(params = {}):
    
    G = nx.read_gml('datasets/dolphins.gml')
    G = nx.convert_node_labels_to_integers(G)     
    for i,j in G.edges:
        G[i][j]['weight']= 1.
        
    return G, None
    

def generate_dumbbell_of_stars(params = {'n': 15, 'm': 10}):
    
    G = nx.Graph()
    G.add_star(np.arange(params['m']))
    G.add_star(np.arange(params['m']-1, params['m'] + params['n']))

    return G, None


def generate_clique_of_cliques(params = {'m':5, 'n': 3, 'L': 500, 'w':[1, 10, 100], 'p':[0.01, 0.1, 1]}):
    
    if 'seed' in params.keys():
        np.random.seed(params['seed'])
        
    m = params['m']
    levels = params['n']
    N = m**levels
    L = params['L']
        
    A = np.zeros([N,N])
    for l in range(levels):   
        for i in range(0,N,N//(m**l)):
            for j in range(N//(m**l)):
                for k in range(j+1,N//(m**l)):
                    A[i+j,i+k] = 0
   
        for i in range(0,N,N//(m**l)):
            for j in range(N//(m**l)):
                for k in range(j+1,N//(m**l)):
                    A[i+j,i+k] = \
                    params['w'][l]*np.random.binomial(1,params['p'][l])
                        
    pos = np.zeros([N,2])
        
    for i in range(levels):
        for k in range(m**(i+1)):
            pos[k*N//(m**(i+1)) : (k+1)*N//(m**(i+1)), :] += \
            [L/(3**i)*np.cos((k % m)*2*np.pi/m), L/(3**i)*np.sin((k % m)*2*np.pi/m)]
        
    A = A + A.T
    G = nx.from_numpy_matrix(A)    

    return G, pos


def generate_erdos_renyi(params={'n': 50, 'p': 0.1}):
    
    if 'seed' in params.keys():
        seed = params['seed']
        
    G = nx.erdos_renyi_graph(params['n'], params['p'], seed=seed)  
    
    return G, None


def generate_Fan(params = {'w_in': 1.5, 'l': 4, 'g': 32, 'p_in': 0.125, 'p_out': 0.125}):
    
    if 'seed' not in params.keys():
        params['seed']=None
        
    G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], seed=params['seed'])   
    
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

    return G, None


def generate_football(params = {}): 
    
    G = nx.read_gml('datasets/football.gml')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    
    return G, None


def generate_frucht(params = {}):
    
    G = nx.frucht_graph() 
    
    return G, None


def generate_GN(params = {'l': 4, 'g': 32, 'p_in': 0.4, 'p_out': 0.2}):
    
    if 'seed' not in params.keys():
        params['seed']=None
        
    G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], seed=params['seed'])
        
    labels_gt = []
    for i in range(params['l']):
        labels_gt = np.append(labels_gt, i*np.ones(params['g']))
            
    for n in G.nodes:
        G.nodes[n]['block'] = labels_gt[n-1]

    return G, None


def generate_geometric(params = {'n': 50, 'p': 0.3}):   
    
    if 'seed' in params.keys():
        np.random.seed(params['seed'])
        
    G = nx.random_geometric_graph(params['n'], params['p'])
    
    return G, None


def generate_grid(params = {'n': 10, 'm': 10}):
    
    G = nx.grid_2d_graph(params['n'], params['m'], periodic=False)
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
    pos = {}
    for i in G:
        pos[i] = np.array(G.nodes[i]['old_label']) 
        
    return G, pos
 
    
def generate_2grid(params = {'n': 10}):
    
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
        
    return G, pos


def generate_delaunay_grid(params = {'n': 10}):
    
    from scipy.spatial import Delaunay
    
    if 'seed' in params.keys():
        np.random.seed(params['seed'])
    
    x = np.linspace(0,1,params['n'])
        
    pos = []
    for i in range(params['n']):
        for j in range(params['n']):
            pos.append([x[j],x[i]])
            
    pos = np.array(pos)

    tri = Delaunay(pos)

    edge_list = []
    for t in tri.simplices:
        edge_list.append([t[0],t[1]])
        edge_list.append([t[0],t[2]])
        edge_list.append([t[1],t[2]])
            
    G = nx.Graph()
    G.add_nodes_from(np.arange(params['n']))
    G.add_edges_from(edge_list)
    
    return G, pos


def generate_gnr(params = {'n': 20, 'p': 0.2}):
    
    G = nx.gnr_graph(params['n'], params['p'])

    return G, None


def generate_grid_delaunay_nonunif(params = {'n': 10}):
    
    from scipy.spatial import Delaunay
        
    if 'seed' in params.keys():
        np.random.seed(params['seed'])
        
    x = np.linspace(0,1,params['n'])
        
    pos = []
    for i in range(params['n']):
        for j in range(params['n']):
            pos.append([x[j],x[i]])
                
    pos = np.array(pos)
        
    gauss_pos = [.5, 0.5]
    gauss_pos2 = [0.7, 0.7]
    gauss_var = [.05,.05]
    new_points = np.random.normal(gauss_pos, gauss_var , [20,2])
    #new_points = np.concatenate( (new_points, np.random.normal( gauss_pos2, gauss_var, [50,2])) )
        
    for p in new_points:
        if p[0]>0 and p[0]<1. and p[1]>0 and p[1]<1:
            pos = np.concatenate( (pos, [p,]) )
                
    #pos = np.concatenate( (pos, np.random.normal(.5,.1, [200,2])) )
        
    tri = Delaunay(pos)
        
    edge_list = []
    for t in tri.simplices:
        edge_list.append([t[0],t[1]])
        edge_list.append([t[0],t[2]])
        edge_list.append([t[1],t[2]])
            
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(pos)))
    G.add_edges_from(edge_list)

    return G, pos


def generate_krackhardt(params = {}):
    
    G = nx.Graph(nx.krackhardt_kite_graph())
    for i,j in G.edges:
        G[i][j]['weight']= 1. 
        
    return G, None
    

def generate_LFR(params = {'n':1000, 'tau1': 2, 'tau2': 1, 'mut': 0.5, 'k': 20, 
                           'minc': 10, 'maxc': 50}):

    if 'seed' not in params.keys():
        params['seed'] = 0
        
    cwd = os.getcwd()
    f= open(os.path.join(cwd,'seed.txt'),"w+")
    f.write(str(params['seed']))
        
    root_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
    script_folder = os.path.join(root_dir,'datasets/LFR-Benchmark/lfrbench_udwov')
    command = script_folder + \
        " -N " + str(params['n']) + \
        " -t1 " + str(params['tau1']) + \
        " -t2 " + str(params['tau2']) + \
        " -mut " + str(params['mut']) + \
        " -muw " + str(params['mut']) + \
        " -maxk " + str(params['n']) + \
        " -k " + str(params['k']) + \
        " -name " + os.path.join(cwd,'data') + \
        " -seed " + os.path.join(cwd,'seed.txt')
        
    os.system(command)
    
    G = nx.read_weighted_edgelist(os.path.join(cwd,'data.nse'), nodetype=int, encoding='utf-8')
    
    for e in G.edges:
        G.edges[e]['weight'] = 1
        
    labels = np.loadtxt(os.path.join(cwd,'data.nmc'),usecols=1,dtype=int)    
    for n in G.nodes:
        G.nodes[n]['block'] = labels[n-1]
        
    return G, None   


def generate_miserable(params = {}):
    
    G = nx.read_gml('datasets/lesmis.gml')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
    for i,j in G.edges:
        G[i][j]['weight']= 1.
        
    return G, None
    

def generate_netscience(params = {}):
    
    G_full = nx.read_gml('datasets/netscience.gml')
    G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='old_label')
    largest_cc = sorted(max(nx.connected_components(G_full), key=len))
    G = G_full.subgraph(largest_cc)
    G = nx.convert_node_labels_to_integers(G) 
    
    return G, None 
 
    
def generate_powergrid(params = {}):
#
    edges    = np.genfromtxt(os.path.join(_path, '../datasets/UCTE_edges.txt'))
    location = np.genfromtxt(os.path.join(_path, '../datasets/UCTE_nodes.txt'))
    posx = location[:,1]
    posy = location[:,2]
    pos  = {}
        
    edges = np.array(edges,dtype=np.int32)
    G = nx.Graph() #empty graph
    G.add_edges_from(edges) #add edges        
        
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    #create the position vector for plotting
    for i in G.nodes():
        pos[i] = [posx[G.nodes[i]['old_label']-1],posy[G.nodes[i]['old_label']-1]]
        #pos[i]= [posx[i-1],posy[i-1]] 

    return G, pos 

def generate_powerlaw(params = {}):

    G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    return G, None 
  

def generate_ring_of_cliques(params = {'m': 5, 'n': 6}):

    G = nx.ring_of_cliques(params['m'], params['n'])
    
    x1 = np.linspace(-np.pi,np.pi,params['m'])
    x2 = np.linspace(0,2*np.pi,params['n'])[::-1]
               
    posx = np.zeros(params['m']*params['n'])
    posy = np.zeros(params['m']*params['n'])
    for i in range(params['m']):         
        for j in range(params['n']):
            posx[i*params['n'] + j] = np.cos(x1[i]) + 0.5*np.cos(x2[j] + x1[i] + 2*np.pi*3/5)
            posy[i*params['n'] + j] = np.sin(x1[i]) + 0.5*np.sin(x2[j] + x1[i] + 2*np.pi*3/5)
                
    pos = [ [posx[i], posy[i]] for i in range(params['m']*params['n'])] 
    
    return G, pos 


def generate_SBM(params = {'n':[30,30,30,30],
                           'p':[[0.8, 0.1, 0.02, 0.02],
                                [0.1, 0.8, 0.02, 0.02],
                                [0.02, 0.02, 0.8, 0.1],
                                [0.02, 0.02, 0.1, 0.8]]}):
    
    if 'seed' not in params.keys():
        params['seed'] = None
    np.random.seed(params['seed'])
    
    G = nx.stochastic_block_model(params['n'], params['p'], seed=params['seed'])

    def random_disk(circle_x=0, circle_y=0, circle_r=1):       
        alpha = 2 * np.pi * np.random.uniform() # random angle
        r = circle_r * np.sqrt(np.random.uniform()) # random radius
        x = r * np.cos(alpha) + circle_x # x coord
        y = r * np.sin(alpha) + circle_y # y coord
        
        return np.array([x,y])
    
    R=3

    pos = {}
    nc = len(params['n'])
    node_seq = np.cumsum(params['n'])
    for i in range(nc):
        if i==0:
            for n in range(node_seq[0]):
                pos[n] = random_disk(R*np.cos(2*np.pi/nc*i - np.pi/4), R*np.sin(2*np.pi/nc*i - np.pi/4))
        else:
            for n in range(node_seq[i-1], node_seq[i]): 
                pos[n] = random_disk(R*np.cos(2*np.pi/nc*i - np.pi/4), R*np.sin(2*np.pi/nc*i - np.pi/4))
    
    return G, pos


def generate_scale_free(params = {'n': 100}):
    
    if 'seed' not in params.keys():
        params['seed'] = None

    G = nx.scale_free_graph(params['n'],seed=params['seed'])
    G = G.to_undirected()
    
    return G, None


def generate_swiss_roll(params = {'n': 100, 'noise': 0., 'elev': 10, 'azim': 270,
                                  'similarity': 'knn', 'similarity_par': 10}):
    if 'seed' not in params.keys():
        params['seed'] = None
    
    G = nx.Graph()
    
    pos, color = skd.make_swiss_roll(n_samples=params['n'], noise=params['noise'], random_state=params['seed'])    
    for i, _pos in enumerate(pos):
        G.add_node(i, pos = _pos, color = color[i])

    return G, pos


def generate_S(params = {'n': 300, 'elev': 10, 'azim': 290,
                         's': 1.2, 'similarity': 'knn', 'similarity_par': 10}):
    
    if 'seed' not in params.keys():
        params['seed'] = None
        
    G = nx.Graph()
    
    pos, color = skd.samples_generator.make_s_curve(params['n'], random_state=params['seed'])
    
    for i, _pos in enumerate(pos):
        G.add_node(i, pos = _pos, color = color[i])
        
    return G, pos


def generate_NWS(params = {'n': 50, 'k': 2, 'p': 0.3}):
    
    G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'])    
          
    return G, None


def generate_star(params = {'n': 10}): 

    G = nx.star_graph(params['n'])
    
    return G, None


def generate_star_of_circles(params={'n': 5, 'm': 10}):
    
    G = nx.Graph()
    G.add_star(np.arange(params['m']))
    for i in range(1,params['m']):
        G.add_cycle([i] + list(range(params['m'] + (i-1)*params['n'], params['m'] + i*params['n'] )))

    return G, None


def generate_torus(params = {'m': 5, 'n': 4}):  
    
    G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)
    
    pos = {}
    for i in G:
        pos[i] = np.array(G.nodes[i]['old_label'])

    return G, pos


def generate_tree(params = {'r': 2, 'h': 5}):
    
    G = nx.balanced_tree(params['r'], params['h'])  

    return G, None
 

def generate_tutte(params = {}):       

    G = nx.tutte_graph() 
    
    return G, None


def generate_triangle_of_triangles(params = {'m': 1, 'N': 3}):
    
    m = params['m']
    N = params['N']
    A = np.ones([N, N])-np.eye(N)
    A = np.kron(np.eye(N**m),A)
    A[2,3]=1; A[3,2]=1; A[1,6]=1; A[6,1]=1; A[4,8]=1; A[8,4]=1
    A = np.vstack((np.hstack((A, np.zeros([9, 9]))), np.hstack((np.zeros([9, 9]), A))))
    A[0,9] = 1; A[9,0] = 1
    
    G = nx.Graph(A)
    
    return G, None

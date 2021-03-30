import graph_library as gl


#generate one Girvan-Newman graph
gl.generate('GN', params=None, plot=True, save=True)

#generate 10 Girvan-Newman graph
n=10
params = {'l': 4, 'g': 32, 'p_in': 0.4, 'p_out': 0.2}
gl.generate_graph_family('GN', params=params, nsamples = n, seed=None, plot=True, save=True)
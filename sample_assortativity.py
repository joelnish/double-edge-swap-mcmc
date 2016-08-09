# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:06:33 2016

@author: Joel Nishimura

This module contains functions to sample the assortativity values of graphs 
with the same degree sequence as an input graph. The functions use the  
dbl_edge_mcmc module to perform double edge swaps.

Running this module as a script samples the assortativity of simple graphs
with the same degree sequence as Zachary's karate club at 50k different graphs
spaced over 5 million double edge swaps.

Running the function 'sample_geometers' performs a more resource intensive 
MCMC sampling of a collaboration network of geometers.

"""

__docformat__ = 'reStructuredText'

import numpy as np
import networkx as nx
import dbl_edge_mcmc as mcmc

def r_sample_MCMC(G,allow_loops,allow_multi,is_v_labeled =True, its = 10000, n_recs = 100,filename = 'temp'):
    '''
    
    Samples the graph assortativity of graphs in a specified graph space  with 
    the same degree sequence as the input graph. Output is saved in 
    subdirectory 'output/'.
    
    | Args:
    |     G (networkx graph or multigraph): Starts the MCMC at graph G. Node 
            names be the integers 0 to n.            
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph
            space.
    |     uniform (bool): True if the space is vertex labeled, False for
            stub-labeled.
    |     its (int): The total number of MCMC steps
    |     n_recs (int): The number of samples from the MCMC sampler, spaced out
            evenly over the total number of its.
    |     filename (str): the name for the output file.
    
    | Returns:
    |    (array) An array recording the assortativity at n_recs number of 
        sampled graphs.
        
    '''  
    
    G = mcmc.flatten_graph(G,allow_loops,allow_multi)
    A = nx.adjacency_matrix(G) 
    A = A.toarray()
    A += np.diag(np.diag(A)) #the row sums should sum to degree
    edge_list = np.array(G.edges())
    swaps = np.zeros(4,dtype=np.int64)
    degree = [G.degree(i) for i in range(0,G.number_of_nodes())]
    
    if is_v_labeled:
        stepper = mcmc.MCMC_step_vertex
    else:
        stepper = mcmc.MCMC_step_stub

    inner_loop_size = its/n_recs
    
    r_samples = np.zeros(n_recs)
    
    
    for j in xrange(0,n_recs):
        for i in xrange(0,int(inner_loop_size)):
            
            stepper(A,edge_list,swaps,allow_loops,allow_multi)
        
        r_samples[j] = calc_r(degree,edge_list)

                
    data_suffix = ['','_wloops'][allow_loops] +['','_wmulti'][allow_multi] + ['_stub','_vertex'][is_v_labeled]   
    filename = filename + data_suffix
    f = file('output/'+filename+'.txt','w')
    f.write(str(list(r_samples))+'\n')
    f.close()
    
    return r_samples





@mcmc.jit(nopython=True,nogil=True)
def calc_r(degree,edges):
    '''
    
    Calculates the assortativity r based on a network's edgelist and degrees.
    
    | Args:
    |     degree (dict): Keys are node names, values are degrees.
    |     edges (list): A list of the edges (u,v) in the graph.
    
    | Returns:
    |     (float) The assortativity of the graph.
        
    '''     
    
    
    s = 0   
    
    dsq = 0
    dsum = 0 
    m = len(edges)       
    for ll in xrange(0,m):
        e = edges[ll]
        i = e[0]
        j = e[1]
        di = degree[i]
        dj = degree[j]
        s += di*dj
        dsq += di**2 + dj**2
        dsum += di + dj
        
        dssq = dsum**2
    if (2.0*m*dsq-dssq) == 0: 
        if 4*m*s-dssq < 0:
            r = 1.0
        elif 4*m*s-dssq > 0:
            r = 1.0
        else:
            r = 0.0
    else:
        r = (4*m*s-dssq)/(2.0*m*dsq-dssq) 
    
    return r


def load_geometers():
    '''
    
    This loads the geometers graph from file and returns a networkx multigraph.
    
    '''
    G = nx.read_weighted_edgelist('geomnet_edgelist',nodetype=int)
    MG = nx.MultiGraph()

    rename = {}
    name = 0
    for u,v,data in G.edges_iter(data=True):

        for w in [u,v]:
            try: 
                rename[w]
            except:
                rename[w] = name
                name += 1
                
        for i in range(int(data['weight'])):
            MG.add_edge(rename[u],rename[v])
            
    return MG


def sample_geometers():
    '''
    
    This calculates the assortativity on a collaboration network of geometers,
    on each of the 7 possible graphs spaces which allow/disallow self-loops,
    multiedges, and are either stub or vertex-labeled. 10 thousand samples are drawn
    over the course of 5 billion double edge swaps. Output is saved in the
    'output' subdirectory with the name 'geo'. 
    
    References:    
    Bill Jones. Computational geometry database 
    (http://jeffe.cs.illinois.edu/compgeom/biblios.html), 2002.
    
    '''
    
    G = load_geometers() 
    for allow_loops in [True,False]:
        for allow_multi in [True,False]:
            r_sample_MCMC(G,allow_loops,allow_multi,False,its = 5000000000, n_recs = 10000,
                          filename = 'geo')
    
    


if __name__ == '__main__':
    
    import pylab as py
    
    G = nx.karate_club_graph()
    r_vals = r_sample_MCMC(G, allow_loops = False, allow_multi = False, is_v_labeled = False,
                  its = 5000000, n_recs = 50000,filename = 'karate')
    
    py.hist(r_vals,bins = 50)
    py.xlabel('assortativity')
    py.ylabel('counts')
    py.title('simple graphs with karate\'s deg seq')
        

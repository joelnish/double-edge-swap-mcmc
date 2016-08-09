# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:20:04 2016

@author: Joel Nishimura


This module contains functions to test the uniformity of the MCMC sampling
in dbl_edge_mcmc.py.  

Running this as a script performs a test on the path graph with degree sequence
1,2,2,2,1. Output is saved to subdirectory 'verification'.

A more thorough, though time-consuming test, is available in the function 
'test_sampling_seven_node'. 

"""

__docformat__ = 'reStructuredText'

import numpy as np
import networkx as nx
import scipy.misc as mis
import matplotlib.pyplot as plt
import dbl_edge_mcmc as mcmc


def determine_relative_freq(G):    
    '''
    
    Returns the ratio of stub-matchings for the input graph divided by the 
    number of stub-matchings for a simple graph with the same degree sequence. 
    
    | Args:
    |     G (networkx_class): The input graph.
    
    | Returns:
    |     The likelihood of the input graph relative to a simple graph with the 
        same degree sequence.
    
    '''
    
    G = nx.multidigraph.MultiGraph(G)
    degs = G.degree()
    prob = 1
    seen = set()
    
    for u in G.nodes_iter():
        for v in G[u]:
            if (u,v) in seen:
                continue
            
            l = len(G[u][v])
            
            du = degs[u]
            
            if u == v:

                temp = mis.comb(du,2*l)                
                for i in range(0,l):
                    temp = temp *(2*l-2*i-1)
                degs[u] += -2*l

            elif l == 1:

                degs[u] += -1
        
                dv = degs[v]
                degs[v] += -1
                
                temp = du*dv
            else:
                temp = mis.comb(du,l)
                
                degs[u] += -l
                
                dv = degs[v]
                degs[v] += -l
                for i in range(0,l):
                    temp = temp *(dv-i)
            
            if temp > 0 and du > 0:
                prob = prob*temp
            
            seen.add((u,v))
            seen.add((v,u))
    
    return prob


def test_sampling(G, self_loops=False, multi_edges=False, is_v_labeled=True, its = 100000): 
    '''
    
    Tests the uniformity of the MCMC sampling on an input graph.
    
    | Args:
    |     G (networkx graph or multigraph): The starting point of the mcmc double
            edges swap method.
    |     self_loops (bool): True only if loops allowed in the graph space.
    |     multi_edges (bool): True only if multiedges are allowed in the graph
            space.
    |     is_v_labeled (bool): True if the space is vertex labeled, False for
            stub-labeled.
    |     its (int): The number of samples from the MCMC sampler.
    
    | Returns:
    |     dict: Keys correspond to each visited graph, with values being a list 
            giving the number of times the graph was sampled along with a 
            weight proportional to the expected number of samplings (relevant
            for stub-labeled samplings) 
        
    '''    
    
    print 'Testing sampling with selfloops= ' + str(self_loops) +' multi_edges= ' +str(multi_edges) 
    
    config = mcmc.MCMC_class(G,self_loops,multi_edges, is_v_labeled)
    visited_graphs = {}

    for i in range(0,its):
        try:
            visited_graphs[tuple(config.G.edges())][0] += 1

        except:
            visited_graphs[tuple(config.G.edges())] = [1,  determine_relative_freq(config.G)  ]      
    
        config.step_and_get_graph()
              
    print 'number of graphs visited: ' +str(len(visited_graphs))

    return visited_graphs

def plot_vals(samples, uniform, name):
    '''
    
    Plots the output of test_sampling as a histogram of the number of times 
    each graph was visited in the MCMC process. Creates a figure in 
    subdirectory 'verification/'.
    
    | Args:
    |     samples (dict): Output from test_sampling. Has a length 2 list as 
            values corresponding to [num_samples,sampling_weight].
    |     uniform (bool): True if the space is vertex labeled, False for
            stub-labeled.
    |     name (str): Name for output.
    
    | Returns:
    |     None
        
    '''   
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    samples = samples.values()
    samples.sort()
    if not uniform:
        samples.sort(key = lambda x: x[1])
        
    samples = np.array(samples)
    num_samples = sum(samples[:,0])
    num_graphs = len(samples)
    
    ax.bar(range(num_graphs), samples[:,0],alpha = .7)
    ax.set_xlabel('graphs')
    ax.set_ylabel('samples')
    ax.set_title(name)
    if uniform:
        average_samples = num_samples/(1.0*num_graphs)
        ax.plot([0,num_graphs],2*[average_samples],'r',linewidth = 2,
                label = 'expected')
    else:
        tot_weight = sum(samples[:,1])
        
        weights = []
        for w in samples[:,1]:
            weights.append((w/(1.0*tot_weight))*num_samples)
            weights.append((w/(1.0*tot_weight))*num_samples)
        ax.plot(np.round(np.linspace(0,num_graphs,2*num_graphs)), weights,'r',
                linewidth = 2,label = 'expected')   

    ax.legend(loc=4)

    fig.savefig('verification/'+ name +'.png')  

def test_sampling_seven_node():
    '''
    
    This tests the MCMC's ability to sample graphs uniformly, on degree seq.
    5,3,2,2,2,1,1.   Output is saved to subdirectory verification with name
    beginning in 'SevenNode'.
    
    '''

    G = nx.MultiGraph()
    G.add_edge(0,1)
    G.add_edge(0,5)
    G.add_edge(2,3)
    G.add_edge(0,4)
    G.add_edge(0,3)
    G.add_edge(2,4)
    G.add_edge(6,1)
    G.add_edge(2,0)

    samples = 8000000
 
    for allow_loops in [False,True]:
        for allow_multi in [False,True]:
            for uniform in [False, True]:
                name = 'SevenNode'  
                name = name + ['','_w_loops'][allow_loops] + ['','_w_multi'][allow_multi] + ['_stub-labeled','_vertex-labeled'][uniform]
        
                vals = test_sampling(G,allow_loops,allow_multi,its=samples, is_v_labeled = uniform)
                plot_vals(vals, uniform, name)
                
                
def test_sampling_five_node():
    '''
    
    This tests the MCMC's ability to sample graphs uniformly, on degree seq.
    1,2,2,2,1.   Output is saved to subdirectory verification with name
    beginning in 'FiveNode'.
    
    '''

    G = nx.MultiGraph()
    G.add_path([0,1,2,3,4])

    samples = 200000
 
    for allow_loops in [False,True]:
        for allow_multi in [False,True]:
            for uniform in [False, True]:
                name = 'FiveNode'   
                name = name + ['','_w_loops'][allow_loops] + ['','_w_multi'][allow_multi] + ['_stub-labeled','_vertex-labeled'][uniform]
        
                vals = test_sampling(G,allow_loops,allow_multi,its=samples, is_v_labeled = uniform)
                plot_vals(vals, uniform, name)


if __name__ == '__main__':

    test_sampling_five_node()
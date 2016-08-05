# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 07:43:30 2016

@author: Joel Nishimura


This module contains the core methods used to uniformly sample graphs with 
fixed degree sequences via a double edge swap Markov chain Monte Carlo sampler.

"""
#
#There are two primary ways to utilize this module, either though providing and
#maintaining a networkx graph/multigraph object as in MCMC_class, or by providing 
#and maintaining an adjacency matrix and an edge list as in the functions 
#MCMC_step_stub and MCMC_step. In either approach it is necessary to specify
#the desired graph space by stating whether self-loops and/or multiedges are 
#allowed as well deciding whether the graph is stub- or vertex-labeled. 
# 
#For a list of function arguments please see the function's specific 
#documentation.
#
#MCMC_step and MCMC_step_uniform
#-------------------------------
#
#Functions MCMC_step_stub and MCMC_step perform a single stub-labeled and 
#vertex-labeled (respectively) double edge swap and correspond to Algorithms 1 
#and 3 in the accompanying paper. These functions modify a full (non-sparse) 
#graph adjacency matrix, a list of edges, and a length 4 list, all in place. 
#Both take the same arguments (as detailed below).
#
#    Example:  
#    
#::    
#    
#  import numpy as np
#  A = np.array([[0,1,0],[1,2,2],[0,2,0 ]])
#  edge_list = np.array([[0,1],[1,2],[1,2],[1,1]])
#  swaps = [0,0,0,0]
#  MCMC_step_stub(A, edge_list, swaps, loops = True, multi = True)
#
#This performs a single MCMC step on a stub-labeled loopy multigraph, 
#potentially updating A, edge_list and swap with new, post-swap values. 
#
#Both functions return a boolean, which is true only if the Markov chain step
#altered the adjacency matrix (as opposed to resampling the current graph).
#If the adjacency matrix is altered the swaps argument will be changed in place,
#storing the nodes that were swapped. 
#
#MCMC_class
#----------
#
#The MCMC_class is initialized with a networkx graph, along with the three choices
#that define the graph space.  Calling the class function 'get_graph' 
#advances the Markov chain and returns the current networkx graph 
#
#    Example:  
#
#::    
#
#  import networkx as nx
#  G = nx.Graph()
#  G.add_path([0,1,2,3,4])
#  MC = MCMC_class(G, loops = True, multi = True, v_labeled = False)
#  G2 = MC.get_graph()
#
#This takes a path graph on 4 nodes, instantiates a MCMC_class based on this 
#graph and returns a pointer to a graph G2 which differs from G by one double
#edge swap. Notice that this samples from the space of stub-labeled loopy 
#multigraphs, but can be easily adjusted to other spaces.    
#
#
#Notes
#-----
#In general, directly calling MCMC_step or MCMC_step_stub is faster than 
#using MCMC_class, since updating networkx data structures doesn't benefit 
#from numba acceleration.
#
#For large graphs, the full adjacency matrix may not be able to be stored 
#in memory. If so, the ‘@nb.jit’ function decorator can be deleted and a sparse
#matrix can be passed into these functions as an argument, though at a 
#significant cost in speed.
#
#We use the convention that a self-loop (u,u) contributes 2 to the diagonal of 
#an adjacency matrix, not 1 as in Networkx.  
#
#References
#----------
#Bailey K. Fosdick, Daniel B. Larremore, Joel Nishimura, Johan Ugander. 
#Configuring Random Graph Models with Fixed Degree Sequences (2016)



__docformat__ = 'reStructuredText'

import numpy as np
import networkx as nx
#import numba as nb


#If numba can't load, the following do-nothing decorator will apply
def __no_jit__(*dargs,**dkwargs):
    def decorate(func):
        def call(*args,**kwargs):
            return func(*args,**kwargs)
        return call
    return decorate

try:
    import numba as nb
    jit =  nb.jit
except:
    jit = __no_jit__



@jit(nopython=True,nogil=True)
def MCMC_step(A,edge_list,swaps,loops,multi):
    '''
    
    Performs a vertex-labeled double edge swap.
    
    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be 
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     loops (bool): True only if loops allowed in the graph space.
    |     multi (bool): True only if multiedges are allowed in the graph space.
    
    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled. 
    
    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this 
    to work a sparse adjacency matrix simply requires removing the '@jit'
    decorator.  This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.
            
    '''
    # Choose two edges uniformly at random
    m= len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m-1)
    if p1 == p2: # Prevents picking the same edge twice
        p2 = m-1
        
    u,v = edge_list[p1]        
    if np.random.rand()<0.5: #Pick either swap orientation 50% at random
        x,y = edge_list[p2]        
    else:
        y,x = edge_list[p2]
        
    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_uv = A[u,v]
    w_xy = A[x,y]
    w_ux = A[u,x]
    w_vy = A[v,y]

    # If multiedges are not allowed, resample if swap would replicate an edge
    if not multi and ( w_ux>=1 or w_vy>=1 ):
        return False
    
    num_loops = 0
    unique_nodes = 4
    # Here we count the number of self-loops and unique nodes in our proposed 2 
    # edges. Using the convention where self-loops contribute 2 to the adj matrix
    # requies dividing entries on the diagonal by 2 to get the number of 
    # self-loops.   
    if u == v:        
        num_loops += 1   
        unique_nodes += -1
        w_uv = w_uv/2
    if x == y:
        num_loops += 1
        unique_nodes += -1
        w_xy = w_xy/2
    if u == x:
        unique_nodes += -1
        w_ux = w_ux/2
    if v == y:
        unique_nodes += -1
        w_vy = w_vy/2
    if u == y:
        unique_nodes += -1
    if v == x:
        unique_nodes += -1
    
    if unique_nodes == -2: # If u=v=x=y then unique_nodes currently is -2
        return False
    if unique_nodes == 1: # Correcting u=v=x~=y or similar rotation
        unique_nodes = 2


    # We now run through the different possible edge swap cases
    if unique_nodes == 2:
        if num_loops == 2:
            if multi: # Swapping two self-loops creates would create a multiedge
                swapsTo = 2*w_uv*w_xy
                swapsFrom = ((w_ux+2)*(w_vy+1))/2
            else:
                return False
        elif num_loops == 1:
            return False # The only swap on two nodes with 1 self-loop doesn't change graph
        else: # No_loops on 2 nodes
            if loops:
                swapsTo = (w_uv*(w_xy-1))/2 
                swapsFrom = 2*(w_ux+1)*(w_vy+1) 
            else: # Only change would make 2 self-loops
                return False
    elif unique_nodes == 3:
        if num_loops == 0:
            if loops: # Swapping adjacent edges creates a self-loop
                swapsTo = w_uv*w_xy
                swapsFrom = 2*(w_ux+1)*(w_vy+1)
            else: # Only change would make a self-loop
                return False
        else: # Num_loops==1
            swapsTo = 2*w_uv*w_xy
            swapsFrom = (w_ux+1)*(w_vy+1)
    else: # Unique_nodes ==4
        swapsTo = w_uv*w_xy
        swapsFrom = (w_ux+1)*(w_vy+1)
    
    # Based upon the above cases we calculate the minimum probability of the 
    # swap and the reverse swap.
    probTo = np.float(swapsTo)
    probFrom = np.float(swapsFrom)
    
    if probFrom/probTo < 1.0:
        P = probFrom/probTo
    else:
        P = 1.0
    

    # If we proceed with the swap we update A, swaps and edge_list
    if np.random.rand() < P:
        swaps[0] = u # Numba currently is having trouble with slicing
        swaps[1] = v
        swaps[2] = x
        swaps[3] = y
        
        A[u,v] += -1
        A[v,u] += -1
        A[x,y] += -1
        A[y,x] += -1
        
        A[u,x] += 1
        A[x,u] += 1
        A[v,y] += 1
        A[y,v] += 1
        
        edge_list[p1,0] = u
        edge_list[p1,1] = x
        edge_list[p2,0] = v
        edge_list[p2,1] = y
        return True
    else:
        return False
            
@jit(nopython=True,nogil=True)
def MCMC_step_stub(A,edge_list,swaps,loops,multi):
    '''
    
    Performs a stub-labeled double edge swap.
    
    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be 
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     loops (bool): True only if loops allowed in the graph space.
    |     multi (bool): True only if multiedges are allowed in the graph space.
    
    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled. 
    
    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this 
    to work a sparse adjacency matrix simply requires removing the '@nb.jit'
    decorator. This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.
        
    '''
    # Choose two edges uniformly at random
    m= len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m-1)
    if p1 == p2: # Prevents picking the same edge twice
        p2 = m-1
        
    u,v = edge_list[p1]        
    if np.random.rand()<0.5: #Pick either swap orientation 50% at random
        x,y = edge_list[p2]        
    else:
        y,x = edge_list[p2]

    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_ux = A[u,x]
    w_vy = A[v,y]

    # If multiedges are not allowed, resample if swap would replicate an edge
    if not multi:
        if ( w_ux>=1 or w_vy>=1 ):
            return False
            
        if u == v and x == y:
            return False
    
    #If loops are not allowed then only swaps on 4 distinct nodes are possible
    if not loops:
        if u == x or u == y or v == x or v == y:
            return False
    
   
    swaps[0] = u # Numba currently is having trouble with slicing
    swaps[1] = v
    swaps[2] = x
    swaps[3] = y
    
    A[u,v] += -1
    A[v,u] += -1
    A[x,y] += -1
    A[y,x] += -1
    
    A[u,x] += 1
    A[x,u] += 1
    A[v,y] += 1
    A[y,v] += 1   

    edge_list[p1,0] = u
    edge_list[p1,1] = x
    edge_list[p2,0] = v
    edge_list[p2,1] = y
    
    return True



   
class MCMC_class:
    '''
    
    MCMC_class stores the objects necessary for MCMC steps. This
    implementation maintains a networkx version of the graph, though at some
    cost in speed.
    
    | Args:
    |     G (networkx_class): This graph initializes the Markov chain. All 
             sampled graphs will have the same degree sequence as G.
    |     loops (bool): True only if loops allowed in the graph space.
    |     multi (bool): True only if multiedges are allowed in the graph space.
    |     v_labeled (bool): True only if the graph space is vertex-labeled. 
            True by default.
    
    | Returns:
    |     None 
        
    Notes
    -----
    MCMC_class copies the instance of the graph used to initialize it. This 
    class supports loopy graphs, but depending on the degree sequence, it may 
    not be able to sample from all loopy graphs.     
        
    '''
    def __init__(self,G,loops,multi,v_labeled = True):
        self.G = flatten_graph(G,loops,multi)
        self.loops = loops
        self.multi = multi
        self.uniform = v_labeled
        if self.uniform:
            self.step = MCMC_step
        else:
            self.step = MCMC_step_stub
        
        self.A = nx.adjacency_matrix(self.G) 
        self.A = self.A.toarray()
        self.A += np.diag(np.diag(self.A))
        self.edge_list = np.array(self.G.edges())
        self.swaps = np.zeros(4,dtype=np.int64)
        
        

    def get_graph(self):    
        '''
        
        The Markov chains will attempt a double edge swap, after which the next
        graph/multigraph in the chain is returned.
        
        | Args:
        |     None
        
        | Returns:
        |     The Markov chain's current graph.
        
        Notes
        -----
        Modifying the returned graph will cause errors in repeated calls of 
        this function.
            
        '''
        new = self.step(self.A,self.edge_list,self.swaps,self.loops,self.multi)
        if new:
#            print swaps, new
#            print A
            u,v,x,y = self.swaps
            self.G.add_edge(u,x)
            self.G.add_edge(v,y)
            self.G.remove_edge(u,v)
            self.G.remove_edge(x,y)
        
        return self.G
        
def flatten_graph(graph,loops,multi):
    '''
    
    Takes an input graph and returns a version w/ or w/o loops and 
    multiedges.
    
    | Args:
    |     G (networkx_class): The original graph.
    |     loops (bool): True only if loops are allowed in output graph.
    |     multi (bool): True only if multiedges are allowed in output graph.
    
    | Returns:
    |     A graph with or without multiedges and/or self-loops, as specified. 
        
    '''
    
    graph = nx.convert_node_labels_to_integers(graph)  
    
    if loops and multi:
        return nx.MultiGraph(graph)
    elif loops and not multi:
        return nx.Graph(graph)
    elif not multi and not loops:
        G = nx.Graph(graph)
    else:
        G = nx.MultiGraph(graph)
        
        
    for e in G.selfloop_edges():
        G.remove_edge(*e)
    if G.selfloop_edges() != []:
        print "Cannot remove all self-loops"

    return G        
        


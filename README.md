# Double Edge Swap MCMC Graph Sampler

## What is it?

For a fixed degree sequence and a given graph space, a configuration model is a uniform distribution of graphs with that degree sequence in that space. This code package implements Markov chain Monte Carlo methods to sample from configuration models, as discussed in the associated paper [1]. Given an input graph (and its implicit degree sequence), these methods can sample graphs from the configuration model on the space of (either stub or vertex-labeled): simple graphs, loopy graphs, multigraphs or loopy multigraphs. 

[1] Bailey K. Fosdick, Daniel B. Larremore, Joel Nishimura, Johan Ugander (2016) Configuring Random Graph Models with Fixed Degree Sequences. https://arxiv.org/abs/1608.00607

## How to Cite

If this code is utilized in work contributing to a academic publication please cite the associated paper [1].

## Included

-  README.md
-  dbl_edge_doc.pdf
-  [dbl_edge_worksheet.ipynb](./dbl_edge_worksheet.ipynb) (Jupyter Notebook demonstration)
-  dbl_edge_mcmc.py
-  dist_verification.py
-  sample_assortativity.py
-  geomnet_edgelist
-  /output
-  /verification
-  /html_doc

## Documentation

Please see the documentation available in either 'html_doc/index.hmtl' or 'dbl_edge_doc.pdf’. Alternatively, jump immediately into ‘dbl_edge_worksheet.ipynb’. 

##  Dependencies

This package utilizes Python 2.7.x, Numba 0.24.0, Numpy, Scipy, Networkx, and Pylab.  The tutorial script runs in Jupyter. Note that this packages are currently included in the Anaconda distribution of python 2.7, available at https://www.continuum.io/downloads.  Failure to load Numba will negatively impact performance, but should not affect output.


## Installation

Simply import desired python modules into an instance of Python 2.7.x.


## History

-  8/9/2016 v.1 Renamed some functions and variables for added clarity.
-  8/1/2016  v.1 More documentation added.
-  7/29/2016 v.1 Additional documentation added.
-  7/23/2016 v.1 iPython worksheet added.


## Authors

- Bailey K. Fosdick, Department of Statistics, Colorado State University, Ft. Collins, CO 80523 USA (bailey [at] stat.colostate.edu)
- Daniel B. Larremore, Santa Fe Institute, 1399 Hyde Park Rd. Sante Fe, NM, 87501 USA (larremore [at] santafe.edu)
- Joel Nishimura, School of Mathematical and Natural Sciences, Arizona State University, Glendale, AZ 85306 USA (joel.nishimura [at] asu.edu)
- Johan Ugander, Management Science & Engineering, Stanford University, Stanford, CA, 94305 USA (jugander [at] stanford.edu) 


## Contact

Please direct questions and bug reports to joel.nishimura [at] asu.edu.


  

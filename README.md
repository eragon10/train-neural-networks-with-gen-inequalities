# lipnet

This repository contains the code for bachelor thesis "Training neural networks with generalized inequalities". The thesis was written at [Institute for Systems Theory and Automatic Control (IST)](https://www.ist.uni-stuttgart.de/de) at the [University of Stuttgart](https://www.uni-stuttgart.de/).


### content
 - train neural networks with lipschitz bound
 - three different methods:
  - ADMM method
  - projected gradient descent
  - barrier method
 - use MNIST dataset to train neural network with lipschitz bound
	
### dependencies
 - [Blaze](https://bitbucket.org/blaze-lib/blaze/src/master/) is an open-source, high-performance C++ math library for dense and sparse arithmetic (v3.7.0).
 - [MOSEK](https://www.mosek.com/) solves all your LPs, QPs, SOCPs, SDPs and MIPs (v9.2.25).
 - [cereal](https://uscilab.github.io/cereal/) is a header-only C++11 serialization library (v1.3.0).
 - [Lyra](https://github.com/bfgroup/Lyra) A simple to use, composing, header only, command line arguments parser for C++ 11 and beyond (v1.4.0).
 - [csv2](https://github.com/p-ranav/csv2) csv loader lib.
	
### usage
 - clone repository
 - download additional libraries
 - patch blaze library
 - build: cmake and make

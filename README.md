# Adversarial Surrogate Losses for Ordinal Regression (NIPS 2017)
This repository is a code example of a the paper: 
[Adversarial Surrogate Losses for Ordinal Regression]()

Full paper: [https://www.cs.uic.edu/~rfathony/pdf/fathony2017adversarial.pdf](https://www.cs.uic.edu/~rfathony/pdf/fathony2017adversarial.pdf)

### Abstract

Ordinal regression seeks class label predictions when the penalty incurred for mistakes increases according to an ordering over the labels. The absolute error is a canonical example. Many existing methods for this task reduce to binary classification problems and employ surrogate losses, such as the hinge loss. We instead derive uniquely defined surrogate ordinal regression loss functions by seeking the predictor that is robust to the worst-case approximations of training data labels, subject to matching certain provided training data statistics. We demonstrate the advantages of our approach over other surrogate losses based on hinge loss approximations using UCI ordinal prediction tasks.

# Setup

The source code is written in [Julia](http://julialang.org/) version 0.6.X.

### Dependency
The primal optimization uses stochastic averaged gradient algorithm. It does not requires any dependency.
The dual optimization is in a quadratic program (QP) form. [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl) is needed to solve a QP problem. Please refer to Gurobi.jl instruction for the installation.

### Example

Three example files are provided: 

* `example_th.jl` :
run primal optimzation on adversarial ordinal regression loss with thresholded features

* `example_mc.jl` :
run primal optimzation on adversarial ordinal regression loss with multiclass features

* `example_kernel.jl`: 
run dual quadratic programming optimzation on adversarial ordinal regression loss using polynomial / Gaussian kernel

In each file, the code will run training with k-fold cross validation for the example dataset (`diabetes`). 
After finding the best setting, it will run testing phase.

To change the training settings, please directly edit the setting values in the given example.

To run the code, execute (in terminal):
```
julia example_th.jl
```

# Citation (BibTeX)
```
@incollection{fathony2017adversarial,
title = {Adversarial Surrogate Losses for Ordinal Regression},
author = {Fathony, Rizal and Bashiri, Mohammad and Ziebart, Brian},
booktitle = {Advances in Neural Information Processing Systems 30},
year = {2017},
}
```
# Acknowledgements 
This research was supported as part of the Future of Life Institute (futureoflife.org) FLI-RFP-AI1 program, grant\#2016-158710 and by NSF grant RI-\#1526379.

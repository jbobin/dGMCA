# DGMCA
**(or Distributed Generalized Morphological Component Analysis)**

New DGMCA [article](https://hal.archives-ouvertes.fr/hal-02426991/) available! 

Algorithm to solve the Blind Source Separation (BSS) problem in a parallelized way.

Article of the [SPARS](http://www.spars-workshop.org/en/index.html) 2019 conference:Tobias Liaudat, Jerome Bobin, Christophe Kervazo. Distributed sparse BSS for large-scale datasets.2019. hal-02088466. [(pdf)](https://hal.archives-ouvertes.fr/hal-02088466/document)

The main theoretical framework of the algorith is taken from the method GMCA [1]. This new algorithm allows to tackle the BSS problem in a faster way as well as for very-large datasets that could not be treated before. 

The original problem of factorizing a matrix X (observation matrix) into two matrices A (mixing matrix) and S (source matrix) is divided into sub-problems as it can be seen in the following figure:

<p align="center">
  <img src="./Fig/v1.png?raw=true" width="500" title="hover text">
</p>

A (very) basic scheme of the algorithm follows:

<p align="center">
  <img src="./Fig/dgmca2_schema.png?raw=true" width="500" title="hover text">
</p>

One of the essential points in this novel method is the fusion of the different estimations of the mixing matrices. It is done by doing an optimization on the hypersphere, a Riemannian manifold, by means of a Fréchet Mean follwing [2]. The setup of the algorith forces the columns of the mixing matrix to live in that manifold. 

The next scheme illustrates the fusion of the mixing matrices.

<p align="center">
  <img src="./Fig/ill_frechetmean.png?raw=true" width="500" title="hover text">
</p>

Finally, if the observations are not sparse (or approximatively sparse) in its natural domain, by means of the parameter *J*, a wavelet decomposition (starlets or Isotropic Undecimated Wavelets) is preformed in order to solve the BSS problem.

A basic scheme of the wavelet decomposition is presented next:

<p align="center">
  <img src="./Fig/transform_decomp.png?raw=true" width="500" title="hover text">
</p>


* [1] [J.Bobin, J.-L. Starck, Y.Moudden, J. Fadili, Blind Source Separation: the Sparsity Revolution](http://jbobin.cosmostat.org/docs/aiep08.pdf)
* [2] [B. Asfari, R. Tron, R. Vidal. On The Convergence of Gradient Descent for Finding the Riemannian Center of Mass.](https://arxiv.org/pdf/1201.0925.pdf)
* [3] [C.Kervazo, T.Liaudat, J.Bobin. Faster and better sparse blind source separation through mini-batch optimization.](https://hal.archives-ouvertes.fr/hal-02426991)

## Setting things up

To use the python/C++ binders, it is important to compile the codes available in the src folder. For that purpose, you need to have boost python and boost numpy properly set up. This will be updated soon with a more convenient distributed implementation.

## Testing

*(Up to now)* There is a jupyter notebook (notebooks folder) with an application to generalized gaussian sources - more to come soon

#### Acknowledgements
The work was done by **Christophe Kervazo**, **Tobias Liaudat** and **Jérôme Bobin**.

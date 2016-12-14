Reaction Fingerprints
=============

This algorithm uses a new reaction fingerprint to predict the reaction type of a reaction. The fingerprint is made up of a concatenation of the fingerprints of the reactant and reagent molecules in the reaction.

The code in the neural fingerprint directory is heavily based on :

[Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)

The code for the above paper can be found at :
[HIPS/neural-fingerprint](https://github.com/HIPS/neural-fingerprint)


## How to install

This package requires:
* Scipy version >= 0.15.0
* [RDkit](http://www.rdkit.org/docs/Install.html)
* [Autograd](http:github.com/HIPS/autograd) (Just run `pip install autograd`)
* [Hyperopt](https://github.com/hyperopt/hyperopt)


## Authors

This software was primarily written by [Jennifer Wei](mailto:jenniferwei@fas.harvard.edu) and [David Duvenaud](http://www.cs.toronto.edu/~duvenaud/)
We'd also love to hear about your experiences with this package in general.
Drop us an email!

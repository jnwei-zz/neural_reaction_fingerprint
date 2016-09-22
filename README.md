Reaction Fingerprints
=============

This algorithm uses a new reaction fingerprint to predict the reaction type of a reaction. The fingerprint is made up of a concatenation of the fingerprints of the reactant and reagent molecules in the reaction.

The code used in this algorithm is heavily based on :

[Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)


## How to install

This package requires:
* Scipy version >= 0.15.0
* [RDkit](http://www.rdkit.org/docs/Install.html)
* [Autograd](http:github.com/HIPS/autograd) (Just run `pip install autograd`)

## Examples

This package includes a [regression example](examples/regression.py) and a [visualization example](examples/vizualization.py) in the examples directory.

## Authors

This software was primarily written by [David Duvenaud](http://people.seas.harvard.edu/~dduvenaud/), [Dougal Maclaurin](mailto:maclaurin@physics.harvard.edu), and [Ryan P. Adams](http://www.seas.harvard.edu/directory/rpa).
Please feel free to submit any bugs or feature requests.
We'd also love to hear about your experiences with this package in general.
Drop us an email!

We want to thank Jennifer Wei for helpful contributions and advice, and Analog Devices International and Samsung Advanced Institute of Technology for their generous support.
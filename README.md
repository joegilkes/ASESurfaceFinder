# ASESurfaceFinder

A utility for determining surface facets and absorption points of ASE-based systems consisting of molecules on surfaces.

[ASE](https://wiki.fysik.dtu.dk/ase/) comes with an excellent selection of utilities for working with atomic surfaces, enabling the construction of many common surface facets, the definition of symmetrically equivalent points across these surfaces, and the adsorption of arbitrary molecules to these surface sites. However, determining which of these sites an absorbed molecule is bound to without prior knowledge is a non-trivial task for computers.

ASESurfaceFinder implements automated tools for training and validating random forest classification models (implemented in [scikit-learn](https://scikit-learn.org/stable/index.html)) that can identify surface sites based on the local atomic environment of adsorbed atoms. Given unseen adsorbed systems, it then enables these models to be used for prediction of both surface facet and high-symmetry absorption site, to be used when cataloguing externally-generated adsorbed systems.

## Installation

TBC

## Usage

TBC


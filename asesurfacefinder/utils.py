from dscribe.descriptors import LMBTR, SOAP
import numpy as np

from ase import Atoms
from numpy.typing import ArrayLike
from collections.abc import Sequence

def descgen_mbtr(elements: Sequence[str]):
    '''Constructs a local MBTR descriptor generator for the requested surface elements.'''
    lmbtr = LMBTR(
        species=elements,
        geometry={"function": "distance"},
        grid={"min": 0.1, "max": 10.0, "n": 500, "sigma": 0.05},
        weighting={"function": "exp", "scale": 1, "threshold": 1e-2},
        periodic=True,
        normalization="none",
    )
    return lmbtr


def descgen_soap(elements: Sequence[str]):
    '''Constructs a SOAP descriptor generator for the requested surface elements.'''
    soap = SOAP(
        species=elements,
        periodic=True,
        r_cut=10.0,
        n_max=8,
        l_max=6,
    )
    return soap


def sample_ads_pos(xy_pos: ArrayLike, z_bounds: tuple[float, float], xy_noise: float):
    '''Sample an adsorbate position.
    
    Given the absolute XY position of a high-symmetry point, samples
    a new XY point by adding normally distributed random noise, and
    an adsorption height from a uniform distribution between upper
    and lower bounds.

    Returns a tuple of new XY position and adsorption height.
    '''
    new_xy_pos = np.copy(xy_pos)
    new_xy_pos += np.random.normal(0.0, xy_noise, 2)

    z = np.random.uniform(z_bounds[0], z_bounds[1])

    return new_xy_pos, z


def get_absolute_abspos(slab: Atoms, site: str):
    '''Determine the absolute position of a high-symmetry adsorption site in a given unit cell.'''
    spos = slab.info['adsorbate_info']['sites'][site]
    cell = slab.info['adsorbate_info']['cell']
    pos = np.dot(spos, cell)

    return pos


def check_tags(tags: Sequence[int], natoms: int):
    if len(tags) != natoms:
        tag_errmsg = 'Slab-adsorbate system has malformed tags (wrong length).'
    elif np.sum(tags) <= 0:
        tag_errmsg = 'Slab-adsorbate system has malformed tags (no surface layers)'
    elif len(np.argwhere(tags==0).flatten()) == 0:
        tag_errmsg = 'Slab-adsorbate system is missing an adsorbate (no ads tags)'
    else:
        tag_errmsg = ''
    return tag_errmsg


def guess_tags(slab: Atoms, surf_elements: Sequence[str]):
    '''Try to approximate tags based on elemental difference between surfaces and adsorbate.'''
    tags = np.zeros(len(slab), dtype=int)
    for i, elem in enumerate(slab.symbols):
        if elem in surf_elements:
            tags[i] = 1

    return tags


def has_elems(atoms: Atoms, elements: Sequence[str]):
    '''Determine if `atoms` contains any of `elements`.'''
    syms = set(atoms.get_chemical_symbols())
    return len(set(elements).intersection(syms)) > 0


def get_element_idxs(atoms: Atoms, element: str):
    '''Returns the indices of atoms with chemical symbol matching `element`.'''
    elems = atoms.get_chemical_symbols()
    idxs = [i for i, e in enumerate(elems) if e == element]
    return idxs


def get_elements_idxs(atoms: Atoms, elements: Sequence[str]):
    '''Returns the indices of atoms with chemical symbols matching those in `elements`.'''
    idxs = np.flatten([get_element_idxs(atoms, e) for e in elements]).sort()
    return idxs


def _get_surface_idxs(slab: Atoms, base_surface: Atoms, tol: float=5e-2):
    '''Test function.
    
    Attempts to find atoms in a given `slab` which match the element and
    height of atoms in `base_surface`.
    
    While this should be able to identify atoms matching in surfaces of the
    same height (i.e. same number of layers), different slab thicknesses will
    cause problems.
    
    This should therefore only be used for testing.
    '''
    surf_elements = base_surface.get_chemical_symbols()
    if not has_elems(slab, surf_elements):
        raise ValueError('`slab` does not contain any elements in `base_surface`.')
    
    # Filter atoms and reset remaining to bottom of cell.
    elems_idxs = get_elements_idxs(slab, surf_elements)
    slab_reduced = slab.copy()[elems_idxs]
    reduced_to_full_idxmap = {i: idx for i, idx in enumerate(elems_idxs)}
    slab_lowest_z = np.min(slab_reduced.get_positions()[:, 2])
    slab_reduced.set_positions(slab_reduced.get_positions()-[0.0, 0.0, slab_lowest_z])

    # Reset base surface to bottom of cell
    surf_lowest_z = np.min(base_surface.get_positions()[:, 2])
    bsc = base_surface.copy()
    bsc.set_positions(bsc.get_positions()-[0.0, 0.0, surf_lowest_z])
    surf_heights = np.unique(bsc.get_positions()[:, 2])

    # Find atoms matching heights.
    slab_surf_idxs = []
    for i, atom in enumerate(slab_reduced):
        if np.any(np.abs(surf_heights - atom.position[2]) < tol):
            slab_surf_idxs.append(reduced_to_full_idxmap[i])

    return slab_surf_idxs

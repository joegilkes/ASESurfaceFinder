from asesurfacefinder.sample_bounds import SampleBounds

from dscribe.descriptors import LMBTR, SOAP
import numpy as np
from ase.build import add_adsorbate
from ase.geometry.analysis import Analysis

from ase import Atoms
from numpy.typing import ArrayLike
from collections.abc import Sequence
from typing import Union

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


def sample_ads_pos(xy_pos: ArrayLike, z_bounds: Union[tuple[float, float], tuple[float, float, float]], r_max: float):
    '''Sample an adsorbate position.

    Uniformly samples an XY position displaced from `xy_pos` within a 
    circle of radius `r_max`.
     
    Samples a Z position uniformly within the bounds defined by `z_bounds`.
    If this is a tuple of two numbers `(z_mid, z_max)`, defines a hemispheroidal 
    sampling volume with a flat base by scaling the upper sampling bound based
    on the distance from the center of the circle to the XY position.
    If this is a tuple of three numbers `(z_min, z_mid, z_max)`, defines an
    ovoid sampling volume with different upper and lower halves.

    Returns a tuple of new XY position and adsorption height.
    '''
    r = np.random.uniform(0, r_max)
    theta = np.random.uniform(0, 2 * np.pi)
    xy_sample = np.array([r * np.cos(theta), r * np.sin(theta)])
    new_xy_pos = xy_pos + xy_sample

    if len(z_bounds) == 2:
        z_diff = z_bounds[1] - z_bounds[0]
        z_upper = max(z_bounds[0] + (np.sqrt(r_max**2 - r**2)/r_max)*z_diff, z_bounds[0])
        z = np.random.uniform(z_bounds[0], z_upper)
    elif len(z_bounds) == 3:
        z_min, z_mid, z_max = z_bounds
        z_diff_lower = z_mid - z_min
        z_lower = min(z_mid - (np.sqrt(r_max**2 - r**2)/r_max)*z_diff_lower, z_mid)
        z_diff_upper = z_max - z_mid
        z_upper = max(z_mid + (np.sqrt(r_max**2 - r**2)/r_max)*z_diff_upper, z_mid)
        z = np.random.uniform(z_lower, z_upper)

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


def get_site_coordination(surface: Atoms, site: str, bounds: SampleBounds):
    '''Determines expected coordination of adsorbate atoms on a surface site.

    Places a hydrogen atom at a site's `bounds.z_min` and checks its 
    connectivity to the surface to establish a site coordination number.
    '''
    slab = surface.copy()
    add_adsorbate(slab, 'H', bounds.z_min, site)
    ana = Analysis(slab, self_interaction=False, bothways=True)
    nl = ana.nl[0]
    coord = len(nl.get_neighbors(len(slab)-1)[0])
    return coord

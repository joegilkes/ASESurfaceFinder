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


def guess_tags(slab: Atoms, surf_elements: Sequence[str]):
    '''Try to approximate tags based on elemental difference between surfaces and adsorbate.'''
    tags = np.zeros(len(slab), dtype=int)
    for i, elem in enumerate(slab.symbols):
        if elem in surf_elements:
            tags[i] = 1

    return tags
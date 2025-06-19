import pytest

from asesurfacefinder.utils import *
from asesurfacefinder import SampleBounds

import numpy as np
from ase.build import fcc110, fcc111, molecule, add_adsorbate
from dscribe.descriptors import LMBTR, SOAP


def test_descgens():
    '''Tests that DScribe descriptor generators are being created properly.'''
    elements = ['Au', 'Pt', 'C', 'O', 'H']

    mbtr = descgen_mbtr(elements)
    assert type(mbtr) is LMBTR
    assert mbtr.n_elements == 6
    assert mbtr.get_number_of_features() == 3000

    soap = descgen_soap(elements)
    assert type(soap) is SOAP
    assert soap.n_elements == 5
    assert soap.get_number_of_features() == 5740


def test_sampling():
    '''Tests that adsorbate position sampling correctly generates points 
    and that this can be seeded properly.
    '''
    np.random.seed(12345)
    xy, z = sample_ads_pos([1.0, 5.0], (1.0, 2.5), 0.5)
    assert type(xy) == np.ndarray
    assert len(xy) == 2
    assert all([a - b <= 1e-8 for a, b in zip(xy, [0.81172253, 5.42496837])])
    assert type(z) == float
    assert z - 1.1016692344648185 <= 1e-8

    xy, z = sample_ads_pos([0.0, 1.2], (-0.3, 0.8, 1.5), 0.75)
    assert z - 0.7725729673134012 <= 1e-8

    with pytest.raises(ValueError):
        sample_ads_pos([0.0, 0.0], (1.0, 2.0, 3.0, 4.0), 0.5)


def test_abspos():
    '''Tests conversion of relative to absolute unit cell position.'''
    slab = fcc110('Ag', (3,3,3))
    hollow_pos = get_absolute_abspos(slab, 'hollow')
    assert all([a - b <= 1e-8 for a, b in zip(hollow_pos, [2.045, 1.44603337])])

    slab = fcc111('Au', (5, 4, 3))
    hcp_pos = get_absolute_abspos(slab, 'hcp')
    assert all([a - b <= 1e-8 for a, b in zip(hcp_pos, [2.88499567, 1.66565303])])


def test_tagging():
    '''Tests tagging utilities.'''
    slab = fcc110('Ag', (3,3,3))
    slab.center(10.0, axis=2)
    mol = molecule('CO2')
    add_adsorbate(slab, mol, 1.5, 'ontop', mol_index=2)
    tags = guess_tags(slab, ['Ag'])
    real_tags = (slab.get_tags() >= 1).astype(int)
    assert all(tags == real_tags)

    errmsg = check_tags(tags, len(slab))
    assert len(errmsg) == 0
    errmsg = check_tags(tags, 10)
    assert 'wrong length' in errmsg
    errmsg = check_tags(np.zeros(10), 10)
    assert 'no surface layers' in errmsg
    errmsg = check_tags(np.ones(10), 10)
    assert 'no ads tags' in errmsg


def test_coordination():
    '''Tests site coordination calculation.'''
    slab = fcc110('Ag', (3,3,3))
    ontop_bounds = SampleBounds(0.6, 1.5, 2.4)
    assert get_site_coordination(slab, 'ontop', ontop_bounds) == 1
    hollow_bounds = SampleBounds(0.65, z_min=0.1, z_mid=0.65, z_max=1.3)
    assert get_site_coordination(slab, 'hollow', hollow_bounds) == 1
    longbridge_bounds = SampleBounds(0.5, z_min=0.5, z_mid=1.0, z_max=1.5)
    assert get_site_coordination(slab, 'longbridge', longbridge_bounds) == 2
    shortbridge_bounds = SampleBounds(0.5, z_min=0.6, z_mid=1.75, z_max=2.1)
    assert get_site_coordination(slab, 'shortbridge', shortbridge_bounds) == 2
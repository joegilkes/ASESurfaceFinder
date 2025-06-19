import pytest

from asesurfacefinder import SurfaceFinder, SampleBounds
from asesurfacefinder.exception import NonPeriodicError, NoSurfaceError

import numpy as np
from ase.build import fcc100, fcc110, fcc111
from ase.neighborlist import natural_cutoffs
from ase.io import read
import multiprocessing


def test_surfacefinder():
    # This seed causes sampling that makes sys3 be predicted with the H on an ontop site.
    np.random.seed(1)
    surfaces = [
        fcc100('Au', (1,1,3)),
        fcc110('Au', (1,1,3)),
        fcc111('Au', (1,1,3)),
        fcc100('Pt', (1,1,3))
    ]
    surface_labels = ['Au_fcc100', 'Au_fcc110', 'Au_fcc111', 'Pt_fcc100']
    sample_bounds = [
        {
            'ontop': SampleBounds(0.6, 1.5, 2.4),
            'bridge': SampleBounds(0.45, z_min=0.75, z_mid=1.4, z_max=2.0),
            'hollow': SampleBounds(0.5, z_min=0.3, z_mid=0.8, z_max=1.75)
        },
        {
            'ontop': SampleBounds(0.6, 1.5, 2.4),
            'hollow': SampleBounds(0.65, z_min=0.1, z_mid=0.65, z_max=1.3),
            'longbridge': SampleBounds(0.5, z_min=-0.6, z_mid=1.0, z_max=1.5),
            'shortbridge': SampleBounds(0.5, z_min=0.6, z_mid=1.75, z_max=2.1)
        },
        {
            'ontop': SampleBounds(0.6, 1.5, 2.4),
            'bridge': SampleBounds(0.35, z_min=0.7, z_mid=1.6, z_max=2.2),
            'fcc': SampleBounds(0.35, z_min=0.4, z_mid=1.4, z_max=2.0),
            'hcp': SampleBounds(0.35, z_min=0.4, z_mid=1.4, z_max=2.0)
        },
        {
            'ontop': SampleBounds(0.6, 1.5, 2.4),
            'bridge': SampleBounds(0.45, z_min=0.75, z_mid=1.4, z_max=2.0),
            'hollow': SampleBounds(0.5, z_min=0.4, z_mid=0.8, z_max=1.75)
        }
    ]

    sf = SurfaceFinder(surfaces, labels=surface_labels, sample_bounds=sample_bounds)
    sf.train(
        samples_per_site=4000,
        n_jobs=multiprocessing.cpu_count()
    )
    valid = sf.validate(
        samples_per_site=200,
        surf_mults=[(1,1,1), (3,3,1), (5,5,1)]
    )
    assert valid

    sys1 = read('examples/4MeO_Au_fcc111.xyz')
    slab, molecules, labels = sf.predict(sys1)
    assert len(slab) == 108
    assert len(molecules) == 4 and len(labels) == 4
    assert [len(l) for l in labels] == [1, 1, 1, 0]
    assert [[k for k in l.keys()][0] for l in labels[:-1]] == [0, 0, 0]
    assert [l[0]['site'] for l in labels[:-1]] == ['Au_fcc111_fcc', 'Au_fcc111_fcc', 'Au_fcc111_fcc']
    assert [l[0]['coordination'] for l in labels[:-1]] == [3, 3, 3]
    assert [l[0]['bonded_elem'] for l in labels[:-1]] == ['O', 'O', 'O']

    sys2 = read('examples/CO+H2O+NHCH3_Au_fcc111.xyz')
    slab, molecules, labels = sf.predict(sys2, nl_cutoffs=natural_cutoffs(sys2, mult=1.2))
    assert len(slab) == 108
    assert len(molecules) == 3 and len(labels) == 3
    assert [len(l) for l in labels] == [1, 1, 1]
    assert [[k for k in l.keys()][0] for l in labels] == [0, 0, 0]
    assert [l[0]['site'] for l in labels] == ['Au_fcc111_ontop', 'Au_fcc111_fcc', 'Au_fcc111_bridge']
    assert [l[0]['coordination'] for l in labels] == [1, 3, 2]
    assert [l[0]['bonded_elem'] for l in labels] == ['C', 'O', 'N']

    # Seeded this way, low-lying H is detected on an ontop site and requires reject_bonded_hydrogens to be removed.
    sys3 = read('examples/CH3CH2NH_Pt_fcc100.xyz')
    sys3.center(10.0, axis=2)
    slab, molecules, labels = sf.predict(sys3, nl_cutoffs=natural_cutoffs(sys3, mult=1.15), reject_bonded_hydrogens=True)
    assert len(slab) == 27
    assert len(molecules) == 1 and len(labels) == 1
    assert len(labels[0]) == 1
    assert [k for k in labels[0].keys()][0] == 2
    assert labels[0][2]['site'] == 'Pt_fcc100_ontop'
    assert labels[0][2]['coordination'] == 1
    assert labels[0][2]['bonded_elem'] == 'N' 

    sys4 = read('examples/CO+H2O+NHCH3.xyz')
    with pytest.raises(NonPeriodicError):
        _, _, _ = sf.predict(sys4)
    sys4.set_cell([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 10.0]])
    sys4.set_pbc([True, True, False])
    with pytest.raises(NoSurfaceError):
        _, _, _ = sf.predict(sys4)

    # Reseed with a seed that makes sys3 H be predicted in a bridge site.
    np.random.seed(12345)
    sf = SurfaceFinder(surfaces, labels=surface_labels, sample_bounds=sample_bounds)
    sf.train(
        samples_per_site=4000,
        n_jobs=multiprocessing.cpu_count()
    )
    valid = sf.validate(
        samples_per_site=200,
        surf_mults=[(1,1,1), (3,3,1), (5,5,1)]
    )
    assert valid

    # After retraining, H is predicted as a bridge site and requires reject_wrong_coordination to be removed.
    slab, molecules, labels = sf.predict(sys3, nl_cutoffs=natural_cutoffs(sys3, mult=1.15), reject_wrong_coordination=True)
    assert len(slab) == 27
    assert len(molecules) == 1 and len(labels) == 1
    assert len(labels[0]) == 1
    assert [k for k in labels[0].keys()][0] == 2
    assert labels[0][2]['site'] == 'Pt_fcc100_ontop'
    assert labels[0][2]['coordination'] == 1
    assert labels[0][2]['bonded_elem'] == 'N' 
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from ase.build import add_adsorbate

from asesurfacefinder.utils import *

from ase import Atoms
from collections.abc import Sequence


class SurfaceFinder:
    def __init__(self, surfaces: Sequence[Atoms], 
                 labels: Sequence[str]=None,
                 clf: RandomForestClassifier=None):
        '''Predicts location of adsorbates on surfaces.
        
        Given a list of ASE surfaces with correctly initialised
        high-symmetry adsorbtion points, trains a random forest
        classification model to predict the high-symmetry point
        that adsorbates are bound to.

        Evaluates its own performance on generated surface-adsorbate
        examples as a part of the training procedure.

        If greater control over the hyperparameters of
        `sklearn.ensemble.RandomForestClassifier` is desired, a
        previously setup instance of the class can be passed.

        Arguments:
            surfaces: List of ASE `Atoms` objects representing surfaces with correctly maped high-symmetry adsorbtion points.
            labels: Optional list of names for surfaces, must be of equal length to `surfaces` if provided.
            clf: Optional `RandomForestClassifier` instance.
        '''
        if labels == None:
            self.labels = [str(i+1) for i in range(len(surfaces))]
        elif len(surfaces) != len(labels):
            raise ValueError('Incorrect number of labels for provided number of surfaces.')
        else:
            self.labels = labels
        
        self.elements = []
        self.surface_sites = []
        self.surfaces = []
        for i, surface in enumerate(surfaces):
            for elem in surface.get_chemical_symbols():
                if elem not in self.elements:
                    self.elements.append(elem)

            try:
                info = surface.info['adsorbate_info']
            except KeyError:
                raise ValueError(f'Surface at index {i} is missing "adsorbate_info" dict.')
            
            sites = info['sites'].keys()
            self.surface_sites.append(sites)
            
            if sum(surface.cell[2]) == 0.0:
                surface.center(10.0, axis=2)

            self.surfaces.append(surface)

        self.n_surfaces = len(surfaces)
        self.clf_preconfig = clf


    def train(self, 
              samples_per_site: int=500,
              surf_mults: Sequence[tuple[int, int, int]]=[(1,1,1)],
              ads_z_bounds: tuple[float, float]=(1.2, 2.2),
              ads_xy_noise: float=1e-2,
              n_jobs: int=1
        ):
        '''Trains a random forest classifier to recognise surface sites.
        
        Arguments:
            samples_per_site: Number of adsorbate positions to sample on each surface site during training.
            surf_mults: (X,Y,Z) surface supercell multipliers to sample.
            ads_z_bounds: Tuple of minimum and maximum heights to train for adsorbates binding to surface sites.
            ads_xy_noise: XY-plane noise to add to sampled adsorbate position during training.
        '''
        desc = descgen(self.elements)
        n_mults = len(surf_mults)
        n_samples = sum([n_mults*len(sites)*samples_per_site for sites in self.surface_sites])

        print('ASESurfaceFinder Training')
        print('------------------------------------')
        print('  Constructing LMBTR descriptors for sampled systems...')

        surf_mbtrs = np.zeros((n_samples, desc.get_number_of_features()))
        labels = []
        start_idx = 0
        for i, (surface, sites, label) in enumerate(zip(self.surfaces, self.surface_sites, self.labels)):
            for j, smult in enumerate(surf_mults):
                slab = surface.repeat(smult)
                slab_positions = np.zeros((len(sites)*samples_per_site, 3))

                for k, site in enumerate(sites):
                    site_abspos = get_absolute_abspos(surface, site)
                    for l in range(samples_per_site):
                        slab = surface.copy()
                        xy, z = sample_ads_pos(site_abspos, ads_z_bounds, ads_xy_noise)
                        add_adsorbate(slab, 'H', z, xy)
                        slab_positions[(k*samples_per_site)+l, :] = slab.get_positions()[-1]
                        labels.append(f'{label}_{site}')

                end_idx = start_idx + (len(sites)*samples_per_site)
                slab = surface.copy()
                print(f'  Adding {len(slab_positions)} MBTRs between idxs {start_idx} and {end_idx} ')
                surf_mbtrs[start_idx:end_idx, :] = desc.create(slab, centers=slab_positions, n_jobs=n_jobs)

                start_idx = end_idx

        print('  Training random forest classifier...')
        X, y = shuffle(surf_mbtrs, labels)
        if self.clf_preconfig is None:
            clf = RandomForestClassifier(n_jobs=n_jobs)
        else:
            clf = self.clf_preconfig
            clf.n_jobs = n_jobs
        clf.fit(X, y)
        print('  Training complete.\n')
        
        self.clf = clf
        return clf


    def validate(self,
                 samples_per_site: int=500,
                 surf_mults: Sequence[tuple[int, int, int]]=[(1,1,1), (2,2,1)],
                 ads_z_bounds: tuple[float, float]=(1.2, 2.2),
                 ads_xy_noise: float=1e-2
        ):
        '''Validates a random forest classifier's ability to recognise surface sites.
        
        Arguments:
            samples_per_site: Number of adsorbate positions to sample on each surface site during validation.
            surf_mults: (X,Y,Z) surface supercell multipliers to sample.
            ads_z_bounds: Tuple of minimum and maximum heights to validate for adsorbates binding to surface sites.
            ads_xy_noise: XY-plane noise to add to sampled adsorbate position during validation.
        '''
        if not hasattr(self, 'clf'):
            raise AttributeError('No trained RandomForestClassifier found.')

        desc = descgen(self.elements)
        n_mults = len(surf_mults)
        n_samples = sum([n_mults*len(sites)*samples_per_site for sites in self.surface_sites])

        print('ASESurfaceFinder Validation')
        print('------------------------------------')
        print('  Constructing LMBTR descriptors for sampled systems...')

        surf_mbtrs = np.zeros((n_samples, desc.get_number_of_features()))
        labels = []
        smults = []
        heights = []
        displacements = []
        start_idx = 0
        for i, (surface, sites, label) in enumerate(zip(self.surfaces, self.surface_sites, self.labels)):
            for j, smult in enumerate(surf_mults):
                slab = surface.repeat(smult)
                slab_positions = np.zeros((len(sites)*samples_per_site, 3))

                for k, site in enumerate(sites):
                    site_abspos = get_absolute_abspos(surface, site)
                    for l in range(samples_per_site):
                        slab = surface.copy()
                        xy, z = sample_ads_pos(site_abspos, ads_z_bounds, ads_xy_noise)
                        add_adsorbate(slab, 'H', z, xy)
                        slab_positions[(k*samples_per_site)+l, :] = slab.get_positions()[-1]
                        labels.append(f'{label}_{site}')
                        smults.append(str(smult))
                        heights.append(z)
                        displacements.append(np.linalg.norm(site_abspos-xy))

                end_idx = start_idx + (len(sites)*samples_per_site)
                slab = surface.copy()
                print(f'  Adding {len(slab_positions)} MBTRs between idxs {start_idx} and {end_idx} ')
                surf_mbtrs[start_idx:end_idx, :] = desc.create(slab, centers=slab_positions, n_jobs=self.clf.n_jobs)

                start_idx = end_idx

        assert(len(labels) == n_samples)

        print('  Predicting labels with random forest classifier...')
        pred_labels = self.clf.predict(surf_mbtrs)
        score = self.clf.score(surf_mbtrs, labels)
        print('  Prediction complete.\n')

        correct_acc = 0
        incorrect_idxs = []
        for i, (yt, yp) in enumerate(zip(labels, pred_labels)):
            if yt == yp:
                correct_acc += 1
            else:
                incorrect_idxs.append(i)
        
        print(f'{correct_acc}/{n_samples} sites classified correctly (accuracy = {score}).')

        if len(incorrect_idxs) > 0:
            stat_labels = []
            for i in incorrect_idxs:
                stat_labels.append(f'{labels[i]} {smults[i]} (h = {heights[i]:.2f}, d = {displacements[i]:.2f})')

            stat_clen = np.max([len(lab) for lab in stat_labels])
            print(f'True {' '*(stat_clen-5)} | Predicted')
            print('-'*(stat_clen+18))
            for i, idx in enumerate(incorrect_idxs):
                print(f'{stat_labels[i].ljust(stat_clen)} | {pred_labels[idx]}')

        return 


    def predict(self, ads_slabs: Sequence[Atoms]):
        '''Predicts absorption site and surface facet of adsorbed systems.
        
        Arguments:
            ads_slabs: List of `Atoms` objects representing adsorbates on surface slabs.
        '''
        #TODO: Implement logic for separating adsorbates from surfaces and
        # finding bonded atoms. May require some metric for average A-B bond
        # lengths.
        raise NotImplementedError()
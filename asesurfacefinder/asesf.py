from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from ase.build import add_adsorbate
from ase.geometry.analysis import Analysis
from scipy import sparse

from asesurfacefinder.utils import *

from ase import Atoms
from collections.abc import Sequence
from typing import Union
from dscribe.descriptors.descriptorlocal import DescriptorLocal


class SurfaceFinder:
    def __init__(self, surfaces: Sequence[Atoms], 
                 labels: Sequence[str]=None,
                 clf: RandomForestClassifier=None,
                 descriptor: Union[DescriptorLocal, str]='SOAP'):
        '''Predicts location of adsorbates on surfaces.
        
        Given a list of ASE surfaces with correctly initialised
        high-symmetry adsorption points, trains a random forest
        classification model to predict the high-symmetry point
        that adsorbates are bound to.

        Can evaluate its own performance on generated surface/adsorbate
        examples in a secondary validation step.

        If greater control over the hyperparameters of
        `sklearn.ensemble.RandomForestClassifier` is desired, a
        previously setup instance of the class can be passed.

        Arguments:
            surfaces: List of ASE `Atoms` objects representing surfaces with correctly maped high-symmetry adsorption points.
            labels: Optional list of names for surfaces, must be of equal length to `surfaces` if provided.
            clf: Optional `RandomForestClassifier` instance.
            descriptor: Optional local descriptor type, must be one of ['SOAP', 'LMBTR'] or an instantiated generator fron DScribe.
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
        if type(descriptor) == str:
            if descriptor == 'SOAP':
                self.desc = descgen_soap(self.elements)
            elif descriptor == 'LMBTR' or descriptor == 'MBTR':
                self.desc = descgen_mbtr(self.elements)
            else:
                raise ValueError('Unknown descriptor type specified.')
        else:
            self.desc = descriptor


    def train(self, 
              samples_per_site: int=500,
              surf_mults: Sequence[tuple[int, int, int]]=[(1,1,1)],
              ads_z_bounds: tuple[float, float]=(1.2, 2.75),
              ads_xy_noise: float=1e-2,
              n_jobs: int=1
        ):
        '''Trains a random forest classifier to recognise surface sites.
        
        Arguments:
            samples_per_site: Number of adsorbate positions to sample on each surface site during training.
            surf_mults: (X,Y,Z) surface supercell multipliers to sample.
            ads_z_bounds: Tuple of minimum and maximum heights to train for adsorbates binding to surface sites.
            ads_xy_noise: XY-plane noise to add to sampled adsorbate position during training.
            n_jobs: Number of processes to parallelise descriptor generation and training over.
        '''
        print('ASESurfaceFinder Training')
        print('------------------------------------')

        n_mults = len(surf_mults)
        n_samples = sum([n_mults*len(sites)*samples_per_site for sites in self.surface_sites])

        print('  Constructing local descriptors for sampled systems...')
        surf_descs = np.zeros((n_samples, self.desc.get_number_of_features()))
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
                print(f'  Adding {len(slab_positions)} descs between idxs {start_idx} and {end_idx} ')
                surf_descs[start_idx:end_idx, :] = self.desc.create(slab, centers=slab_positions, n_jobs=n_jobs)

                start_idx = end_idx

        print('  Training random forest classifier...')
        X, y = shuffle(surf_descs, labels)
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
                 ads_z_bounds: tuple[float, float]=(1.3, 2.7),
                 ads_xy_noise: float=1e-2
        ):
        '''Validates a random forest classifier's ability to recognise surface sites.
        
        Arguments:
            samples_per_site: Number of adsorbate positions to sample on each surface site during validation.
            surf_mults: (X,Y,Z) surface supercell multipliers to sample.
            ads_z_bounds: Tuple of minimum and maximum heights to validate for adsorbates binding to surface sites.
            ads_xy_noise: XY-plane noise to add to sampled adsorbate position during validation.
        '''
        print('ASESurfaceFinder Validation')
        print('------------------------------------')
        if not hasattr(self, 'clf'):
            raise AttributeError('No trained RandomForestClassifier found.')
        
        n_mults = len(surf_mults)
        n_samples = sum([n_mults*len(sites)*samples_per_site for sites in self.surface_sites])

        print('  Constructing local descriptors for sampled systems...')
        surf_descs = np.zeros((n_samples, self.desc.get_number_of_features()))
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
                print(f'  Adding {len(slab_positions)} descs between idxs {start_idx} and {end_idx} ')
                surf_descs[start_idx:end_idx, :] = self.desc.create(slab, centers=slab_positions, n_jobs=self.clf.n_jobs)

                start_idx = end_idx

        assert(len(labels) == n_samples)

        print('  Predicting labels with random forest classifier...')
        pred_labels = self.clf.predict(surf_descs)
        score = self.clf.score(surf_descs, labels)
        print('  Prediction complete.\n')

        correct_acc = 0
        incorrect_idxs = []
        for i, (yt, yp) in enumerate(zip(labels, pred_labels)):
            if yt == yp:
                correct_acc += 1
            else:
                incorrect_idxs.append(i)
        
        print(f'  {correct_acc}/{n_samples} sites classified correctly (accuracy = {score}).')

        if len(incorrect_idxs) > 0:
            stat_labels = []
            for i in incorrect_idxs:
                stat_labels.append(f'{labels[i]} {smults[i]} (h = {heights[i]:.2f}, d = {displacements[i]:.2f})')

            stat_clen = np.max([len(lab) for lab in stat_labels])
            print(f'True {' '*(stat_clen-5)} | Predicted')
            print('-'*(stat_clen+18))
            for i, idx in enumerate(incorrect_idxs):
                print(f'{stat_labels[i].ljust(stat_clen)} | {pred_labels[idx]}')

            print(f'\n  {correct_acc}/{n_samples} sites classified correctly (accuracy = {score}).')

        return 


    def predict(self, ads_slab: Atoms, allow_tag_guessing: bool=True):
        '''Predicts adsorption site and surface facet of adsorbed systems.

        Systems may contain multiple adsorbates, as well as non-
        adsorbed, gas-phase molecules. This method separates all
        molecules (adsorbed or otherwise) from the underlying
        surface slab and locates all atoms in these molecules
        that are bonded to the surface.

        Surface-bound atoms are used for predicting adsorption
        sites, which are then mapped back to the correct atoms 
        in the molecules they originated from.

        Returns the isolated surface slab, a list of isolated
        molecule geometries, and a matching list of adsorption
        site dicts. Each dict (one per molecule) is keyed by
        the indices of adsorbed atoms, with dict values representing
        predicted adsorption site and coordination to the surface
        at this site. In the event that a molecule was not
        bound to the surface, its dict remains empty.

        
        Arguments:
            ads_slab: `Atoms` object representing adsorbate(s) on surface slab.
            allow_tag_guessing: Whether to allow surface/adsorbate layer tags to be guessed based on elemental composition if not present or otherwise malformed in input.
        '''
        print('ASESurfaceFinder Prediction')
        print('------------------------------------')
        if not hasattr(self, 'clf'):
            raise AttributeError('No trained RandomForestClassifier found.')

        tags = ads_slab.get_tags()
        tag_errmsg = ''
        if len(tags) != len(ads_slab):
            tag_errmsg = 'Slab-adsorbate system has malformed tags (wrong length).'
        elif np.sum(tags) <= 0:
            tag_errmsg = 'Slab-adsorbate system has malformed tags (no surface layers)'
        elif len(np.argwhere(tags==0).flatten()) == 0:
            tag_errmsg = 'Slab-adsorbate system is missing an adsorbate (no ads tags)'
        
        if len(tag_errmsg) > 0:
            if allow_tag_guessing:
                print(tag_errmsg)
                print('WARNING: Guessing surface/adsorbate separation from elements.')
                tags = guess_tags(ads_slab, self.elements)
            else:
                raise RuntimeError(tag_errmsg)
    
        slabatom_slabidxs = np.argwhere(tags!=0).flatten()
        molatom_slabidxs = np.argwhere(tags==0).flatten()

        # Determine which atoms are bonded to the surface.
        ana = Analysis(ads_slab)
        adj = ana.adjacency_matrix[0]
        bonded_molatom_slabidxs = []
        for molatom_slabidx in molatom_slabidxs:
            for slabatom_slabidx in slabatom_slabidxs:
                bonded = bool(adj[(min(molatom_slabidx, slabatom_slabidx), max(molatom_slabidx, slabatom_slabidx))])
                if bonded:
                    bonded_molatom_slabidxs.append(molatom_slabidx)

        bonded_molatom_slabidxs, bonded_molatom_coordinations = np.unique(bonded_molatom_slabidxs, return_counts=True)
        print(f'  {len(bonded_molatom_slabidxs)} adsorbed atoms found on surface at idxs {bonded_molatom_slabidxs}.')

        # Isolate molecules.
        mol_atoms = ads_slab[molatom_slabidxs]
        mol_atoms.set_pbc([False, False, False])
        mol_atoms.set_cell([0.0, 0.0, 0.0])
        ana = Analysis(mol_atoms)
        nl = ana.nl[0]
        cm = nl.get_connectivity_matrix()
        n_mol, molidx_to_moleculeidx = sparse.csgraph.connected_components(cm) # e.g. 2, array[0, 0, 0, 1, 1]
        print(f'  {n_mol} molecule(s) found on/above surface.')
        molecules = [mol_atoms[np.argwhere(molidx_to_moleculeidx==i).flatten()] for i in range(n_mol)]

        # Predict surface sites.
        pos = ads_slab.get_positions()
        slab = ads_slab[slabatom_slabidxs]
        bonded_positions = pos[bonded_molatom_slabidxs]
        descs = self.desc.create(slab, bonded_positions, n_jobs=self.clf.n_jobs)
        pred_labels = self.clf.predict(descs)

        # Assign surface sites to molecules.
        slabidx_to_molidx = {int(idx): i for i, idx in enumerate(molatom_slabidxs)}
        pred_labels_per_molecule = [{} for _ in range(n_mol)]
        for i, bonded_molatom_slabidx in enumerate(bonded_molatom_slabidxs):
            bonded_molatom_molidx = slabidx_to_molidx[bonded_molatom_slabidx]
            moleculeidx = int(molidx_to_moleculeidx[bonded_molatom_molidx])
            elem = ads_slab.symbols[bonded_molatom_slabidx]

            unique, counts = np.unique(molidx_to_moleculeidx[:bonded_molatom_molidx+1], return_counts=True)
            atomidx_in_mol = int(dict(zip(unique, counts))[moleculeidx])-1

            pred_labels_per_molecule[moleculeidx][atomidx_in_mol] = {
                'site': str(pred_labels[i]), 
                'bonded_elem': elem,
                'coordination': int(bonded_molatom_coordinations[i])
            }

        return slab, molecules, pred_labels_per_molecule

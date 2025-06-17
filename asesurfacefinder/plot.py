from ase import Atoms
from ase.build import add_adsorbate
from ase.visualize.plot import plot_atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.lines import Line2D

from asesurfacefinder.sample_bounds import SampleBounds
from asesurfacefinder.utils import get_absolute_abspos, sample_ads_pos


class SamplePlotter:
    def __init__(self, surface: Atoms,
                 samples_per_site: int=500,
                 sample_bounds: dict={},
                 sample_defaults:SampleBounds=SampleBounds(0.1, 1.0, 2.75)):
        '''Samples surface sites and plots adsorbate positions.
        
        Arguments:
            surface: ASE surface to sample.
            samples_per_site: Number of adsorbate positions to sample on each surface site.
            sample_bounds: Optional dict binding site names to `SampleBounds` instances.
            sample_defaults: Default `SampleBounds` to fall back on when one is not specified for a site in `sample_bounds`.
        '''
        self.surface = surface
        self.sites = surface.info['adsorbate_info']['sites'].keys()

        self.atom_types = [chemical_symbols[n+1] for n in range(len(self.sites))]
        self.ref_colors = [jmol_colors[n+1] for n in range(len(self.sites))]
        self.ref_colors = [(rc[0], rc[1], rc[2], 1.0) for rc in self.ref_colors]
        self.plot_colors = [(0.9, 0.2, 0.2, 0.25), (0.2, 0.2, 0.9, 0.25), (0.2, 0.9, 0.2, 0.25), (0.9, 0.2, 0.9, 0.25)]

        self.radii = [covalent_radii[n] for n in surface.get_atomic_numbers()]
        ads_radius = 0.02

        for i, site in enumerate(self.sites):
            site_abspos = get_absolute_abspos(surface, site)
            bounds = sample_bounds[site] if site in sample_bounds.keys() else sample_defaults
            for _ in range(samples_per_site):
                xy, z = sample_ads_pos(site_abspos, bounds.z_bounds, bounds.r_max)
                add_adsorbate(self.surface, self.atom_types[i], z, xy)
                self.radii.append(ads_radius)

        return
    
    def plot(self, ax=None, **kwargs):
        '''Plot the sampled adsorbate positions on a surface.
        
        Wraps `ase.visualize.plot.plot_atoms` to replace the sizes and
        colours of adsorbate atoms with points unique to each site.
        Generates a figure legend for sites.

        Returns the passed/generated matplotlib axes.

        Arguments:
            ax: Optional matplotlib axes to plot on, creates one if not provided.
            **kwargs: Keyword arguments to pass to `plot_atoms`.
        '''
        ax = plot_atoms(self.surface, ax=ax, radii=self.radii, **kwargs)
        n_sites = len(self.sites)
        for patch in ax.patches:
            c = patch.properties().get('facecolor', None)
            for i, rc in enumerate(self.ref_colors):
                if c == rc:
                    patch.set_edgecolor((0.0, 0.0, 0.0, 0.0))
                    patch.set_facecolor(self.plot_colors[i])
                    continue

        legend_points = [Line2D([0], [0], marker='o', ls='', color=pc) for pc in self.plot_colors[:n_sites]]
        fig = ax.figure
        fig.legend(legend_points, self.sites, loc='lower center', ncols=n_sites)
        
        return ax
        
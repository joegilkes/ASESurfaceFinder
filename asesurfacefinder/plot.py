from ase import Atoms
from ase.build import add_adsorbate
from ase.io.utils import PlottingVariables
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch

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

        self.colors = [jmol_colors[i] for i in surface.get_atomic_numbers()]
        self.remove_edgecolor_mask = [False for _ in surface]
        self.radii = [covalent_radii[n] for n in surface.get_atomic_numbers()]
        ads_radius = 0.02

        for i, site in enumerate(self.sites):
            site_abspos = get_absolute_abspos(surface, site)
            bounds = sample_bounds[site] if site in sample_bounds.keys() else sample_defaults
            for _ in range(samples_per_site):
                xy, z = sample_ads_pos(site_abspos, bounds.z_bounds, bounds.r_max)
                add_adsorbate(self.surface, self.atom_types[i], z, xy)
                self.radii.append(ads_radius)
                self.colors.append(self.plot_colors[i])
                self.remove_edgecolor_mask.append(True)

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
        if ax is None:
            ax = plt.gca()
        Matplotlib(self.surface, ax, radii=self.radii, colors=self.colors, remove_edgecolor_mask=self.remove_edgecolor_mask, **kwargs).write()
        
        n_sites = len(self.sites)
        legend_points = [Line2D([0], [0], marker='o', ls='', color=pc) for pc in self.plot_colors[:n_sites]]
        fig = ax.figure
        fig.legend(legend_points, self.sites, loc='lower center', ncols=n_sites)
        
        return ax
        
# Reimplementation of ASE's matplotlib integration.
# Cuts out a few bits we don't need to speed up plotting with thousands of points,
# and allows for disabling edgecolors of sampled points during figure creation,
# rather than modifying thousands of patches afterwards.
class Matplotlib(PlottingVariables):
    def __init__(self, atoms, ax, rotation='', radii=None,
                 colors=None, remove_edgecolor_mask=None, 
                 scale=1, offset=(0, 0), **parameters):
        PlottingVariables.__init__(
            self, atoms, rotation=rotation,
            radii=radii, colors=colors, scale=scale,
            extra_offset=offset, **parameters)
        
        self.edgecolors = [(0.0, 0.0, 0.0, 0.0) if mask == 1 else 'black' for mask in remove_edgecolor_mask]
        self.ax = ax
        self.figure = ax.figure
        self.ax.set_aspect('equal')

    def write(self):
        self.write_body()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            self.ax.add_patch(patch)

def make_patch_list(writer):
    indices = writer.positions[:, 2].argsort()
    patch_list = []
    for a in indices:
        xy = writer.positions[a, :2]
        if a < writer.natoms:
            r = writer.d[a] / 2
            if ((xy[1] + r > 0) and (xy[1] - r < writer.h) and
                (xy[0] + r > 0) and (xy[0] - r < writer.w)):
                patch = Circle(xy, r, facecolor=writer.colors[a],
                                edgecolor=writer.edgecolors[a])
                patch_list.append(patch)
        else:
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                patch = PathPatch(Path((xy + hxy, xy - hxy)))
                patch_list.append(patch)
    return patch_list
class NoSurfaceError(Exception):
    '''Exception when no valid surfaces found.'''

class SurfaceTagError(Exception):
    '''Exception when provided slab/adsorbate system cannot be separated by tags.'''

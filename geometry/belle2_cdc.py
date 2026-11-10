import numpy as np

'''
Attributes
----------
A, B, C, D, E, F, G, H : numpy.ndarray
    Coordinates of the detector boundary points in the (z, y) plane.
    These points define the longitudinal profile of the drift chamber.
    The geometry follows the A-H convention illustrated in Fig. 6.2 of Belle II Technical Design Report (arXiv:1011.0352).
    For detector geometries that do not require G and H, dummy values may be provided to preserve a common geometry interface.

rhosize_min, rhosize_max : numpy.ndarray
    Cell size of the innermost and outermost layers of each superlayer (SL).
    If rhosize_min != rhosize_max, the cell size is assumed to vary linearly across the layers within the superlayer. Otherwise, a uniform cell size is assumed.

rho_min, rho_max : numpy.ndarray
    Minimum and maximum radial positions of each superlayer in the (x, y) plane. rho_min (rho_max) corresponds to the inner (outer) boundary of the superlayer. Index 0 corresponds to the innermost superlayer.

nLayers : numpy.ndarray
    Number of layers in each superlayer.

nCells : numpy.ndarray
    Number of drift cells in each layer. It is assumed that all layers within the same superlayer contain the same number of cells.

rhosize_threshold : float
    Cell-size threshold used to classify detector layers into inner/small (S) and outer/large (L) cells.

Bz : float
    Axial magnetic field strength (Tesla) used in helix propagation.
'''

BELLE2= {
    'A': np.array([-83.1,  108.2+15.0]),
    'B': np.array([158.6,  108.2+15.0]),
    'C': np.array([-68.2,   37.14]),
    'D': np.array([144.9,   43.80]),
    'E': np.array([-47.4,   24.95]),
    'F': np.array([87.7,   24.95]),
    'G': np.array([-999,    16.0]),
    'H': np.array([-999,   16.0]),

    # Layer size model
    'rhosize_min': np.array([0.88] + [1.51]*8),
    'rhosize_max': np.array([0.88] + [1.51]*8),

    # Superlayer geometry
    'rho_min': np.array([16.80, 25.70, 36.52, 47.69, 58.41, 69.53, 80.25,  91.37, 102.09]),
    'rho_max': np.array([23.80, 34.80, 45.57, 56.69, 67.41, 78.53, 89.25, 100.37, 111.14]),

    # Detector layout
    'nLayers': np.array([8,    6,     6,     6,     6,     6,     6,      6,      6]),
    'nCells': np.array([160,  160,   192,   224,   256,   288,   320,    352,    384]),

    # Hit acceptance
    'rhosize_threshold': 1.0,

    # Magnetic field
    'Bz': 1.5,
}

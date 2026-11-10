import numpy as np

'''
Attributes
----------
A, B, C, D, E, F, G, H : numpy.ndarray
    Coordinates of the detector boundary points in the (z, y) plane.
    These points define the longitudinal profile of the drift chamber.
    The geometry follows the convention based on information in section-4 of Design and Construction of the BESIII Detector report (arXiv:0911.4960):
        A : upper-left
        B : upper-right
        C : middle-left
        D : middle-right
        E : lower-left (boundary of inner layers)
        F : lower-right (boundary of inner layers)
        G : lower extension on the left
        H : lower extension on the right
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



BES3= {
    'A': np.array([-123.2,  81+7.0]),
    'B': np.array([123.2,  81+7.0]),
    'C': np.array([-107.0,   40.0]),
    'D': np.array([107.0,   40.0]),
    'E': np.array([-54.6,   18.35]),
    'F': np.array([54.6,   18.35]),
    'G': np.array([-999,    5.9]),
    'H': np.array([-999,    5.9]),

    # Layer size model
    'rhosize_min': np.array([1.2]*2 + [1.6]*9),
    'rhosize_max': np.array([1.2]*2 + [1.6]*9),

    # Superlayer geometry
    'rho_min': np.array([6.7, 12, 17.7, 25, 32, 39.4, 46.7,  54, 61.3, 68.5, 76]),
    'rho_max': np.array([11.5, 16.8, 24, 31.4, 38.6, 45.8, 53.2, 60.5, 67.8, 75, 81]),

    # Detector layout
    'nLayers': np.array([4,    4,     4,     4,     4,     4,     4,      4,      4,     4,      3]),
    'nCells': np.array([38,  77,   77,   89,   114,   139,   168,    196,    224,     263,    291]),

    # Hit acceptance
    'rhosize_threshold': 1.2,

    # Magnetic field
    'Bz': 1.0,
}

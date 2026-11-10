import numpy as np

'''
Attributes
----------
A, B, C, D, E, F, G, H : numpy.ndarray
    Coordinates of the detector boundary points in the (z, y) plane.
    These points define the longitudinal profile of the drift chamber.
    The geometry follows the convention based on information in section-3.5 of Conceptual Design Report of the STCF Detector (arXiv:2303.15790):
        A : upper-left
        B : upper-right
        C : middle-left
        D : middle-right
        E : lower-left (boundary of first superlayer)
        F : lower-right (boundary of first layer)
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


STCF= {
    'A': np.array([-127.4,  84+10.0]),
    'B': np.array([127.4,  84+10.0]),
    'C': np.array([-118.7,   40.8]),
    'D': np.array([118.7,   40.8]),
    'E': np.array([-79.89,    27.16]),
    'F': np.array([79.89,    27.16]),
    'G': np.array([-999, 19.6]),
    'H': np.array([-999, 19.6]),

    # Layer size model
    'rhosize_min': np.array([0.98, 1.07, 1.12, 1.17, 1.23, 1.26, 1.3, 1.33]),
    'rhosize_max': np.array([1.25, 1.29, 1.32, 1.35, 1.38, 1.4, 1.43, 1.45]),

    # Superlayer geometry
    'rho_min': np.array([20.0, 27.1, 34.2, 41.9, 49.9, 57.8, 66.2,  74.4]),
    'rho_max': np.array([26.6, 34.1, 41.5, 49.4, 57.7, 65.7, 74.3, 82.7]),

    # Detector layout
    'nLayers': np.array([6,  6,  6,  6,  6,  6,  6,  6]),
    'nCells': np.array([128,  160,   192,   224,   256,   288,   320,    352]),

    # Hit acceptance
    'rhosize_threshold': 1.20,

    # Magnetic field
    'Bz': 1.0,
}

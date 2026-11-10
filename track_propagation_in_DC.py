import sys
import numpy as np
import pandas as pd
from scipy import constants

from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection

'''
'''

# convinience functions

def to_deg(rad):
    return rad * 180/np.pi

def compute_point_on_line(P1, P2, y):
    ## y = az + b
    if P1[0]==P2[0]:
        z = P1[0]
    else:
        a = ( P1[1] - P2[1] )/( P1[0] - P2[0])
        if a==0:
            print(f'a = 0: P1 = {P1}     P2 = {P2}, y = {y}')
        b =  P1[1] - a * P1[0]
        z = (y - b)/a
    return np.array([z, y])

def compute_point_on_line_signed(P1, P2, y):
    if y>=0:
        return compute_point_on_line(P1, P2, y)
    else:
        return compute_point_on_line([P1[0], -P1[1]], [P2[0], -P2[1]], y)

def angle_0_to_2pi(angle):
    while True:
        if angle>=0 and angle<2*np.pi:
            break
        elif angle < 0:
            angle += 2*np.pi
        elif angle >= 2*np.pi:
            angle -= 2*np.pi
    return angle


class DC:
    '''Simplified model of the Drift Chamber of the given detector
        all lengths are in cm

    INPUT: dic
        the given model is loaded from the file with decay info (vertices and momenta). Check sample_example.ipynb
        the user can also specify their own dc model by modifying the attibutes

    Attributes
    ----------
    A, B, C, D, E, F, G, H : numpy.ndarray
        Coordinates of the detector boundary points in the (z, y) plane.
        These points define the longitudinal profile of the drift chamber.
        The geometry follows the A-H convention as per the information given in the design reports of the respective detectors.
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
        Cell-size threshold used to classify detector layers into Small (S) and Large (L) cells.

    Bz : float
        Axial magnetic field strength (Tesla) used in helix propagation.

    SL : pandas.DataFrame
        DataFrame containing superlayer information, including radial boundaries, number of layers, and other derived quantities.

    layers : pandas.DataFrame
        DataFrame containing layer-level information, including layer ID, superlayer ID, number of cells, radial position, cell size, and Small/Large (S/L) classification.

    Public Methods
    --------------
    insideDC(x, y, z, verbose=False)
        bool
        determines if a point with coordinates (x, y, z) is within the dc volume
    get_cellID(phi, cell_delta_phi)
        int
        returns the index of a given cell (index 0 corresponds to the cell that has the smallest phi angle)
        input: phi position and cell size (in phi) of the given layer

    zy_contour(fig, ax)
        plot the dc boundaries in the (z-y) plane
    zy_layers(fig, ax)
        plot the dc layers in the (z-y) plane
    xy_contour(fig, ax)
        plot the dc boundaries in the (x-y) plane
    xy_cells(fig, ax)
        plot the dc cells in the (x-y) plane
        REMARK: if there are a lot of cells this method will significanlty slow down the plotting function
    '''
    def __init__(self, dictionary):

        self.unit = '[cm]'
        self.A, self.B = dictionary['A'], dictionary['B']
        self.C, self.D = dictionary['C'], dictionary['D']
        self.E, self.F = dictionary['E'], dictionary['F']
        self.G, self.H = dictionary['G'], dictionary['H']

        self.rhosize_min = dictionary['rhosize_min']
        self.rhosize_max = dictionary['rhosize_max']

        self.rho_min = dictionary['rho_min']
        self.rho_max = dictionary['rho_max']

        self.nLayers = dictionary['nLayers']

        self.nCells  = dictionary['nCells']

        for attr in self.__dict__:
            if attr in dictionary.keys():
                self.__setattr__(attr, dictionary[attr])

        self.G = compute_point_on_line(P1=self.C,  P2=self.E, y=self.G[1])
        self.H = compute_point_on_line(P1=self.D,  P2=self.F, y=self.H[1])

        DC.Bz = dictionary['Bz']
        self.rhosize_threshold = dictionary['rhosize_threshold']

        self.SL = self._superLayers()
        self.layers = self._layer_organisation()

    def _layer_organisation(self):
        Ntot_layers = self.nLayers.sum()
        dc_layers = pd.DataFrame()
        dc_layers['layer_id'] = [i for i in range( 0, Ntot_layers )]
        dc_layers.index = [f'lay{i}'  for i in range(0, dc_layers.shape[0])]
        dc_layers['SL'] = 'SL0'
        layer_min = int(self.SL.loc['SL0'].nLayers)
        for i in range(1, self.SL.shape[0]):
            dc_layers['SL'] = np.where(dc_layers.layer_id.isin(np.arange(layer_min, layer_min+6, 1)), f'SL{i}', dc_layers.SL)
            layer_min += int(self.SL.loc[f'SL{i}'].nLayers)
        dc_layers['nCells']   = 0
        dc_layers['rho_size'] = 0.0
        dc_layers['rho_min']  = 0.0
        dc_layers['rho_max']  = 0.0
        dc_layers['rho_size_class'] = ''


        for i in range(0, self.SL.shape[0]):
            sl = f'SL{i}'

            nL = self.SL.loc[sl].nLayers
            rho_min0 = self.SL.loc[sl].rho_min  # fixed inner radius
            rho_max_sl = self.SL.loc[sl].rho_max  # fixed outer radius

            # step increase in size
            if nL == 1:
                del_rho = 0.0
            else:
                del_rho = (self.SL.loc[sl].rhosize_max - self.SL.loc[sl].rhosize_min)/(nL-1)

            # assign rho_min and rho_size layer by layer
            rho_min_vals = []
            rho_size_vals = []
            rho_max_vals = []

            rho_min = rho_min0
            for j in range(nL):
                rho_size = self.SL.loc[sl].rhosize_min + j * del_rho
                rho_max = rho_min + rho_size

                rho_min_vals.append(rho_min)
                rho_size_vals.append(rho_size)
                rho_max_vals.append(rho_max)

                # next layer starts at this layer’s max
                rho_min = rho_max

            # assign back to DataFrame
            idx = dc_layers[dc_layers.SL == sl].index
            dc_layers.loc[idx, 'rho_min'] = rho_min_vals
            dc_layers.loc[idx, 'rho_size'] = rho_size_vals
            dc_layers.loc[idx, 'rho_max'] = rho_max_vals

            # angular cell size
            dc_layers.loc[idx, 'cell_delta_phi'] = 2*np.pi / self.SL.loc[sl].nCells
            dc_layers.loc[idx, 'nCells'] = self.SL.loc[sl].nCells

        # classifying the cells as small and large with a threshold size
        dc_layers['rho_size_class'] = np.where(dc_layers['rho_size'] <= self.rhosize_threshold, 'S', 'L')
        dc_layers = self._add_zShape_info(dc_layers)
        return dc_layers

    def _layer_length_z(P1, P2):
        return abs(P2[0] - P1[0])

    def _superLayers(self):
        dc_SL = pd.DataFrame()
        dc_SL['rho_min'] = self.rho_min
        dc_SL['rho_max'] = self.rho_max
        dc_SL['nLayers'] = self.nLayers
        dc_SL['nCells']  = self.nCells
        dc_SL.index = [f'SL{i}'  for i in range(0, dc_SL.shape[0])]
        dc_SL['maxLayerID'] = dc_SL.nLayers.cumsum() - 1
        dc_SL['minLayerID'] = dc_SL.maxLayerID - dc_SL.nLayers + 1
        dc_SL['rhosize_min'] = self.rhosize_min
        dc_SL['rhosize_max'] = self.rhosize_max
        dc_SL['del_rho_Lsize'] = (dc_SL.rhosize_max - dc_SL.rhosize_min)/(dc_SL.nLayers - 1)
        dc_SL['rho_size_calc'] = ((dc_SL.rho_max - dc_SL.rho_min) / dc_SL.nLayers) * dc_SL.del_rho_Lsize
        dc_SL = self._add_zShape_info(dc_SL)
        return dc_SL

    def _add_zShape_info(self, df):
        ## p1, p2 are the points defining the lines in the TDR (belle II) (used to get z_bwd and z_fwd)
        # z_bwd(fwd) is the x position in the bwd(fwd) direction of the (super)Layer
        df['p1_bwd'] = 0
        df['p1_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.E, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.C, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.A, axis=1),  df.p1_bwd)
        df['p2_bwd'] = 0
        df['p2_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.G, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.E, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.C, axis=1),  df.p2_bwd)
        df['z_bwd'] = 0
        df['z_bwd'] = df.apply(lambda row: compute_point_on_line(P1 = row.p1_bwd, P2 = row.p2_bwd, y = row.rho_max)[0], axis=1)

        df['p1_fwd'] = 0
        df['p1_fwd'] = np.where(df.rho_max.between(self.H[1], self.F[1]), df.apply(lambda r: self.F, axis=1),  df.p1_fwd)
        df['p1_fwd'] = np.where(df.rho_max.between(self.F[1], self.D[1]), df.apply(lambda r: self.D, axis=1),  df.p1_fwd)
        df['p1_fwd'] = np.where(df.rho_max.between(self.D[1], self.B[1]), df.apply(lambda r: self.B, axis=1),  df.p1_fwd)
        df['p2_fwd'] = 0
        df['p2_fwd'] = np.where(df.rho_max.between(self.H[1], self.F[1]), df.apply(lambda r: self.H, axis=1),  df.p2_fwd)
        df['p2_fwd'] = np.where(df.rho_max.between(self.F[1], self.D[1]), df.apply(lambda r: self.F, axis=1),  df.p2_fwd)
        df['p2_fwd'] = np.where(df.rho_max.between(self.D[1], self.B[1]), df.apply(lambda r: self.D, axis=1),  df.p2_fwd)
        df['z_fwd'] = 0
        df['z_fwd'] = df.apply(lambda row: compute_point_on_line(P1 = row.p1_fwd, P2 = row.p2_fwd, y = row.rho_max)[0], axis=1)
        return df


    def __repr__(self):
        # what is displayed when writing the name of the cass in the command line
        return f'<dc simplified model>'

    def __str__(self):
        # what is displayed when using the print() function
        return f'dc simplified model\n  {dc.__dict__}\n'

    def insideDC(self, x, y, z, verbose=False):
        inDC = True
        rho = np.sqrt(x**2 + y**2)
        id_last_layer = self.layers.shape[0]-1
        zL, zR = 0, 0
        ## check in z
        if rho<=self.rho_min[0]:
            # actually, the particle is not in DC but between the beam pipe and DC... å
            pass
        else:
            # compute the limits in z at a given y
            zL = compute_point_on_line_signed(self.C, self.E, rho)[0]  if abs(rho)<=self.C[1] else compute_point_on_line_signed(self.A, self.C, rho)[0]
            zR = compute_point_on_line_signed(self.F, self.D, rho)[0]  if abs(rho)<=self.D[1] else compute_point_on_line_signed(self.D, self.B, rho)[0]
            if z<zL or z>zR:
                inDC=False
                if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), zL = {zL:.1f}, zR = {zR:.1f}, rho = {rho:.1f}, rhoM = {self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:.1f}' )
                if verbose: print(f'  ==> particle exited in the z direction' )

        ## check in rho direction
        if  rho>self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:
            inDC=False
            if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), zL = {zL:.1f}, zR = {zR:.1f}, rho = {rho:.1f}, rhoM = {self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:.1f}' )
            if verbose: print(f' ==> particle exited in the  rho direction')

        return inDC

    def get_cellID(self, phi, cell_delta_phi):
        if cell_delta_phi > 0:
            cell_id = int(np.floor(angle_0_to_2pi(phi)/cell_delta_phi))
        else:
            cell_id = -1
        return cell_id

    # plots
    def zy_contour(self, fig, ax):
         ## upper-half (y>0)
        ax.plot( [self.A[0],  self.B[0]],   [self.A[1],  self.B[1]]  )  ## AB
        ax.plot( [self.D[0],  self.B[0]],   [self.D[1],  self.B[1]]  )  ## DB
        ax.plot( [self.A[0],  self.C[0]],   [self.A[1],  self.C[1]]  )  ## AC
        ax.plot( [self.C[0],  self.E[0]],   [self.C[1],  self.E[1]]  )  ## CE
        ax.plot( [self.F[0],  self.D[0]],   [self.F[1],  self.D[1]]  )  ## FD
        ax.plot( [self.E[0],  self.G[0]],   [self.E[1],  self.G[1]]  )  ## EG
        ax.plot( [self.F[0],  self.H[0]],   [self.F[1],  self.H[1]]  )  ## FH
        ax.plot( [self.G[0],  self.H[0]],   [self.G[1],  self.H[1]]  )  ## GH

        ## lower-half (y<0)
        ax.plot( [self.A[0],  self.B[0]],   [-self.A[1],  -self.B[1]]  )  ## AB
        ax.plot( [self.D[0],  self.B[0]],   [-self.D[1],  -self.B[1]]  )  ## DB
        ax.plot( [self.A[0],  self.C[0]],   [-self.A[1],  -self.C[1]]  )  ## AC
        ax.plot( [self.C[0],  self.E[0]],   [-self.C[1],  -self.E[1]]  )  ## CE
        ax.plot( [self.F[0],  self.D[0]],   [-self.F[1],  -self.D[1]]  )  ## FD
        ax.plot( [self.E[0],  self.G[0]],   [-self.E[1],  -self.G[1]]  )  ## EG
        ax.plot( [self.F[0],  self.H[0]],   [-self.F[1],  -self.H[1]]  )  ## FH
        ax.plot( [self.G[0],  self.H[0]],   [-self.G[1],  -self.H[1]]  )  ## GH

        ax.set_xlabel(f'z {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)


    def zy_layers(self, fig, ax):
        # Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
        #       +------------------+
        # :     |                  |
        # :   height               |
        # :     |                  |
        # :    (xy)---- width -----+
        patches  = [ Rectangle((self.SL.loc[sl].z_bwd,  self.SL.loc[sl].rho_min),
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd,
                            height=(self.SL.loc[sl].rho_max - self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        patches += [ Rectangle((self.SL.loc[sl].z_bwd,  - self.SL.loc[sl].rho_max),
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd,
                            height=(self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        p = PatchCollection(patches, alpha=0.0)
        ax.add_collection(p)

        patches  = [ Rectangle((self.layers.loc[lay].z_bwd,  self.layers.loc[lay].rho_min),
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd,
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        patches += [ Rectangle((self.layers.loc[lay].z_bwd,  -self.layers.loc[lay].rho_max),
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd,
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        p2 = PatchCollection(patches, alpha=0.2, linewidth=1, edgecolor='b')
        ax.add_collection(p2)


        ax.set_xlabel(f'z {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)

        ax.set_xlim(self.A[0], self.B[0])
        ax.set_ylim(-self.A[1], self.A[1])

    def xy_contour(self, fig, ax):
        # a wedge centered at x, y center with radius r that sweeps theta1 to theta2 (in degrees)
        # if width is given, then a partial wedge is drawn from inner radius r - width to outer radius r
        patches = [ Wedge((0, 0),
                      self.SL.loc[sl].rho_max,
                      0, 360,
                      width=( self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min  ))  for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])] ]
        ax.add_collection( PatchCollection(patches,  alpha=0.1, linewidth=1, edgecolor='b') )


        patches = [ Wedge((0, 0),
                        self.layers.loc[lay].rho_max,
                        0, 360,
                        width=( self.layers.loc[lay].rho_max-self.layers.loc[lay].rho_min  ))  for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])] ]
        ax.add_collection( PatchCollection(patches, alpha=0.2, linewidth=1, edgecolor='b', ls='-') )

        ax.set_xlim(-self.A[1], self.A[1])
        ax.set_ylim(-self.A[1], self.A[1])
        ax.set_xlabel(f'x {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        ax.set_aspect( 1 )

    def xy_cells(self, fig, ax):
        patches = []
        for l in range(0, self.layers.shape[0]):
            lay = f'lay{l}'
            patches += [ Wedge((0, 0),
                            self.layers.loc[lay].rho_max,
                            i*to_deg(self.layers.loc[lay].cell_delta_phi), i*to_deg(self.layers.loc[lay].cell_delta_phi)+to_deg(self.layers.loc[lay].cell_delta_phi),
                            width=( self.layers.loc[lay].rho_max-self.layers.loc[lay].rho_min  )) for i in range(0, self.layers.loc[lay].nCells) ]

        ax.add_collection( PatchCollection(patches, linewidth=2, edgecolor='b', ls=':') )
        ax.set_xlim(-self.A[1], self.A[1])
        ax.set_ylim(-self.A[1], self.A[1])
        ax.set_xlabel(f'x {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        ax.set_aspect( 1 )


class Particle:
    '''
    Given the charge, 3-momentum and production vertex of the particle, this class
    compute the information related to the point of closest approach (POCA)

    Attributes
    ----------
    q: int
        charge
    px, py, pz: float
        3-momentum of the particle
    prodVtxX, prodVtxY, prodVtxZ: float
        coordinates of the production vertex of the particle

    x0, y0, z0: float
        coordinates of the POCA
    d0, phi0, omega, tanLambda: float
        helix parametrs
    s0: float
        arclength in the (x-y) plan at the POCA

    '''
    def __init__(self, dictionary):
        self.q = None
        self.px, self.py, self.pz = None, None, None
        self.prodVtxX, self.prodVtxY, self.prodVtxZ =  None, None, None

        self.x0, self.y0, self.z0 = None, None, None
        self.d0, self.phi0, self.omega, self.tanLambda, self.s0 = None, None, None, None, None

        for attr in self.__dict__:
            if attr in dictionary.keys():
                self.__setattr__(attr, dictionary[attr])

        if self.x0 is None:
            self.x0, self.y0, self.z0, self.d0, self.phi0, self.omega, self.tanLambda, self.s0 = self._get_helix_paramters_at_POCA()


    def _get_helix_paramters_at_POCA(self):

        phi = np.arctan2(self.py, self.px)
        pt = np.sqrt(self.px**2 + self.py**2)

        # track curvature
        Bz = DC.Bz
        a = Bz/(constants.c*1e-6) # constants.c is in m.s-1
        omega     = a * self.q / pt   # [cm-1]   constants.c*1e-9 = 0.3 (which corresponds to cyclotron radius in m, we want cm)
        tanLambda = self.pz/pt

        Dpar  = -self.prodVtxX * np.cos(phi) - self.prodVtxY * np.sin(phi)
        Dperp = -self.prodVtxY * np.cos(phi) + self.prodVtxX * np.sin(phi)
        A = 2 * Dperp + omega * ( Dperp**2 + Dpar**2 )
        U = np.sqrt( 1 + omega * A )

        dphi = np.arctan2( omega * Dpar, 1 + omega * Dperp )
        s = dphi/omega

        d0 = A / ( 1 + U )
        phi0 = phi - dphi # np.arctan2(py, px) - np.arctan2( omega * Dpar, 1 + omega * Dperp )
        z0 = self.prodVtxZ + s * tanLambda

        x0 = d0 * np.sin(phi0)
        y0 = -d0 * np.cos(phi0)

        return x0, y0, z0, d0, phi0, omega, tanLambda, s

class Point(Particle):
    '''
    Get any point on the trajectory of a given particle and also computes its position in the dc


    Attributes
    ----------
    instanceDC: dc
        dc model
    inDC: bool
        is the point in the dc?
    layer_id: int
        dc layer corresponding to the point
        the special value -1 means that no layer is associated to this point
    cell_id: int
        cell corresponding to the point
        the special value -1 means that no cell is associated to this point

    x, y, z, rho, phi, s: float
        coordinate, rho (vector from the origin the x-y plane), phi angle and arc-length corresponding to the point

    '''
    def __init__(self, dictionary, instanceDC, s):
        super().__init__(dictionary)
        self.instanceDC = instanceDC
        self.layer_id = -1
        self.cell_id = -1
        self.rho_size = -1
        self.rho_size_class = -1


        self.x, self.y, self.z, self.phi, self.s = self._get_particle_position(s)
        self.inDC = instanceDC.insideDC(self.x, self.y, self.z, verbose=False)
        self.rho =  np.sqrt(self.x**2 + self.y**2)

        self.azimutal_angle = np.arctan2(self.y, self.x)

        if self.inDC:
            vals = instanceDC.layers.loc[(instanceDC.layers['rho_min'] <= self.rho) & (instanceDC.layers['rho_max'] > self.rho)].layer_id.values
            if len(vals)==1:
                self.layer_id = instanceDC.layers.loc[
                    (instanceDC.layers['rho_min'] <= self.rho) &
                    (instanceDC.layers['rho_max'] > self.rho)
                ].layer_id.values[0]

                layer = instanceDC.layers.loc[f'lay{self.layer_id}']

                self.cell_id = instanceDC.get_cellID(
                    self.azimutal_angle,
                    layer.cell_delta_phi
                )
                self.rho_size = layer.rho_size
                self.rho_size_class = layer.rho_size_class

            elif len(vals)>1:
                sys.exit('More than one dc layer correspond to that point. That should not happend. Aborting.')

    def _get_particle_position(self, s):
        # returns particle position at a given arc length

        phi = self.phi0 - s*self.omega # not used in the computation

        z_prime = self.z0 + s * self.tanLambda

        Cs = self.omega * s
        x_prime = self.x0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.cos(self.phi0 - 0.5 * Cs) if Cs!=0 else self.x0
        y_prime = self.y0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.sin(self.phi0 - 0.5 * Cs) if Cs!=0 else self.y0

        return x_prime, y_prime, z_prime, phi, s

    def __repr__(self):
        return f'<Point at phi = {self.phi:.3f}>'


class Trajectory(Particle):
    '''
        collection of points representing the trajectory of the particle
        computes the number of hits in the dc
        the distance between the points (steps) is given as a function of the arc length (s)

    Attributes
    ----------
        signed_step: float
            for steps in arc length, this is equal to step_size
        steps: numpy array
            array containing the steps
        DChits: int
            number of dc hits
        meanStepsPerCell_all, stdStepsPerCell_all, minStepsPerCell_all, maxStepsPerCell_all: float
            mean, standard deviation, min and max number on steps per cell
            can be used to tune the cut on the number of steps
        meanStepsPerCell_small_cell, stdStepsPerCell_small_cell, minStepsPerCell_small_cell, maxStepsPerCell_small_cell: float
            mean, standard deviation, min and max number on steps per cell for the smaller layers (Small 'S')
            can be used to tune the cut on the number of steps
        meanStepsPerCell_large_cell, stdStepsPerCell_large_cell, minStepsPerCell_large_cell, maxStepsPerCell_large_cell: float
            mean, standard deviation, min and max number on steps per cell for the larger layers in superlayers (Large 'L')
            can be used to tune the cut on the number of steps
        min_setps_in_small_cell: int
            minimum number of steps in a small cell to be counted as a hit
        min_setps_in_large_cell: int
            minimum number of steps in a largeger cell to be counted as a hit
        list_of_points: list
            list of Point() objects forming the trajectory
        points: pandas dataframe
            for each point, provide the coordinates, the angle phi, the arc length s and the corresponding dc layer and cell
        hit_layers: list
            sorted list of detector layer IDs satisfying the hit acceptance criteria.
        not_hit_layers: list
            sorted list of traversed detector layer IDs failing the hit acceptance criteria.
        SL_hit_info: dict
            dictionary containing the number of hit layers and the total number of layers in each superlayer.
            Each entry has the form SL_hit_info[SL] = (n_hit_layers, n_total_layers).

    Public Methods
    --------------
        xy_trajectory(self, ax, label='', color='b')
            plot the trajectory in the (x-y) plane
        zy_trajectory(self, ax, label='', color='b')
            plot the trajectory in the (z-y) plane
        xy_vertex(self, ax, label='', color='b')
            plot the vertex position in the (x-y) plane
        zy_vertex(self, ax, label='', color='b')
            plot the vertex position in the (z-y) plane
        xy_poca(self, ax, label='', color='b')
            plot the POCA position in the (x-y) plane
        zy_poca(self, ax, label='', color='b')
            plot the POCA position in the (x-y) plane
        SL_hits()
            Returns the number of hit layers in each superlayer as a list ordered by increasing superlayer index.

    '''

    def __init__(self, dictionary, instanceDC, step_size, min_steps_in_small_cell=5, min_steps_in_large_cell=10):
        super().__init__(dictionary)

        self.instanceDC = instanceDC

        self.signed_step = step_size
        self.steps = np.arange(self.s0, self.s0 + 1000, self.signed_step)  # 1000 is a dummy value

        self.points  = None
        self.DChits = None

        self.min_steps_in_small_cell = min_steps_in_small_cell
        self.min_steps_in_large_cell   = min_steps_in_large_cell

        self.meanStepsPerCell, self.stdStepsPerCell, self.minStepsPerCell, self.maxStepsPerCell  = None, None, None, None
        self.meanStepsPerCell_small_cell, self.stdStepsPerCell_small_cell, self.minStepsPerCell_small_cell, self.maxStepsPerCell_small_cell  = None, None, None, None
        self.meanStepsPerCell_large_cell, self.stdStepsPerCell_large_cell, self.minStepsPerCell_large_cell, self.maxStepsPerCell_large_cell  = None, None, None, None

        self.list_of_points = []
        rho_Vtx = np.sqrt(self.prodVtxX**2 + self.prodVtxY**2)


        for step in self.steps:
            p = Point(dictionary, instanceDC, step)
            if p.inDC:
                if rho_Vtx==0:
                    # save all the points from the POCA
                    self.list_of_points.append(p)
                else:
                    # save points from the production vertex of the particle
                    if p.rho>=rho_Vtx and rho_Vtx>0:
                        if self.pz>=0 and p.z>=self.prodVtxZ:
                            self.list_of_points.append(p)
                        if self.pz<0 and p.z<=self.prodVtxZ:
                            self.list_of_points.append(p)
            else:
                break



        dct = {'phi': [], 's': [], 'x': [], 'y': [], 'z': [], 'layer_id': [], 'cell_id': [], 'rho_size':[], 'rho_size_class': []}

        for i in range(len(self.list_of_points)):
            p = self.list_of_points[i]
            dct['phi'].append(p.phi)
            dct['s'].append(p.s)
            dct['x'].append(p.x)
            dct['y'].append(p.y)
            dct['z'].append(p.z)
            dct['layer_id'].append(p.layer_id)
            dct['cell_id'].append(p.cell_id)
            dct['rho_size'].append(p.rho_size)
            dct['rho_size_class'].append(p.rho_size_class)

        self.points = pd.DataFrame(dct)
        # get number of steps in one cell
        # https://stackoverflow.com/questions/59946601/groupby-consecutive-occurrences-of-two-column-values-in-pandas
        self.points['cumSum'] = (self.points[['layer_id', 'cell_id']].ne(self.points[['layer_id', 'cell_id']].shift()).any(axis=1).cumsum())
        self.points['nStepsIncell'] = (self.points.groupby('cumSum')['layer_id'].transform('count'))
        self._count_dc_hits()


    def _count_dc_hits(self):
        mask = (self.points.cell_id > -1)
        df_valid = self.points[mask]
        # as is, drop_duplicates will also drop cells if the track 'came back to the same cell after a time'
        # could use nStepsIncell to help with that or write something that drops only consecutive duplicates
        cells = df_valid.drop_duplicates(['layer_id', 'cell_id'], keep='first')
        self.DChits = cells.loc[(((cells.rho_size_class == 'S') & (cells.nStepsIncell >= self.min_steps_in_small_cell)) | ((cells.rho_size_class == 'L') & (cells.nStepsIncell >= self.min_steps_in_large_cell)))].shape[0]

        self.meanStepsPerCell = df_valid['nStepsIncell'].mean()
        self.stdStepsPerCell  = df_valid['nStepsIncell'].std()
        self.minStepsPerCell  = df_valid['nStepsIncell'].min()
        self.maxStepsPerCell  = df_valid['nStepsIncell'].max()

        df_S = df_valid[df_valid.rho_size_class == 'S']
        df_L = df_valid[df_valid.rho_size_class == 'L']

        self.meanStepsPerCell_small_cell = df_S['nStepsIncell'].mean()
        self.stdStepsPerCell_small_cell  = df_S['nStepsIncell'].std()
        self.minStepsPerCell_small_cell  = df_S['nStepsIncell'].min()
        self.maxStepsPerCell_small_cell  = df_S['nStepsIncell'].max()

        self.meanStepsPerCell_large_cell = df_L['nStepsIncell'].mean()
        self.stdStepsPerCell_large_cell  = df_L['nStepsIncell'].std()
        self.minStepsPerCell_large_cell  = df_L['nStepsIncell'].min()
        self.maxStepsPerCell_large_cell  = df_L['nStepsIncell'].max()

        # Layer-level hit logic
        hit_layers = []
        for layer, group in df_valid.groupby('layer_id'):
            max_steps = group['nStepsIncell'].max()
            layer_class = group['rho_size_class'].iloc[0]
            if (
                (layer_class == 'S' and max_steps >= self.min_steps_in_small_cell) or
                (layer_class == 'L' and max_steps >= self.min_steps_in_large_cell)
            ):
                hit_layers.append(layer)
        self.hit_layers = sorted(hit_layers)
        all_layers = sorted(df_valid['layer_id'].unique())
        self.not_hit_layers = sorted(set(all_layers) - set(self.hit_layers))

        # SuperLayer (SL) hit summary
        SL_hit_info = {}
        for SL in self.instanceDC.SL.index:
            SL_layers = self.instanceDC.layers[
                self.instanceDC.layers['SL'] == SL
            ]['layer_id'].values
            total_layers = len(SL_layers)
            hit_count = len(set(SL_layers) & set(self.hit_layers))
            SL_hit_info[SL] = (hit_count, total_layers)
        self.SL_hit_info = SL_hit_info

    def SL_hits(self):
        SLhits = []
        for SL in sorted(self.SL_hit_info.keys()):
            hit, total = self.SL_hit_info[SL]
            SLhits.append(hit)
        return SLhits



    # plots
    def xy_trajectory(self, ax, label='', color='b'):
        ax.scatter(self.points['x'], self.points['y'], marker ='o', s=1, color=color, label=label)

    def zy_trajectory(self, ax, label='', color='b'):
        ax.scatter(self.points['z'], self.points['y'], marker ='o', s=1, color=color, label=label)

    def xy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxX ], [self.prodVtxY], marker='x', markersize=15, color=color, label=label)

    def zy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxZ ], [self.prodVtxY], marker='x', markersize=15, color=color, label=label)

    def xy_poca(self, ax, label='', color='b'):
        ax.plot( [self.x0 ], [self.y0], marker='*', markersize=15, color=color, label=label)

    def zy_poca(self, ax, label='', color='b'):
        ax.plot( [self.z0 ], [self.y0], marker='*', markersize=15, color=color, label=label)

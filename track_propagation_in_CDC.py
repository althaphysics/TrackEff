import sys
import numpy as np
import pandas as pd
from scipy import constants
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
from numba import njit
from numba import prange

# ---------------------------------------------------------
# 1. OPTIMIZED HELPER FUNCTIONS
# ---------------------------------------------------------

def to_deg(rad):
    return rad * 180/np.pi

def compute_point_on_line(P1, P2, y):
    ## y = az + b
    if P1[0]==P2[0]:
        z = P1[0]
    else:
        a = ( P1[1] - P2[1] )/( P1[0] - P2[0])
        if a==0:
            # Just a print for safety, usually handled
            pass 
        b =  P1[1] - a * P1[0]
        z = (y - b)/a
    return np.array([z, y])

def compute_point_on_line_signed(P1, P2, y):
    if y>=0:
        return compute_point_on_line(P1, P2, y)
    else:
        return compute_point_on_line([P1[0], -P1[1]], [P2[0], -P2[1]], y)

def angle_0_to_2pi(angle):
    # Kept for backward compatibility if used elsewhere
    while True:
        if angle>=0 and angle<2*np.pi:
            break
        elif angle < 0:
            angle += 2*np.pi
        elif angle >= 2*np.pi:
            angle -= 2*np.pi 
    return angle

def angle_0_to_2pi_vectorized(angles):
    """Fast vectorized modulo for arrays."""
    return angles % (2 * np.pi)


#------------------------------------------------------------------------------------
# Number kernels
#------------------------------------------------------------------------------------

#@njit(fastmath=True)
@njit
def compute_track_numba(
    s0, step_size, omega, tanLambda, phi0, x0, y0, z0, 
    prodVtxX, prodVtxY, prodVtxZ, pz,
    rho_min_0, max_rho,
    Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Ex, Ey, Fx, Fy,
    layer_edges, bin_to_layer_id, cell_dphis
):
    """
    JIT-compiled core that propagates the particle, checks boundaries,
    identifies cells, and counts hits in a single pass.
    """
    
    # Pre-allocate maximum possible arrays (e.g., 1000 steps)
    max_steps = 10000
    
    # Result buffers
    phi_buf = np.zeros(max_steps, dtype=np.float64)
    s_buf = np.zeros(max_steps, dtype=np.float64)
    x_buf = np.zeros(max_steps, dtype=np.float64)
    y_buf = np.zeros(max_steps, dtype=np.float64)
    z_buf = np.zeros(max_steps, dtype=np.float64)
    lid_buf = np.full(max_steps, -1, dtype=np.int32)
    cid_buf = np.full(max_steps, -1, dtype=np.int32)
    
    count = 0
    rho_vtx = np.sqrt(prodVtxX**2 + prodVtxY**2)
    
    # Pre-compute constants
    inv_omega = 1.0 / omega if omega != 0 else 0.0
    # Loop through steps
    for i in range(max_steps):
        s = s0 + i * step_size
        Cs = omega * s
        
        # 1. Propagate Helix
        # Hand-written sin/cos expansion for speed
        sin_val = np.sin(0.5 * Cs)
        # Handle limit Cs -> 0
        if abs(Cs) < 1e-9:
            sinc_term = 1.0
        else:
            sinc_term = sin_val / (0.5 * Cs)
            
        phi_curr = phi0 - Cs
        # Normalize phi to 0-2pi
        #phi_curr = phi_curr % (2 * np.pi)
        
        x = x0 + s * sinc_term * np.cos(phi0 - 0.5 * Cs)
        y = y0 + s * sinc_term * np.sin(phi0 - 0.5 * Cs)
        z = z0 + s * tanLambda
        rho = np.sqrt(x**2 + y**2)
        
        # 2. Check Global Bounds (Early Exit potential)
        if rho > max_rho + 5.0: # Margin for error
             break
        
        # 3. Check Vertex Condition
        '''if rho_vtx > 0:
            if rho < rho_vtx: continue # Before vertex
            if pz >= 0 and z < prodVtxZ: continue
            if pz < 0 and z > prodVtxZ: continue'''

        # 3. Check Vertex Condition
        if rho_vtx > 0:
            if rho < rho_vtx: 
                continue # Before vertex in rho
            # Check z condition based on pz direction
            if pz >= 0:
                if z < prodVtxZ:
                    continue
            else:  # pz < 0
                if z > prodVtxZ:
                    continue

        # 4. Check Inside CDC (Geometric)
        # Z-boundaries logic manually inlined for speed
        # Left boundary zL
        if rho <= Cy: # Below C/E split
             # Line CE: P1=C(Cx, Cy), P2=E(Ex, Ey)
             slope = (Cy - Ey)/(Cx - Ex)
             zL = (rho - (Cy - slope*Cx)) / slope
        else:
             # Line AC: P1=A(Ax, Ay), P2=C(Cx, Cy)
             slope = (Ay - Cy)/(Ax - Cx)
             zL = (rho - (Ay - slope*Ax)) / slope
             
        # Right boundary zR
        if rho <= Dy:
             # Line FD: P1=F(Fx, Fy), P2=D(Dx, Dy)
             slope = (Fy - Dy)/(Fx - Dx)
             zR = (rho - (Fy - slope*Fx)) / slope
        else:
             # Line DB: P1=D(Dx, Dy), P2=B(Bx, By)
             slope = (Dy - By)/(Dx - Bx)
             zR = (rho - (Dy - slope*Dx)) / slope
             
        if z < zL or z > zR:
            break
            #continue # Outside Z bounds
        
        if rho < rho_min_0:
            continue # Inside inner wall
            
        if rho > max_rho:
            break # Exited outer wall -> STOP CALCULATION (Early Exit)

        '''# 5. Layer Lookup (Binary Search)
        # np.searchsorted is supported in Numba
        idx = np.searchsorted(layer_edges, rho)
        layer_id = bin_to_layer_id[idx]
        
        cell_id = -1
        if layer_id != -1:
            dphi = cell_dphis[layer_id]
            if dphi > 0:
                cell_id = int(phi_curr // dphi)'''
        
        # 5. Layer Lookup (Binary Search)
        idx = np.searchsorted(layer_edges, rho)
        layer_id = bin_to_layer_id[idx]
        
        # Calculate azimuthal angle for ALL valid points
        azimuthal_phi = np.arctan2(y, x)
        if azimuthal_phi < 0:
            azimuthal_phi += 2 * np.pi
        
        cell_id = -1
        if layer_id != -1:
            dphi = cell_dphis[layer_id]
            if dphi > 0:
                cell_id = np.int32(azimuthal_phi // dphi)

        # Store valid point
        phi_buf[count] = azimuthal_phi
        #phi_buf[count] = phi_curr
        s_buf[count] = s
        x_buf[count] = x
        y_buf[count] = y
        z_buf[count] = z
        lid_buf[count] = layer_id
        cid_buf[count] = cell_id
        count += 1
        
    return count, phi_buf, s_buf, x_buf, y_buf, z_buf, lid_buf, cid_buf

#@njit(fastmath=True)
@njit
def count_hits_numba(count, lid_buf, cid_buf, min_sl0, min_outer):
    """
    Replaces Pandas groupby/drop_duplicates for hit counting.
    """
    hits = 0
    
    if count == 0:
        return 0
    
    # Current tracking
    current_lid = -2
    current_cid = -2
    steps_in_cell = 0
    
    # We iterate through the valid points
    # (The problematic line `step_counts = []` has been removed here)
    
    # Loop for hit counting
    for i in range(count):
        l = lid_buf[i]
        c = cid_buf[i]
        
        if l == -1 or c == -1:
            # Gap between layers, reset counter
            current_lid = -2
            current_cid = -2
            steps_in_cell = 0
            continue
            
        if l == current_lid and c == current_cid:
            steps_in_cell += 1
        else:
            # New cell entered, check previous cell
            if current_lid != -2:
                # Check thresholds
                threshold = min_sl0 if current_lid < 8 else min_outer
                if steps_in_cell > threshold:
                    hits += 1
            
            # Reset for new cell
            current_lid = l
            current_cid = c
            steps_in_cell = 1
            
    # Check final cell
    if current_lid != -2:
        threshold = min_sl0 if current_lid < 8 else min_outer
        if steps_in_cell > threshold:
            hits += 1
            
    return hits

@njit(parallel=True)
def batch_count_hits(
    step_size, 
    s0s, # <-- NEW ARGUMENT
    omegas, tanLambdas, phi0s, x0s, y0s, z0s, 
    vtxXs, vtxYs, vtxZs, pzs,
    rho_min, rho_max,
    Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Ex, Ey, Fx, Fy,
    layer_edges, bin_to_layer_id, cell_dphis,
    min_sl0, min_outer
):
    #print('layer edges are = ', layer_edges)
    #print('bin_to_layer_id = ', bin_to_layer_id)
    n_tracks = len(omegas)
    hit_counts = np.zeros(n_tracks, dtype=np.int32)
    
    # Loop over all tracks in parallel
    for i in prange(n_tracks):
        # UPDATE: pass the correct starting arc length (s0s[i])
        count, _, _, _, _, _, lids, cids = compute_track_numba(
            s0s[i], step_size, omegas[i], tanLambdas[i], phi0s[i], # <-- Use s0s[i] here
            x0s[i], y0s[i], z0s[i],
            vtxXs[i], vtxYs[i], vtxZs[i], pzs[i],
            rho_min, rho_max,
            Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Ex, Ey, Fx, Fy,
            layer_edges, bin_to_layer_id, cell_dphis
        )
        
        hit_counts[i] = count_hits_numba(count, lids, cids, min_sl0, min_outer)
        
    return hit_counts

# ---------------------------------------------------------
# 2. UPDATED CDC CLASS
# ---------------------------------------------------------
class CDC:
    """Simplified model of the CDC"""
    
    def __init__(self, dictionary):
        self.unit = '[cm]'
        self.A, self.B = np.array([ -83.1,  108.2+15.0  ]),  np.array([  158.6,  108.2+15.0  ])     
        self.C, self.D = np.array([ -68.2,   37.14 ]),       np.array([  144.9,   43.80 ])      
        self.E, self.F = np.array([ -47.4,   24.95 ]),       np.array([   87.7,   24.95 ]) 
        self.G, self.H = np.array([ -999,    16.0  ]),       np.array([   -999,   16.0  ]) 
        # super-layers (SL) geomertry
        self.rho_min = np.array([16.80, 25.70, 36.52, 47.69, 58.41, 69.53, 80.25,  91.37, 102.09])
        self.rho_max = np.array([23.80, 34.80, 45.57, 56.69, 67.41, 78.53, 89.25, 100.37, 111.14])
        # number of layers in a given super-layer 
        self.nLayers = np.array([   8,    6,     6,     6,     6,     6,     6,      6,      6]) 
        # number of cells in a given layer
        self.nCells  = np.array([ 160,  160,   192,   224,   256,   288,   320,    352,    384]) 

        for attr in self.__dict__:
            if attr in dictionary.keys():
                self.__setattr__(attr, dictionary[attr])
                
        self.G = compute_point_on_line(P1=self.C,  P2=self.E, y=self.G[1])
        self.H = compute_point_on_line(P1=self.D,  P2=self.F, y=self.H[1])
        
        self.SL = self._superLayers() 
        self.layers = self._layer_organisation() 
        
        # --- OPTIMIZATION: Prepare Fast Lookups ---
        self._prepare_fast_lookups()

    '''def _prepare_fast_lookups(self):
        # Create sorted array of all layer boundaries for binary search
        self.fast_layer_edges = np.sort(np.unique(np.concatenate([
            self.layers.rho_min.values, 
            self.layers.rho_max.values
        ])))
        
        # Map bins to layer IDs
        # If edges has N elements, searchsorted returns indices 0 to N.
        # This creates N+1 possible bins.
        # Index 0: x < edge[0] (Outside inner boundary)
        # Index N: x >= edge[N-1] (Outside outer boundary)
        self.fast_bin_to_layer_id = np.full(len(self.fast_layer_edges) + 1, -1, dtype=int)
        
        # We calculate midpoints for the intervals BETWEEN edges.
        # If edges has N elements, there are N-1 intervals.
        mids = 0.5 * (self.fast_layer_edges[:-1] + self.fast_layer_edges[1:])
        
        for idx, row in self.layers.iterrows():
            # Determine which intervals belong to this layer
            mask = (mids >= row.rho_min) & (mids < row.rho_max)
            
            # Assign layer_id to the corresponding bins.
            # The first interval (mids[0]) corresponds to searchsorted index 1.
            # The last interval (mids[N-2]) corresponds to searchsorted index N-1.
            # We slice fast_bin_to_layer_id from 1 to -1 to exclude the "underflow" and "overflow" bins.
            self.fast_bin_to_layer_id[1:-1][mask] = int(row.layer_id)

        # Fast access to cell_delta_phi
        max_lid = int(self.layers.layer_id.max())
        self.fast_cell_dphi = np.zeros(max_lid + 1)
        self.fast_cell_dphi[self.layers.layer_id.values.astype(int)] = self.layers.cell_delta_phi.values'''
    
    def _prepare_fast_lookups(self):
        # Use ALL boundaries, not only rho_max (there could be gaps in between the layers, so rho_max is not enough)
        edges = np.sort(np.unique(np.concatenate([
            self.layers.rho_min.values,
            self.layers.rho_max.values
        ])))
        self.fast_layer_edges = edges

        # searchsorted gives indices 0..N for N edges, so we need N+1 bins
        self.fast_bin_to_layer_id = np.full(len(edges) + 1, -1, dtype=np.int32)

        # Fill bins corresponding to the intervals between edges using midpoints
        mids = 0.5 * (edges[:-1] + edges[1:])
        for _, row in self.layers.iterrows():
            mask = (mids >= row.rho_min) & (mids < row.rho_max)
            self.fast_bin_to_layer_id[1:-1][mask] = int(row.layer_id)

        # Fast access to cell_delta_phi
        max_lid = int(self.layers.layer_id.max())
        self.fast_cell_dphi = np.zeros(max_lid + 1)
        self.fast_cell_dphi[self.layers.layer_id.values.astype(int)] = self.layers.cell_delta_phi.values

    def _layer_organisation(self):
        Ntot_layers = self.nLayers.sum()
        cdc_layers = pd.DataFrame()
        cdc_layers['layer_id'] = [i for i in range( 0, Ntot_layers )]
        cdc_layers.index = [f'lay{i}'  for i in range(0, cdc_layers.shape[0])]
        cdc_layers['SL'] = 'SL0'
        layer_min = int(self.SL.loc['SL0'].nLayers)
        for i in range(1, self.SL.shape[0]):
            cdc_layers['SL'] = np.where(cdc_layers.layer_id.isin(np.arange(layer_min, layer_min+6, 1)), f'SL{i}', cdc_layers.SL)
            layer_min += int(self.SL.loc[f'SL{i}'].nLayers)
        cdc_layers['nCells']   = 0
        cdc_layers['rho_size'] = 0
        cdc_layers['rho_min']  = 0
        cdc_layers['rho_max']  = 0
        for i in range(0, self.SL.shape[0]):
            sl = f'SL{i}'
            cdc_layers['nCells'] = np.where(cdc_layers.SL==sl, int(self.SL.loc[sl].nCells), cdc_layers.nCells)
            rho_size = (self.SL.loc[sl].rho_max - self.SL.loc[sl].rho_min)/self.SL.loc[sl].nLayers # layer size in rho
            cdc_layers['rho_size'] = np.where(cdc_layers.SL==sl, rho_size, cdc_layers.rho_size)
            rho_min0 = self.SL.loc[sl].rho_min
            relative_index = cdc_layers[cdc_layers.SL==sl].layer_id.values[0]
            cdc_layers['rho_min'] = np.where(cdc_layers.SL==sl, rho_min0+(cdc_layers.layer_id - relative_index)*cdc_layers.rho_size, cdc_layers.rho_min)
            cdc_layers['rho_max'] = np.where(cdc_layers.SL==sl, rho_min0+(cdc_layers.layer_id - relative_index)*cdc_layers.rho_size + cdc_layers.rho_size, cdc_layers.rho_max)
        cdc_layers['cell_delta_phi']  = 2*np.pi/cdc_layers.nCells ## size of one cell in phi
        
        cdc_layers = self._add_zShape_info(cdc_layers)
            
        return cdc_layers
        
    def _layer_length_z(P1, P2):
        return abs(P2[0] - P1[0])
    
    def _superLayers(self):
        cdc_SL = pd.DataFrame()
        cdc_SL['rho_min'] = self.rho_min
        cdc_SL['rho_max'] = self.rho_max
        cdc_SL['nLayers'] = self.nLayers
        cdc_SL['nCells']  = self.nCells
        cdc_SL.index = [f'SL{i}'  for i in range(0, cdc_SL.shape[0])]
        cdc_SL['maxLayerID'] = cdc_SL.nLayers.cumsum()
        cdc_SL['minLayerID'] = cdc_SL.maxLayerID - cdc_SL.nLayers
        cdc_SL['layer_size_rho'] = (cdc_SL.rho_max - cdc_SL.rho_min) / cdc_SL.nLayers
        cdc_SL = self._add_zShape_info(cdc_SL)
        return cdc_SL
    
    def _add_zShape_info(self, df):
        df['p1_bwd'] = 0
        df['p1_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.E, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.C, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.A, axis=1),  df.p1_bwd)
        df['p2_bwd'] = 0
        df['p2_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.G, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.E, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.C, axis=1),  df.p2_bwd)
        df['z_bwd'] = 0
        # Optimizing this scalar apply is possible but less critical than Trajectory
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
        return f"<CDC simplified model>"

    def __str__(self):
        return f"CDC simplified model\n  {self.__dict__}\n"

    def insideCDC(self, x, y, z, verbose=False):    
        # Original scalar implementation kept for backward compatibility
        inCDC = True
        rho = np.sqrt(x**2 + y**2)
        id_last_layer = self.layers.shape[0]-1
        zL, zR = 0, 0
        if rho<=self.rho_min[0]: 
            pass
        else:
            zL = compute_point_on_line_signed(self.C, self.E, rho)[0]  if abs(rho)<=self.C[1] else compute_point_on_line_signed(self.A, self.C, rho)[0]
            zR = compute_point_on_line_signed(self.F, self.D, rho)[0]  if abs(rho)<=self.D[1] else compute_point_on_line_signed(self.D, self.B, rho)[0] 
            if z<zL or z>zR:
                inCDC=False
                if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), zL = {zL:.1f}, zR = {zR:.1f}, rho = {rho:.1f}, rhoM = {self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:.1f}' )

        if  rho>self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]: 
            inCDC=False
            if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), rho = {rho:.1f}' )

        return inCDC

    def get_cellID(self, phi, cell_delta_phi):
        if cell_delta_phi > 0:
            cell_id = int(np.floor(angle_0_to_2pi(phi)/cell_delta_phi))
        else:
            cell_id = -1
        return cell_id

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
        patches  = [ Rectangle((self.SL.loc[sl].z_bwd,  self.SL.loc[sl].rho_min), 
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd, 
                            height=(self.SL.loc[sl].rho_max - self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        patches += [ Rectangle((self.SL.loc[sl].z_bwd,  - self.SL.loc[sl].rho_max), 
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd, 
                            height=(self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)
        
        patches  = [ Rectangle((self.layers.loc[lay].z_bwd,  self.layers.loc[lay].rho_min), 
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd, 
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        patches += [ Rectangle((self.layers.loc[lay].z_bwd,  -self.layers.loc[lay].rho_max), 
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd, 
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        p2 = PatchCollection(patches, alpha=0.4, linewidth=2, edgecolor='b')
        ax.add_collection(p2)
            
        ax.set_xlabel(f'z {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        ax.set_xlim(self.A[0], self.B[0])
        ax.set_ylim(-self.A[1], self.A[1])

    def xy_contour(self, fig, ax):
        patches = [ Wedge((0, 0), 
                      self.SL.loc[sl].rho_max, 
                      0, 360, 
                      width=( self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min  ))  for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])] ]
        ax.add_collection( PatchCollection(patches,  alpha=0.4, linewidth=2, edgecolor='b') )
        
        patches = [ Wedge((0, 0), 
                        self.layers.loc[lay].rho_max, 
                        0, 360, 
                        width=( self.layers.loc[lay].rho_max-self.layers.loc[lay].rho_min  ))  for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])] ]
        ax.add_collection( PatchCollection(patches, alpha=0.4, linewidth=2, edgecolor='b', ls=':') )

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
    """
    Base Particle Class.
    """
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
        Bz = 1.5
        a = Bz/(constants.c*1e-6) # constants.c is in m.s-1
        omega     = a * self.q / pt   # [cm-1]   
        tanLambda = self.pz/pt 

        Dpar  = -self.prodVtxX * np.cos(phi) - self.prodVtxY * np.sin(phi)
        Dperp = -self.prodVtxY * np.cos(phi) + self.prodVtxX * np.sin(phi)
        A = 2 * Dperp + omega * ( Dperp**2 + Dpar**2 )
        U = np.sqrt( 1 + omega * A )

        dphi = np.arctan2( omega * Dpar, 1 + omega * Dperp ) 
        s = dphi/omega 

        d0 = A / ( 1 + U )
        phi0 = phi - dphi 
        z0 = self.prodVtxZ + s * tanLambda 

        x0 = d0 * np.sin(phi0) 
        y0 = -d0 * np.cos(phi0) 

        return x0, y0, z0, d0, phi0, omega, tanLambda, s 

class Point(Particle):
    """
    Single Point Logic.
    (Not used by the vectorized Trajectory anymore, but kept for debugging/compatibility)
    """
    def __init__(self, dictionary, instanceCDC, s):
        super().__init__(dictionary)
        self.instanceCDC = instanceCDC 
        self.layer_id = -1 
        self.cell_id = -1 

        self.x, self.y, self.z, self.phi, self.s = self._get_particle_position(s)
        self.inCDC = instanceCDC.insideCDC(self.x, self.y, self.z, verbose=False) 
        self.rho =  np.sqrt(self.x**2 + self.y**2)

        self.azimutal_angle = np.arctan2(self.y, self.x)
        
        if self.inCDC:
            vals = instanceCDC.layers.loc[(instanceCDC.layers['rho_min'] <= self.rho) & (instanceCDC.layers['rho_max'] > self.rho)].layer_id.values
            if len(vals)==1:
                self.layer_id = instanceCDC.layers.loc[(instanceCDC.layers['rho_min'] <= self.rho) & (instanceCDC.layers['rho_max'] > self.rho)].layer_id.values[0]
                self.cell_id = instanceCDC.get_cellID(self.azimutal_angle, instanceCDC.layers.loc[f'lay{self.layer_id}'].cell_delta_phi)
            elif len(vals)>1:
                sys.exit('More than one CDC layer correspond to that point. That should not happend. Aborting.')

    def _get_particle_position(self, s):
        phi = self.phi0 - s*self.omega 
        z_prime = self.z0 + s * self.tanLambda

        Cs = self.omega * s
        if Cs != 0:
             x_prime = self.x0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.cos(self.phi0 - 0.5 * Cs)
             y_prime = self.y0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.sin(self.phi0 - 0.5 * Cs)
        else:
             x_prime = self.x0
             y_prime = self.y0

        return x_prime, y_prime, z_prime, phi, s
    
    def __repr__(self):
        return f"<Point at phi = {self.phi:.3f}>"

# ---------------------------------------------------------
# 3. OPTIMIZED TRAJECTORY CLASS
# ---------------------------------------------------------
class Trajectory(Particle):
    """
    Vectorized Trajectory Calculation.
    Calculates all points in one go using numpy arrays instead of looping.
    """
    def __init__(self, dictionary, instanceCDC, step_size, min_steps_in_SL0_cell=2, min_steps_in_outer_cell=10):
        super().__init__(dictionary)

        self.min_steps_in_SL0_cell   = min_steps_in_SL0_cell 
        self.min_steps_in_outer_cell = min_steps_in_outer_cell 
        self.signed_step = step_size
        
        # 1. Generate step array (vector of s values)
        self.steps = np.arange(self.s0, self.s0 + 1000, self.signed_step)
        
        # 2. Vectorized Helix Propagation
        Cs = self.omega * self.steps
        
        # Handle sinc singularity at Cs=0
        with np.errstate(divide='ignore', invalid='ignore'):
            sin_term = np.sin(0.5 * Cs) / (0.5 * Cs)
            sin_term = np.where(Cs == 0, 1.0, sin_term)
            
        self.phi = self.phi0 - Cs 
        self.x_arr = self.x0 + self.steps * sin_term * np.cos(self.phi0 - 0.5 * Cs)
        self.y_arr = self.y0 + self.steps * sin_term * np.sin(self.phi0 - 0.5 * Cs)
        self.z_arr = self.z0 + self.steps * self.tanLambda
        self.rho_arr = np.sqrt(self.x_arr**2 + self.y_arr**2)
        
        # 3. Vectorized "Inside CDC" Checks
        
        # Z-Boundary helper (vectorized)
        def get_z_boundary(P1, P2, y_arr):
            y_abs = np.abs(y_arr)
            slope = (P1[1] - P2[1]) / (P1[0] - P2[0])
            intercept = P1[1] - slope * P1[0]
            return (y_abs - intercept) / slope

        # Left Boundary
        zL = np.where(self.rho_arr <= instanceCDC.C[1],
                      get_z_boundary(instanceCDC.C, instanceCDC.E, self.rho_arr),
                      get_z_boundary(instanceCDC.A, instanceCDC.C, self.rho_arr))
        # Right Boundary
        zR = np.where(self.rho_arr <= instanceCDC.D[1],
                      get_z_boundary(instanceCDC.F, instanceCDC.D, self.rho_arr),
                      get_z_boundary(instanceCDC.D, instanceCDC.B, self.rho_arr))
        
        valid_z = (self.z_arr >= zL) & (self.z_arr <= zR)
        # THIS LINE IS CULPRIT FOR TRACK OFFSET
        valid_rho = (self.rho_arr <= instanceCDC.layers.rho_max.max()) #& (self.rho_arr >= instanceCDC.rho_min[0])
        self.inCDC_mask = valid_z & valid_rho
        
        # Vertex check logic
        rho_Vtx = np.sqrt(self.prodVtxX**2 + self.prodVtxY**2)
        if rho_Vtx > 0:
            vertex_mask = (self.rho_arr >= rho_Vtx)
            if self.pz >= 0: vertex_mask &= (self.z_arr >= self.prodVtxZ)
            else:            vertex_mask &= (self.z_arr <= self.prodVtxZ)
            self.inCDC_mask &= vertex_mask

        # 4. Construct Points DataFrame
        if not np.any(self.inCDC_mask):
            self.points = pd.DataFrame({'phi':[], 's':[], 'x':[], 'y':[], 'z':[], 'layer_id':[], 'cell_id':[]})
            self.CDChits = 0
            self._set_empty_stats()
            return

        # Filter arrays to only valid points
        v_rho = self.rho_arr[self.inCDC_mask]
        v_x = self.x_arr[self.inCDC_mask]
        v_y = self.y_arr[self.inCDC_mask]
        v_z = self.z_arr[self.inCDC_mask]
        v_s = self.steps[self.inCDC_mask]
        v_phi = angle_0_to_2pi_vectorized(np.arctan2(v_y, v_x))

        # Fast Layer Lookup (Binary Search)
        bin_indices = np.searchsorted(instanceCDC.fast_layer_edges, v_rho)
        v_layer_ids = instanceCDC.fast_bin_to_layer_id[bin_indices]
        
        # Fast Cell Lookup
        v_cell_ids = np.full_like(v_layer_ids, -1)
        has_layer = v_layer_ids >= 0
        if np.any(has_layer):
            rel_layers = v_layer_ids[has_layer]
            dphis = instanceCDC.fast_cell_dphi[rel_layers]
            v_cell_ids[has_layer] = np.floor(v_phi[has_layer] / dphis).astype(int)

        self.points = pd.DataFrame({
            'phi': v_phi, 's': v_s, 'x': v_x, 'y': v_y, 'z': v_z,
            'layer_id': v_layer_ids, 'cell_id': v_cell_ids
        })

        # 5. Count Hits (Vectorized Grouping)
        self.points['cumSum'] = self.points[["layer_id","cell_id"]].ne(self.points[["layer_id","cell_id"]].shift()).any(axis=1).cumsum() 
        self.points['nStepsIncell'] = self.points.groupby('cumSum')['layer_id'].transform('count')
        
        self._count_cdc_hits()
    
    def _set_empty_stats(self):
        self.meanStepsPerCell = 0
        self.stdStepsPerCell = 0
        self.minStepsPerCell = 0
        self.maxStepsPerCell = 0
        self.meanStepsPerCell_SL0 = 0
        self.stdStepsPerCell_SL0 = 0
        self.minStepsPerCell_SL0 = 0
        self.maxStepsPerCell_SL0 = 0
        self.meanStepsPerCell_SL1_SL8 = 0
        self.stdStepsPerCell_SL1_SL8 = 0
        self.minStepsPerCell_SL1_SL8 = 0
        self.maxStepsPerCell_SL1_SL8 = 0

    def _count_cdc_hits(self):
        mask = (self.points.cell_id > -1)
        df_valid = self.points[mask]
        cells = df_valid.drop_duplicates(['layer_id', 'cell_id'], keep='first')
        # cells.to_csv('new_cells.txt', sep='\t')
        self.CDChits = cells.loc[( ((cells.layer_id<8)&(cells.nStepsIncell>self.min_steps_in_SL0_cell)) | ((cells.layer_id>=7)&(cells.nStepsIncell>self.min_steps_in_outer_cell)) ) ].shape[0]
    
        self.meanStepsPerCell = df_valid['nStepsIncell'].mean()
        self.stdStepsPerCell  = df_valid['nStepsIncell'].std()
        self.minStepsPerCell  = df_valid['nStepsIncell'].min()
        self.maxStepsPerCell  = df_valid['nStepsIncell'].max()

        self.meanStepsPerCell_SL0 = df_valid.query('layer_id<8')['nStepsIncell'].mean()
        self.stdStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].std()
        self.minStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].min()
        self.maxStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].max()

        self.meanStepsPerCell_SL1_SL8 = df_valid.query('layer_id>=8')['nStepsIncell'].mean()
        self.stdStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].std()
        self.minStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].min()
        self.maxStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].max()

    def xy_trajectory(self, ax, label='', color='b'):
        #print(f'in new code')
        #print(self.points['x'], self.points['y'])
        ax.scatter(self.points['x'], self.points['y'], color=color, label=label)

    def zy_trajectory(self, ax, label='', color='b'):
        ax.scatter(self.points['z'], self.points['y'], color=color, label=label)

    def xy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxX ], [self.prodVtxY], marker="*", markersize=15, color=color, label=label)
        
    def zy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxZ ], [self.prodVtxY], marker="*", markersize=15, color=color, label=label)
        
    def xy_poca(self, ax, label='', color='b'):
        ax.plot( [self.x0 ], [self.y0], marker="x", markersize=15, color=color, label=label)
        
    def zy_poca(self, ax, label='', color='b'):
        ax.plot( [self.z0 ], [self.y0], marker="x", markersize=15, color=color, label=label)
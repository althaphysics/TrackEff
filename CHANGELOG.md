
# Changelog – TrackEff

## Modifications to TrackEff

The TrackEff package, originally developed for the Belle II Central Drift Chamber (CDC) geometry, has been generalized to support the drift chamber geometries of Belle II (Section 6 of [arXiv:1011.0352](https://arxiv.org/abs/1011.0352)), BESIII (Section 4 of [arXiv:0911.4960](https://arxiv.org/abs/0911.4960)), and the proposed STCF (Section 3.5 of [arXiv:2303.15790](https://arxiv.org/abs/2303.15790)).

The major modifications are summarized below:

1. **Detector-dependent layer size modelling**
   - Added support for detector-specific drift-cell size modelling within each superlayer (SL).
   - Layer sizes are now computed from detector geometry parameters, allowing both uniform and gradually varying cell sizes.

2. **Generalized layer classification**
   - Replaced the Belle II-specific inner/outer layer classification with a detector-independent Small (S)/Large (L) layer classification based on configurable cell-size thresholds.

3. **Superlayer-wise hit counting**
   - Implemented superlayer (SL) hit counting by determining the number of hit layers within each superlayer, enabling SL-based track selection.

4. **Support for multiple detector geometries**
   - Refactored the detector geometry implementation to support Belle II CDC, BESIII MDC, and the proposed STCF MDC through external detector configuration files.

---

## Implementation Details

The following modifications were implemented to accommodate different drift chamber geometries and detector-specific hit selection criteria.

*See the detector geometry files in the `geometry/` directory and `track_propagation_in_DC.py` for the corresponding implementation.*

### 1. Boundary Point Definition

**Files:** Detector geometry files for BESIII and STCF

- Introduced boundary points E and F so that all detector geometries are represented using a common eight-point boundary description (A–H).
- This provides a unified geometry interface across Belle II, BESIII, and STCF.

### 2. Magnetic Field

**Files:** Detector geometry files and `Particle._get_helix_paramters_at_POCA()`

- The magnetic field (`Bz`) is now defined within the detector geometry configuration.
- Helix parameter calculations automatically use the detector-specific magnetic field.

### 3. Layer Size Modelling Within a Superlayer (SL)

**Files:** Detector geometry files

New detector parameters:

- `layersize_min`
- `layersize_max`

These correspond to the cell size of the innermost and outermost layers of each superlayer.

- If `layersize_min != layersize_max`, the layer size is assumed to increase linearly within the superlayer.
- If `layersize_min == layersize_max`, a uniform cell size is assumed throughout the superlayer.

### 4. Layer Size Calculation

**Function:** `MDC._layer_organisation()`

For detectors with non-uniform cell sizes:

$$
\text{rhosize} = \text{rhosize}_{\min} + j \times \Delta \rho
$$

where:

- `rhosize_min`: Cell size of the first layer in the superlayer.
- `j`: Layer index within the superlayer.
- \(\Delta \rho\):

$$
\Delta \rho =
\frac{\text{rhosize}_{\max}-\text{rhosize}_{\min}}
{n_{\text{layers}}-1}
$$

### 5. Cell Size Threshold

**Files:** Detector geometry files

New detector parameter:

- `layer_size_threshold`

This parameter replaces the previous inner/outer layer distinction and defines the boundary between:

- **S** — Small cells
- **L** — Large cells

Because the classification is based on physical cell size, smaller cells may also occur in outer superlayers for detectors with gradually increasing layer sizes.

---

## Hit Acceptance Criteria

### 6. Layer Classification

**Function:** `MDC._layer_organisation()`

New layer property:

- `rho_size_class`

Each layer is classified according to the detector-specific `rho_size_threshold`:

- **S** — Small cell
- **L** — Large cell

### 7. Hit Selection

**Classes:** `Trajectory.__init__()`, `Point.__init__()`

The hit acceptance criteria were updated to use:

- `min_steps_in_small_cell`
- `min_steps_in_large_cell`

These replace the previous parameters:

- `min_steps_in_inner_cell`
- `min_steps_in_outer_cell`

### 8. Superlayer-wise Hit Counting

**Function:** `Trajectory._count_dc_hits()`

Added layer-level hit evaluation to identify detector layers satisfying the hit selection criteria:

- Determination of hit layers.
- Determination of non-hit layers.
- Computation of superlayer-wise hit counts.
- `SL_hits()` function for retrieving the number of hit layers in each superlayer.

This information can be used to implement superlayer-based track selection criteria, such as requiring a minimum fraction of hit layers within individual superlayers.


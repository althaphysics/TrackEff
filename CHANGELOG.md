# MODIFICATIONS TO TRACKEFF
**The TrackEff package, originally developed for the Belle II Central Drift Chamber (CDC) geometry, has been generalized to support the drift chamber geometries of Belle II( Section.6 of arXiv:1011.0352), BESIII(section.4 of arXiv:0911.4960), and the proposed STCF (section.3.5 of arXiv:2303.15790).**

The major modifications are summarized below:

1. Detector-dependent layer size modelling
    - Added support for detector-specific drift-cell size modelling within each superlayer (SL).
    - Layer sizes are now computed from detector geometry parameters, allowing both uniform and gradually varying cell sizes.
2. Generalized layer classification
    - Replaced the Belle II-specific inner/outer layer classification with a detector-independent Small (S)/Large (L) layer classification based on configurable cell-size thresholds.
3. Superlayer-wise hit counting
    - Implemented superlayer (SL) hit counting by determining the number of hit layers within each superlayer, enabling SL-based track selection.
4. Support for multiple detector geometries
    - Refactored the detector geometry implementation to support Belle II CDC, BESIII MDC, and the proposed STCF MDC through external detector configuration files.


## IMPLEMENTATION DETAILS
**The following modifications were implemented to accommodate del_rhoerent drift chamber geometries and detector-specific hit selection criteria**
*See the detector geometry files in the geometry/ directory and track_propagation_in_DC.py for the corresponding implementation.*

## Boundary Point Definition
[Geometry files for BESIII and STCF]:
    - Introduced boundary points E and F so that all detector geometries are represented using a common eight-point boundary description (A–H).
    - This provides a unified geometry interface across Belle II, BESIII, and STCF.

## Magnetic Field
[Detector geometry files and Particle._get_helix_paramters_at_POCA()]:
    - The magnetic field (Bz) is now defined within the detector geometry configuration.
    - Helix parameter calculations automatically use the detector-specific magnetic field.

## Layer Size Modelling Within a Superlayer (SL)
[Detector geometry files]:
New detector parameters:
    *layersize_min
    layersize_max*
These correspond to the cell size of the innermost and outermost layers of each superlayer.
    - If *layersize_min ≠ layersize_max*, the layer size is assumed to increase linearly within the superlayer.
    - If *layersize_min = layersize_max*, a uniform cell size is assumed throughout the superlayer.

## Layer size calculation
[MDC._layer_organisation()]:
For detectors with non-uniform cell sizes,
 **rhosize = rhosize_min + j * del_rho**
    *where
        rhosize_min : cell size of the first layer in the superlayer,
        j : layer index within the superlayer,
        del_rho = (rhosize_max − rhosize_min)/(nlayers − 1)*

## Cell Size Threshold
[Detector geometry files]:
New detector parameter:
    *layer_size_threshold*

This parameter replaces the previous inner/outer layer distinction and defines the boundary between
    - Small (S) cells
    - Large (L) cells
Because the classification is based on physical cell size, smaller cells may also occur in outer superlayers for detectors with gradually increasing layer sizes.

## HIT ACCEPTANCE CRITERIA
## Layer Classification
[MDC._layer_organisation()]:
New layer property:
    *rho_size_class*
Each layer is classified as
    S — Small cell
    L — Large cell
according to the detector-specific *rho_size_threshold*.

## Hit Selection
[Trajectory.__init__(), Point.__init__()]:
The hit acceptance criteria were updated to use
    *min_steps_in_small_cell
    min_steps_in_large_cell*
instead of the previous
    *min_steps_in_inner_cell
    min_steps_in_outer_cell*

## Superlayer-wise Hit Counting
[Trajectory._count_dc_hits()]:
Added layer-level hit evaluation to identify detector layers satisfying the hit selection criteria:
    - determination of hit layers,
    - determination of non-hit layers,
    - computation of superlayer-wise hit counts,
    - *SL_hits()* function for retrieving the number of hit layers in each superlayer.
This information can be used to implement superlayer-based track selection criteria, such as requiring a minimum fraction of hit layers within individual superlayers.

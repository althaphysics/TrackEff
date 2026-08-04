# TrackEff

TrackEff is a lightweight Python package for estimating the number of hits deposited by a charged track passing through the Central Drift Chamber (CDC) of the Belle II detector.

Given the particle charge, three-momentum, and production vertex, TrackEff propagates the particle through the detector magnetic field using a helical trajectory model and estimates the number of CDC hits using a simplified detector geometry.

This implementation has been significantly redesigned compared to version 1.1.1:

- Numba JIT compilation for high-performance track propagation.
- Vectorized calculations using NumPy whenever possible.
- Pre-computed lookup tables for fast layer and cell identification.
- Single-pass hit counting to minimize memory allocations and improve execution speed.

These improvements make the package suitable for processing large samples of simulated tracks while preserving the same physics model.

# Installation

This package uses the following Python Packages:

- numpy
- pandas
- scipy
- matplotlib
- numba

~~~
pip install numpy scipy pandas matplotlib numba
~~~

# Quick Start

The easiest way to learn the package is through the notebook ```simple_example.ipynb```.

# Basic Usage

## 1. Create a detector model

A CDC object describes the detector geometry.

Using an empty dictionary loads the default geometry based on the Belle II Technical Design Report (TDR).

~~~python
cdc = CDC({})
~~~

Detector parameters can be modified by supplying a configuration dictionary.

## 2. Define a particle

~~~python
track = {
    "q": 1,
    "px": 1.10,
    "py": 0.89,
    "pz": -0.29,
    "prodVtxX": 4.39,
    "prodVtxY": 3.21,
    "prodVtxZ": -1.27
}
~~~

## 3. Propagate the trajectory

~~~python
traj = Trajectory(
    track,
    cdc,
    step_size=0.1
)
~~~

The trajectory object computes the particle path and stores the propagated points together with the corresponding detector information.

The estimated number of CDC hits is available through

~~~python
traj.CDChits
~~~

## 4. Visualization (optional)

TrackEff includes utilities for visualizing

- the CDC geometry,
- propagated trajectories,
- detector layers,
- crossed cells.



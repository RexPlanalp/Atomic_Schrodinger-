# Input File Documentation

## Box:

- grid_size: Defines the radius of spherically shaped grid in atomi units

- grid_spacing: Defines the resolution for the spherical grid. Only used when expanding wavefunctions from bspline basis to position basis. 

- N: Defines duration of laser pulse via number of optical cycles of central frequency

- N_post: Defines the duration of field-free post propagation after laser turns off via number of optical cucles of central frequency

- time_spacing: Defines the resolution of time evolution used for TDSE propagation. 

- R0: Defines the desired distance from the core at which point ECS turns on. Note internally R0 will be rounded to the nearest value in knot vector. 

- eta: Defines the phase with which to rotate the grid for ECS.

## species:

- species: Defines which potential to use in calculations based on atom/ion.

## splines:

- n_basis: Defines the numbere of bspline basis functions to expand in

- order: Defines the order and thus polynomial degree of bspline basis functions.

- knot_spacing: Defines the relative spacing between bspline basis functions over the grid.

## lm:

- lmax: Defines the maximum l value to expand up to

- nmax: Defines the maximum energy of eigenstates to request from TISE calculation.

## TISE:

- tolerance: Defines the tolerance for the generalized eigenvalue problem solver.

- max_iter: Defines the maximum iterations in the generalized eigenvalue problem solver to reach convergence.

- embed: Determines whether matrices necessary for a followup TDSE calculation will be constructed and saved.

# Note, everything after this is only required if TDSE calculation is desired.

## state:

- state: Defines the initial state of the TDSE simulation based on quantum numbers

## lasers: 

- center: Determines where in the time interval the center of the laser pulse is

- w: Defines the frequency of the laser pulse in atomic units

- I: Defines the intensity of the laser pulse in W/cm^2

- polarization: Defines either the major axis of elliptical polarization or the axis for linear polarization depending on the value of ell.

- poynting: Defines the laser propagation direction.

- ell: Defines whether the pulse will be linearly, circularly, or elliptically polarized. When polarization has multiple nonzero components ell will behave differently:

nonzero = 2 and ell > 0:
Circular or elliptical polarization in plane

nonzero = 2 and ell = 0:
Linear polarization along tilted axis in plane

nonzero = 1 and ell > 0:
Circular or elliptical polarization in plane with input axis defaulting to major

nonzero = 1 and ell = 0:
Linear polarization along input axis

## TDSE:

- tolerance: Defines tolerance for the solution of the linear system for time propagation.

## E:

- E: Defines the resolution and max energy for computing photoelectron spectra in atomic units.

## SLICE:

- SLICE: Defines the plane to compute the angle resolved photoelectron spectra in.

## HHG:

- HHG: Defines whether HHG data will be computed at runtime of TDSE


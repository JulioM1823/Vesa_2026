'''
Purpose
-------
Convert magnetic-field orientation and project wavenumbers in the
Cartesian x-z propagation plane used for Newington & Cally (2010).

Inputs
------
None

Outputs
-------
components, orientation, field_aligned_wavenumber: callable
    Coordinate-conversion helpers; the positive z-axis points upward.
'''
# Import the numerical functions used for coordinate conversion
import numpy as np


def components(strength, theta_deg, phi_deg, *, degrees=True):
    '''
    Purpose
    -------
    Convert a field magnitude and its inclination and azimuth to Cartesian
    x, y, and z components.

    Inputs
    ------
    strength: float or np.array
        Field magnitude, in the desired output units.
    theta_deg: float or np.array
        Inclination from the positive z-axis.
    phi_deg: float or np.array
        Azimuth from the positive x-axis toward the positive y-axis.
    degrees: bool, optional
        Interpret both angles in degrees when True and in radians otherwise.

    Outputs
    -------
    bx, by, bz: tuple of float or np.array
        Cartesian components in the units of strength. Array inputs follow
        NumPy broadcasting.
    '''

    # Convert the supplied angles before evaluating the direction cosines
    theta, phi = (np.deg2rad(theta_deg), np.deg2rad(phi_deg)) if degrees else (theta_deg, phi_deg)
    # Resolve the vector along the horizontal axes and the upward z-axis
    return (strength*np.sin(theta)*np.cos(phi),
            strength*np.sin(theta)*np.sin(phi), strength*np.cos(theta))


def orientation(bx, by, bz):
    '''
    Purpose
    -------
    Recover the magnitude, inclination, and canonical azimuth of a
    Cartesian magnetic-field vector.

    Inputs
    ------
    bx, by, bz: float or np.array
        Cartesian field components in consistent units; inputs may broadcast.

    Outputs
    -------
    strength: float or np.array
        Nonnegative field magnitude in the input units.
    theta: float or np.array
        Inclination from the positive z-axis in degrees, from 0 to 180.
    phi: float or np.array
        Azimuth in degrees from arctan2(by, bx), between -180 and 180.

    Notes
    -----
    A zero field receives inclination zero. A vertical or zero field receives
    azimuth zero because its horizontal direction is undefined.
    '''
    # Calculate horizontal and total magnitudes without squaring large values
    horizontal = np.hypot(bx, by)
    strength = np.hypot(horizontal, bz)
    # Set the undefined angles of zero and vertical fields to zero
    theta = np.where(strength == 0, 0., np.degrees(np.arctan2(horizontal, bz)))
    phi = np.where(horizontal == 0, 0., np.degrees(np.arctan2(by, bx)))
    return strength, theta, phi


def field_aligned_wavenumber(kx, kz, theta_deg, phi_deg):
    '''
    Purpose
    -------
    Project a wavenumber in the x-z propagation plane onto the field
    direction.

    Inputs
    ------
    kx, kz: float, complex, or np.array
        Horizontal and vertical wavenumber components in consistent units.
    theta_deg, phi_deg: float or np.array
        Field inclination and azimuth in degrees, as defined by components().

    Outputs
    -------
    k_parallel: float, complex, or np.array
        Signed field-parallel wavenumber in the units of kx and kz, with
        NumPy broadcasting applied.
    '''

    # Use a unit field so that only the direction enters the projection
    bx, _, bz = components(1., theta_deg, phi_deg)
    # Omit the y component because the wavevector lies in the x-z plane
    return kx*bx + kz*bz

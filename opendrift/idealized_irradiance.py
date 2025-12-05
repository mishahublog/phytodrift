import numpy as np
from datetime import datetime, timedelta


# --------------------------------------------------------
# 1. Extract time-of-day in hours (0–24)
# --------------------------------------------------------
def time_of_day_hours(time):
    """
    Convert a datetime object or timestamp to hours since midnight.
    """
    if isinstance(time, datetime):
        return time.hour + time.minute/60 + time.second/3600
    else:
        # If user supplies a float hour directly
        return float(time)


# --------------------------------------------------------
# 2. Idealized surface shortwave radiation (W/m²)
# --------------------------------------------------------
def idealized_surface_flux(time, daylength=24, Fmax=1000):
    """
    Produce a daily cycle of incoming light following a sine curve.

    Parameters
    ----------
    time : datetime or float
        Time of day. 
        If datetime → converted to hour-of-day automatically.
        If float → must be hours (0–24).
    daylength : float
        Length of day in hours. Default: 24 (full sine cycle).
    Fmax : float
        Maximum surface flux at solar noon (W/m²).

    Returns
    -------
    float
        Surface downward shortwave flux (W/m²).
    """
    t = time_of_day_hours(time)

    phase = np.pi * t / daylength
    flux = Fmax * np.maximum(0, np.sin(phase))  # zero at night
    return flux


# --------------------------------------------------------
# 3. Irradiance at depth using exponential attenuation
# --------------------------------------------------------
def irradiance_at_depth(time, lon, z, tau, nu, k_water,
                        daylength=24, Fmax=1000):
    """
    Complete irradiance model combining:
    - Gaussian solar-angle distribution
    - Idealized surface cycle
    - Exponential decay with depth

    Parameters
    ----------
    time : datetime or float
    lon : array-like
        Longitudes (only mean is used here).
    z : array-like
        Depths (negative values).
    tau : float
        Width parameter for Gaussian distribution.
    nu : float
        Biological/optical scaling parameter.
    k_water : float
        Light attenuation coefficient.
    daylength : float
        Hours per day for the light cycle.
    Fmax : float
        Max surface flux.

    Returns
    -------
    array-like
        Irradiance at depth (µmol photon m⁻² s⁻¹).
    """

    # Step 1: get surface flux (W/m²)
    surface_flux = idealized_surface_flux(time, daylength, Fmax)

    # Step 2: compute solar angle using simplified hour-angle approximation
    mean_lon = np.mean(lon)
    hour_angle_val = (time_of_day_hours(time) - 12) * 15 - mean_lon
    solar_angle = np.deg2rad(hour_angle_val)

    # Step 3: Gaussian weighting based on solar angle
    solar_coeff = np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * solar_angle**2 / 2)

    # Step 4: Convert W/m² → µmol photons m⁻² s⁻¹ (factor = 0.00836)
    # And apply depth attenuation
    light = (
        solar_coeff
        * surface_flux
        * nu
        * 0.00836
        * np.exp(k_water * z)
    )

    return light

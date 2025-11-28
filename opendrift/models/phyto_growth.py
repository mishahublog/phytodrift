import numpy as np

def phyto_growth(
    P,
    I,
    N,
    T,
    mu_max=1.2,      # d^-1
    alpha=0.03,      # PI-curve slope
    K_N=0.5,         # half-sat for nutrient (mmol/m3)
    Q10=2.0,
    T_ref=20.0,
    mortality=0.05   # d^-1
):
    """
    Computes phytoplankton growth rate at a point location.
    
    Parameters
    ----------
    P : float
        Phytoplankton biomass (any units)
    I : float
        Irradiance (e.g. µE m^-2 s^-1)
    N : float
        Nutrient concentration (e.g. mmol m^-3)
    T : float
        Temperature (°C)
    mu_max : float
        Maximum growth rate (d^-1)
    alpha : float
        Light-limitation parameter
    K_N : float
        Half-saturation constant (mmol m^-3)
    Q10 : float
        Temperature Q10 coefficient
    T_ref : float
        Reference temperature for Q10
    mortality : float
        Mortality rate (d^-1)
    
    Returns
    -------
    dPdt : float
        Time rate of change of phytoplankton
    mu : float
        Total specific growth rate
    """

    # Light limitation (no photoinhibition)
    mu_I = 1.0 - np.exp(-alpha * I)

    # Nutrient limitation
    mu_N = N / (K_N + N)

    # Temperature dependence
    mu_T = Q10 ** ((T - T_ref) / 10.0)

    # Combined specific growth rate
    mu = mu_max * mu_I * mu_N * mu_T

    # dP/dt
    dPdt = (mu - mortality) * P

    return dPdt, mu

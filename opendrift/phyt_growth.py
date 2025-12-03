import numpy as np

def phyt_growth(
    P,
    I,
    N,
    Ph,
    Fe,
    Si,
    T,
    mu_max= 3,       # d^-1
    alpha=0.03,       # light-slope
    K_N=0.5,          # half-sat nitrate (mmol/m3)
    K_Ph=0.03,         # half-sat phosphate (mmol/m3)
    K_Fe=0.001,       # half-sat iron (mmol/m3)
    K_Si=0.001,
    Q10=2.0,
    T_ref=20.0,
    mortality=0.05 ):
    """
    Computes phytoplankton growth with multi-nutrient limitation (N, P, Fe).
    Liebig minimum rule is used for nutrient limitation.

    Parameters
    ----------
    P : float
        Phytoplankton concentration
    I : float
        Light (µE m^-2 s^-1)
    N, Pnut, Fe : float
        Nutrient concentrations (mmol m^-3)
    T : float
        Temperature (°C)
    K_N, K_P, K_Fe : float
        Half-saturation constants
    """

    # --------------------
    # Light limitation
    # --------------------
    mu_I = 1.0 - np.exp(-alpha * I)

    # --------------------
    # Multi-nutrient limitation (Liebig minimum)
    # --------------------
    lim_N  = N   / (K_N  + N)
    lim_Ph  = Ph   / (K_Ph  + Ph)
    lim_Fe = Fe  / (K_Fe + Fe)
    lim_Si = Si / (K_Si + Si)


    mu_nut = np.minimum.reduce([lim_N, lim_Ph, lim_Fe,lim_Si])

    # --------------------
    # Temperature dependence
    # --------------------
    mu_T = Q10 ** ((T - T_ref) / 10.0)

    # --------------------
    # Combined specific growth rate
    # --------------------
    mu = mu_max * mu_I * mu_nut * mu_T

    # dP/dt including mortality
    dPdt = (mu - mortality) * P

    return dPdt


# ---------------------------------------------------------




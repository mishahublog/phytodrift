import numpy as np

def phyto_growth_multinutrient(
    P,
    I,
    N,
    Pnut,
    Fe,
    T,
    mu_max=1.2,       # d^-1
    alpha=0.03,       # light-slope
    K_N=0.5,          # half-sat nitrate (mmol/m3)
    K_P=0.03,         # half-sat phosphate (mmol/m3)
    K_Fe=0.001,       # half-sat iron (mmol/m3)
    Q10=2.0,
    T_ref=20.0,
    mortality=0.05
):
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
    lim_P  = Pnut / (K_P  + Pnut)
    lim_Fe = Fe  / (K_Fe + Fe)

    mu_nut = min(lim_N, lim_P, lim_Fe)

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

    return dPdt, mu, mu_nut, lim_N, lim_P, lim_Fe


# ---------------------------------------------------------
# Example usage
P0   = 1.0
I    = 80
N    = 1.2
Pnut = 0.05
Fe   = 0.002
T    = 18

results = phyto_growth_multinutrient(P0, I, N, Pnut, Fe, T)

(dPdt, mu, mu_nut, lim_N, lim_P, lim_Fe) = results

print("Specific growth rate (1/day):", mu)
print("dP/dt:", dPdt)
print("N limitation:", lim_N)
print("P limitation:", lim_P)
print("Fe limitation:", lim_Fe)
print("Overall nutrient limitation:", mu_nut)

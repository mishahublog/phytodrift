# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import numpy as np
import logging; logger = logging.getLogger(__name__)

from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED


# Defining the oil element properties
class PelagicEgg(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for pelagic eggs
    """

    variables = Lagrangian3DArray.add_variables([
        ('diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}),  # for NEA Cod
        ('neutral_buoyancy_salinity', {'dtype': np.float32,
                                       'units': '[]',
                                       'default': 31.25}),  # for NEA Cod
        ('density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),
        ('hatched', {'dtype': np.float32,
                     'units': '',
                     'default': 0.})])


class PelagicEggDrift(OceanDrift):
    """Buoyant particle trajectory model based on the OpenDrift framework.
    """

    ElementType = PelagicEgg

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_ice_area_fraction': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 100},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'surface_downward_x_stress': {'fallback': 0},
        'surface_downward_y_stress': {'fallback': 0},
        'turbulent_kinetic_energy': {'fallback': 0},
        'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'r': {'fallback': 0},
        'K': {'fallback': 0}
      }

    status_colors = {'initial': 'green', 'active': 'blue',
                     'hatched': 'red', 'eaten': 'yellow', 'died': 'magenta'}

    def __init__(self, *args, **kwargs):
        super(PelagicEggDrift, self).__init__(*args, **kwargs)
        self._set_config_default('general:coastline_action', 'previous')
        self._set_config_default('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_mixing_at_surface', True)
        self._set_config_default('drift:vertical_advection_at_surface', True)

    def update_terminal_velocity(self, Tprofiles=None,
                                 Sprofiles=None, z_index=None):
        g = 9.81  # ms-2

        eggsize = self.elements.diameter
        eggsalinity = self.elements.neutral_buoyancy_salinity

        if not (Tprofiles is None and Sprofiles is None):
            if z_index is None:
                z_i = range(Tprofiles.shape[0])
                z_index = interp1d(-self.environment_profiles['z'],
                                   z_i, bounds_error=False)
            zi = z_index(-self.elements.z)
            upper = np.maximum(np.floor(zi).astype(np.uint8), 0)
            lower = np.minimum(upper+1, Tprofiles.shape[0]-1)
            weight_upper = 1 - (zi - upper)

        if Tprofiles is None:
            T0 = self.environment.sea_water_temperature
        else:
            T0 = Tprofiles[upper, range(Tprofiles.shape[1])] * weight_upper + \
                 Tprofiles[lower, range(Tprofiles.shape[1])] * (1-weight_upper)
        if Sprofiles is None:
            S0 = self.environment.sea_water_salinity
        else:
            S0 = Sprofiles[upper, range(Sprofiles.shape[1])] * weight_upper + \
                 Sprofiles[lower, range(Sprofiles.shape[1])] * (1-weight_upper)

        DENSw = self.sea_water_density(T=T0, S=S0)
        DENSegg = self.sea_water_density(T=T0, S=eggsalinity)
        dr = DENSw - DENSegg

        my_w = 0.001*(1.7915 - 0.0538*T0 + 0.007*T0*T0 - 0.0023*S0)
        W = (1.0/my_w)*(1.0/18.0)*g*(eggsize**2) * dr

        highRe = np.where(W*1000*eggsize/my_w > 0.5)

        my_w = 0.01854 * np.exp(-0.02783 * T0)
        d0 = (eggsize*100) - 0.4 * (9.0 * my_w**2 /
                                   (100 * g) * DENSw / dr)**(1.0/3.0)
        W2 = 19.0*d0*(0.001*dr)**(2.0/3.0)*(my_w*0.001*DENSw)**(-1.0/3.0)
        W2 = W2/100.

        W[highRe] = W2[highRe]
        self.elements.terminal_velocity = W

    # ============================================================
    # NEW SECTION: LOGISTIC GROWTH FOR EGG POPULATION
    # ============================================================
    def logistic_growth(self, r=0.0005, K=5000):

        """Logistic growth of egg population.

        Parameters
        ----------
        r : float
            Intrinsic growth rate (per second).
        K : float
            Carrying capacity (maximum number of eggs).
        """

        N = self.elements.
        dNdt = r * N * (1 - N / K)
        N_new = N + dNdt * self.dt_seconds

        self.elements.number = N_new



    def update(self):
        """Update positions and properties of buoyant particles."""

        # Turbulent Mixing
        self.update_terminal_velocity()
        self.vertical_mixing()

        # Horizontal advection
        self.advect_ocean_current()

        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()

        # -------------------------------
        # Add logistic growth each step
        # -------------------------------
        self.logistic_growth()

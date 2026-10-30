'''
Purpose
-------
Define the physical constants, magnetic geometry, and model parameter
dictionaries used by the diagnostic diagrams.

Inputs
------
None

Outputs
-------
params: dict of dict
    Model settings indexed by solver model key. Each entry includes
    polynomial storage orders, a title, and the required physical inputs.
cs, dz, g, gamma, H, N, tau, wac, B, epsilon, phi, theta, a, ax, ay, az:
    Shared defaults used to construct the model dictionaries.

Notes
-----
Distances and speeds are expressed in km and km/s; angular frequencies
are expressed in rad/s and field angles in radians. N is the legacy
Brunt-Vaisala frequency input; DiagnosticDiagram derives N2 as N**2
when an explicit N2 value is absent. The dictionaries are built at import.
Changing a shared constant later does not rebuild their existing values.
'''

# Import the numerical constants and trigonometric functions
from numpy import sin, cos, sqrt, pi

# Use same constants as @vesaMultiheightObservationsAtmospheric2023 Figure 3
cs = 7.8 # km/s
dz = 220  # km
g  = 0.274 # km/s^2
# Define the dimensionless specific-heat ratio for the default atmosphere
gamma = 5*3**(-1)  # dimensionless
H = 125.00 # km
N = sqrt((g*H**(-1)) - (g**2*cs**(-2))) # rad/s (angular frequency; legacy N input for notebooks)
tau = 300 # s
wac = cs*(2*H)**(-1) # rad/s (angular frequency)

# Magnetic parameters
B = 100 # Gauss
# Retain the magnetic-profile parameter used in model metadata
epsilon = 1.0  # dimensionless
phi = 10.0 * pi*(180)**(-1) # deg -> rads
theta = -80.0 * pi*(180)**(-1) # deg -> rads

# Calculate the Alfven velocity components (km/s)
a  = 0.0*cs # km/s
ax = a*sin(theta)*cos(phi)
ay = a*sin(theta)*sin(phi)
az = a*cos(theta)

# Define the input parameters
# Collect the acoustic-gravity inputs for the Souffrin model
params_sf1966 = {'cs': cs, 
                 'dz': dz,
                  'g': g,
                  'N': N,
                'wac': wac,
           'kz_order': 2,
        'omega_order': 4,
              'model': 'sf1966',
              'title': '(SF, 1966)'}

# Collect the adiabatic acoustic-gravity inputs with an explicit scale height
params_mt1981 = {'cs': cs, 
                 'dz': dz,
                  'g': g,
                  'H': H,
                  'N': N,
           'kz_order': 2,
        'omega_order': 4,
              'model': 'mt1981',
              'title': '(M\\&T, 1981)'}

# Add the radiative relaxation time and specific-heat ratio
params_mt1982 = {'cs': cs, 
                 'dz': dz,
                  'g': g,
              'gamma': gamma,
                  'H': H,
                  'N': N,
                'tau': tau,
           'kz_order': 2,
        'omega_order': 8,
              'model': 'mt1982',
              'title': r'$\tau = \,$' + f'{tau:.0f}' + r's, (M\&T, 1982)' }

# Combine magnetic, atmospheric, and radiative inputs for Bunte and Bogdan
params_bunte1993 = {'a': a,
                    'B': B,
                   'cs': cs, 
                   'dz': dz,
              'epsilon': epsilon,
                    'g': g,
                'gamma': gamma,
                    'N': N,
                    'H': H,
                  'tau': tau,
                  'wac': wac,
             'kz_order': 2,
          'omega_order': 8,
                'model': 'bunte1993',
                'title': r'$a = \,$' + f'{a*cs**(-1):.2f}' + r'$c_s$' + r', $\tau = \,$' + f'{tau:.0f}' + r's, $\epsilon = \,$' + f'{epsilon:.1f}' + ' (B\\&B, 1993)'}

# Supply the full field vector and orientation for the coupled MHD model
params_nc2009 = {'a': a,
                'ax': ax,
                'ay': ay,
                'az': az,
                'cs': cs, 
                'dz': dz,
                 'g': g,
                 'N': N, 
               'phi': phi,
             'theta': theta, 
               'wac': wac,
          'kz_order': 6,
       'omega_order': 6,
             'model': 'nc2009',
             'title': r'$a = \,$' + f'{a*cs**(-1):.2f}' + r'$c_s$' +  r', $\theta = \,$' + f'{theta*180*pi**(-1):.0f}' + r'$^\circ$, $\phi = \,$' + f'{phi*180*pi**(-1):.0f}' + r'$^\circ$ (N\&C, 2009)'}

# Define the homogeneous Alfven relation and its two vertical-root slots
params_alfven_mode = {'a': a,
                    'cs': cs,
                    'dz': dz,
                    'g': g,
                    'phi': phi,
                    'theta': theta,
                    'kz_order': 2,
                    'omega_order': 2,
                    'model': 'alfven_mode',
                    'title': r'$a = \, $' + f'{a*cs**(-1):.2f}' + r'$c_s$, $\theta = \, $' + f'{theta*180*pi**(-1):.0f}' + r'$^\circ$, $\phi = \, $' + f'{phi*180*pi**(-1):.0f}' + r'$^\circ$ (Alfven mode)'}

# Reserve four vertical roots and two nonnegative frequency boundaries
params_slow_fast_mode = {'a': a,
                    'cs': cs,
                    'dz': dz,
                    'g': g,
                    'phi': phi,
                    'theta': theta,
                    'kz_order': 4,
                    'omega_order': 2,
                    'model': 'slow_fast_mode',
                    'title': r'$a = \, $' + f'{a*cs**(-1):.2f}' + r'$c_s$, $\theta = \, $' + f'{theta*180*pi**(-1):.0f}' + r'$^\circ$, $\phi = \, $' + f'{phi*180*pi**(-1):.0f}' + r'$^\circ$ (Slow-fast mode)'}

# Select the slow family from the homogeneous magnetoacoustic root set
params_slow_mode = {**params_slow_fast_mode, 'a': 3.0, 'kz_order': 2,
                    'omega_order': 1, 'model': 'slow_mode',
                    'title': 'Exact homogeneous-MHD slow mode'}
# Reuse the slow-mode physical defaults while selecting the fast family
params_fast_mode = {**params_slow_mode, 'model': 'fast_mode',
                    'title': 'Exact homogeneous-MHD fast mode'}

# Dictionary of all models
params = {'sf1966': params_sf1966,     # @schatzmanWavesSolarAtmosphere1967
          'mt1981': params_mt1981,     # @mihalasInternalGravityWaves1981
          'mt1982': params_mt1982,     # @mihalasInternalGravityWaves1982
       'bunte1993': params_bunte1993,  # @bunteMagnetoatmosphericWavesSubject1994
          'nc2009': params_nc2009,     # @newingtonReflectionConversionMagnetogravity2010
      'alfven_mode': params_alfven_mode,
       'slow_mode': params_slow_mode,
       'fast_mode': params_fast_mode,
   'slow_fast_mode': params_slow_fast_mode}

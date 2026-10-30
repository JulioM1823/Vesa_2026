# Import the relevant libraries
import heapq
import numpy as np
from numpy import sin, cos, sqrt, pi
from scipy import optimize

class DiagnosticDiagram:

    '''
    Purpose
    -------
    Construct dispersion-relation solvers and diagnostics on a k-omega grid.

    The first row of kh_grid and first column of omega_grid define the coordinate
    axes. Model parameters use km, s, and radians; signed N2 is retained when
    provided, and legacy N is squared otherwise.

    Inputs
    ------
    kh_grid: np.array, float
        Horizontal-wavenumber mesh (n_omega, n_kh), in km^-1.

    omega_grid: np.array, float
        Angular-frequency mesh matching kh_grid, in rad/s.

    params: dict
        Model identifier, physical parameters, and configured root counts.

    Outputs
    -------
    diagram: DiagnosticDiagram
        Stateful solver exposing boundary, branch-tracking, and plotting methods.
    '''

    def __init__(self, kh_grid, omega_grid, params):

        '''
        Purpose
        -------
        Store the sampled grid and initialize parameters for the selected model.

        Inputs
        ------
        kh_grid: np.array, float
            Horizontal-wavenumber mesh (n_omega, n_kh), in km^-1.

        omega_grid: np.array, float
            Angular-frequency mesh matching kh_grid, in rad/s.

        params: dict
            Physical parameters and model name; copied before legacy N conversion.

        Outputs
        -------
        None
            Store coordinate axes and model attributes on this instance. Unsupported
            model names raise ValueError.
        '''

        # Store the grid axes and copy the model parameters
        self.kh = kh_grid[0]  # km^-1
        self.omega  = omega_grid[:, 0]  # rad/s
        self.params = dict(params)
        # Legacy callers may supply N; all equations consume signed N2.
        if 'N2' not in self.params and 'N' in self.params:
            self.params['N2'] = self.params['N']**2
        self.model  = self.params['model']
        self.cs = self.params['cs']  # km/s
        self.g  = self.params['g']   # km/s^2

        # Initialize parameters according to the model
        # Load the buoyancy frequency and acoustic cutoff for the SF model.
        if self.model == 'sf1966':
            self.N2, self.wac = self.params['N2'], self.params['wac']
        # Load the isothermal scale height and buoyancy term for MT1981.
        elif self.model == 'mt1981':
            self.H, self.N2 = self.params['H'], self.params['N2']
        # Add the heat-capacity ratio and radiative relaxation time for MT1982.
        elif self.model == 'mt1982':
            self.gamma, self.N2, self.H, self.tau = self.params['gamma'], self.params['N2'], self.params['H'], self.params['tau']
        # Load the magnetic and radiative parameters for the Bunte model.
        elif self.model == 'bunte1993':
            self.a, self.epsilon, self.gamma, self.H, self.tau, self.wac = self.params['a'], self.params['epsilon'], self.params['gamma'], self.params['H'], self.params['tau'], self.params['wac']
        # Load the Cartesian field components and orientation for NC2009.
        elif self.model == 'nc2009':
            self.a, self.ax, self.ay, self.az, self.theta, self.phi, self.N2, self.wac = self.params['a'], self.params['ax'], self.params['ay'], self.params['az'], self.params['theta'], self.params['phi'], self.params['N2'], self.params['wac']
        # Load the speed and field direction for the analytic Alfvén branch.
        elif self.model == 'alfven_mode':
            self.a, self.theta, self.phi = self.params['a'], self.params['theta'], self.params['phi']
        # Use the same magnetic geometry for the magnetoacoustic model variants.
        elif self.model in ('slow_mode', 'fast_mode', 'slow_fast_mode'):
            self.a, self.theta, self.phi = self.params['a'], self.params['theta'], self.params['phi']
        else:
            raise ValueError("Model not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009', 'alfven_mode', 'slow_mode', or 'fast_mode'.")

    @staticmethod
    def _finite_complex_mask(values):

        '''
        Purpose
        -------
        Return a mask selecting values with finite real and imaginary parts.

        Inputs
        ------
        values: array-like
            Real or complex values to check.

        Outputs
        -------
        finite: np.array, bool
            Mask with the input shape; both real and imaginary parts must be finite.
        '''

        # Convert array-like input before checking both complex components.
        values = np.asarray(values)
        return np.isfinite(values.real) & np.isfinite(values.imag)

    @staticmethod
    def _sorted_finite_indices(values):

        '''
        Purpose
        -------
        Return finite-value indices sorted by real part, then imaginary part.

        Inputs
        ------
        values: array-like
            One-dimensional real or complex root set.

        Outputs
        -------
        indices: np.array, int
            Indices of finite entries, sorted by real part and then imaginary part.
            An entirely nonfinite root set returns an empty array.
        '''

        # Discard nonfinite root slots while retaining their original indices.
        values = np.asarray(values)
        finite = DiagnosticDiagram._finite_complex_mask(values)
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size == 0:
            return finite_indices

        # Use the imaginary part only to break equal-real-part ties.
        finite_values = values[finite_indices]
        sort_order = np.lexsort((finite_values.imag, finite_values.real))
        return finite_indices[sort_order]

    @staticmethod
    def _min_pairwise_distance(values):

        '''
        Purpose
        -------
        Return the minimum distance between finite values, or NaN if unavailable.

        Inputs
        ------
        values: array-like
            One-dimensional root set; nonfinite entries are ignored.

        Outputs
        -------
        distance: float
            Smallest complex-plane distance in the input units, or NaN when fewer
            than two finite roots are available.
        '''

        # Restrict the distance calculation to finite roots.
        values = np.asarray(values)
        finite_values = values[DiagnosticDiagram._finite_complex_mask(values)]
        if finite_values.size < 2:
            return np.nan

        # Construct all pairwise distances and exclude self-distances.
        distances = np.abs(finite_values[:, None] - finite_values[None, :])
        distances[np.diag_indices_from(distances)] = np.inf
        return np.min(distances)

    @staticmethod
    def _trim_polynomial_coefficients(coefficients, rtol=1e-12, atol=0.0):

        '''
        Purpose
        -------
        Remove negligible leading coefficients and identify invalid or zero polynomials.

        Inputs
        ------
        coefficients: array-like
            Polynomial coefficients in descending powers.

        rtol: float, optional
            Relative threshold multiplied by the largest coefficient magnitude.

        atol: float, optional
            Absolute coefficient threshold; the larger absolute/relative value is used.

        Outputs
        -------
        trimmed: np.array, complex or None
            Coefficients after negligible leading terms are removed. Return None for
            nonfinite coefficients, an empty array for empty input, or [0j] for a
            polynomial whose coefficients all lie below the tolerance.
        '''

        # Normalize the coefficient type and reject nonfinite input.
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.size == 0:
            return coefficients
        if not np.all(DiagnosticDiagram._finite_complex_mask(coefficients)):
            return None

        # Find the first leading term above the absolute/relative threshold.
        scale = np.max(np.abs(coefficients))
        tolerance = max(atol, rtol*scale)
        first = 0
        while first < coefficients.size - 1 and np.abs(coefficients[first]) <= tolerance:
            first += 1

        # Retain a single zero coefficient for an identically negligible polynomial.
        trimmed = coefficients[first:]
        if np.all(np.abs(trimmed) <= tolerance):
            return np.asarray([0.0 + 0.0j])

        return trimmed

    @staticmethod
    def _grid_component(values, component):

        '''
        Purpose
        -------
        Extract the requested real, imaginary, magnitude, or phase component.

        Inputs
        ------
        values: array-like
            Complex grid or root values.

        component: str
            Real, imaginary, magnitude, or phase selector; common aliases are accepted.

        Outputs
        -------
        component_values: np.array
            Selected component with the input shape. Phase is in radians; all other
            components retain the input units. Unsupported selectors raise ValueError.
        '''

        # Normalize the selector and extract the requested complex-grid component.
        component = component.lower()
        if component in ('real', 're'):
            return np.real(values)
        if component in ('imag', 'imaginary', 'im'):
            return np.imag(values)
        if component in ('abs', 'absolute', 'magnitude'):
            return np.abs(values)
        if component in ('phase', 'angle', 'arg'):
            return np.angle(values)
        raise ValueError("component must be one of 'real', 'imag', 'abs', or 'phase'.")

    @staticmethod
    def _update_max_map(target, values):

        '''
        Purpose
        -------
        Update finite target cells with the elementwise maximum of new values.

        Inputs
        ------
        target: np.array, float
            Writable target map updated in place.

        values: np.array, float
            New values matching the target shape; nonfinite entries are ignored.

        Outputs
        -------
        None
            Replace finite targets by the larger value and initialize previously
            nonfinite targets from finite new values.
        '''

        # Update only finite incoming values, initializing missing target cells.
        valid = np.isfinite(values)
        current_valid = np.isfinite(target)
        target[valid & current_valid] = np.maximum(target[valid & current_valid], values[valid & current_valid])
        target[valid & ~current_valid] = values[valid & ~current_valid]

    def fmode_dispersion(self):

        '''
        Purpose
        -------
        Calculate the theoretical frequency of the f-mode as a function of wavenumber
        (k).

        Inputs
        ------
        None

        Outputs
        -------
        omega: np.array, float
            Theoretical f-mode angular frequency (rad/s), with one value per self.kh
            sample, using omega = sqrt(g*kh).

        Author(s)
        ---------
        Julio M. Morales, August 30, 2022
        '''

        # Calculate omega
        omega = sqrt(self.g*np.array(self.kh)) # rad/s

        return omega
    
    def omega_coefficients(self, kh):

        '''
        Purpose
        -------
        Construct polynomial coefficients in descending powers of omega.

        Inputs
        ------
        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        coefficients: np.array, complex
            Model coefficients in descending omega powers, ordered for np.roots and
            np.polyval. For mt1982, this method sets self.a to zero before constructing
            the nonmagnetic coefficients.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # Determine the model
        # Construct the acoustic-gravity quartic with the explicit acoustic cutoff.
        if (self.model == 'sf1966'):
            c4 = self.cs**(-2)
            c3 = 0
            c2 = -(kh**2 + self.wac**2*(self.cs)**(-2))
            c1 = 0
            c0 = self.N2*kh**2
            coefficients = [c4, c3, c2, c1, c0]
        # Express the isothermal acoustic cutoff through the scale height.
        elif (self.model == 'mt1981'):
            c4 = self.cs**(-2)
            c3 = 0
            c2 = -(kh**2 + (4*self.H**2)**(-1))
            c1 = 0
            c0 = self.N2*kh**2
            coefficients = [c4, c3, c2, c1, c0]
        # Construct the eighth-order radiative/magnetic boundary polynomial.
        elif (self.model == 'bunte1993') or (self.model == 'mt1982'):
            # Recover the nonmagnetic radiative case by setting the Alfvén speed to zero.
            if (self.model == 'mt1982'):
                self.a = 0
            # Store the even-power coefficients; odd powers vanish in this boundary equation.
            c8 = 4*self.H**2*self.a**2*self.gamma**2*self.tau**2 + 4*self.H**2*self.cs**2*self.gamma**2*self.tau**2
            c7 = 0
            c6 = -4*self.H**2*self.a**4*self.gamma**2*kh**2*self.tau**2 - 12*self.H**2*self.a**2*self.cs**2*self.gamma**2*kh**2*self.tau**2 + 4*self.H**2*self.a**2*self.gamma**2 - 4*self.H**2*self.cs**4*self.gamma**2*kh**2*self.tau**2 + 4*self.H**2*self.cs**2*self.gamma - self.a**4*self.gamma**2*self.tau**2 - 2*self.a**2*self.cs**2*self.gamma**2*self.tau**2 - self.cs**4*self.gamma**2*self.tau**2
            c5 = 0
            c4 = 8*self.H**2*self.a**4*self.cs**2*self.gamma**2*kh**4*self.tau**2 - 4*self.H**2*self.a**4*self.gamma**2*kh**2 + 8*self.H**2*self.a**2*self.cs**4*self.gamma**2*kh**4*self.tau**2 - 12*self.H**2*self.a**2*self.cs**2*self.gamma*kh**2 - 4*self.H**2*self.a**2*self.g**2*self.gamma**2*kh**2*self.tau**2 - 4*self.H**2*self.cs**4*kh**2 - 4*self.H**2*self.cs**2*self.g**2*self.gamma**2*kh**2*self.tau**2 + 4*self.H*self.a**2*self.cs**2*self.g*self.gamma**2*kh**2*self.tau**2 + 4*self.H*self.cs**4*self.g*self.gamma**2*kh**2*self.tau**2 + 2*self.a**4*self.cs**2*self.gamma**2*kh**2*self.tau**2 - self.a**4*self.gamma**2 + 2*self.a**2*self.cs**4*self.gamma**2*kh**2*self.tau**2 - 2*self.a**2*self.cs**2*self.gamma - self.cs**4
            c3 = 0
            c2 = -4*self.H**2*self.a**4*self.cs**4*self.gamma**2*kh**6*self.tau**2 + 8*self.H**2*self.a**4*self.cs**2*self.gamma*kh**4 + 8*self.H**2*self.a**2*self.cs**4*kh**4 + 4*self.H**2*self.a**2*self.cs**2*self.g**2*self.gamma**2*kh**4*self.tau**2 - 4*self.H**2*self.a**2*self.g**2*self.gamma**2*kh**2 - 4*self.H**2*self.cs**2*self.g**2*self.gamma*kh**2 - 4*self.H*self.a**2*self.cs**4*self.g*self.gamma**2*kh**4*self.tau**2 + 4*self.H*self.a**2*self.cs**2*self.g*self.gamma*kh**2 + 4*self.H*self.cs**4*self.g*kh**2 - self.a**4*self.cs**4*self.gamma**2*kh**4*self.tau**2 + 2*self.a**4*self.cs**2*self.gamma*kh**2 + 2*self.a**2*self.cs**4*kh**2
            c1 = 0
            c0 = -4*self.H**2*self.a**4*self.cs**4*kh**6 + 4*self.H**2*self.a**2*self.cs**2*self.g**2*self.gamma*kh**4 - 4*self.H*self.a**2*self.cs**4*self.g*kh**4 - self.a**4*self.cs**4*kh**4
            coefficients = [c8, c7, c6, c5, c4, c3, c2, c1, c0]
        # Construct the sixth-order boundary equation using Cartesian field components.
        elif (self.model == 'nc2009'):
            c6 = 1
            c5 = 0
            c4 = -self.ax**2*kh**2 - self.a**2*kh**2 - self.cs**2*kh**2 - self.wac**2
            c3 = 0
            c2 = self.N2*self.cs**2*kh**2 + self.a**2*self.ax**2*kh**4 + 2*self.ax**2*self.cs**2*kh**4 + self.ay**2*kh**2*self.wac**2 + self.ax**2*kh**2*self.wac**2 + self.az**2*kh**2*self.wac**2
            c1 = 0
            c0 = -self.N2*self.ax**2*self.cs**2*kh**4 - self.ax**4*self.cs**2*kh**6 - self.ax**2*self.az**2*kh**4*self.wac**2
            coefficients = [c6, c5, c4, c3, c2, c1, c0]
        # Use the field-aligned horizontal projection for the Alfvén boundary.
        elif (self.model == 'alfven_mode'):
            alfven_frequency = self.a*kh*np.sin(self.theta)*np.cos(self.phi)
            c2 = 1
            c1 = 0
            c0 = -alfven_frequency**2
            coefficients = [c2, c1, c0]
        # Retain both magnetoacoustic frequency families in the quartic coefficients.
        elif (self.model in ('slow_mode', 'fast_mode', 'slow_fast_mode')):
            c4 = 1
            c3 = 0
            c2 = -self.a**2*kh**2 - self.cs**2*kh**2
            c1 = 0
            c0 = self.a**2*self.cs**2*kh**4*sin(self.theta)**2*cos(self.phi)**2
            coefficients = [c4, c3, c2, c1, c0]
        else:
            raise ValueError("Model for omega not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009', 'alfven_mode', 'slow_mode', or 'fast_mode'.")

        # Return descending powers with a consistent complex dtype for all models.
        return np.asarray(coefficients, dtype=np.complex128)

    def omega_poly(self, omega, kh):

        '''
        Purpose
        -------
        Evaluate the omega-boundary polynomial at omega and kh.

        Inputs
        ------
        omega: float or np.array
            Angular frequency (rad/s); arrays follow NumPy broadcasting where supported.

        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        poly: scalar or np.array
            Omega-boundary polynomial evaluated with NumPy broadcasting; nearly real
            results are converted from complex values by np.real_if_close.
        '''

        # Evaluate the boundary polynomial and remove negligible imaginary roundoff.
        return np.real_if_close(np.polyval(self.omega_coefficients(kh), omega))
        
    def kz_coefficients(self, omega, kh):

        '''
        Purpose
        -------
        Calculate the polynomial coefficients in descending powers of kz.

        Inputs
        ------
        omega: float
            Angular frequency at the polynomial solve/evaluation point (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        coefficients: np.array, complex
            Model coefficients in descending kz powers, ordered for np.roots and
            np.polyval. Leading coefficients can vanish at degenerate field geometries.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # A right angle is represented by a small nonzero cosine in floating
        # point. Snap only this geometric degeneracy, before forming powers of
        # cos(theta) that determine the polynomial degree.
        cos_theta = cos(self.theta) if hasattr(self, 'theta') else None
        if cos_theta is not None and abs(cos_theta) <= 1e-14:
            cos_theta = 0.0

        # Determine the model
        # Move the SF acoustic-gravity dispersion relation into kz-polynomial form.
        if (self.model == 'sf1966'):
            c2 = 1
            c1 = 0
            c0 = kh**2*(omega**2 - self.N2)*omega**(-2) - (omega**2 - self.wac**2)*self.cs**(-2)
            coefficients = [c2, c1, c0]
        # Use the scale-height cutoff in the adiabatic quadratic.
        elif (self.model == 'mt1981'):
            c2 = 1
            c1 = 0
            c0 = kh**2 + (4*self.H**2)**(-1) - self.N2*kh**2*omega**(-2) - omega**2*self.cs**(-2)
            coefficients = [c2, c1, c0]
        # Evaluate the radiatively modified quadratic coefficient for MT1982.
        elif (self.model == 'mt1982'):
            c2 = 1
            c1 = 0
            c0 = -0.5*(self.N2*kh**2*omega**(-2) - kh**2 + (omega**2*self.tau**2*(-self.N2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))**2*(omega**2*self.tau**2 + 1)**(-2) + (self.N2*kh**2*omega**(-2) - kh**2 + (-self.N2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))**2)**(0.5) + (-self.N2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))
            coefficients = [c2, c1, c0]
        elif (self.model == 'bunte1993'):
            # Use the same rational dispersion relation at a=0 and a>0.
            # The former zero-field special case mixed incompatible cutoff
            # and buoyancy definitions and did not satisfy omega_coefficients.
            a0 = self.a
            p1 = (self.cs**2 + a0**2)*omega**2 - self.cs**2*a0**2*kh**2
            p_gamma = (self.cs**2 + self.gamma*a0**2)*omega**2 - self.cs**2*a0**2*kh**2
            g1 = omega**4 + self.g**2*(self.cs**2*(self.g*self.H)**(-1) - 1)*kh**2
            g_gamma = self.gamma*omega**4 + self.gamma*self.g**2*(self.cs**2*(self.gamma*self.g*self.H)**(-1) - 1)*kh**2
            c0 = (4*self.H**2)**(-1) + kh**2 - (p_gamma*g_gamma + (omega*self.tau*self.gamma)**2*p1*g1) / (p_gamma**2 + (omega*self.tau*self.gamma)**2*p1**2)
            coefficients = [1, 0, c0]
        # Expand the three-dimensional magnetic relation through sixth order in kz.
        elif (self.model == 'nc2009'):
            c6 = -self.a**4*self.cs**2*cos_theta**4
            c5 = -4*kh*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos_theta**3
            c4 = -6*kh**2*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos_theta**2 - kh**2*self.a**4*self.cs**2*cos_theta**4 + omega**2*self.a**4*cos_theta**2 + 2*omega**2*self.a**2*self.cs**2*cos_theta**2 - self.a**2*self.az**2*self.wac**2*cos_theta**2
            c3 = -4*kh**3*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos_theta - 4*kh**3*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos_theta**3 + 2*kh*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos_theta + 4*kh*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos_theta - 2*kh*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos_theta
            c2 = -kh**4*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 - 6*kh**4*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos_theta**2 + kh**2*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.a**4*cos_theta**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*cos_theta**2 - kh**2*self.N2*self.a**2*self.cs**2*cos_theta**2 - kh**2*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*self.a**2*self.az**2*self.wac**2*cos_theta**2 - omega**4*self.a**2*cos_theta**2 - omega**4*self.a**2 - omega**4*self.cs**2 + omega**2*self.a**2*self.wac**2*cos_theta**2 + omega**2*self.az**2*self.wac**2
            c1 = -4*kh**5*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos_theta + 2*kh**3*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos_theta + 4*kh**3*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos_theta - 2*kh**3*self.N2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos_theta - 2*kh**3*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos_theta - 2*kh*omega**4*self.a**2*sin(self.theta)*cos(self.phi)*cos_theta + 2*kh*omega**2*self.a**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos_theta
            c0 = -kh**6*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 + kh**4*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**4*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.N2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2 - kh**2*omega**4*self.cs**2 + kh**2*omega**2*self.N2*self.cs**2 + kh**2*omega**2*self.a**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.ay**2*self.wac**2 + kh**2*omega**2*self.az**2*self.wac**2 + omega**6 - omega**4*self.wac**2
            coefficients = [c6, c5, c4, c3, c2, c1, c0]
        # Construct the field-aligned Alfvén quadratic without dividing by vertical speed.
        elif (self.model == 'alfven_mode'):
            alfven_frequency = self.a*kh*np.sin(self.theta)*np.cos(self.phi)
            vertical_speed = self.a*cos_theta
            # Keep the original dispersion polynomial. Dividing by the
            # vertical speed creates infinities at a=0 or horizontal field.
            coefficients = [vertical_speed**2,
                            2*vertical_speed*alfven_frequency,
                            alfven_frequency**2 - omega**2]
        # Construct the homogeneous-MHD quartic containing both slow and fast roots.
        elif (self.model in ('slow_mode', 'fast_mode', 'slow_fast_mode')):
            c4 = self.a**2*self.cs**2*cos_theta**2
            c3 = 2*self.a**2*self.cs**2*kh*sin(self.theta)*cos(self.phi)*cos_theta
            c2 = self.a**2*self.cs**2*kh**2*sin(self.theta)**2*cos(self.phi)**2 + self.a**2*self.cs**2*kh**2*cos_theta**2 - self.a**2*omega**2 - self.cs**2*omega**2
            c1 = 2*self.a**2*self.cs**2*kh**3*sin(self.theta)*cos(self.phi)*cos_theta
            c0 = self.a**2*self.cs**2*kh**4*sin(self.theta)**2*cos(self.phi)**2 - self.a**2*kh**2*omega**2 - self.cs**2*kh**2*omega**2 + omega**4
            coefficients = [c4, c3, c2, c1, c0]
        else:
            raise ValueError("Model for kz not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009', 'alfven_mode', 'slow_mode', or 'fast_mode'.")

        # Keep descending powers and complex storage even for real-valued coefficients.
        return np.asarray(coefficients, dtype=np.complex128)

    def kz_poly(self, kz, omega, kh):

        '''
        Purpose
        -------
        Calculate the polynomial in kz.

        Inputs
        ------
        kz: float, complex or np.array
            Vertical wavenumber (km^-1).

        omega: float
            Angular frequency at the polynomial solve/evaluation point (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        poly: scalar or np.array, complex
            Value of the model kz polynomial, with the broadcast shape of its inputs.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # Construct the model coefficients and evaluate the polynomial at the supplied kz.
        coefficients = self.kz_coefficients(omega, kh)
        poly = np.polyval(coefficients, kz)

        return poly
    
    def omega_solve(self, kh):

        '''
        Purpose
        -------
        Calculate one legacy, single-kh boundary row.

        Inputs
        ------
        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        omega_bounds: np.array, float
            Nonnegative roots found by sign scanning (rad/s), with omega_order slots.
            Unused slots and failed Brent solves are returned as zero by this legacy
            compatibility method; it does not continuity-track branches.

        Author(s)
        ---------
        Julio M. Morales, July 30th, 2025
        '''

        # Define a fine grid for omega
        fine_grid = np.linspace(max(0.0, min(self.omega)), max(self.omega), 5_000)  # rad/s

        # List to store roots
        roots = np.zeros(self.params['omega_order'])

        # Compute the polynomial being solved on a fine grid and find sign changes
        poly = self.omega_poly(fine_grid, kh)
        sign_changes = np.nonzero(np.diff(np.sign(poly)))[0]
        
        # Loop through the sign changes to find roots by brentq method
        for ind, idx in enumerate(sign_changes):
            A0, B0 = fine_grid[idx], fine_grid[idx + 1]
            try:
                root = optimize.brentq(self.omega_poly, A0, B0, args=(kh,))
                roots[ind] = root
            except ValueError:
                root = np.nan
                roots[ind] = np.nan

        return np.nan_to_num(roots, nan=0.0)

    def _omega_raw_root_count(self):

        '''
        Purpose
        -------
        Return storage width for the complete direct-solver root set.

        Inputs
        ------
        None

        Outputs
        -------
        root_count: int
            Largest coefficient degree found over self.kh, bounded below by the
            configured omega_order; coefficient-construction failures are skipped.
        '''

        # Reserve at least the configured public boundary width.
        root_count = int(self.params['omega_order'])
        # Inspect every coefficient degree to retain negative and complex partner roots.
        for kh_i in self.kh:
            try:
                coefficients = np.asarray(self.omega_coefficients(kh_i)).reshape(-1)
            except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
                continue
            root_count = max(root_count, max(coefficients.size - 1, 0))

        return root_count

    def _solve_omega_roots_from_coefficients(
        self,
        coefficients,
        root_count,
        coefficient_rtol=1e-12,
        coefficient_atol=0.0
    ):

        '''
        Purpose
        -------
        Solve one omega polynomial and return padded roots with solver diagnostics.

        Inputs
        ------
        coefficients: array-like
            One omega polynomial in descending powers.

        root_count: int
            Length of the padded complex output.

        coefficient_rtol: float, optional
            Relative threshold for trimming leading omega-polynomial coefficients.

        coefficient_atol: float, optional
            Absolute threshold for trimming leading omega-polynomial coefficients.

        Outputs
        -------
        roots: np.array, complex
            Complete polynomial roots with NaN padding for unavailable roots.

        info: dict
            Coefficient validity and scale, nominal/effective degree, trimmed term
            count, zero-polynomial status, solver failure, and nonfinite-root count.
            Invalid or zero polynomials retain NaN roots and flag the failure.
        '''

        # Initialize padded roots and explicit coefficient/solver status fields.
        roots = np.full(root_count, np.nan + 0.0j, dtype=np.complex128)
        info = {
            'coefficient_valid': False,
            'coefficient_degree': -1,
            'effective_degree': -1,
            'trimmed_leading_count': 0,
            'coefficient_scale': np.nan,
            'zero_polynomial': False,
            'solver_failed': False,
            'nonfinite_root_count': 0
        }

        # Coerce input to a flat complex vector before checking polynomial validity.
        try:
            coefficients = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
        except (TypeError, ValueError):
            info['solver_failed'] = True
            return roots, info

        # Reject empty or nonfinite coefficient vectors without attempting a solve.
        info['coefficient_degree'] = max(coefficients.size - 1, 0)
        if coefficients.size == 0 or not np.all(self._finite_complex_mask(coefficients)):
            info['solver_failed'] = True
            return roots, info

        # Measure the coefficient scale and detect the identically negligible polynomial.
        info['coefficient_valid'] = True
        coefficient_scale = float(np.max(np.abs(coefficients)))
        info['coefficient_scale'] = coefficient_scale
        zero_tolerance = max(coefficient_atol, coefficient_rtol*coefficient_scale)
        if np.all(np.abs(coefficients) <= zero_tolerance):
            info['zero_polynomial'] = True
            info['effective_degree'] = 0
            info['trimmed_leading_count'] = max(coefficients.size - 1, 0)
            info['solver_failed'] = True
            return roots, info

        # Remove negligible leading terms and record the resulting effective degree.
        trimmed = self._trim_polynomial_coefficients(
            coefficients, rtol=coefficient_rtol, atol=coefficient_atol
        )
        if trimmed is None:
            info['solver_failed'] = True
            return roots, info

        info['trimmed_leading_count'] = coefficients.size - trimmed.size
        info['effective_degree'] = max(trimmed.size - 1, 0)
        if trimmed.size <= 1:
            return roots, info

        try:
            # Normalize the retained coefficients before forming the companion-matrix roots.
            scale = np.max(np.abs(trimmed))
            normalized = trimmed / scale
            point_roots = np.roots(normalized)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            info['solver_failed'] = True
            return roots, info

        # Keep finite roots in solver order and preserve unused NaN padding.
        finite = self._finite_complex_mask(point_roots)
        info['nonfinite_root_count'] = int(np.count_nonzero(~finite))
        if not np.all(finite):
            info['solver_failed'] = True
        point_roots = point_roots[finite]
        roots[:min(root_count, point_roots.size)] = point_roots[:root_count]
        return roots, info

    def _solve_omega_roots_at_point(
        self,
        kh,
        root_count=None,
        coefficient_rtol=1e-12,
        coefficient_atol=0.0,
        return_diagnostics=False
    ):

        '''
        Purpose
        -------
        Solve all omega-polynomial roots directly at one kh sample.

        Inputs
        ------
        kh: float
            Horizontal wavenumber (km^-1).

        root_count: int or None, optional
            Padded output width; infer the complete model root count when None.

        coefficient_rtol: float, optional
            Relative threshold for trimming leading omega-polynomial coefficients.

        coefficient_atol: float, optional
            Absolute threshold for trimming leading omega-polynomial coefficients.

        return_diagnostics: bool, optional
            Return the result and its diagnostic dictionary when True.

        Outputs
        -------
        roots: np.array, complex
            Complete polynomial roots with NaN padding for unavailable roots.

        info: dict, optional
            Per-polynomial metadata returned only when return_diagnostics is True.
            Coefficient-construction failures are passed to the solver as NaN input.
        '''

        # Infer storage for the full model root set when the caller does not supply it.
        if root_count is None:
            root_count = self._omega_raw_root_count()

        # Convert coefficient-construction failures into an explicit invalid polynomial.
        try:
            coefficients = self.omega_coefficients(kh)
        except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
            coefficients = np.asarray([np.nan + 0.0j])

        # Solve the point polynomial and optionally return its status metadata.
        roots, info = self._solve_omega_roots_from_coefficients(
            coefficients,
            root_count,
            coefficient_rtol=coefficient_rtol,
            coefficient_atol=coefficient_atol
        )
        if return_diagnostics:
            return roots, info
        return roots

    def _omega_solve_raw_grid(self, coefficient_rtol=1e-12, coefficient_atol=0.0):

        '''
        Purpose
        -------
        Solve the omega polynomial at every horizontal-wavenumber sample.

        Inputs
        ------
        coefficient_rtol: float, optional
            Relative threshold for trimming leading omega-polynomial coefficients.

        coefficient_atol: float, optional
            Absolute threshold for trimming leading omega-polynomial coefficients.

        Outputs
        -------
        raw_roots: np.array, complex
            Untracked omega roots (rad/s), shape (n_kh, raw_root_count), with NaN
            padding.

        solve_info: dict
            Per-kh coefficient/solver metadata arrays and the scalar raw_root_count.
        '''

        # Allocate all root slots and per-sample solve diagnostics.
        root_count = self._omega_raw_root_count()
        raw_roots = np.full(
            (len(self.kh), root_count), np.nan + 0.0j, dtype=np.complex128
        )
        solve_info = {
            'coefficient_valid': np.zeros(len(self.kh), dtype=bool),
            'coefficient_degree': np.full(len(self.kh), -1, dtype=int),
            'effective_degree': np.full(len(self.kh), -1, dtype=int),
            'trimmed_leading_count': np.zeros(len(self.kh), dtype=int),
            'coefficient_scale': np.full(len(self.kh), np.nan),
            'zero_polynomial': np.zeros(len(self.kh), dtype=bool),
            'solver_failed': np.zeros(len(self.kh), dtype=bool),
            'nonfinite_root_count': np.zeros(len(self.kh), dtype=int)
        }

        # Solve each horizontal-wavenumber sample without assigning global branch labels.
        for j, kh_j in enumerate(self.kh):
            point_roots, point_info = self._solve_omega_roots_at_point(
                kh_j,
                root_count=root_count,
                coefficient_rtol=coefficient_rtol,
                coefficient_atol=coefficient_atol,
                return_diagnostics=True
            )
            raw_roots[j, :] = point_roots
            for key in solve_info:
                solve_info[key][j] = point_info[key]

        # Attach the complete solver width alongside the per-sample metadata.
        solve_info['raw_root_count'] = root_count
        return raw_roots, solve_info

    def track_omega_branches(
        self,
        raw_roots,
        kh=None,
        degenerate_rtol=1e-6,
        degenerate_atol=1e-12,
        uncertainty_rtol=1e-3,
        uncertainty_atol=1e-12,
        return_diagnostics=False
    ):

        '''
        Purpose
        -------
        Assign one-dimensional omega root sets to continuous complex branches.

        Adjacent sets are matched one-to-one with the Hungarian algorithm and
        the complete complex distance. A branch that appears without a finite
        predecessor is reinitialized into a free label and explicitly marked;
        no maximum continuation distance is imposed.

        Inputs
        ------
        raw_roots: np.array, complex
            Complete untracked omega root sets (rad/s), shape (n_kh, branch_count).

        kh: np.array or None, optional
            One horizontal-wavenumber coordinate (km^-1) per row; use self.kh when None.

        degenerate_rtol: float, optional
            Relative root-separation tolerance used to flag near-coincident roots.

        degenerate_atol: float, optional
            Absolute root-separation tolerance in the units of the supplied roots.

        uncertainty_rtol: float, optional
            Relative assignment-margin threshold used to flag ambiguous continuation.

        uncertainty_atol: float, optional
            Absolute assignment-margin threshold in the units of the supplied roots.

        return_diagnostics: bool, optional
            Return the result and its diagnostic dictionary when True.

        Outputs
        -------
        tracked_roots: np.array, complex
            Continuity-assigned roots with the input shape and NaN for absent branches.

        diagnostics: dict, optional
            Local/raw branch mappings, connected-component labels, initialization
            masks, edge permutations and costs, physical-step cost rates, degeneracy,
            and uncertainty masks; returned only when return_diagnostics is True.
        '''

        # Validate the root-set array and its physical horizontal-wavenumber coordinates.
        raw_roots = np.asarray(raw_roots, dtype=np.complex128)
        if raw_roots.ndim != 2:
            raise ValueError("raw_roots must have shape (kh, branch).")

        n_kh, branch_count = raw_roots.shape
        if kh is None:
            kh = self.kh
        kh = np.asarray(kh, dtype=float)
        if kh.shape != (n_kh,):
            raise ValueError("kh must have one coordinate per raw-root row.")

        # Allocate branch mappings, component identities, and initialization masks.
        tracked_roots = np.full_like(raw_roots, np.nan + 0.0j)
        local_index_by_branch = np.full(raw_roots.shape, -1, dtype=int)
        raw_index_to_branch = np.full(raw_roots.shape, -1, dtype=int)
        branch_component_id = np.full(raw_roots.shape, -1, dtype=int)
        component_id = np.full(n_kh, -1, dtype=int)
        initialization_mask = np.zeros(raw_roots.shape, dtype=bool)
        reinitialization_mask = np.zeros(raw_roots.shape, dtype=bool)
        component_seed_mask = np.zeros(n_kh, dtype=bool)

        # Measure local finite-root counts and separation scales before matching neighbors.
        finite_root_count = np.zeros(n_kh, dtype=int)
        min_root_separation = np.full(n_kh, np.nan)
        local_scale = np.full(n_kh, degenerate_atol)
        # Inspect each row to establish the local root-separation scale.
        for j in range(n_kh):
            point_roots = raw_roots[j, self._finite_complex_mask(raw_roots[j])]
            finite_root_count[j] = point_roots.size
            if point_roots.size == 0:
                continue
            median_abs = float(np.median(np.abs(point_roots)))
            min_separation = self._min_pairwise_distance(point_roots)
            if np.isfinite(min_separation):
                min_root_separation[j] = min_separation
                local_scale[j] = max(median_abs, min_separation, degenerate_atol)
            else:
                local_scale[j] = max(median_abs, degenerate_atol)

        # Flag root collisions using the absolute and local relative separation tolerances.
        degenerate_threshold = degenerate_atol + degenerate_rtol*local_scale
        degenerate_mask = (
            (finite_root_count > 1)
            & np.isfinite(min_root_separation)
            & (min_root_separation <= degenerate_threshold)
        )

        # Allocate diagnostics for each adjacent pair of horizontal-wavenumber samples.
        edge_count = max(n_kh - 1, 0)
        edge_permutation = np.full((edge_count, branch_count), -1, dtype=int)
        edge_assignment_cost = np.full(edge_count, np.nan)
        edge_max_assignment_cost = np.full(edge_count, np.nan)
        edge_assignment_margin = np.full(edge_count, np.nan)
        edge_assignment_confidence = np.full(edge_count, np.nan)
        edge_matched_count = np.zeros(edge_count, dtype=int)
        kh_step = np.abs(np.diff(kh))
        assignment_cost_rate = np.full(edge_count, np.nan)
        max_assignment_cost_rate = np.full(edge_count, np.nan)
        zero_spacing_edge_mask = ~np.isfinite(kh_step) | (kh_step <= 0)

        # Track disconnected intervals independently while preserving branch-component IDs.
        next_branch_component = 0
        current_component = -1
        component_seeds = []

        for j in range(n_kh):
            # Visit finite root sets in coordinate order; empty rows break continuity.
            target_indices = self._sorted_finite_indices(raw_roots[j])
            if target_indices.size == 0:
                continue

            # Initialize deterministic labels at the start of each connected finite interval.
            starts_component = j == 0 or finite_root_count[j - 1] == 0
            if starts_component:
                current_component += 1
                component_id[j] = current_component
                component_seed_mask[j] = True
                component_seeds.append(j)
                for branch, target_index in enumerate(target_indices[:branch_count]):
                    local_index_by_branch[j, branch] = target_index
                    tracked_roots[j, branch] = raw_roots[j, target_index]
                    raw_index_to_branch[j, target_index] = branch
                    branch_component_id[j, branch] = next_branch_component
                    next_branch_component += 1
                    initialization_mask[j, branch] = True
                    if j > 0:
                        reinitialization_mask[j, branch] = True
                continue

            # Match adjacent complete root sets using one-to-one complex-plane assignment.
            component_id[j] = component_id[j - 1]
            permutation, stats = self._match_root_sets(raw_roots[j - 1], raw_roots[j])
            edge_permutation[j - 1, :] = permutation
            edge_assignment_cost[j - 1] = stats['assignment_cost']
            edge_max_assignment_cost[j - 1] = stats['max_assignment_cost']
            edge_assignment_margin[j - 1] = stats['assignment_margin']
            edge_matched_count[j - 1] = stats['matched_count']
            # Treat a unique root on each side as an unambiguous one-to-one continuation.
            if (
                stats['matched_count'] == 1
                and finite_root_count[j - 1] == 1
                and finite_root_count[j] == 1
            ):
                confidence = 1.0
            else:
                confidence = self._edge_confidence(
                    stats['max_assignment_cost'], stats['assignment_margin']
                )
            edge_assignment_confidence[j - 1] = confidence

            # Normalize assignment distances by the physical kh step where it is positive.
            if not zero_spacing_edge_mask[j - 1]:
                assignment_cost_rate[j - 1] = stats['assignment_cost']/kh_step[j - 1]
                max_assignment_cost_rate[j - 1] = stats['max_assignment_cost']/kh_step[j - 1]

            # Propagate existing branch identities through the assigned source-to-target
            # indices.
            used_target_indices = set()
            for branch, source_index in enumerate(local_index_by_branch[j - 1]):
                if source_index < 0:
                    continue
                target_index = permutation[source_index]
                if target_index < 0:
                    continue
                local_index_by_branch[j, branch] = target_index
                tracked_roots[j, branch] = raw_roots[j, target_index]
                raw_index_to_branch[j, target_index] = branch
                branch_component_id[j, branch] = branch_component_id[j - 1, branch]
                used_target_indices.add(int(target_index))

            # Reinitialize newly appearing roots only in free branch slots.
            unmatched_targets = [
                int(index) for index in target_indices if int(index) not in used_target_indices
            ]
            free_branches = np.flatnonzero(local_index_by_branch[j] < 0)
            for branch, target_index in zip(free_branches, unmatched_targets):
                local_index_by_branch[j, branch] = target_index
                tracked_roots[j, branch] = raw_roots[j, target_index]
                raw_index_to_branch[j, target_index] = branch
                branch_component_id[j, branch] = next_branch_component
                next_branch_component += 1
                reinitialization_mask[j, branch] = True

        # Combine assignment margins, confidence, and root spacing into edge ambiguity flags.
        uncertainty_threshold = uncertainty_atol + uncertainty_rtol*np.maximum(
            local_scale[:-1], local_scale[1:]
        )
        edge_ambiguous_mask = zero_spacing_edge_mask.copy()
        edge_ambiguous_mask |= (
            np.isfinite(edge_assignment_margin)
            & (edge_assignment_margin <= uncertainty_threshold)
        )
        edge_ambiguous_mask |= (
            np.isfinite(edge_assignment_confidence)
            & (edge_assignment_confidence < 0.2)
        )
        edge_separation = np.fmin(min_root_separation[:-1], min_root_separation[1:])
        edge_ambiguous_mask |= (
            np.isfinite(edge_max_assignment_cost)
            & np.isfinite(edge_separation)
            & (edge_max_assignment_cost >= 0.5*edge_separation)
        )

        # Mark both samples incident to each ambiguous continuation edge.
        assignment_ambiguous_mask = np.zeros(n_kh, dtype=bool)
        if edge_count > 0:
            assignment_ambiguous_mask[:-1] |= edge_ambiguous_mask
            assignment_ambiguous_mask[1:] |= edge_ambiguous_mask

        # Project the largest edge jump and smallest edge margin onto neighboring samples.
        root_set_jump = np.full(n_kh, np.nan)
        root_set_margin = np.full(n_kh, np.nan)
        for edge in range(edge_count):
            cost = edge_max_assignment_cost[edge]
            margin = edge_assignment_margin[edge]
            for point in (edge, edge + 1):
                if np.isfinite(cost):
                    if np.isfinite(root_set_jump[point]):
                        root_set_jump[point] = max(root_set_jump[point], cost)
                    else:
                        root_set_jump[point] = cost
                if np.isfinite(margin):
                    if np.isfinite(root_set_margin[point]):
                        root_set_margin[point] = min(root_set_margin[point], margin)
                    else:
                        root_set_margin[point] = margin

        # Align edge statistics with their destination rows for per-sample inspection.
        assignment_cost = np.full(n_kh, np.nan)
        max_assignment_cost = np.full(n_kh, np.nan)
        assignment_margin = np.full(n_kh, np.nan)
        assignment_confidence = np.full(n_kh, np.nan)
        matched_count = np.zeros(n_kh, dtype=int)
        if edge_count > 0:
            assignment_cost[1:] = edge_assignment_cost
            max_assignment_cost[1:] = edge_max_assignment_cost
            assignment_margin[1:] = edge_assignment_margin
            assignment_confidence[1:] = edge_assignment_confidence
            matched_count[1:] = edge_matched_count

        # Include degeneracy, missing roots, and reinitializations in the uncertainty mask.
        uncertain_mask = degenerate_mask | assignment_ambiguous_mask
        uncertain_mask |= finite_root_count < branch_count
        uncertain_mask |= np.any(reinitialization_mask, axis=1)

        # Return complete mappings and metrics without discarding negative or complex roots.
        diagnostics = {
            'tracked_roots': tracked_roots,
            'local_index_by_branch': local_index_by_branch,
            'raw_index_to_branch': raw_index_to_branch,
            'component_id': component_id,
            'branch_component_id': branch_component_id,
            'component_seeds': component_seeds,
            'component_seed_mask': component_seed_mask,
            'initialization_mask': initialization_mask,
            'reinitialization_mask': reinitialization_mask,
            'finite_root_count': finite_root_count,
            'min_root_separation': min_root_separation,
            'degenerate_threshold': degenerate_threshold,
            'degenerate_mask': degenerate_mask,
            'edge_permutation': edge_permutation,
            'edge_assignment_cost': edge_assignment_cost,
            'edge_max_assignment_cost': edge_max_assignment_cost,
            'edge_assignment_margin': edge_assignment_margin,
            'edge_assignment_confidence': edge_assignment_confidence,
            'edge_matched_count': edge_matched_count,
            'kh_step': kh_step,
            'assignment_cost_rate': assignment_cost_rate,
            'max_assignment_cost_rate': max_assignment_cost_rate,
            'zero_spacing_edge_mask': zero_spacing_edge_mask,
            'edge_ambiguous_mask': edge_ambiguous_mask,
            'assignment_ambiguous_mask': assignment_ambiguous_mask,
            'assignment_cost': assignment_cost,
            'max_assignment_cost': max_assignment_cost,
            'assignment_margin': assignment_margin,
            'assignment_confidence': assignment_confidence,
            'matched_count': matched_count,
            'root_set_jump': root_set_jump,
            'root_set_margin': root_set_margin,
            'uncertainty_threshold': uncertainty_threshold,
            'uncertain_mask': uncertain_mask
        }
        if return_diagnostics:
            return tracked_roots, diagnostics
        return tracked_roots

    def omega_root_residuals(self, roots, coefficient_rtol=1e-12, coefficient_atol=0.0):

        '''
        Purpose
        -------
        Calculate normalized omega-polynomial residuals in raw-root order.

        Inputs
        ------
        roots: np.array, complex
            Omega roots (rad/s), shape (n_kh, branch_count); any branch order is
            accepted.

        coefficient_rtol: float, optional
            Relative threshold for trimming leading omega-polynomial coefficients.

        coefficient_atol: float, optional
            Absolute threshold for trimming leading omega-polynomial coefficients.

        Outputs
        -------
        residuals: np.array, float
            Dimensionless absolute polynomial residual divided by the coefficient
            magnitude sum weighted by max(1, abs(root)) powers. Shape matches roots;
            unavailable evaluations remain NaN.
        '''

        # Validate that each horizontal-wavenumber sample has one root-set row.
        roots = np.asarray(roots, dtype=np.complex128)
        if roots.ndim != 2 or roots.shape[0] != len(self.kh):
            raise ValueError("roots must have shape (kh, branch).")

        # Reconstruct and trim each point polynomial before evaluating finite roots.
        residuals = np.full(roots.shape, np.nan)
        for j, kh_j in enumerate(self.kh):
            try:
                coefficients = self.omega_coefficients(kh_j)
                coefficients = self._trim_polynomial_coefficients(
                    coefficients, rtol=coefficient_rtol, atol=coefficient_atol
                )
            except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
                continue
            if coefficients is None or coefficients.size <= 1:
                continue

            # Skip unavailable roots while preserving their NaN residual entries.
            valid = self._finite_complex_mask(roots[j])
            if not np.any(valid):
                continue
            point_roots = roots[j, valid]
            # Normalize the polynomial value by a bounded-below coefficient magnitude sum.
            powers = np.arange(coefficients.size - 1, -1, -1)
            values = np.polyval(coefficients, point_roots)
            scale = np.sum(
                np.abs(coefficients)[None, :]
                * np.maximum(1.0, np.abs(point_roots))[:, None]**powers[None, :],
                axis=1
            )
            residuals[j, valid] = np.abs(values)/np.maximum(scale, 1e-300)

        return residuals

    def _physical_omega_boundaries(
        self,
        tracked_roots,
        output_order,
        real_rtol=1e-8,
        real_atol=1e-12,
        frequency_rtol=1e-10,
        frequency_atol=1e-12
    ):

        '''
        Purpose
        -------
        Select real omega roots inside the sampled nonnegative frequency interval.

        Inputs
        ------
        tracked_roots: np.array, complex
            Tracked omega roots (rad/s), shape (n_kh, raw_branch_count).

        output_order: int
            Number of public boundary slots to retain.

        real_rtol: float, optional
            Relative tolerance for treating a complex omega root as real.

        real_atol: float, optional
            Absolute imaginary-part tolerance (rad/s).

        frequency_rtol: float, optional
            Relative tolerance at the sampled frequency limits.

        frequency_atol: float, optional
            Absolute tolerance at the sampled frequency limits (rad/s).

        Outputs
        -------
        boundaries: np.array, float
            Numerically real roots in the sampled nonnegative frequency interval
            (rad/s), shape (n_kh, output_order), with NaN for unavailable values.
            Branches rank by valid-sample count, then median frequency, then index.

        diagnostics: dict
            Frequency limits, tolerance arrays, root-selection masks, valid counts,
            and the mapping from public slots to complete tracked branch indices.
        '''

        # Determine the nonnegative interval covered by finite sampled frequencies.
        tracked_roots = np.asarray(tracked_roots, dtype=np.complex128)
        finite_omega = self.omega[np.isfinite(self.omega)]
        if finite_omega.size == 0:
            raise ValueError("omega grid must contain at least one finite frequency.")

        frequency_min = max(0.0, float(np.min(finite_omega)))
        frequency_max = float(np.max(finite_omega))
        if frequency_max < frequency_min:
            raise ValueError("omega grid does not include a nonnegative frequency interval.")

        # Apply frequency-scaled tolerances to imaginary parts and interval endpoints.
        finite_mask = self._finite_complex_mask(tracked_roots)
        frequency_scale = max(abs(frequency_min), abs(frequency_max), 1e-30)
        real_tolerance = real_atol + real_rtol*np.maximum(
            np.abs(tracked_roots.real), frequency_scale
        )
        frequency_tolerance = frequency_atol + frequency_rtol*np.maximum(
            np.abs(tracked_roots.real), frequency_scale
        )
        real_root_mask = finite_mask & (np.abs(tracked_roots.imag) <= real_tolerance)
        in_frequency_range_mask = (
            real_root_mask
            & (tracked_roots.real >= frequency_min - frequency_tolerance)
            & (tracked_roots.real <= frequency_max + frequency_tolerance)
        )
        physical_root_mask = in_frequency_range_mask

        # Rank tracked curves by coverage and representative real frequency.
        valid_count = np.count_nonzero(physical_root_mask, axis=0)
        candidates = np.flatnonzero(valid_count > 0)
        branch_median = np.full(tracked_roots.shape[1], np.nan)
        for branch in candidates:
            branch_median[branch] = np.median(
                tracked_roots.real[physical_root_mask[:, branch], branch]
            )

        ranked = sorted(
            candidates,
            key=lambda branch: (-valid_count[branch], branch_median[branch], int(branch))
        )[:output_order]
        # Put complete curves before isolated in-range points (for example the
        # negative partner touching omega=0 at one degenerate endpoint), then
        # order equally complete curves by their representative frequency.
        selected = ranked

        # Copy qualified samples into public slots and clip only tolerance-sized endpoint
        # excess.
        boundaries = np.full((tracked_roots.shape[0], output_order), np.nan)
        physical_branch_indices = np.full(output_order, -1, dtype=int)
        for output_branch, tracked_branch in enumerate(selected):
            mask = physical_root_mask[:, tracked_branch]
            values = tracked_roots.real[mask, tracked_branch]
            boundaries[mask, output_branch] = np.clip(
                values, frequency_min, frequency_max
            )
            physical_branch_indices[output_branch] = tracked_branch

        # Preserve selection masks and the public-to-complete branch mapping.
        diagnostics = {
            'frequency_interval': (frequency_min, frequency_max),
            'real_root_rtol': float(real_rtol),
            'real_root_atol': float(real_atol),
            'frequency_rtol': float(frequency_rtol),
            'frequency_atol': float(frequency_atol),
            'real_tolerance': real_tolerance,
            'frequency_tolerance': frequency_tolerance,
            'real_root_mask': real_root_mask,
            'in_frequency_range_mask': in_frequency_range_mask,
            'physical_root_mask': physical_root_mask,
            'complex_root_mask': finite_mask & ~real_root_mask,
            'negative_root_mask': (
                real_root_mask
                & (tracked_roots.real < frequency_min - frequency_tolerance)
            ),
            'above_frequency_range_mask': (
                real_root_mask
                & (tracked_roots.real > frequency_max + frequency_tolerance)
            ),
            'physical_valid_count_by_tracked_branch': valid_count,
            'physical_branch_indices': physical_branch_indices
        }
        return boundaries, diagnostics

    def omega_solve_grid(
        self,
        return_diagnostics=False,
        coefficient_rtol=1e-12,
        coefficient_atol=0.0,
        real_rtol=1e-8,
        real_atol=1e-12,
        frequency_rtol=1e-10,
        frequency_atol=1e-12,
        degenerate_rtol=1e-6,
        degenerate_atol=1e-12,
        uncertainty_rtol=1e-3,
        uncertainty_atol=1e-12,
        residual_tolerance=1e-8
    ):

        '''
        Purpose
        -------
        Solve and continuity-track dispersion-boundary frequencies over kh.

        Inputs
        ------
        return_diagnostics: bool, optional
            Return the result and its diagnostic dictionary when True.

        coefficient_rtol: float, optional
            Relative threshold for trimming leading omega-polynomial coefficients.

        coefficient_atol: float, optional
            Absolute threshold for trimming leading omega-polynomial coefficients.

        real_rtol: float, optional
            Relative tolerance for treating a complex omega root as real.

        real_atol: float, optional
            Absolute imaginary-part tolerance (rad/s).

        frequency_rtol: float, optional
            Relative tolerance at the sampled frequency limits.

        frequency_atol: float, optional
            Absolute tolerance at the sampled frequency limits (rad/s).

        degenerate_rtol: float, optional
            Relative root-separation tolerance used to flag near-coincident roots.

        degenerate_atol: float, optional
            Absolute root-separation tolerance in the units of the supplied roots.

        uncertainty_rtol: float, optional
            Relative assignment-margin threshold used to flag ambiguous continuation.

        uncertainty_atol: float, optional
            Absolute assignment-margin threshold in the units of the supplied roots.

        residual_tolerance: float, optional
            Dimensionless normalized-residual threshold for the solver-instability mask.

        Outputs
        -------
        omega_boundaries: np.array, float
            Boundary frequencies (rad/s), shape (n_kh, omega_order). The general
            selection retains numerically real, nonnegative in-range roots and uses
            NaN for missing values. The separate slow/fast models replace this array
            with their analytic positive-frequency boundary, without range clipping.

        diagnostics: dict, optional
            Raw/tracked complex roots, branch mappings, assignment and degeneracy
            metrics, residuals, solve status, and component data. The slow/fast analytic
            output replacement does not recompute the general selection masks.
        '''

        # Solve complete complex root sets across the horizontal-wavenumber grid.
        raw_roots, solve_info = self._omega_solve_raw_grid(
            coefficient_rtol=coefficient_rtol,
            coefficient_atol=coefficient_atol
        )
        # Assign continuous labels before selecting the public frequency boundaries.
        tracked_roots, diagnostics = self.track_omega_branches(
            raw_roots,
            kh=self.kh,
            degenerate_rtol=degenerate_rtol,
            degenerate_atol=degenerate_atol,
            uncertainty_rtol=uncertainty_rtol,
            uncertainty_atol=uncertainty_atol,
            return_diagnostics=True
        )
        # Select numerically real roots within the sampled nonnegative interval.
        boundaries, physical_diagnostics = self._physical_omega_boundaries(
            tracked_roots,
            output_order=int(self.params['omega_order']),
            real_rtol=real_rtol,
            real_atol=real_atol,
            frequency_rtol=frequency_rtol,
            frequency_atol=frequency_atol
        )
        # Use the analytic kz = 0 frequency for the separately selected slow or fast mode.
        if self.model in ('slow_mode', 'fast_mode'):
            mu = sin(self.theta)*cos(self.phi)
            discriminant = (self.a**2 + self.cs**2)**2 - 4*self.a**2*self.cs**2*mu**2
            branch_sign = 1.0 if self.model == 'fast_mode' else -1.0
            omega_squared = 0.5*np.asarray(self.kh)**2*(
                self.a**2 + self.cs**2 + branch_sign*np.sqrt(max(discriminant, 0.0))
            )
            boundaries = np.sqrt(np.maximum(omega_squared, 0.0))[:, None]

        # Evaluate normalized polynomial residuals independently of branch assignment.
        root_residuals = self.omega_root_residuals(
            raw_roots,
            coefficient_rtol=coefficient_rtol,
            coefficient_atol=coefficient_atol
        )
        max_root_residual = np.full(len(self.kh), np.nan)
        residual_valid = np.any(np.isfinite(root_residuals), axis=1)
        if np.any(residual_valid):
            max_root_residual[residual_valid] = np.nanmax(
                root_residuals[residual_valid], axis=1
            )

        # Distinguish reduced polynomial degree, missing roots, and excessive residuals.
        raw_root_count = int(solve_info['raw_root_count'])
        degree_reduced_mask = (
            solve_info['coefficient_valid']
            & (solve_info['effective_degree'] >= 0)
            & (solve_info['effective_degree'] < solve_info['coefficient_degree'])
        )
        missing_root_mask = diagnostics['finite_root_count'] < raw_root_count
        solver_instability_mask = (
            np.isfinite(max_root_residual)
            & (max_root_residual > residual_tolerance)
        )

        # Combine solve, tracking, selection, and residual diagnostics for the caller.
        diagnostics.update(solve_info)
        diagnostics.update(physical_diagnostics)
        diagnostics.update({
            'raw_roots': raw_roots,
            'tracked_roots': tracked_roots,
            'omega_boundaries': boundaries,
            'root_residuals': root_residuals,
            'max_root_residual': max_root_residual,
            'polynomial_solve_failed_mask': solve_info['solver_failed'],
            'solve_failed_mask': solve_info['solver_failed'],
            'degree_reduced_mask': degree_reduced_mask,
            'missing_root_mask': missing_root_mask,
            'solver_instability_mask': solver_instability_mask,
            'residual_tolerance': float(residual_tolerance),
            'coefficient_rtol': float(coefficient_rtol),
            'coefficient_atol': float(coefficient_atol),
            'physical_output_order': int(self.params['omega_order'])
        })
        diagnostics['uncertain_mask'] |= solve_info['solver_failed']
        diagnostics['uncertain_mask'] |= solver_instability_mask

        if return_diagnostics:
            return boundaries, diagnostics
        return boundaries

    @staticmethod
    def _contiguous_index_runs(mask):

        '''
        Purpose
        -------
        Return contiguous true-index runs from a one-dimensional mask.

        Inputs
        ------
        mask: array-like, bool
            One-dimensional mask selecting samples.

        Outputs
        -------
        runs: list of np.array, int
            Consecutive selected-index runs; an empty list if no sample is selected.
        '''

        # Split selected indices wherever neighboring samples are separated by a gap.
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if indices.size == 0:
            return []
        return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)

    @staticmethod
    def _kh_run_span(kh, run):

        '''
        Purpose
        -------
        Return physical path length covered by a contiguous kh-index run.

        Inputs
        ------
        kh: array-like, float
            One-dimensional horizontal-wavenumber coordinates (km^-1).

        run: array-like, int
            Indices defining a contiguous coordinate run.

        Outputs
        -------
        span: float
            Sum of absolute coordinate steps (km^-1), or zero for fewer than two
            indices.
        '''

        # Sum absolute physical steps so descending coordinate runs retain positive span.
        run = np.asarray(run, dtype=int)
        if run.size < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(np.asarray(kh, dtype=float)[run]))))

    def _continue_kz_zero_roots(
        self,
        exact_roots,
        zero_root_indices,
        boundary_omega,
        reference_omega,
        kh,
        substeps
    ):

        '''
        Purpose
        -------
        Continue exact kz=0 root indices to one sampled omega row.

        Inputs
        ------
        exact_roots: np.array, complex
            Complete kz root set at the exact boundary frequency (km^-1).

        zero_root_indices: array-like, int
            Indices of roots identified as kz = 0 in exact_roots.

        boundary_omega: float
            Starting angular frequency (rad/s).

        reference_omega: float
            Target sampled angular frequency (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        substeps: int
            Positive number of equally spaced intermediate polynomial solves.

        Outputs
        -------
        current_roots: np.array, complex
            Complete kz root set at the last continuation step (km^-1).

        current_indices: np.array, int
            Surviving zero-root descendants in current_roots; missing descendants
            are removed and cause failed metadata to be set.

        stats: dict
            Failure flag, mean assignment cost, maximum individual assignment cost,
            and minimum assignment margin encountered during continuation.
        '''

        # Copy the exact boundary root set and the zero-root indices to continue.
        current_roots = np.asarray(exact_roots, dtype=np.complex128).copy()
        current_indices = np.asarray(zero_root_indices, dtype=int).copy()
        costs = []
        max_costs = []
        margins = []
        failed = current_indices.size == 0

        # Advance in equal frequency increments and solve all roots at each step.
        for step in range(1, substeps + 1):
            step_fraction = step/substeps
            step_omega = boundary_omega + step_fraction*(reference_omega - boundary_omega)
            next_roots = self._solve_kz_roots_at_point(step_omega, kh)
            permutation, stats = self._match_root_sets(current_roots, next_roots)

            # Carry surviving zero-root descendants through each local assignment
            # permutation.
            mapped_indices = np.full(current_indices.shape, -1, dtype=int)
            valid_current = (
                (current_indices >= 0) & (current_indices < permutation.size)
            )
            mapped_indices[valid_current] = permutation[current_indices[valid_current]]
            if np.any(mapped_indices < 0):
                failed = True
            current_indices = mapped_indices[mapped_indices >= 0]
            current_roots = next_roots

            # Accumulate continuation costs and margins without replacing unavailable
            # metrics.
            if np.isfinite(stats['assignment_cost']):
                costs.append(stats['assignment_cost'])
            if np.isfinite(stats['max_assignment_cost']):
                max_costs.append(stats['max_assignment_cost'])
            if np.isfinite(stats['assignment_margin']):
                margins.append(stats['assignment_margin'])
            if current_indices.size == 0:
                break

        # Report endpoint roots, surviving descendants, and the worst continuation evidence.
        return current_roots, current_indices, {
            'failed': bool(failed),
            'assignment_cost': float(np.mean(costs)) if costs else np.nan,
            'max_assignment_cost': float(np.max(max_costs)) if max_costs else np.nan,
            'assignment_margin': float(np.min(margins)) if margins else np.nan
        }

    def associate_omega_boundaries_with_kz(
        self,
        omega_boundaries,
        tracked_kz_roots,
        kz_zero_rtol=1e-4,
        kz_zero_atol=1e-10,
        omega_zero_atol=1e-9,
        boundary_match_substeps=8,
        stitch_boundary_curves=True,
        minimum_curve_support_points=3,
        minimum_curve_support_kh_span=None,
        return_diagnostics=False
    ):

        '''
        Purpose
        -------
        Associate continuous omega curves with the tracked kz/phase families.

        At every positive boundary frequency the kz polynomial is solved again,
        and only an actual kz=0 root is accepted as local evidence. Each zero
        root is continued through ``boundary_match_substeps`` intermediate
        solves to the strict sampled-frequency rows immediately below and above
        the boundary. The endpoint root sets are matched to the globally tracked
        kz labels with full complex Hungarian assignment. Both incident labels
        are retained when a crossing changes identity across the boundary.

        A global phase panel can remain bounded by the same continuous omega
        curve after its local zero-root label hands off at a crossing. With
        ``stitch_boundary_curves=True``, a kz/omega-curve pair is therefore
        promoted across each connected valid kh interval only after repeated
        contiguous local support. Isolated endpoint/degeneracy matches are not
        promoted. The unpromoted pointwise evidence and every promotion remain
        available in the returned diagnostics.

        Missing values stay NaN. Zero-frequency endpoints are attached only as
        limits of an adjacent, qualified positive-frequency curve.

        Inputs
        ------
        omega_boundaries: np.array, float
            Boundary frequencies (rad/s), shape (n_kh, boundary_count).

        tracked_kz_roots: np.array, complex
            Globally tracked kz roots (km^-1), shape (n_omega, n_kh, kz_order).

        kz_zero_rtol: float, optional
            Relative zero-root threshold using the median positive magnitude in each
            tracked-kz column.

        kz_zero_atol: float, optional
            Absolute zero-root threshold (km^-1).

        omega_zero_atol: float, optional
            Boundary frequencies at or below this value (rad/s) are treated as zero
            endpoints.

        boundary_match_substeps: int, optional
            Positive continuation-step count from a boundary to each neighboring row.

        stitch_boundary_curves: bool, optional
            Promote qualified local support over each connected verified curve interval.

        minimum_curve_support_points: int, optional
            Required contiguous support count, capped at each curve component length.

        minimum_curve_support_kh_span: float or None, optional
            Required contiguous support span (km^-1), capped at the component span.
            None uses median positive kh step times minimum_curve_support_points - 1.

        return_diagnostics: bool, optional
            Return the result and its diagnostic dictionary when True.

        Outputs
        -------
        associated: np.array, float
            Boundary frequencies (rad/s), shape (n_kh, kz_order, boundary_count),
            with NaN wherever a boundary is not associated with that kz branch.

        diagnostics: dict, optional
            Local and two-sided evidence, stitched associations, zero-endpoint
            extensions, continuation quality, support counts/spans, connected-component
            records, and ambiguity masks, returned when return_diagnostics is True.
        '''

        # Check grid compatibility and validate continuation/support settings.
        omega_boundaries = np.asarray(omega_boundaries, dtype=float)
        tracked_kz_roots = np.asarray(tracked_kz_roots, dtype=np.complex128)
        if omega_boundaries.ndim != 2 or omega_boundaries.shape[0] != len(self.kh):
            raise ValueError("omega_boundaries must have shape (kh, omega_boundary).")
        expected_kz_shape = (len(self.omega), len(self.kh), int(self.params['kz_order']))
        if tracked_kz_roots.shape != expected_kz_shape:
            raise ValueError(
                "tracked_kz_roots must have shape "
                f"{expected_kz_shape}, matching kz_solve_grid()."
            )
        if not isinstance(boundary_match_substeps, (int, np.integer)) or boundary_match_substeps < 1:
            raise ValueError("boundary_match_substeps must be a positive integer.")
        if (
            not isinstance(minimum_curve_support_points, (int, np.integer))
            or minimum_curve_support_points < 1
        ):
            raise ValueError("minimum_curve_support_points must be a positive integer.")
        if minimum_curve_support_kh_span is not None:
            if not np.isfinite(minimum_curve_support_kh_span) or minimum_curve_support_kh_span < 0:
                raise ValueError("minimum_curve_support_kh_span must be finite and nonnegative.")

        # Allocate local/two-sided associations and pointwise continuation diagnostics.
        n_kh, boundary_count = omega_boundaries.shape
        kz_branch_count = tracked_kz_roots.shape[2]
        association_shape = (n_kh, kz_branch_count, boundary_count)
        local_association_mask = np.zeros(association_shape, dtype=bool)
        side_association_mask = np.zeros(association_shape + (2,), dtype=bool)
        boundary_condition_residual = np.full((n_kh, boundary_count), np.nan)
        boundary_kz_zero_count = np.zeros((n_kh, boundary_count), dtype=int)
        boundary_reference_omega_indices = np.full(
            (n_kh, boundary_count, 2), -1, dtype=int
        )
        boundary_reference_omega_index = np.full(
            (n_kh, boundary_count), -1, dtype=int
        )
        boundary_side_assignment_cost = np.full((n_kh, boundary_count, 2), np.nan)
        boundary_side_max_assignment_cost = np.full(
            (n_kh, boundary_count, 2), np.nan
        )
        boundary_side_assignment_margin = np.full((n_kh, boundary_count, 2), np.nan)
        boundary_side_assignment_confidence = np.full(
            (n_kh, boundary_count, 2), np.nan
        )
        boundary_side_matched_count = np.zeros((n_kh, boundary_count, 2), dtype=int)
        boundary_continuation_failed_mask = np.zeros(
            (n_kh, boundary_count, 2), dtype=bool
        )
        boundary_assignment_cost = np.full((n_kh, boundary_count), np.nan)
        boundary_max_assignment_cost = np.full((n_kh, boundary_count), np.nan)
        boundary_assignment_margin = np.full((n_kh, boundary_count), np.nan)
        boundary_assignment_confidence = np.full((n_kh, boundary_count), np.nan)
        boundary_matched_count = np.zeros((n_kh, boundary_count), dtype=int)
        boundary_reference_kz_distance = np.full(
            (n_kh, boundary_count, kz_branch_count), np.nan
        )
        zero_frequency_mask = np.zeros((n_kh, boundary_count), dtype=bool)
        invalid_kz_solve_mask = np.zeros((n_kh, boundary_count), dtype=bool)
        boundary_label_transition_mask = np.zeros((n_kh, boundary_count), dtype=bool)

        # Estimate a representative positive kz magnitude for each horizontal-wavenumber
        # column.
        column_kz_scale = np.full(n_kh, np.nan)
        for kh_index in range(n_kh):
            column_values = tracked_kz_roots[:, kh_index, :]
            finite_values = np.abs(column_values[self._finite_complex_mask(column_values)])
            if finite_values.size > 0:
                positive_values = finite_values[finite_values > kz_zero_atol]
                if positive_values.size > 0:
                    column_kz_scale[kh_index] = float(np.median(positive_values))
                else:
                    column_kz_scale[kh_index] = float(np.max(finite_values))

        # Sort finite frequency samples while retaining their original grid indices.
        finite_omega_indices = np.flatnonzero(np.isfinite(self.omega))
        omega_sort = np.argsort(self.omega[finite_omega_indices], kind='stable')
        sorted_omega_indices = finite_omega_indices[omega_sort]
        sorted_omega = self.omega[sorted_omega_indices]

        # Verify each positive-frequency boundary against the actual kz polynomial.
        for kh_index, kh_value in enumerate(self.kh):
            kz_scale = column_kz_scale[kh_index]
            if not np.isfinite(kz_scale):
                continue
            zero_tolerance = kz_zero_atol + kz_zero_rtol*kz_scale

            for boundary_index, boundary_omega in enumerate(omega_boundaries[kh_index]):
                if not np.isfinite(boundary_omega):
                    continue
                if boundary_omega <= omega_zero_atol:
                    zero_frequency_mask[kh_index, boundary_index] = True
                    continue

                # Accept local evidence only when the exact boundary solve contains a near-
                # zero kz root.
                exact_roots = self._solve_kz_roots_at_point(boundary_omega, kh_value)
                finite_exact = self._finite_complex_mask(exact_roots)
                if not np.any(finite_exact):
                    invalid_kz_solve_mask[kh_index, boundary_index] = True
                    continue

                min_abs_kz = float(np.min(np.abs(exact_roots[finite_exact])))
                boundary_condition_residual[kh_index, boundary_index] = (
                    min_abs_kz/max(kz_scale, kz_zero_atol, 1e-300)
                )
                zero_root_mask = finite_exact & (np.abs(exact_roots) <= zero_tolerance)
                zero_root_indices = np.flatnonzero(zero_root_mask)
                boundary_kz_zero_count[kh_index, boundary_index] = zero_root_indices.size
                if zero_root_indices.size == 0 or sorted_omega.size == 0:
                    continue

                # Find strict frequency neighbors on both sides of the boundary.
                left_position = int(np.searchsorted(sorted_omega, boundary_omega, side='left')) - 1
                right_position = int(np.searchsorted(sorted_omega, boundary_omega, side='right'))
                reference_pairs = []
                if left_position >= 0:
                    reference_pairs.append((0, int(sorted_omega_indices[left_position])))
                if right_position < sorted_omega.size:
                    reference_pairs.append((1, int(sorted_omega_indices[right_position])))
                if len(reference_pairs) == 0:
                    nearest_position = int(np.argmin(np.abs(sorted_omega - boundary_omega)))
                    reference_pairs.append((0, int(sorted_omega_indices[nearest_position])))

                # Continue each zero root to both neighboring rows and match the global
                # branch labels.
                side_branch_sets = []
                for side, reference_index in reference_pairs:
                    reference_omega = self.omega[reference_index]
                    boundary_reference_omega_indices[
                        kh_index, boundary_index, side
                    ] = reference_index
                    reference_roots = tracked_kz_roots[reference_index, kh_index, :]
                    finite_reference = self._finite_complex_mask(reference_roots)
                    reference_distances = np.abs(reference_roots)
                    current_distances = boundary_reference_kz_distance[
                        kh_index, boundary_index
                    ]
                    replace = finite_reference & (
                        ~np.isfinite(current_distances)
                        | (reference_distances < current_distances)
                    )
                    current_distances[replace] = reference_distances[replace]

                    # Resolve descendants through intermediate solves before comparing
                    # endpoint root sets.
                    continued_roots, continued_zero_indices, continuation_stats = (
                        self._continue_kz_zero_roots(
                            exact_roots,
                            zero_root_indices,
                            boundary_omega,
                            reference_omega,
                            kh_value,
                            int(boundary_match_substeps)
                        )
                    )
                    endpoint_permutation, endpoint_stats = self._match_root_sets(
                        reference_roots, continued_roots
                    )
                    selected_branches = np.flatnonzero(
                        np.isin(endpoint_permutation, continued_zero_indices)
                    )
                    side_association_mask[
                        kh_index, selected_branches, boundary_index, side
                    ] = True
                    side_branch_sets.append(set(selected_branches.tolist()))
                    boundary_side_matched_count[
                        kh_index, boundary_index, side
                    ] = selected_branches.size
                    boundary_continuation_failed_mask[
                        kh_index, boundary_index, side
                    ] = continuation_stats['failed'] or continued_zero_indices.size == 0

                    # Combine intermediate and endpoint costs using conservative
                    # maximum/minimum summaries.
                    cost_values = [
                        value for value in (
                            continuation_stats['assignment_cost'],
                            endpoint_stats['assignment_cost']
                        ) if np.isfinite(value)
                    ]
                    max_cost_values = [
                        value for value in (
                            continuation_stats['max_assignment_cost'],
                            endpoint_stats['max_assignment_cost']
                        ) if np.isfinite(value)
                    ]
                    margin_values = [
                        value for value in (
                            continuation_stats['assignment_margin'],
                            endpoint_stats['assignment_margin']
                        ) if np.isfinite(value)
                    ]
                    if cost_values:
                        boundary_side_assignment_cost[
                            kh_index, boundary_index, side
                        ] = float(np.mean(cost_values))
                    if max_cost_values:
                        boundary_side_max_assignment_cost[
                            kh_index, boundary_index, side
                        ] = float(np.max(max_cost_values))
                    if margin_values:
                        boundary_side_assignment_margin[
                            kh_index, boundary_index, side
                        ] = float(np.min(margin_values))
                    boundary_side_assignment_confidence[
                        kh_index, boundary_index, side
                    ] = self._edge_confidence(
                        boundary_side_max_assignment_cost[kh_index, boundary_index, side],
                        boundary_side_assignment_margin[kh_index, boundary_index, side]
                    )

                # Retain the union of both incident branch labels and flag label handoffs.
                local_association_mask[kh_index, :, boundary_index] = np.any(
                    side_association_mask[kh_index, :, boundary_index, :], axis=1
                )
                boundary_matched_count[kh_index, boundary_index] = np.count_nonzero(
                    local_association_mask[kh_index, :, boundary_index]
                )
                if len(side_branch_sets) == 2:
                    boundary_label_transition_mask[
                        kh_index, boundary_index
                    ] = side_branch_sets[0] != side_branch_sets[1]

                # Record the nearest sampled reference row for compatibility diagnostics.
                reference_indices = boundary_reference_omega_indices[
                    kh_index, boundary_index
                ]
                valid_reference_indices = reference_indices[reference_indices >= 0]
                if valid_reference_indices.size > 0:
                    nearest = np.argmin(
                        np.abs(self.omega[valid_reference_indices] - boundary_omega)
                    )
                    boundary_reference_omega_index[
                        kh_index, boundary_index
                    ] = int(valid_reference_indices[nearest])

                # Reduce two-sided assignment evidence into one summary per boundary point.
                side_costs = boundary_side_assignment_cost[kh_index, boundary_index]
                finite_side_costs = side_costs[np.isfinite(side_costs)]
                if finite_side_costs.size > 0:
                    boundary_assignment_cost[kh_index, boundary_index] = float(
                        np.mean(finite_side_costs)
                    )
                side_max_costs = boundary_side_max_assignment_cost[kh_index, boundary_index]
                finite_side_max_costs = side_max_costs[np.isfinite(side_max_costs)]
                if finite_side_max_costs.size > 0:
                    boundary_max_assignment_cost[kh_index, boundary_index] = float(
                        np.max(finite_side_max_costs)
                    )
                side_margins = boundary_side_assignment_margin[kh_index, boundary_index]
                finite_side_margins = side_margins[np.isfinite(side_margins)]
                if finite_side_margins.size > 0:
                    boundary_assignment_margin[kh_index, boundary_index] = float(
                        np.min(finite_side_margins)
                    )
                side_confidence = boundary_side_assignment_confidence[
                    kh_index, boundary_index
                ]
                finite_side_confidence = side_confidence[np.isfinite(side_confidence)]
                if finite_side_confidence.size > 0:
                    boundary_assignment_confidence[kh_index, boundary_index] = float(
                        np.min(finite_side_confidence)
                    )

        # Separate verified positive boundaries from zero endpoints and unmatched candidates.
        finite_boundary_mask = np.isfinite(omega_boundaries) & ~zero_frequency_mask
        verified_boundary_mask = boundary_kz_zero_count > 0
        unmatched_boundary_mask = finite_boundary_mask & ~verified_boundary_mask

        # Derive a physical support span from the typical positive kh step when unspecified.
        if minimum_curve_support_kh_span is None:
            kh_steps = np.abs(np.diff(np.asarray(self.kh, dtype=float)))
            finite_positive_steps = kh_steps[np.isfinite(kh_steps) & (kh_steps > 0)]
            typical_kh_step = (
                float(np.median(finite_positive_steps))
                if finite_positive_steps.size > 0 else 0.0
            )
            curve_support_kh_span = typical_kh_step*(minimum_curve_support_points - 1)
        else:
            typical_kh_step = np.nan
            curve_support_kh_span = float(minimum_curve_support_kh_span)

        # Prepare optional curve stitching while retaining the original local evidence.
        association_mask = local_association_mask.copy() if not stitch_boundary_curves else np.zeros(
            association_shape, dtype=bool
        )
        endpoint_extension_mask = np.zeros(association_shape, dtype=bool)
        boundary_component_index = np.full((n_kh, boundary_count), -1, dtype=int)
        curve_branch_support_count = np.zeros((kz_branch_count, boundary_count), dtype=int)
        curve_branch_max_contiguous_support = np.zeros(
            (kz_branch_count, boundary_count), dtype=int
        )
        curve_branch_max_support_kh_span = np.zeros(
            (kz_branch_count, boundary_count), dtype=float
        )
        curve_branch_qualified_mask = np.zeros(
            (kz_branch_count, boundary_count), dtype=bool
        )
        component_records = []
        component_index = 0

        # Process connected verified curve intervals without bridging NaN or invalid gaps.
        for boundary_index in range(boundary_count):
            valid_curve = verified_boundary_mask[:, boundary_index] & finite_boundary_mask[:, boundary_index]
            for component in self._contiguous_index_runs(valid_curve):
                boundary_component_index[component, boundary_index] = component_index
                component_span = self._kh_run_span(self.kh, component)
                required_points = min(int(minimum_curve_support_points), component.size)
                required_span = min(curve_support_kh_span, component_span)
                qualified_branches = []
                branch_support = {}

                # Measure repeated contiguous support for each kz branch within this curve
                # component.
                for kz_branch in range(kz_branch_count):
                    local_component_mask = np.zeros(n_kh, dtype=bool)
                    local_component_mask[component] = local_association_mask[
                        component, kz_branch, boundary_index
                    ]
                    support_runs = self._contiguous_index_runs(local_component_mask)
                    support_count = int(np.count_nonzero(local_component_mask))
                    max_run_points = max(
                        (run.size for run in support_runs), default=0
                    )
                    max_run_span = max(
                        (self._kh_run_span(self.kh, run) for run in support_runs),
                        default=0.0
                    )
                    curve_branch_support_count[kz_branch, boundary_index] += support_count
                    curve_branch_max_contiguous_support[
                        kz_branch, boundary_index
                    ] = max(
                        curve_branch_max_contiguous_support[kz_branch, boundary_index],
                        max_run_points
                    )
                    curve_branch_max_support_kh_span[
                        kz_branch, boundary_index
                    ] = max(
                        curve_branch_max_support_kh_span[kz_branch, boundary_index],
                        max_run_span
                    )
                    branch_support[int(kz_branch)] = {
                        'point_count': support_count,
                        'max_contiguous_points': int(max_run_points),
                        'max_kh_span': float(max_run_span)
                    }

                    # Require both a sufficient support count and a sufficient physical kh
                    # span.
                    qualifies = (
                        max_run_points >= required_points
                        and max_run_span + np.finfo(float).eps >= required_span
                    )
                    if qualifies:
                        qualified_branches.append(int(kz_branch))
                        curve_branch_qualified_mask[kz_branch, boundary_index] = True
                        if stitch_boundary_curves:
                            association_mask[component, kz_branch, boundary_index] = True

                # Extend qualified stitched curves only through directly adjacent finite zero
                # endpoints.
                zero_endpoint_indices = []
                if stitch_boundary_curves and qualified_branches:
                    before = int(component[0]) - 1
                    while (
                        before >= 0
                        and zero_frequency_mask[before, boundary_index]
                        and np.isfinite(omega_boundaries[before, boundary_index])
                    ):
                        zero_endpoint_indices.append(before)
                        before -= 1
                    after = int(component[-1]) + 1
                    while (
                        after < n_kh
                        and zero_frequency_mask[after, boundary_index]
                        and np.isfinite(omega_boundaries[after, boundary_index])
                    ):
                        zero_endpoint_indices.append(after)
                        after += 1
                    if zero_endpoint_indices:
                        zero_endpoint_indices = np.asarray(zero_endpoint_indices, dtype=int)
                        for kz_branch in qualified_branches:
                            association_mask[
                                zero_endpoint_indices, kz_branch, boundary_index
                            ] = True
                            endpoint_extension_mask[
                                zero_endpoint_indices, kz_branch, boundary_index
                            ] = True

                # Retain the support evidence and endpoint extensions for this connected
                # component.
                component_records.append({
                    'component_index': int(component_index),
                    'omega_branch': int(boundary_index),
                    'kh_indices': component.copy(),
                    'required_support_points': int(required_points),
                    'required_support_kh_span': float(required_span),
                    'qualified_kz_branches': qualified_branches,
                    'branch_support': branch_support,
                    'zero_endpoint_indices': np.asarray(
                        zero_endpoint_indices, dtype=int
                    )
                })
                component_index += 1

        # Expose only associated frequencies, keeping all other curve/branch entries as NaN.
        associated = np.where(
            association_mask,
            omega_boundaries[:, None, :],
            np.nan
        )
        stitched_association_mask = association_mask & ~local_association_mask
        unassigned_verified_mask = verified_boundary_mask & ~np.any(
            association_mask, axis=1
        )
        # Flag multiple zero roots, branch handoffs, weak matches, and failed continuations.
        low_confidence_mask = (
            np.isfinite(boundary_assignment_confidence)
            & (boundary_assignment_confidence < 0.2)
        )
        boundary_ambiguous_mask = verified_boundary_mask & (
            (boundary_kz_zero_count > 1)
            | boundary_label_transition_mask
            | low_confidence_mask
            | np.any(boundary_continuation_failed_mask, axis=2)
            | unassigned_verified_mask
        )
        # Package local evidence separately from stitching and ambiguity diagnostics.
        diagnostics = {
            'association_mask': association_mask,
            'local_association_mask': local_association_mask,
            'side_association_mask': side_association_mask,
            'stitched_association_mask': stitched_association_mask,
            'endpoint_extension_mask': endpoint_extension_mask,
            'boundaries_by_kz_branch': associated,
            'boundary_condition_residual': boundary_condition_residual,
            'boundary_kz_zero_count': boundary_kz_zero_count,
            'boundary_reference_omega_index': boundary_reference_omega_index,
            'boundary_reference_omega_indices': boundary_reference_omega_indices,
            'boundary_assignment_cost': boundary_assignment_cost,
            'boundary_max_assignment_cost': boundary_max_assignment_cost,
            'boundary_assignment_margin': boundary_assignment_margin,
            'boundary_assignment_confidence': boundary_assignment_confidence,
            'boundary_matched_count': boundary_matched_count,
            'boundary_side_assignment_cost': boundary_side_assignment_cost,
            'boundary_side_max_assignment_cost': boundary_side_max_assignment_cost,
            'boundary_side_assignment_margin': boundary_side_assignment_margin,
            'boundary_side_assignment_confidence': boundary_side_assignment_confidence,
            'boundary_side_matched_count': boundary_side_matched_count,
            'boundary_reference_kz_distance': boundary_reference_kz_distance,
            # Compatibility alias for the first association implementation.
            'boundary_interpolated_kz_distance': boundary_reference_kz_distance,
            'boundary_continuation_failed_mask': boundary_continuation_failed_mask,
            'boundary_label_transition_mask': boundary_label_transition_mask,
            'boundary_component_index': boundary_component_index,
            'component_records': component_records,
            'curve_branch_support_count': curve_branch_support_count,
            'curve_branch_max_contiguous_support': curve_branch_max_contiguous_support,
            'curve_branch_max_support_kh_span': curve_branch_max_support_kh_span,
            'curve_branch_qualified_mask': curve_branch_qualified_mask,
            'unassigned_verified_mask': unassigned_verified_mask,
            'column_kz_scale': column_kz_scale,
            'kz_zero_rtol': float(kz_zero_rtol),
            'kz_zero_atol': float(kz_zero_atol),
            'omega_zero_atol': float(omega_zero_atol),
            'boundary_match_substeps': int(boundary_match_substeps),
            'stitch_boundary_curves': bool(stitch_boundary_curves),
            'minimum_curve_support_points': int(minimum_curve_support_points),
            'minimum_curve_support_kh_span': float(curve_support_kh_span),
            'typical_kh_step': float(typical_kh_step),
            'zero_frequency_mask': zero_frequency_mask,
            'invalid_kz_solve_mask': invalid_kz_solve_mask,
            'verified_boundary_mask': verified_boundary_mask,
            'unmatched_boundary_mask': unmatched_boundary_mask,
            'boundary_ambiguous_mask': boundary_ambiguous_mask
        }
        if return_diagnostics:
            return associated, diagnostics
        return associated

    def plot_omega_boundaries_for_kz_branch(
        self,
        ax,
        boundaries_by_kz_branch,
        kz_branch,
        kh_scale=1e3,
        frequency_scale=(2*pi)**(-1)*1e3,
        **plot_kwargs
    ):

        '''
        Purpose
        -------
        Plot only omega curves associated with one tracked kz/phase branch.

        Inputs
        ------
        ax: matplotlib.axes.Axes
            Axes receiving the boundary lines; modified in place.

        boundaries_by_kz_branch: np.array, float
            Associated frequencies (rad/s), shape (n_kh, kz_order, boundary_count).

        kz_branch: int
            Zero-based tracked kz/phase branch to plot.

        kh_scale: float, optional
            Factor converting km^-1 to horizontal plot units; 1e3 gives Mm^-1.

        frequency_scale: float, optional
            Factor converting rad/s to vertical plot units; default gives mHz.

        **plot_kwargs: dict
            Matplotlib line options overriding the black solid-line defaults.

        Outputs
        -------
        lines: list of matplotlib.lines.Line2D
            Line artists added for curves containing at least one finite sample.
        '''

        # Validate the association cube and requested branch before adding artists.
        boundaries_by_kz_branch = np.asarray(boundaries_by_kz_branch, dtype=float)
        expected_prefix = (len(self.kh), int(self.params['kz_order']))
        if boundaries_by_kz_branch.ndim != 3 or boundaries_by_kz_branch.shape[:2] != expected_prefix:
            raise ValueError(
                "boundaries_by_kz_branch must have shape "
                "(kh, kz_branch, omega_boundary)."
            )
        if kz_branch < 0 or kz_branch >= boundaries_by_kz_branch.shape[1]:
            raise IndexError("kz_branch is outside the tracked branch range.")

        # Apply caller styling over the default black boundary-line appearance.
        defaults = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5}
        defaults.update(plot_kwargs)
        lines = []
        # Draw finite associated curves with the requested coordinate conversions.
        for boundary in range(boundaries_by_kz_branch.shape[2]):
            values = boundaries_by_kz_branch[:, kz_branch, boundary]
            if not np.any(np.isfinite(values)):
                continue
            lines.extend(ax.plot(
                self.kh*kh_scale,
                values*frequency_scale,
                **defaults
            ))
        return lines

    def plot_omega_tracking_comparison(
        self,
        omega_boundaries=None,
        diagnostics=None,
        kh_scale=1e3,
        frequency_scale=(2*pi)**(-1)*1e3,
        save_path=None
    ):

        '''
        Purpose
        -------
        Plot legacy, raw, tracked, and flagged omega-boundary results.

        Inputs
        ------
        omega_boundaries: np.array or None, optional
            Public boundary frequencies (rad/s). If either this input or diagnostics
            is None, solve and replace both inputs.

        diagnostics: dict or None, optional
            Metadata from omega_solve_grid(return_diagnostics=True).

        kh_scale: float, optional
            Horizontal-wavenumber scale factor; default gives Mm^-1.

        frequency_scale: float, optional
            Angular-frequency scale factor; default gives mHz.

        save_path: str or path-like or None, optional
            Save the figure at 160 dpi when a path is supplied.

        Outputs
        -------
        fig: matplotlib.figure.Figure
            Figure comparing legacy sign scans, raw roots, tracked roots, and flags.

        axes: np.array of matplotlib.axes.Axes
            Axes array with shape (2, 2); the figure remains open for the caller.
        '''

        # Import plotting dependencies only when a comparison figure is requested.
        import matplotlib.pyplot as plt

        # Compute a consistent boundary/diagnostic pair if either input is absent.
        if omega_boundaries is None or diagnostics is None:
            omega_boundaries, diagnostics = self.omega_solve_grid(
                return_diagnostics=True
            )

        # Recompute the legacy sign-scan result for comparison with direct complex roots.
        legacy = np.vstack([self.omega_solve(kh_j) for kh_j in self.kh])
        raw_roots = diagnostics['raw_roots']
        kh_plot = self.kh*kh_scale

        # Compare legacy scanning, raw solver slots, physical branches, and status flags.
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for branch in range(legacy.shape[1]):
            axes[0, 0].plot(kh_plot, legacy[:, branch]*frequency_scale, lw=1)
        axes[0, 0].set_title('legacy Brent/sign-scan boundaries')

        # Show raw real parts before any branch continuity assignment.
        for raw_index in range(raw_roots.shape[1]):
            axes[0, 1].plot(
                kh_plot, raw_roots[:, raw_index].real*frequency_scale,
                lw=1, alpha=0.75
            )
        axes[0, 1].set_title('raw direct roots (solver-index real part)')

        # Show only the public tracked boundary curves in the third panel.
        for branch in range(omega_boundaries.shape[1]):
            axes[1, 0].plot(
                kh_plot, omega_boundaries[:, branch]*frequency_scale, lw=1.5
            )
        axes[1, 0].set_title('continuity-tracked physical boundaries')

        # Display uncertainty, degeneracy, and solve failure at their sampled kh positions.
        axes[1, 1].step(
            kh_plot, diagnostics['uncertain_mask'].astype(int), where='mid',
            label='uncertain', color='tab:orange'
        )
        axes[1, 1].step(
            kh_plot, diagnostics['degenerate_mask'].astype(int), where='mid',
            label='degenerate', color='tab:red'
        )
        axes[1, 1].step(
            kh_plot, diagnostics['solve_failed_mask'].astype(int), where='mid',
            label='solve failed', color='black', linestyle='--'
        )
        axes[1, 1].set_ylim(-0.05, 1.15)
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_title('flagged kh samples')
        axes[1, 1].legend(loc='upper right')

        # Label physical axes using the default Mm^-1 and mHz plot conventions.
        for ax in axes.flat:
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$\nu\,[\mathrm{mHz}]$')
        axes[0, 1].set_ylabel(r'$\mathrm{Re}(\omega)/(2\pi)\,[\mathrm{mHz}]$')

        # Optionally write the figure and keep its artists available to the caller.
        if save_path is not None:
            fig.savefig(save_path, dpi=160)
        return fig, axes

    @staticmethod
    def _scaled_kz_coefficients(coefficients):
        '''
        Purpose
        -------
        Scale kz = radius*y before constructing the companion matrix.

        Only exactly zero leading coefficients reduce the degree. Coefficients
        of different powers have different units, so comparing their raw
        magnitudes with a relative tolerance is not a valid degree test.

        Inputs
        ------
        coefficients: array-like
            Kz-polynomial coefficients in descending powers.

        Outputs
        -------
        scaled: np.array, complex or None
            Normalized coefficients after substituting kz = radius*y; None for
            invalid, constant, identically zero, or unrepresentable scaling cases.

        radius: float
            Positive kz scale (km^-1), or NaN when unavailable.

        degree: int
            Degree after exactly zero leading coefficients are removed; -1 denotes
            nonfinite input and zero denotes a constant or zero polynomial.
        '''

        # Reject nonfinite input and remove only exactly zero leading coefficients.
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if not np.all(DiagnosticDiagram._finite_complex_mask(coefficients)):
            return None, np.nan, -1
        nonzero = np.flatnonzero(coefficients != 0)
        if nonzero.size == 0:
            return None, np.nan, 0
        coefficients = coefficients[nonzero[0]:]
        degree = coefficients.size - 1
        if degree == 0:
            return None, np.nan, 0

        # Choose a logarithmic kz scale from the nonzero lower-order coefficients.
        magnitudes = np.abs(coefficients)
        lower = np.flatnonzero(magnitudes[1:] > 0) + 1
        if lower.size:
            log_radius = np.max(
                (np.log(magnitudes[lower]) - np.log(magnitudes[0])) / lower
            )
        else:
            log_radius = 0.0
        # A root outside the floating-point range cannot be represented.
        if not np.isfinite(log_radius) or abs(log_radius) > 690:
            return None, np.nan, degree
        # Apply the variable substitution and normalize in log space to limit overflow.
        radius = float(np.exp(log_radius))
        powers = np.arange(degree, -1, -1)
        log_terms = np.full(coefficients.size, -np.inf)
        valid = magnitudes > 0
        log_terms[valid] = np.log(magnitudes[valid]) + powers[valid]*log_radius
        scaled = np.zeros(coefficients.size, dtype=np.complex128)
        scaled[valid] = (
            np.exp(log_terms[valid] - np.max(log_terms))
            * np.exp(1j*np.angle(coefficients[valid]))
        )
        # Reject a scaled leading coefficient that underflows to zero.
        if scaled[0] == 0:
            return None, radius, degree
        return scaled, radius, degree

    def _solve_kz_roots_at_point(self, omega, kh, coefficient_rtol=1e-12,
                                 return_info=False):

        '''
        Purpose
        -------
        Solve the kz polynomial at one grid point, optionally returning solve metadata.

        Inputs
        ------
        omega: float
            Angular frequency at the polynomial solve/evaluation point (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        coefficient_rtol: float, optional
            Compatibility argument retained in the signature; unused by this solver.
            Only exactly zero leading coefficients reduce the kz polynomial degree.

        return_info: bool, optional
            Return the roots and solve metadata when True.

        Outputs
        -------
        roots: np.array, complex
            Kz roots (km^-1), padded to _raw_kz_root_count(). Failed or degenerate
            solves retain NaN slots. Alfvén roots use the direct analytic expression;
            other models use scaled polynomial roots with guarded Newton polishing.

        info: dict, optional
            Nominal/effective degree, degree reduction, zero/nonfinite coefficient
            flags, solver status, and root scale, returned when return_info is True.
        '''

        # Allocate complete raw-root storage and initialize pointwise solve status.
        root_count = self._raw_kz_root_count()
        roots = np.full(root_count, np.nan + 0.0j, dtype=np.complex128)
        info = {'coefficient_degree': -1, 'effective_degree': -1,
                'degree_reduced': False, 'zero_polynomial': False,
                'solver_failed': False, 'nonfinite_coefficients': False,
                'root_scale': np.nan}

        # Use analytic Alfvén roots and preserve the horizontal/zero-field degree degeneracy.
        if self.model == 'alfven_mode':
            cos_theta = cos(self.theta)
            if abs(cos_theta) <= 1e-14:
                cos_theta = 0.0
            mu = sin(self.theta)*cos(self.phi)
            offset = self.a*mu*kh
            mismatch = omega**2 - offset**2
            roots = np.full(root_count, np.nan + 0.0j, dtype=np.complex128)
            info.update({'coefficient_degree': 2, 'effective_degree': 0,
                         'degree_reduced': True})
            # Distinguish a constant mismatch from the identically zero Alfvén polynomial.
            if self.a <= np.finfo(float).tiny or abs(cos_theta) <= 1e-14:
                scale = max(abs(omega**2), abs(offset**2), np.finfo(float).tiny)
                if abs(mismatch) <= 100.0*np.finfo(float).eps*scale:
                    info.update({'zero_polynomial': True, 'solver_failed': True})
                return (roots, info) if return_info else roots

            # Evaluate the two field-aligned signs without a numerical companion matrix.
            roots[:] = (
                -kh*np.tan(self.theta)*cos(self.phi)
                + np.asarray([-1.0, 1.0])[:root_count]*omega/(self.a*cos_theta)
            )
            info.update({'effective_degree': 2, 'degree_reduced': False,
                         'root_scale': float(np.max(np.abs(roots)))})
            return (roots, info) if return_info else roots

        # Construct the model polynomial and preserve failures as padded NaN roots.
        try:
            coefficients = self.kz_coefficients(omega, kh)
        except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
            info['solver_failed'] = True
            return (roots, info) if return_info else roots

        # Check finite coefficients before applying the kz variable scaling.
        info['coefficient_degree'] = coefficients.size - 1
        if not np.all(self._finite_complex_mask(coefficients)):
            info['nonfinite_coefficients'] = True
            info['solver_failed'] = True
            return (roots, info) if return_info else roots

        # Record exact degree reduction and reject invalid or unrepresentable scaling.
        info['zero_polynomial'] = bool(np.all(coefficients == 0))
        scaled, radius, degree = self._scaled_kz_coefficients(coefficients)
        info['effective_degree'] = degree
        info['degree_reduced'] = degree < info['coefficient_degree']
        info['root_scale'] = radius
        if scaled is None:
            info['solver_failed'] = degree > 0 or info['zero_polynomial']
            return (roots, info) if return_info else roots

        try:
            # Solve the normalized polynomial in the scaled variable.
            scaled_roots = np.roots(scaled)
            # Two guarded Newton steps polish isolated roots without changing
            # the polynomial or mixing roots near a multiple-root cluster.
            derivative = np.polyder(scaled)
            for index, value in enumerate(scaled_roots):
                for _ in range(2):
                    slope = np.polyval(derivative, value)
                    if not np.isfinite(slope) or abs(slope) < 1e-14:
                        break
                    candidate = value - np.polyval(scaled, value)/slope
                    if not np.isfinite(candidate):
                        break
                    if abs(np.polyval(scaled, candidate)) >= abs(np.polyval(scaled, value)):
                        break
                    value = candidate
                scaled_roots[index] = value
            point_roots = scaled_roots*radius
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            info['solver_failed'] = True
            return (roots, info) if return_info else roots

        # Rescale finite roots to km^-1 and retain missing slots as NaN.
        point_roots = point_roots[self._finite_complex_mask(point_roots)]
        roots[:min(root_count, point_roots.size)] = point_roots[:root_count]
        info['solver_failed'] = point_roots.size < degree
        return (roots, info) if return_info else roots

    def _raw_kz_root_count(self):

        '''
        Purpose
        -------
        Return the padded root count used by the direct kz solver.

        Inputs
        ------
        None

        Outputs
        -------
        root_count: int
            Four roots for the separate slow/fast models, otherwise configured kz_order.
        '''

        # Keep all four roots until the requested magnetoacoustic family is selected.
        if self.model in ('slow_mode', 'fast_mode'):
            return 4
        return int(self.params['kz_order'])

    def _kz_solve_column_raw(self, kh, return_info=False):

        '''
        Purpose
        -------
        Solve raw, untracked kz roots for one kh column.

        Inputs
        ------
        kh: float
            Horizontal wavenumber (km^-1).

        return_info: bool, optional
            Return the roots and solve metadata when True.

        Outputs
        -------
        roots: np.array, complex
            Untracked kz roots (km^-1), shape (n_omega, raw_root_count), with NaN
            padding.

        point_info: list of dict, optional
            One metadata dictionary per frequency, returned when return_info is True.

        Author(s)
        ---------
        Julio M. Morales, August 07th, 2025
        '''

        # Allocate raw roots and preserve one solve-status dictionary per sampled frequency.
        roots = np.full((len(self.omega), self._raw_kz_root_count()), np.nan + 0.0j, dtype=np.complex128)
        point_info = []

        # Loop through frequency
        for j, omega_j in enumerate(self.omega):
            roots[j, :], info = self._solve_kz_roots_at_point(
                omega_j, kh, return_info=True
            )
            point_info.append(info)

        return (roots, point_info) if return_info else roots

    def kz_solve(self, *args, return_diagnostics=False, **tracking_kwargs):

        '''
        Purpose
        -------
        Solve for continuity-tracked vertical-wavenumber branches over the full
        omega-kh grid.

        The previous public use of kz_solve(kh) returned one untracked column
        and encouraged downstream stacking/manual branch fixes. That path is
        intentionally disabled so kz_solve() and kz_solve_grid() both use the
        same global branch-tracking workflow.

        Inputs
        ------
        *args: tuple
            No positional arguments are accepted; legacy single-kh calls raise
            TypeError.

        return_diagnostics: bool, optional
            Return the result and its diagnostic dictionary when True.

        **tracking_kwargs: dict
            Keyword arguments forwarded unchanged to kz_solve_grid().

        Outputs
        -------
        roots: np.array, complex
            Full-grid tracked/selected kz branches (km^-1), shape
            (n_omega, n_kh, kz_order), as returned by kz_solve_grid().

        diagnostics: dict, optional
            Full-grid diagnostics when return_diagnostics is True.
        '''

        # Reject the former single-column call to ensure public results use global tracking.
        if len(args) > 0:
            raise TypeError("kz_solve() now solves the full grid with branch tracking. Use kz_solve_grid(return_diagnostics=True) instead of passing one kh value.")

        # Forward all accepted keyword options to the full-grid solver.
        return self.kz_solve_grid(return_diagnostics=return_diagnostics, **tracking_kwargs)

    def _kz_solve_raw_grid(self, return_info=False):

        '''
        Purpose
        -------
        Solve the kz polynomial at every omega-kh grid point before branch
        tracking. Raw roots are retained only for diagnostics.

        Inputs
        ------
        return_info: bool, optional
            Return the roots and solve metadata when True.

        Outputs
        -------
        roots: np.array, complex
            Raw kz solver roots (km^-1), shape (n_omega, n_kh, raw_root_count).
            The separate slow/fast models retain all four raw roots before selection.

        info: dict, optional
            Two-dimensional coefficient-degree, degree-reduction, zero-polynomial,
            nonfinite-coefficient, solver-failure, and root-scale maps.
        '''

        # Allocate the complete raw-root cube and two-dimensional solve-status maps.
        roots = np.full((len(self.omega), len(self.kh), self._raw_kz_root_count()), np.nan + 0.0j, dtype=np.complex128)
        info = {
            'coefficient_degree': np.full(roots.shape[:2], -1, dtype=int),
            'effective_degree': np.full(roots.shape[:2], -1, dtype=int),
            'degree_reduced': np.zeros(roots.shape[:2], dtype=bool),
            'zero_polynomial': np.zeros(roots.shape[:2], dtype=bool),
            'solver_failed': np.zeros(roots.shape[:2], dtype=bool),
            'nonfinite_coefficients': np.zeros(roots.shape[:2], dtype=bool),
            'root_scale': np.full(roots.shape[:2], np.nan),
        }
        # Solve each kh column and place its point metadata at matching grid coordinates.
        for i, kh_i in enumerate(self.kh):
            roots[:, i, :], column_info = self._kz_solve_column_raw(
                kh_i, return_info=True
            )
            for j, point_info in enumerate(column_info):
                for key in info:
                    info[key][j, i] = point_info[key]

        return (roots, info) if return_info else roots

    def _root_geometry_maps(self, raw_roots, degenerate_rtol, degenerate_atol):

        '''
        Purpose
        -------
        Measure finite-root counts, local separations, scales, and degeneracies.

        Inputs
        ------
        raw_roots: np.array, complex
            Root cube (km^-1), shape (n_omega, n_kh, branch_count).

        degenerate_rtol: float, optional
            Relative root-separation tolerance used to flag near-coincident roots.

        degenerate_atol: float, optional
            Absolute root-separation tolerance in the units of the supplied roots.

        Outputs
        -------
        finite_count: np.array, int
            Finite-root count per (omega, kh) cell.

        min_separation: np.array, float
            Minimum root separation (km^-1); NaN with fewer than two finite roots.

        local_scale: np.array, float
            Largest of median root magnitude, finite minimum separation, and absolute
            tolerance (km^-1) at each grid cell.

        degenerate_mask: np.array, bool
            Cells with multiple roots separated by no more than the local tolerance.
        '''

        # Allocate local root-count, separation, and scale maps.
        n_omega, n_kh, _ = raw_roots.shape
        finite_count = np.zeros((n_omega, n_kh), dtype=int)
        min_separation = np.full((n_omega, n_kh), np.nan)
        local_scale = np.full((n_omega, n_kh), degenerate_atol)

        # Measure pairwise separation using only finite roots in each cell.
        for index in np.ndindex(n_omega, n_kh):
            roots = raw_roots[index]
            finite_roots = roots[self._finite_complex_mask(roots)]
            finite_count[index] = finite_roots.size

            if finite_roots.size == 0:
                continue

            median_abs = np.median(np.abs(finite_roots))
            min_sep = self._min_pairwise_distance(finite_roots)
            if np.isfinite(min_sep):
                min_separation[index] = min_sep
                local_scale[index] = max(median_abs, min_sep, degenerate_atol)
            else:
                local_scale[index] = max(median_abs, degenerate_atol)

        # Compare the minimum separation with the local absolute/relative degeneracy
        # threshold.
        threshold = degenerate_atol + degenerate_rtol*local_scale
        degenerate_mask = (finite_count > 1) & np.isfinite(min_separation) & (min_separation <= threshold)

        return finite_count, min_separation, local_scale, degenerate_mask

    def _select_tracking_seed(self, raw_roots, assigned, finite_count, min_separation):

        '''
        Purpose
        -------
        Choose an unassigned grid point with well-separated roots as a new seed.

        Inputs
        ------
        raw_roots: np.array, complex
            Root cube retained for signature compatibility; not read in this method.

        assigned: np.array, bool
            Cells that already have global branch labels, shape (n_omega, n_kh).

        finite_count: np.array, int
            Finite-root count at each cell.

        min_separation: np.array, float
            Minimum pairwise root separation (km^-1) at each cell.

        Outputs
        -------
        seed: tuple of int or None
            Unassigned (omega_index, kh_index), preferring the most roots, largest
            separation, shortest Manhattan distance to grid center, then lower indices.
            Return None when no unassigned cell contains a finite root.
        '''

        # Restrict candidates to unassigned cells containing at least one finite root.
        available = (~assigned) & (finite_count > 0)
        if not np.any(available):
            return None

        # Prefer the fullest root sets before comparing separation and grid-center distance.
        max_count = np.max(finite_count[available])
        candidates = np.argwhere(available & (finite_count == max_count))
        center = 0.5*np.asarray(available.shape)

        # Break remaining ties deterministically by row and column indices.
        best_index = None
        best_key = None
        for i, j in candidates:
            separation = min_separation[i, j]
            if not np.isfinite(separation):
                separation = -np.inf
            center_distance = np.sum(np.abs(np.asarray([i, j]) - center))
            key = (separation, -center_distance, -i, -j)
            if best_key is None or key > best_key:
                best_key = key
                best_index = (int(i), int(j))

        return best_index

    def _sorted_branch_values(self, roots, branch_count):

        '''
        Purpose
        -------
        Place finite roots into deterministic order and pad missing branches with NaN.

        Inputs
        ------
        roots: np.array, complex
            One-dimensional local root set (km^-1).

        branch_count: int
            Output width; truncate surplus roots or pad missing ones.

        Outputs
        -------
        values: np.array, complex
            Finite roots sorted by real then imaginary part, followed by NaN slots.
        '''

        # Place sorted finite roots into a fixed-width complex row with NaN padding.
        values = np.full(branch_count, np.nan + 0.0j, dtype=np.complex128)
        sorted_indices = self._sorted_finite_indices(roots)
        for branch, root_index in enumerate(sorted_indices[:branch_count]):
            values[branch] = roots[root_index]

        return values

    def _match_root_sets(self, source_roots, target_roots):

        '''
        Purpose
        -------
        Match neighboring root sets by minimum complex-plane distance.

        Inputs
        ------
        source_roots: np.array, complex
            One-dimensional roots at the source cell.

        target_roots: np.array, complex
            One-dimensional roots at the neighboring target cell; same physical units.

        Outputs
        -------
        permutation: np.array, int
            Target index for each source slot, or -1 when unmatched. Finite roots
            are assigned one-to-one by Hungarian complex-plane distance minimization.

        stats: dict
            Matched count, mean/max assigned distances, and minimum alternative-match
            margin, in the root units. Unavailable statistics remain NaN.
        '''

        # Initialize unmatched source slots and unavailable assignment statistics.
        branch_count = source_roots.size
        permutation = np.full(branch_count, -1, dtype=int)
        stats = {
            'matched_count': 0,
            'assignment_cost': np.nan,
            'assignment_margin': np.nan,
            'max_assignment_cost': np.nan
        }

        # Use only finite roots while retaining the original source and target indices.
        source_indices = self._sorted_finite_indices(source_roots)
        target_indices = self._sorted_finite_indices(target_roots)
        if source_indices.size == 0 or target_indices.size == 0:
            return permutation, stats

        # Construct the complete complex-plane distance matrix.
        source_values = source_roots[source_indices]
        target_values = target_roots[target_indices]
        cost = np.abs(source_values[:, None] - target_values[None, :])

        # Add a machine-scale deterministic tie break before Hungarian assignment.
        scale_values = np.concatenate((source_values, target_values))
        scale_values = scale_values[self._finite_complex_mask(scale_values)]
        cost_scale = max(np.median(np.abs(scale_values)), self._min_pairwise_distance(target_values), 1e-30)
        tie_break = np.finfo(float).eps*cost_scale*(
            source_indices[:, None]*(branch_count + 1) + target_indices[None, :]
        )
        rows, cols = optimize.linear_sum_assignment(cost + tie_break)

        # Map the assigned finite positions back into the original root slots.
        for row, col in zip(rows, cols):
            permutation[source_indices[row]] = target_indices[col]

        # Summarize distances without the numerical tie-breaking perturbation.
        matched_costs = cost[rows, cols]
        if matched_costs.size > 0:
            stats['matched_count'] = int(matched_costs.size)
            stats['assignment_cost'] = float(np.mean(matched_costs))
            stats['max_assignment_cost'] = float(np.max(matched_costs))

            # Measure the nearest alternative assignment in each matched row and column.
            margins = []
            for row, col, matched_cost in zip(rows, cols, matched_costs):
                if target_values.size > 1:
                    other_cols = np.arange(target_values.size) != col
                    margins.append(np.min(cost[row, other_cols]) - matched_cost)
                if source_values.size > 1:
                    other_rows = np.arange(source_values.size) != row
                    margins.append(np.min(cost[other_rows, col]) - matched_cost)
            if len(margins) > 0:
                stats['assignment_margin'] = float(np.min(margins))

        return permutation, stats

    def _invert_permutation(self, permutation):

        '''
        Purpose
        -------
        Return the inverse mapping while retaining -1 for unmatched indices.

        Inputs
        ------
        permutation: np.array, int
            Source-to-target indices, with -1 for unmatched entries.

        Outputs
        -------
        inverse: np.array, int
            Target-to-source indices with the same shape and -1 for unmatched slots.
        '''

        # Reverse matched indices and preserve -1 for target slots with no source.
        inverse = np.full_like(permutation, -1)
        for source_index, target_index in enumerate(permutation):
            if target_index >= 0:
                inverse[target_index] = source_index

        return inverse

    def _edge_confidence(self, cost, margin):

        '''
        Purpose
        -------
        Convert assignment cost and margin into a bounded confidence score.

        Inputs
        ------
        cost: float
            Maximum assigned root distance on an edge.

        margin: float
            Smallest alternative-match margin, in the same units as cost.

        Outputs
        -------
        confidence: float
            Bounded margin/(margin + cost + 1e-30) score; nonfinite costs or absent
            positive margins produce zero confidence.
        '''

        # Assign zero confidence when no finite cost or positive alternative margin is
        # available.
        if not np.isfinite(cost):
            return 0.0
        if not np.isfinite(margin):
            margin = 0.0

        # Convert the nonnegative margin into a bounded dimensionless confidence score.
        margin = max(0.0, margin)
        return margin / (margin + cost + 1e-30)

    def _edge_match_maps(self, raw_roots):

        '''
        Purpose
        -------
        Build root assignments and quality metrics for adjacent grid cells.

        Inputs
        ------
        raw_roots: np.array, complex
            Untracked root cube, shape (n_omega, n_kh, branch_count).

        Outputs
        -------
        edge_maps: dict
            Rightward and downward root permutations, mean/max costs, margins, and
            matched counts. Horizontal edge maps have shape (n_omega, n_kh - 1) and
            vertical maps (n_omega - 1, n_kh), with a final branch axis on permutations;
            zero-length axes are retained when the corresponding grid dimension is one.
        '''

        # Allocate source-to-neighbor assignments and quality maps for both grid directions.
        n_omega, n_kh, branch_count = raw_roots.shape
        right_perm = np.full((n_omega, max(n_kh - 1, 0), branch_count), -1, dtype=int)
        down_perm = np.full((max(n_omega - 1, 0), n_kh, branch_count), -1, dtype=int)
        right_cost = np.full((n_omega, max(n_kh - 1, 0)), np.nan)
        down_cost = np.full((max(n_omega - 1, 0), n_kh), np.nan)
        right_max_cost = np.full_like(right_cost, np.nan)
        down_max_cost = np.full_like(down_cost, np.nan)
        right_margin = np.full_like(right_cost, np.nan)
        down_margin = np.full_like(down_cost, np.nan)
        right_matched = np.zeros(right_cost.shape, dtype=int)
        down_matched = np.zeros(down_cost.shape, dtype=int)

        # Match every horizontal edge from its left cell to its right cell.
        for i in range(n_omega):
            for j in range(n_kh - 1):
                permutation, stats = self._match_root_sets(raw_roots[i, j, :], raw_roots[i, j + 1, :])
                right_perm[i, j, :] = permutation
                right_cost[i, j] = stats['assignment_cost']
                right_max_cost[i, j] = stats['max_assignment_cost']
                right_margin[i, j] = stats['assignment_margin']
                right_matched[i, j] = stats['matched_count']

        # Match every vertical edge from its lower-index row to the next row.
        for i in range(n_omega - 1):
            for j in range(n_kh):
                permutation, stats = self._match_root_sets(raw_roots[i, j, :], raw_roots[i + 1, j, :])
                down_perm[i, j, :] = permutation
                down_cost[i, j] = stats['assignment_cost']
                down_max_cost[i, j] = stats['max_assignment_cost']
                down_margin[i, j] = stats['assignment_margin']
                down_matched[i, j] = stats['matched_count']

        # Return permutations separately from costs, margins, and matched-root counts.
        return {
            'right_perm': right_perm,
            'down_perm': down_perm,
            'right_cost': right_cost,
            'down_cost': down_cost,
            'right_max_cost': right_max_cost,
            'down_max_cost': down_max_cost,
            'right_margin': right_margin,
            'down_margin': down_margin,
            'right_matched': right_matched,
            'down_matched': down_matched
        }

    def _edge_max_to_pixel_map(self, right_values, down_values, shape):

        '''
        Purpose
        -------
        Project edge values to both neighboring pixels using their maximum.

        Inputs
        ------
        right_values: np.array, float
            Horizontal edge values, shape (n_omega, n_kh - 1).

        down_values: np.array, float
            Vertical edge values, shape (n_omega - 1, n_kh).

        shape: tuple of int
            Output pixel-map shape (n_omega, n_kh).

        Outputs
        -------
        pixel_map: np.array, float
            Finite incident-edge maximum at each pixel, retaining the edge units.
            Pixels without finite incident edges remain NaN.
        '''

        # Start with unavailable pixels, then accumulate finite incident-edge maxima.
        pixel_map = np.full(shape, np.nan)
        # Apply each horizontal edge to both adjacent pixels.
        if right_values.size > 0:
            self._update_max_map(pixel_map[:, :-1], right_values)
            self._update_max_map(pixel_map[:, 1:], right_values)
        # Apply each vertical edge to both adjacent pixels.
        if down_values.size > 0:
            self._update_max_map(pixel_map[:-1, :], down_values)
            self._update_max_map(pixel_map[1:, :], down_values)

        return pixel_map

    def _edge_min_to_pixel_map(self, right_values, down_values, shape):

        '''
        Purpose
        -------
        Project edge values to both neighboring pixels using their minimum.

        Inputs
        ------
        right_values: np.array, float
            Horizontal edge values, shape (n_omega, n_kh - 1).

        down_values: np.array, float
            Vertical edge values, shape (n_omega - 1, n_kh).

        shape: tuple of int
            Output pixel-map shape (n_omega, n_kh).

        Outputs
        -------
        pixel_map: np.array, float
            Finite incident-edge minimum at each pixel, retaining the edge units.
            Pixels without finite incident edges remain NaN.
        '''

        # Start with unavailable pixels before accumulating finite incident-edge minima.
        pixel_map = np.full(shape, np.nan)

        def update_min(target, values):

            '''
            Purpose
            -------
            Update finite target cells with the elementwise minimum of new values.

            Inputs
            ------
            target: np.array, float
                Writable view of the pixel map to update in place.

            values: np.array, float
                Incident edge values matching the view; nonfinite values are ignored.

            Outputs
            -------
            None
                Keep the smaller finite value and initialize previously nonfinite
                targets.
            '''
            # Keep finite minima while initializing targets that have no previous finite
            # value.
            valid = np.isfinite(values)
            current_valid = np.isfinite(target)
            target[valid & current_valid] = np.minimum(target[valid & current_valid], values[valid & current_valid])
            target[valid & ~current_valid] = values[valid & ~current_valid]

        # Apply horizontal-edge minima to both neighboring pixels.
        if right_values.size > 0:
            update_min(pixel_map[:, :-1], right_values)
            update_min(pixel_map[:, 1:], right_values)
        # Apply vertical-edge minima to both neighboring pixels.
        if down_values.size > 0:
            update_min(pixel_map[:-1, :], down_values)
            update_min(pixel_map[1:, :], down_values)

        return pixel_map

    def _values_from_local_indices(self, raw_roots, local_index_by_branch):

        '''
        Purpose
        -------
        Gather tracked branch values from each cell's raw-root indices.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        local_index_by_branch: np.array, int
            Raw-root index for each labeled branch, with shape (n_omega, n_kh,
            n_branch); -1 marks an unavailable root.

        Outputs
        -------
        values: np.array, complex
            Root grid matching raw_roots.shape, in km^-1; negative branch indices
            remain NaN. Positive out-of-range indices raise IndexError.
        '''

        # Initialize missing values before gathering each branch from its raw slot.
        values = np.full(raw_roots.shape, np.nan + 0.0j, dtype=np.complex128)
        # Copy only assigned indices; retain NaN for unavailable branches.
        for i, j in np.ndindex(raw_roots.shape[:2]):
            for branch, root_index in enumerate(local_index_by_branch[i, j, :]):
                if root_index >= 0:
                    values[i, j, branch] = raw_roots[i, j, root_index]

        return values

    class _ParityDisjointSet:

        '''
        Purpose
        -------
        Track connected components and pair-swap parity constraints.

        Inputs
        ------
        size: int
            Number of nodes passed to the constructor.

        Outputs
        -------
        instance: _ParityDisjointSet
            Mutable parent, rank, and parity_to_parent arrays used for binary
            branch-swap constraints.
        '''

        def __init__(self, size):

            '''
            Purpose
            -------
            Initialize singleton sets with zero parity to each parent.

            Inputs
            ------
            size: int
                Number of grid nodes represented by the disjoint-set arrays.

            Outputs
            -------
            None
                Initialize parent, rank, and parity_to_parent on this instance.
            '''

            # Start each node as its own component with no relative branch swap.
            self.parent = np.arange(size, dtype=int)
            self.rank = np.zeros(size, dtype=int)
            self.parity_to_parent = np.zeros(size, dtype=int)

        def find(self, node):

            '''
            Purpose
            -------
            Return a node's representative and parity to that representative.

            Inputs
            ------
            node: int
                Zero-based node index in the disjoint-set structure.

            Outputs
            -------
            root, parity: tuple of int
                Representative node and XOR parity relative to it. Compress the
                traversed parent path in place.
            '''

            # Accumulate XOR parity while compressing the path to the representative.
            parent = self.parent[node]
            if parent != node:
                root, parity_to_root = self.find(parent)
                self.parity_to_parent[node] ^= parity_to_root
                self.parent[node] = root

            return self.parent[node], self.parity_to_parent[node]

        def union(self, node_a, node_b, parity):

            '''
            Purpose
            -------
            Join two sets with the requested relative parity, detecting conflicts.

            Inputs
            ------
            node_a: int
                Zero-based index of the first node to constrain.

            node_b: int
                Zero-based index of the second node to constrain.

            parity: int
                Required relative parity: zero keeps the pair order and one swaps it.

            Outputs
            -------
            consistent: bool
                True when the requested relation is accepted or already satisfied;
                False for a contradictory relation. Accepted joins update the set in
                place.
            '''

            # Resolve both representatives and their current relative parities.
            root_a, parity_a = self.find(node_a)
            root_b, parity_b = self.find(node_b)

            # Check an existing relation without changing an already joined component.
            if root_a == root_b:
                return (parity_a ^ parity_b) == parity

            # Attach the shallower tree and preserve the requested XOR relation.
            if self.rank[root_a] < self.rank[root_b]:
                root_a, root_b = root_b, root_a
                parity_a, parity_b = parity_b, parity_a

            # Store the parity between the two representatives after joining them.
            self.parent[root_b] = root_a
            self.parity_to_parent[root_b] = parity_a ^ parity_b ^ parity
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_a] += 1

            return True

    def _pair_swap_constraints(
        self,
        branch_values,
        branch_a,
        branch_b,
        ratio_threshold,
        complex_weight=1.0,
        real_weight=0.0
    ):

        '''
        Purpose
        -------
        Score same-order and swapped continuation for a pair of branches.

        Inputs
        ------
        branch_values: np.array, complex
            Labeled vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        branch_a: int
            Zero-based index of the first branch in the pair.

        branch_b: int
            Zero-based index of the second branch in the pair.

        ratio_threshold: float
            Minimum ratio between competing continuation costs for an edge to supply
            a parity constraint.

        complex_weight: float
            Weight of squared complex-root displacement in the continuation cost.

        real_weight: float
            Additional weight of squared real-part displacement in the continuation
            cost.

        Outputs
        -------
        constraints: list of tuple
            Edge votes (confidence, cost gain, source node, target node, parity),
            sorted by decreasing confidence and gain.
        '''

        # Collect edge votes from both sampled grid directions.
        n_omega, n_kh, _ = branch_values.shape
        constraints = []

        def append_constraints(source_values, target_values, source_nodes, target_nodes):

            '''
            Purpose
            -------
            Add reliable neighboring-cell parity constraints for the branch pair.

            Inputs
            ------
            source_values: np.array, complex
                Root values at the source of each edge, with shape (n_edges, n_branch).

            target_values: np.array, complex
                Root values at the target of each edge, with shape (n_edges, n_branch).

            source_nodes: np.array, int
                Flattened grid indices for the source cells of the supplied edges.

            target_nodes: np.array, int
                Flattened grid indices for the target cells of the supplied edges.

            Outputs
            -------
            None
                Append finite, sufficiently separated edge votes to the enclosing
                constraints list.
            '''
            # Require both branches to be finite at both ends of an edge.
            valid = (
                self._finite_complex_mask(source_values[:, branch_a])
                & self._finite_complex_mask(source_values[:, branch_b])
                & self._finite_complex_mask(target_values[:, branch_a])
                & self._finite_complex_mask(target_values[:, branch_b])
            )
            if not np.any(valid):
                return

            # Compare complex displacement for unchanged and exchanged branch labels.
            same_cost = complex_weight*(
                np.abs(source_values[:, branch_a] - target_values[:, branch_a])**2
                + np.abs(source_values[:, branch_b] - target_values[:, branch_b])**2
            )
            swap_cost = complex_weight*(
                np.abs(source_values[:, branch_a] - target_values[:, branch_b])**2
                + np.abs(source_values[:, branch_b] - target_values[:, branch_a])**2
            )
            # Include the optional real-part cost in both competing assignments.
            if real_weight != 0.0:
                same_cost = same_cost + real_weight*(
                    (source_values[:, branch_a].real - target_values[:, branch_a].real)**2
                    + (source_values[:, branch_b].real - target_values[:, branch_b].real)**2
                )
                swap_cost = swap_cost + real_weight*(
                    (source_values[:, branch_a].real - target_values[:, branch_b].real)**2
                    + (source_values[:, branch_b].real - target_values[:, branch_a].real)**2
                )
            # Keep edges only when the competing costs are sufficiently separated.
            min_cost = np.minimum(same_cost, swap_cost)
            max_cost = np.maximum(same_cost, swap_cost)
            separated = max_cost > ratio_threshold*np.maximum(min_cost, 1e-300)
            use = valid & separated & np.isfinite(min_cost) & np.isfinite(max_cost)
            if not np.any(use):
                return

            # Convert the winning cost into a confidence score and binary swap vote.
            gain = max_cost[use] - min_cost[use]
            confidence = gain / np.maximum(max_cost[use] + min_cost[use], 1e-300)
            parity = (swap_cost[use] < same_cost[use]).astype(int)

            # Append the accepted votes with their flattened source and target nodes.
            for item in zip(confidence, gain, source_nodes[use], target_nodes[use], parity):
                constraints.append(item)

        # Flatten horizontal neighbors with their corresponding grid-node indices.
        if n_kh > 1:
            source = branch_values[:, :-1, :].reshape(-1, branch_values.shape[2])
            target = branch_values[:, 1:, :].reshape(-1, branch_values.shape[2])
            source_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:, :-1].ravel()
            target_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:, 1:].ravel()
            append_constraints(source, target, source_nodes, target_nodes)

        # Repeat the edge construction along the frequency direction.
        if n_omega > 1:
            source = branch_values[:-1, :, :].reshape(-1, branch_values.shape[2])
            target = branch_values[1:, :, :].reshape(-1, branch_values.shape[2])
            source_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:-1, :].ravel()
            target_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[1:, :].ravel()
            append_constraints(source, target, source_nodes, target_nodes)

        # Process the most confident, largest-gain constraints first.
        constraints.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return constraints

    def _pair_swap_parity(
        self,
        branch_values,
        branch_a,
        branch_b,
        ratio_threshold,
        complex_weight=1.0,
        real_weight=0.0
    ):

        '''
        Purpose
        -------
        Infer a consistent swap region from weighted neighboring-cell constraints.

        Inputs
        ------
        branch_values: np.array, complex
            Labeled vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        branch_a: int
            Zero-based index of the first branch in the pair.

        branch_b: int
            Zero-based index of the second branch in the pair.

        ratio_threshold: float
            Minimum ratio between competing continuation costs for an edge to supply
            a parity constraint.

        complex_weight: float
            Weight of squared complex-root displacement in the continuation cost.

        real_weight: float
            Additional weight of squared real-part displacement in the continuation
            cost.

        Outputs
        -------
        parity: np.array, bool
            Swap selection with shape (n_omega, n_kh), referenced to each accepted
            component's representative.

        stats: dict
            Branch pair, accepted/rejected edge counts, accepted flip count, and
            number of selected cells.
        '''

        # Build pairwise edge constraints for the full two-dimensional grid.
        n_omega, n_kh, _ = branch_values.shape
        constraints = self._pair_swap_constraints(
            branch_values,
            branch_a,
            branch_b,
            ratio_threshold,
            complex_weight=complex_weight,
            real_weight=real_weight
        )
        # Allocate a parity union-find and counters for accepted and conflicting votes.
        disjoint_set = self._ParityDisjointSet(n_omega*n_kh)
        accepted = 0
        rejected = 0
        accepted_flips = 0

        # Join components in confidence order, rejecting contradictory parity votes.
        for _, _, source_node, target_node, parity in constraints:
            if disjoint_set.union(int(source_node), int(target_node), int(parity)):
                accepted += 1
                accepted_flips += int(parity)
            else:
                rejected += 1

        # Recover each node's swap state relative to its component representative.
        parity = np.zeros(n_omega*n_kh, dtype=bool)
        for node in range(n_omega*n_kh):
            _, node_parity = disjoint_set.find(node)
            parity[node] = bool(node_parity)

        # Summarize the evidence and the resulting region size.
        stats = {
            'pair': (int(branch_a), int(branch_b)),
            'constraints': int(len(constraints)),
            'accepted_constraints': int(accepted),
            'rejected_constraints': int(rejected),
            'accepted_flip_constraints': int(accepted_flips),
            'swap_pixels': int(np.count_nonzero(parity))
        }

        return parity.reshape(n_omega, n_kh), stats

    def _apply_pair_swap(self, local_index_by_branch, branch_a, branch_b, parity):

        '''
        Purpose
        -------
        Swap two branch indices in cells selected by the parity mask.

        Inputs
        ------
        local_index_by_branch: np.array, int
            Raw-root index for each labeled branch, with shape (n_omega, n_kh,
            n_branch); -1 marks an unavailable root.

        branch_a: int
            Zero-based index of the first branch in the pair.

        branch_b: int
            Zero-based index of the second branch in the pair.

        parity: np.array, bool
            Cells where the two branches are to be exchanged, shape (n_omega, n_kh).

        Outputs
        -------
        corrected_indices: np.array, int
            Copy of the index grid with the two valid branch indices exchanged where
            parity is True.

        changed_pixels: int
            Number of cells where both branch indices were valid and swapped.
        '''

        # Copy the label map so the caller's input remains unchanged.
        corrected_indices = local_index_by_branch.copy()
        swap_mask = np.asarray(parity, dtype=bool)
        if not np.any(swap_mask):
            return corrected_indices, 0

        # Exclude cells where either requested branch lacks a raw-root assignment.
        valid_swap = (
            swap_mask
            & (corrected_indices[:, :, branch_a] >= 0)
            & (corrected_indices[:, :, branch_b] >= 0)
        )
        if not np.any(valid_swap):
            return corrected_indices, 0

        # Exchange paired indices using a temporary copy to avoid overwriting values.
        branch_a_view = corrected_indices[:, :, branch_a]
        branch_b_view = corrected_indices[:, :, branch_b]
        branch_a_indices = branch_a_view[valid_swap].copy()
        branch_a_view[valid_swap] = branch_b_view[valid_swap]
        branch_b_view[valid_swap] = branch_a_indices

        return corrected_indices, int(np.count_nonzero(valid_swap))

    def _pair_identity_objective(
        self,
        branch_values,
        branch_a,
        branch_b,
        complex_weight=1.0,
        real_weight=1.0,
        imag_weight=0.25,
        curvature_weight=6.0
    ):

        '''
        Purpose
        -------
        Score a candidate assignment for one branch pair.

        Inputs
        ------
        branch_values: np.array, complex
            Labeled vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        branch_a: int
            Zero-based index of the first branch in the pair.

        branch_b: int
            Zero-based index of the second branch in the pair.

        complex_weight: float
            Weight of squared complex-root displacement in the continuation cost.

        real_weight: float
            Additional weight of squared real-part displacement in the continuation
            cost.

        imag_weight: float
            Weight of imaginary-part differences in the global identity objective.

        curvature_weight: float
            Weight of second differences relative to first differences in the
            identity objective.

        Outputs
        -------
        objective: dict
            Weighted total and mean squared complex, real, and imaginary jumps and
            curvatures. Unavailable components are NaN and do not contribute to
            total.

        Notes
        -----
        The local branch-swap pass minimizes nearest-neighbor displacement in
        the complex plane. A broad, coherent misassignment can still look
        locally continuous while giving a worse 2D sheet. This objective is
        deliberately global: it measures first-neighbor continuity and second
        differences of the pair over the whole grid. Candidate region swaps are
        accepted only when this score decreases, so ambiguous true crossings are
        reported instead of being relabeled by noise.
        '''

        # Restrict the global continuity objective to the selected branch pair.
        values = np.asarray(branch_values[:, :, [branch_a, branch_b]], dtype=np.complex128)

        def diff_energy(array, order):

            '''
            Purpose
            -------
            Calculate the mean squared finite difference over both grid axes.

            Inputs
            ------
            array: np.array
                Real or complex paired-branch grid to difference along the first two
                axes.

            order: int
                Order of the finite difference; one measures jumps and two measures
                curvature.

            Outputs
            -------
            energy: float
                Mean squared finite difference over both grid axes and both selected
                branches; NaN when no differences are available.
            '''
            # Accumulate squared finite differences and normalize by their sample count.
            total = 0.0
            count = 0
            # Skip grid directions too short for the requested difference order.
            for axis in (0, 1):
                if array.shape[axis] <= order:
                    continue
                diff = np.diff(array, n=order, axis=axis)
                valid = np.isfinite(diff)
                if not np.any(valid):
                    continue
                total += float(np.sum(np.abs(diff[valid])**2))
                count += int(np.count_nonzero(valid))
            # Represent an unavailable difference statistic with NaN.
            if count == 0:
                return np.nan
            return total / count

        # Measure first differences and second differences across both grid axes.
        complex_jump = diff_energy(values, 1)
        real_jump = diff_energy(values.real, 1)
        imag_jump = diff_energy(values.imag, 1)
        real_curvature = diff_energy(values.real, 2)
        imag_curvature = diff_energy(values.imag, 2)

        # Combine only finite objective components with their configured weights.
        total = 0.0
        for weight, item in [
            (complex_weight, complex_jump),
            (real_weight, real_jump),
            (imag_weight, imag_jump),
            (curvature_weight, real_curvature),
            (curvature_weight*imag_weight, imag_curvature)
        ]:
            if np.isfinite(item):
                total += weight*item

        # Retain individual terms alongside the total for candidate diagnostics.
        return {
            'total': float(total),
            'complex_jump': float(complex_jump) if np.isfinite(complex_jump) else np.nan,
            'real_jump': float(real_jump) if np.isfinite(real_jump) else np.nan,
            'imag_jump': float(imag_jump) if np.isfinite(imag_jump) else np.nan,
            'real_curvature': float(real_curvature) if np.isfinite(real_curvature) else np.nan,
            'imag_curvature': float(imag_curvature) if np.isfinite(imag_curvature) else np.nan
        }

    def _refine_branch_identity(
        self,
        raw_roots,
        local_index_by_branch,
        max_iterations=2,
        ratio_threshold=1.2,
        complex_weight=1.0,
        real_weight=2.0,
        objective_improvement_rtol=1e-3
    ):

        '''
        Purpose
        -------
        Test and apply large-scale pair swaps missed by local continuity.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        local_index_by_branch: np.array, int
            Raw-root index for each labeled branch, with shape (n_omega, n_kh,
            n_branch); -1 marks an unavailable root.

        max_iterations: int
            Maximum number of full passes through all branch pairs.

        ratio_threshold: float
            Minimum ratio between competing continuation costs for an edge to supply
            a parity constraint.

        complex_weight: float
            Weight of squared complex-root displacement in the continuation cost.

        real_weight: float
            Additional weight of squared real-part displacement in the continuation
            cost.

        objective_improvement_rtol: float
            Minimum fractional reduction of the global identity objective required
            to accept a region swap.

        Outputs
        -------
        branch_values: np.array, complex
            Relabeled raw roots, with the input shape and units (km^-1).

        corrected_indices: np.array, int
            Corresponding branch-to-raw-root index grid; the input grid is not modified.

        stats: dict
            Iteration counts and per-pair candidate, objective, acceptance, and
            rejection records.

        Notes
        -----
        For each branch pair, edges vote for a binary parity just as in the
        branch-cut correction pass, but the vote includes the real part of
        k_z because Delta phi is proportional to Re(k_z). The proposed
        contiguous-region swap is then accepted only if a global first- and
        second-difference objective improves. This catches coherent branch
        exchanges while rejecting true near-degeneracies where either labeling
        is locally ambiguous and no global continuity is gained.
        '''

        # Copy the current labeling and initialize per-pair refinement records.
        corrected_indices = local_index_by_branch.copy()
        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
        branch_count = raw_roots.shape[2]
        pair_stats = []
        total_changed_pixels = 0
        # Stop once a complete pass makes no accepted changes.
        completed_iterations = 0

        # Test every branch pair until no region swap is accepted.
        for iteration in range(max_iterations):
            iteration_changed_pixels = 0
            for branch_a in range(branch_count):
                for branch_b in range(branch_a + 1, branch_count):
                    # Construct a coherent candidate using complex and real-part edge
                    # costs.
                    parity, stats = self._pair_swap_parity(
                        branch_values,
                        branch_a,
                        branch_b,
                        ratio_threshold,
                        complex_weight=complex_weight,
                        real_weight=real_weight
                    )
                    # Record candidates without supported flip regions and continue.
                    if stats['accepted_flip_constraints'] == 0 or stats['swap_pixels'] == 0:
                        pair_stats.append({
                            **stats,
                            'iteration': int(iteration),
                            'changed_pixels': 0,
                            'accepted': False,
                            'rejected_reason': 'no coherent swap candidate'
                        })
                        continue

                    # Apply the candidate only to cells where both raw-root indices
                    # exist.
                    updated_indices, changed_pixels = self._apply_pair_swap(
                        corrected_indices,
                        branch_a,
                        branch_b,
                        parity
                    )
                    if changed_pixels == 0:
                        pair_stats.append({
                            **stats,
                            'iteration': int(iteration),
                            'changed_pixels': 0,
                            'accepted': False,
                            'rejected_reason': 'candidate contained no valid pixels'
                        })
                        continue

                    # Compare current and candidate labels using the same global
                    # objective.
                    before_objective = self._pair_identity_objective(branch_values, branch_a, branch_b)
                    candidate_values = self._values_from_local_indices(raw_roots, updated_indices)
                    after_objective = self._pair_identity_objective(candidate_values, branch_a, branch_b)
                    improvement = before_objective['total'] - after_objective['total']
                    improvement_rtol = improvement / max(abs(before_objective['total']), 1e-300)
                    accepted = (
                        np.isfinite(improvement_rtol)
                        and improvement_rtol > objective_improvement_rtol
                    )

                    # Commit only candidates whose fractional objective improvement
                    # passes the threshold.
                    if accepted:
                        corrected_indices = updated_indices
                        branch_values = candidate_values
                        iteration_changed_pixels += changed_pixels
                        total_changed_pixels += changed_pixels

                    # Keep acceptance, rejection, and objective evidence for this pair.
                    pair_stats.append({
                        **stats,
                        'iteration': int(iteration),
                        'changed_pixels': int(changed_pixels if accepted else 0),
                        'candidate_pixels': int(changed_pixels),
                        'accepted': bool(accepted),
                        'rejected_reason': None if accepted else 'global objective did not improve',
                        'objective_before': before_objective,
                        'objective_after': after_objective,
                        'objective_improvement': float(improvement),
                        'objective_improvement_rtol': float(improvement_rtol)
                    })

            completed_iterations = iteration + 1
            if iteration_changed_pixels == 0:
                break

        # Return the final relabeling and a complete refinement summary.
        stats = {
            'enabled': True,
            'iterations': int(completed_iterations),
            'total_changed_pixels': int(total_changed_pixels),
            'ratio_threshold': float(ratio_threshold),
            'complex_weight': float(complex_weight),
            'real_weight': float(real_weight),
            'objective_improvement_rtol': float(objective_improvement_rtol),
            'pair_stats': pair_stats
        }

        return branch_values, corrected_indices, stats

    def _branch_pair_geometry_maps(self, branch_values, degenerate_threshold=None, close_rtol=0.1, close_atol=0.0):

        '''
        Purpose
        -------
        Map nearest branch-pair separations and identify close-root regions.

        Inputs
        ------
        branch_values: np.array, complex
            Labeled vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        degenerate_threshold: np.array or None
            Per-cell root-separation tolerance in km^-1. If omitted, use 1e-10 plus
            1e-3 times the local median root magnitude.

        close_rtol: float
            Relative close-root tolerance, multiplied by the local median root
            magnitude.

        close_atol: float
            Absolute close-root tolerance (km^-1), added to the relative tolerance.

        Outputs
        -------
        geometry: dict
            Two-dimensional minimum separation (km^-1), nearest zero-based branch
            indices (-1 if absent), thresholds, and degeneracy/close-root masks.
        '''

        # Allocate nearest-pair maps with missing-index sentinels.
        branch_values = np.asarray(branch_values, dtype=np.complex128)
        n_omega, n_kh, branch_count = branch_values.shape
        min_separation = np.full((n_omega, n_kh), np.nan)
        nearest_branch_a = np.full((n_omega, n_kh), -1, dtype=int)
        nearest_branch_b = np.full((n_omega, n_kh), -1, dtype=int)

        # Scale degeneracy and close-root thresholds by local finite root magnitudes.
        abs_values = np.where(self._finite_complex_mask(branch_values), np.abs(branch_values), np.nan)
        local_scale = np.nanmedian(abs_values, axis=2)
        local_scale[~np.isfinite(local_scale)] = 1.0
        if degenerate_threshold is None:
            degenerate_threshold = 1e-10 + 1e-3*local_scale
        close_threshold = np.maximum(degenerate_threshold, close_atol + close_rtol*local_scale)

        # Compare all finite branch pairs and retain the smallest separation.
        for branch_a in range(branch_count):
            for branch_b in range(branch_a + 1, branch_count):
                separation = np.abs(branch_values[:, :, branch_a] - branch_values[:, :, branch_b])
                valid = np.isfinite(separation)
                update = valid & (~np.isfinite(min_separation) | (separation < min_separation))
                min_separation[update] = separation[update]
                nearest_branch_a[update] = branch_a
                nearest_branch_b[update] = branch_b

        # Report the nearest pair and both separation-based masks.
        return {
            'branch_min_separation': min_separation,
            'nearest_branch_a': nearest_branch_a,
            'nearest_branch_b': nearest_branch_b,
            'branch_degenerate_threshold': degenerate_threshold,
            'branch_close_threshold': close_threshold,
            'branch_degenerate_mask': (
                np.isfinite(min_separation)
                & (min_separation <= degenerate_threshold)
            ),
            'branch_close_mask': (
                np.isfinite(min_separation)
                & (min_separation <= close_threshold)
            )
        }

    def _pair_swap_gain_map(
        self,
        branch_values,
        branch_a=None,
        branch_b=None,
        complex_weight=1.0,
        real_weight=2.0
    ):

        '''
        Purpose
        -------
        Measure the local objective improvement available from swapping branch pairs.

        Inputs
        ------
        branch_values: np.array, complex
            Labeled vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        branch_a: int or None
            Zero-based first branch index. If either branch index is None,
            evaluate all branch pairs.

        branch_b: int or None
            Zero-based second branch index; supply both indices to restrict
            the calculation to one pair.

        complex_weight: float
            Weight of squared complex-root displacement in the continuation cost.

        real_weight: float
            Additional weight of squared real-part displacement in the continuation
            cost.

        Outputs
        -------
        gain_map: np.array, float
            Largest positive neighboring swap-cost reduction touching each cell, in
            km^-2; NaN where no positive gain is present.
        '''

        # Allocate a map for the largest positive swap gain touching each cell.
        branch_values = np.asarray(branch_values, dtype=np.complex128)
        n_omega, n_kh, branch_count = branch_values.shape
        gain_map = np.full((n_omega, n_kh), np.nan)

        # Evaluate every pair unless both branch indices are explicitly supplied.
        if branch_a is None or branch_b is None:
            pairs = [
                (a, b)
                for a in range(branch_count)
                for b in range(a + 1, branch_count)
            ]
        else:
            pairs = [(branch_a, branch_b)]

        def edge_gain(source_values, target_values):

            '''
            Purpose
            -------
            Return same-order continuation cost minus swapped continuation cost.

            Inputs
            ------
            source_values: np.array, complex
                Root values at the source of each edge, with shape (n_edges, n_branch).

            target_values: np.array, complex
                Root values at the target of each edge, with shape (n_edges, n_branch).

            Outputs
            -------
            gain: np.array, float
                Same-order cost minus swapped-order cost per edge (km^-2); nonfinite
                root pairs are NaN.
            '''

            # Require a finite selected pair at each end of the edge.
            valid = (
                self._finite_complex_mask(source_values[:, branch_a])
                & self._finite_complex_mask(source_values[:, branch_b])
                & self._finite_complex_mask(target_values[:, branch_a])
                & self._finite_complex_mask(target_values[:, branch_b])
            )
            # Compute unchanged and swapped costs with the selected component weights.
            same_cost = complex_weight*(
                np.abs(source_values[:, branch_a] - target_values[:, branch_a])**2
                + np.abs(source_values[:, branch_b] - target_values[:, branch_b])**2
            )
            swap_cost = complex_weight*(
                np.abs(source_values[:, branch_a] - target_values[:, branch_b])**2
                + np.abs(source_values[:, branch_b] - target_values[:, branch_a])**2
            )
            if real_weight != 0.0:
                same_cost = same_cost + real_weight*(
                    (source_values[:, branch_a].real - target_values[:, branch_a].real)**2
                    + (source_values[:, branch_b].real - target_values[:, branch_b].real)**2
                )
                swap_cost = swap_cost + real_weight*(
                    (source_values[:, branch_a].real - target_values[:, branch_b].real)**2
                    + (source_values[:, branch_b].real - target_values[:, branch_a].real)**2
                )
            # Store the signed cost reduction, leaving invalid edges as NaN.
            gain = np.full(source_values.shape[0], np.nan)
            gain[valid] = same_cost[valid] - swap_cost[valid]
            return gain

        # Compute horizontal and vertical gains for each requested branch pair.
        for pair_a, pair_b in pairs:
            branch_a, branch_b = pair_a, pair_b
            right_gain = np.full((n_omega, max(n_kh - 1, 0)), np.nan)
            down_gain = np.full((max(n_omega - 1, 0), n_kh), np.nan)

            if n_kh > 1:
                source = branch_values[:, :-1, :].reshape(-1, branch_count)
                target = branch_values[:, 1:, :].reshape(-1, branch_count)
                right_gain = edge_gain(source, target).reshape(n_omega, n_kh - 1)
            if n_omega > 1:
                source = branch_values[:-1, :, :].reshape(-1, branch_count)
                target = branch_values[1:, :, :].reshape(-1, branch_count)
                down_gain = edge_gain(source, target).reshape(n_omega - 1, n_kh)

            # Keep beneficial edges and project their maxima onto adjacent grid cells.
            positive_right_gain = np.where(right_gain > 0.0, right_gain, np.nan)
            positive_down_gain = np.where(down_gain > 0.0, down_gain, np.nan)
            pair_gain_map = self._edge_max_to_pixel_map(
                positive_right_gain,
                positive_down_gain,
                (n_omega, n_kh)
            )
            self._update_max_map(gain_map, pair_gain_map)

        return gain_map

    def _branch_topology_maps(self, edge_maps, local_index_by_branch):

        '''
        Purpose
        -------
        Diagnose branch cuts forced by the root-sheet topology.

        Inputs
        ------
        edge_maps: dict
            Neighboring raw-root assignments returned by _edge_match_maps(),
            including right_perm and down_perm.

        local_index_by_branch: np.array, int
            Raw-root index for each labeled branch, with shape (n_omega, n_kh,
            n_branch); -1 marks an unavailable root.

        Outputs
        -------
        topology: dict
            Edge label-mismatch counts, per-pixel branch-cut masks/strengths, and
            plaquette and pixel monodromy masks. Only complete loop permutations are
            assessed.

        Notes
        -----
        For each adjacent grid edge, the raw root sets define a local analytic
        continuation through the Hungarian permutation. If the current global
        labels do not follow that local permutation, the edge lies on a label
        branch cut. Around a true root coalescence/exceptional point, the
        product of local permutations around a plaquette can be non-identity;
        no single global labeling can satisfy all such local continuations.
        The maps describe the sampled assignment permutations; numerical
        ambiguity can also produce a nonidentity loop. These maps alone do not
        establish an analytic branch point or a polynomial-solver failure.
        '''

        # Allocate label-mismatch counts for horizontal and vertical edges.
        local_index_by_branch = np.asarray(local_index_by_branch, dtype=int)
        n_omega, n_kh, branch_count = local_index_by_branch.shape
        right_mismatch_count = np.zeros((n_omega, max(n_kh - 1, 0)), dtype=int)
        down_mismatch_count = np.zeros((max(n_omega - 1, 0), n_kh), dtype=int)

        # Compare horizontal global labels with the local raw-root continuation.
        if n_kh > 1:
            for i in range(n_omega):
                for j in range(n_kh - 1):
                    permutation = edge_maps['right_perm'][i, j, :]
                    for branch in range(branch_count):
                        source_index = local_index_by_branch[i, j, branch]
                        target_index = local_index_by_branch[i, j + 1, branch]
                        if source_index < 0 or target_index < 0:
                            continue
                        continued_index = permutation[source_index]
                        if continued_index >= 0 and continued_index != target_index:
                            right_mismatch_count[i, j] += 1

        # Repeat the continuation check across adjacent frequency rows.
        if n_omega > 1:
            for i in range(n_omega - 1):
                for j in range(n_kh):
                    permutation = edge_maps['down_perm'][i, j, :]
                    for branch in range(branch_count):
                        source_index = local_index_by_branch[i, j, branch]
                        target_index = local_index_by_branch[i + 1, j, branch]
                        if source_index < 0 or target_index < 0:
                            continue
                        continued_index = permutation[source_index]
                        if continued_index >= 0 and continued_index != target_index:
                            down_mismatch_count[i, j] += 1

        # Project nonzero edge mismatches onto their neighboring pixels.
        branch_cut_strength = self._edge_max_to_pixel_map(
            np.where(right_mismatch_count > 0, right_mismatch_count, np.nan),
            np.where(down_mismatch_count > 0, down_mismatch_count, np.nan),
            (n_omega, n_kh)
        )
        branch_cut_mask = np.isfinite(branch_cut_strength)

        def invert_permutation(permutation):

            '''
            Purpose
            -------
            Invert a branch permutation while preserving unmatched entries.

            Inputs
            ------
            permutation: np.array, int
                One-dimensional source-to-target root-index mapping; -1 marks an
                unmatched entry.

            Outputs
            -------
            inverse: np.array, int
                Target-to-source mapping with the input shape and -1 for unmatched
                entries.
            '''

            # Reverse each finite index mapping while retaining unmatched sentinels.
            inverse = np.full_like(permutation, -1)
            for source_index, target_index in enumerate(permutation):
                if target_index >= 0:
                    inverse[target_index] = source_index
            return inverse

        def compose_permutations(first, second):

            '''
            Purpose
            -------
            Compose two branch-index mappings and preserve invalid destinations.

            Inputs
            ------
            first: np.array, int
                First source-to-intermediate root-index mapping, with -1 for
                unmatched entries.

            second: np.array, int
                Second intermediate-to-target root-index mapping, with -1 for
                unmatched entries.

            Outputs
            -------
            composed: np.array, int
                Mapping second[first], retaining -1 when either step is unmatched.
            '''

            # Follow both mappings only when both intermediate and final indices exist.
            composed = np.full_like(first, -1)
            for source_index, intermediate_index in enumerate(first):
                if intermediate_index >= 0 and second[intermediate_index] >= 0:
                    composed[source_index] = second[intermediate_index]
            return composed

        # Allocate one closed-loop diagnostic for each four-cell plaquette.
        monodromy_cell_mask = np.zeros((max(n_omega - 1, 0), max(n_kh - 1, 0)), dtype=bool)
        monodromy_moved_count = np.zeros_like(monodromy_cell_mask, dtype=int)
        expected = np.arange(branch_count, dtype=int)
        # Compose the four oriented edge permutations around each plaquette.
        for i in range(n_omega - 1):
            for j in range(n_kh - 1):
                right_top = edge_maps['right_perm'][i, j, :]
                down_right = edge_maps['down_perm'][i, j + 1, :]
                right_bottom_inverse = invert_permutation(edge_maps['right_perm'][i + 1, j, :])
                down_left_inverse = invert_permutation(edge_maps['down_perm'][i, j, :])
                loop = compose_permutations(
                    compose_permutations(
                        compose_permutations(right_top, down_right),
                        right_bottom_inverse
                    ),
                    down_left_inverse
                )
                # Test only complete loop mappings for a nonidentity root permutation.
                valid = loop >= 0
                if np.all(valid):
                    moved = loop != expected
                    if np.any(moved):
                        monodromy_cell_mask[i, j] = True
                        monodromy_moved_count[i, j] = int(np.count_nonzero(moved))

        # Mark all four pixels touching a plaquette with nontrivial monodromy.
        monodromy_mask = np.zeros((n_omega, n_kh), dtype=bool)
        if monodromy_cell_mask.size > 0:
            monodromy_mask[:-1, :-1] |= monodromy_cell_mask
            monodromy_mask[1:, :-1] |= monodromy_cell_mask
            monodromy_mask[:-1, 1:] |= monodromy_cell_mask
            monodromy_mask[1:, 1:] |= monodromy_cell_mask

        # Return edge, plaquette, and pixel topology diagnostics together.
        return {
            'branch_cut_mask': branch_cut_mask,
            'branch_cut_strength': branch_cut_strength,
            'branch_cut_edge_kh': right_mismatch_count,
            'branch_cut_edge_omega': down_mismatch_count,
            'monodromy_cell_mask': monodromy_cell_mask,
            'monodromy_moved_count': monodromy_moved_count,
            'monodromy_mask': monodromy_mask
        }

    def _branch_jump_excess_mask(self, tracked_roots, root_set_jump, uncertainty_threshold):

        '''
        Purpose
        -------
        Flag tracked-branch jumps that exceed local raw-root motion and uncertainty.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        root_set_jump: np.array, float
            Largest optimal neighboring-root displacement touching each grid cell
            (km^-1).

        uncertainty_threshold: np.array, float
            Per-cell absolute-plus-relative tracking tolerance (km^-1).

        Outputs
        -------
        tracked_jump: np.array, float
            Largest adjacent branch displacement per cell (km^-1).

        branch_jump_ratio: np.array, float
            Tracked jump divided by the local optimal raw-root jump, with a 1e-300
            denominator floor.

        branch_jump_excess_mask: np.array, bool
            Cells exceeding both eight times the raw-root jump and the additive
            uncertainty tolerance.
        '''

        # Compare tracked-label motion against the optimal motion of the raw root sets.
        tracked_jump = self.kz_smoothness_map(tracked_roots)
        branch_jump_ratio = tracked_jump / np.maximum(root_set_jump, 1e-300)
        branch_jump_excess_mask = (
            np.isfinite(branch_jump_ratio)
            # A modest ratio can occur at true low-frequency coalescence
            # curves where the root-set motion itself is nearly singular.
            # Keep this mask for clear assignment artifacts.
            & (branch_jump_ratio > 8.0)
            & (tracked_jump > root_set_jump + uncertainty_threshold)
        )

        return tracked_jump, branch_jump_ratio, branch_jump_excess_mask

    def _synchronize_branch_swaps(self, raw_roots, local_index_by_branch, max_iterations=3, ratio_threshold=2.0):

        '''
        Purpose
        -------
        Correct branch-cut artifacts left by the spanning-tree tracker.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        local_index_by_branch: np.array, int
            Raw-root index for each labeled branch, with shape (n_omega, n_kh,
            n_branch); -1 marks an unavailable root.

        max_iterations: int
            Maximum number of full passes through all branch pairs.

        ratio_threshold: float
            Minimum ratio between competing continuation costs for an edge to supply
            a parity constraint.

        Outputs
        -------
        branch_values: np.array, complex
            Roots after parity-consistent region swaps, in km^-1.

        corrected_indices: np.array, int
            Updated copy of the branch-to-raw-root indices.

        stats: dict
            Iteration counts, changed-cell totals, and per-pair constraint statistics.

        Notes
        -----
        The first tracking pass follows a reliable spanning tree. That can
        leave a branch cut on off-tree edges when several branch pairs interact
        near a crossing. This pass treats each branch pair as a binary swap
        synchronization problem over the full 2D grid. Edges vote for either
        "same labels" or "swap this pair" when one choice is clearly smoother;
        a parity union-find accepts the strongest non-conflicting votes. The
        resulting connected-region swaps remove artificial cuts without hiding
        pixels or changing the polynomial roots.
        '''

        # Copy the branch map and initialize synchronization diagnostics.
        corrected_indices = local_index_by_branch.copy()
        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
        branch_count = raw_roots.shape[2]
        pair_stats = []
        total_changed_pixels = 0
        # Stop after a pass with no changes to any pair.
        completed_iterations = 0

        # Visit every pair repeatedly while supported region swaps remain.
        for iteration in range(max_iterations):
            iteration_changed_pixels = 0
            for branch_a in range(branch_count):
                for branch_b in range(branch_a + 1, branch_count):
                    # Infer swap regions from the strongest consistent neighboring-edge
                    # votes.
                    parity, stats = self._pair_swap_parity(
                        branch_values,
                        branch_a,
                        branch_b,
                        ratio_threshold
                    )
                    if stats['accepted_flip_constraints'] == 0 or stats['swap_pixels'] == 0:
                        pair_stats.append({**stats, 'iteration': int(iteration), 'changed_pixels': 0})
                        continue

                    # Exchange only valid branch indices within the inferred region.
                    updated_indices, changed_pixels = self._apply_pair_swap(
                        corrected_indices,
                        branch_a,
                        branch_b,
                        parity
                    )
                    # Update the gathered root values and count accepted changes.
                    if changed_pixels > 0:
                        corrected_indices = updated_indices
                        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
                        iteration_changed_pixels += changed_pixels
                        total_changed_pixels += changed_pixels

                    pair_stats.append({**stats, 'iteration': int(iteration), 'changed_pixels': int(changed_pixels)})

            completed_iterations = iteration + 1
            if iteration_changed_pixels == 0:
                break

        # Retain per-pair counts and the total number of relabeled cells.
        stats = {
            'enabled': True,
            'iterations': int(completed_iterations),
            'total_changed_pixels': int(total_changed_pixels),
            'ratio_threshold': float(ratio_threshold),
            'pair_stats': pair_stats
        }

        return branch_values, corrected_indices, stats

    def track_kz_branches(self, raw_roots, seed_index=None, degenerate_rtol=1e-3, degenerate_atol=1e-10, uncertainty_rtol=1e-3, uncertainty_atol=1e-10, return_diagnostics=False):

        '''
        Purpose
        -------
        Assign raw kz roots to continuous branches over the omega-kh grid.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        seed_index: tuple or None
            Optional (omega_index, kh_index) for the first tracking seed. A seed
            cell without finite roots falls back to automatic selection.

        degenerate_rtol: float
            Relative root-separation tolerance for detecting degeneracy.

        degenerate_atol: float
            Absolute root-separation tolerance (km^-1).

        uncertainty_rtol: float
            Relative tolerance for assessing ambiguous assignments.

        uncertainty_atol: float
            Absolute tolerance for assessing ambiguous assignments (km^-1).

        return_diagnostics: bool
            Return the diagnostic dictionary alongside the roots when True.

        Outputs
        -------
        tracked_roots: np.array, complex
            Labeled roots with raw_roots.shape, in km^-1; missing branch assignments
            remain NaN.

        diagnostics: dict, optional
            Returned only when requested: component seeds/parents, finite-root
            counts, root separations, edge costs, assignment confidence, uncertainty
            flags, and branch-to-raw-root indices.

        Notes
        -----
        The branch labels are seeded at a grid point where the roots are as
        well-separated as possible. Neighboring raw root sets are first matched
        in the complex plane with the Hungarian assignment algorithm. Branch
        labels are then propagated over a maximum-reliability spanning tree,
        so the traversal follows locally unambiguous edges before crossing
        near-degenerate or high-cost regions. This avoids the staircase
        artifacts produced by path-dependent averaging of already-tracked
        neighbors.
        '''

        # Validate the root-grid shape and allocate missing-value and parent maps.
        raw_roots = np.asarray(raw_roots, dtype=np.complex128)
        if raw_roots.ndim != 3:
            raise ValueError("raw_roots must have shape (omega, kh, branch).")

        n_omega, n_kh, branch_count = raw_roots.shape
        tracked_roots = np.full_like(raw_roots, np.nan + 0.0j)
        local_index_by_branch = np.full(raw_roots.shape, -1, dtype=int)
        assigned = np.zeros((n_omega, n_kh), dtype=bool)
        component_id = np.full((n_omega, n_kh), -1, dtype=int)
        parent_index = np.full((n_omega, n_kh, 2), -1, dtype=int)

        # Measure local root geometry and precompute neighboring root assignments.
        finite_count, min_separation, local_scale, degenerate_mask = self._root_geometry_maps(
            raw_roots, degenerate_rtol, degenerate_atol
        )
        edge_maps = self._edge_match_maps(raw_roots)
        assignment_cost = np.full((n_omega, n_kh), np.nan)
        assignment_margin = np.full((n_omega, n_kh), np.nan)
        max_assignment_cost = np.full((n_omega, n_kh), np.nan)
        assignment_confidence = np.full((n_omega, n_kh), np.nan)
        matched_count = np.zeros((n_omega, n_kh), dtype=int)
        seeds = []

        def assign_seed(i, j, component):
            '''
            Purpose
            -------
            Initialize deterministic branch labels at a tracking-component seed.

            Inputs
            ------
            i: int
                Omega index of the source cell.

            j: int
                Horizontal-wavenumber index of the source cell.

            component: int
                Identifier of the connected tracking component.

            Outputs
            -------
            None
                Populate branch labels, root values, assigned state, and component
                identifier at the seed cell in the enclosing tracking arrays.
            '''

            # Sort finite seed roots deterministically before assigning branch labels.
            root_indices = self._sorted_finite_indices(raw_roots[i, j, :])
            for branch, root_index in enumerate(root_indices[:branch_count]):
                local_index_by_branch[i, j, branch] = root_index
                tracked_roots[i, j, branch] = raw_roots[i, j, root_index]
            assigned[i, j] = True
            component_id[i, j] = component

        def edge_from_to(ui, uj, vi, vj):
            '''
            Purpose
            -------
            Return match metadata oriented from one adjacent cell to another.

            Inputs
            ------
            ui: int
                Omega index of the assigned source cell.

            uj: int
                Horizontal-wavenumber index of the assigned source cell.

            vi: int
                Omega index of the neighboring target cell.

            vj: int
                Horizontal-wavenumber index of the neighboring target cell.

            Outputs
            -------
            edge: tuple
                Source-to-target permutation, mean cost, maximum cost, assignment
                margin, and matched-root count. Reverse edges use inverse
                permutations; nonadjacent cells raise ValueError.
            '''

            # Read forward horizontal edges directly from the precomputed maps.
            if ui == vi and vj == uj + 1:
                return (
                    edge_maps['right_perm'][ui, uj, :],
                    edge_maps['right_cost'][ui, uj],
                    edge_maps['right_max_cost'][ui, uj],
                    edge_maps['right_margin'][ui, uj],
                    edge_maps['right_matched'][ui, uj]
                )
            # Invert horizontal permutations when traversing an edge backward.
            if ui == vi and vj == uj - 1:
                return (
                    self._invert_permutation(edge_maps['right_perm'][vi, vj, :]),
                    edge_maps['right_cost'][vi, vj],
                    edge_maps['right_max_cost'][vi, vj],
                    edge_maps['right_margin'][vi, vj],
                    edge_maps['right_matched'][vi, vj]
                )
            # Read forward frequency edges from the downward-edge maps.
            if vi == ui + 1 and uj == vj:
                return (
                    edge_maps['down_perm'][ui, uj, :],
                    edge_maps['down_cost'][ui, uj],
                    edge_maps['down_max_cost'][ui, uj],
                    edge_maps['down_margin'][ui, uj],
                    edge_maps['down_matched'][ui, uj]
                )
            # Invert frequency-edge permutations when traversing upward.
            if vi == ui - 1 and uj == vj:
                return (
                    self._invert_permutation(edge_maps['down_perm'][vi, vj, :]),
                    edge_maps['down_cost'][vi, vj],
                    edge_maps['down_max_cost'][vi, vj],
                    edge_maps['down_margin'][vi, vj],
                    edge_maps['down_matched'][vi, vj]
                )
            raise ValueError("Pixels are not adjacent.")

        def assign_from_edge(ui, uj, vi, vj, component, permutation, cost, edge_max_cost, margin, edge_matched):
            '''
            Purpose
            -------
            Propagate branch labels and assignment metadata across one grid edge.

            Inputs
            ------
            ui: int
                Omega index of the assigned source cell.

            uj: int
                Horizontal-wavenumber index of the assigned source cell.

            vi: int
                Omega index of the neighboring target cell.

            vj: int
                Horizontal-wavenumber index of the neighboring target cell.

            component: int
                Identifier of the connected tracking component.

            permutation: np.array, int
                One-dimensional source-to-target root-index mapping; -1 marks an
                unmatched entry.

            cost: float
                Mean matched-root displacement across the edge (km^-1).

            edge_max_cost: float
                Largest matched-root displacement across the edge (km^-1).

            margin: float
                Separation of the preferred assignment from competing matches (km^-1).

            edge_matched: int
                Number of finite roots matched across the edge.

            Outputs
            -------
            None
                Update the target's root values, indices, parent/component, and
                assignment diagnostics in the enclosing tracking arrays.
            '''

            # Carry each assigned branch through the source-to-target permutation.
            for branch in range(branch_count):
                source_index = local_index_by_branch[ui, uj, branch]
                if source_index < 0:
                    continue
                target_index = permutation[source_index]
                if target_index < 0:
                    continue
                local_index_by_branch[vi, vj, branch] = target_index
                tracked_roots[vi, vj, branch] = raw_roots[vi, vj, target_index]

            # Record the target cell's parent, component, and edge-quality statistics.
            assigned[vi, vj] = True
            component_id[vi, vj] = component
            parent_index[vi, vj, :] = [ui, uj]
            matched_count[vi, vj] = edge_matched
            assignment_cost[vi, vj] = cost
            max_assignment_cost[vi, vj] = edge_max_cost
            assignment_margin[vi, vj] = margin
            assignment_confidence[vi, vj] = self._edge_confidence(edge_max_cost, margin)

        def push_neighbors(heap, i, j):
            '''
            Purpose
            -------
            Queue unassigned finite neighbors by edge reliability for tree traversal.

            Inputs
            ------
            heap: list
                Priority queue updated in place with unassigned neighboring grid edges.

            i: int
                Omega index of the source cell.

            j: int
                Horizontal-wavenumber index of the source cell.

            Outputs
            -------
            None
                Push eligible edges into heap, prioritizing complete matches,
                confidence, margin, then maximum displacement.
            '''

            # Consider the four in-bounds neighbors that still contain unassigned roots.
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= n_omega or nj < 0 or nj >= n_kh:
                    continue
                if assigned[ni, nj] or finite_count[ni, nj] == 0:
                    continue
                # Rank complete assignments before confidence, margin, and displacement.
                _, cost, edge_max_cost, margin, edge_matched = edge_from_to(i, j, ni, nj)
                complete = int(edge_matched == branch_count)
                confidence = self._edge_confidence(edge_max_cost, margin)
                margin_priority = max(0.0, margin) if np.isfinite(margin) else 0.0
                cost_priority = edge_max_cost if np.isfinite(edge_max_cost) else np.inf
                heapq.heappush(heap, (-complete, -confidence, -margin_priority, cost_priority, i, j, ni, nj))

        # Use the requested seed once, then seed each remaining finite component.
        requested_seed = seed_index
        component = 0
        # Traverse every connected region that contains at least one finite root.
        while np.any((~assigned) & (finite_count > 0)):

            if requested_seed is not None:
                seed = tuple(requested_seed)
                requested_seed = None
                if finite_count[seed] == 0 or assigned[seed]:
                    seed = self._select_tracking_seed(raw_roots, assigned, finite_count, min_separation)
            else:
                seed = self._select_tracking_seed(raw_roots, assigned, finite_count, min_separation)

            if seed is None:
                break

            # Initialize deterministic labels at the selected component seed.
            seeds.append(seed)
            assign_seed(seed[0], seed[1], component)

            # Queue the most reliable boundary edges of the assigned region.
            heap = []
            push_neighbors(heap, seed[0], seed[1])

            # Extend the spanning tree only into cells that are still unassigned.
            while heap:
                _, _, _, _, ui, uj, vi, vj = heapq.heappop(heap)
                if assigned[vi, vj] or not assigned[ui, uj]:
                    continue

                permutation, cost, edge_max_cost, margin, edge_matched = edge_from_to(ui, uj, vi, vj)
                assign_from_edge(ui, uj, vi, vj, component, permutation, cost, edge_max_cost, margin, edge_matched)
                push_neighbors(heap, vi, vj)

            component += 1

        # Project neighboring-edge costs and margins onto the sampled grid.
        root_set_jump_map = self._edge_max_to_pixel_map(
            edge_maps['right_max_cost'], edge_maps['down_max_cost'], (n_omega, n_kh)
        )
        root_set_margin_map = self._edge_min_to_pixel_map(
            edge_maps['right_margin'], edge_maps['down_margin'], (n_omega, n_kh)
        )
        # Flag degeneracy, missing roots, and weak or expensive assignments.
        uncertainty_threshold = uncertainty_atol + uncertainty_rtol*local_scale
        uncertain_mask = degenerate_mask.copy()
        uncertain_mask |= finite_count < branch_count
        uncertain_mask |= np.isfinite(assignment_margin) & (assignment_margin <= uncertainty_threshold)
        uncertain_mask |= np.isfinite(assignment_confidence) & (assignment_confidence < 0.2)
        uncertain_mask |= (
            np.isfinite(max_assignment_cost)
            & np.isfinite(min_separation)
            & (max_assignment_cost >= 0.5*min_separation)
        )

        # Expose the tree, root geometry, assignment scores, and uncertainty maps.
        diagnostics = {
            'seed_index': seeds[0] if len(seeds) > 0 else None,
            'component_seeds': seeds,
            'component_id': component_id,
            'assignment_parent': parent_index,
            'finite_root_count': finite_count,
            'solve_failed_mask': finite_count < branch_count,
            'min_root_separation': min_separation,
            'degenerate_mask': degenerate_mask,
            'root_set_jump': root_set_jump_map,
            'root_set_margin': root_set_margin_map,
            'root_set_jump_kh': edge_maps['right_max_cost'],
            'root_set_jump_omega': edge_maps['down_max_cost'],
            'root_set_margin_kh': edge_maps['right_margin'],
            'root_set_margin_omega': edge_maps['down_margin'],
            'assignment_cost': assignment_cost,
            'max_assignment_cost': max_assignment_cost,
            'assignment_margin': assignment_margin,
            'assignment_confidence': assignment_confidence,
            'matched_count': matched_count,
            'uncertain_mask': uncertain_mask,
            'local_index_by_branch': local_index_by_branch.copy(),
            'degenerate_threshold': degenerate_atol + degenerate_rtol*local_scale,
            'uncertainty_threshold': uncertainty_threshold
        }

        if return_diagnostics:
            return tracked_roots, diagnostics

        return tracked_roots

    def kz_quadratic_roots_grid(self, return_diagnostics=False):
        '''
        Purpose
        -------
        Solve k_z² + c0 = 0 as the two explicit ±sqrt(-c0) sheets.

        Inputs
        ------
        return_diagnostics: bool
            Return the diagnostic dictionary alongside the roots when True.

        Outputs
        -------
        roots: np.array, complex
            Explicit positive/negative square-root sheets, shape (n_omega, n_kh, 2),
            in km^-1.

        diagnostics: dict, optional
            When requested, return solve-failure, residual-instability,
            plot-suspicion, and degeneracy masks, plus the maximum residual.

        Notes
        -----
        This retains the analytic sheet identities at real/imaginary crossings,
        where matching unordered numerical roots has equal assignment costs.
        The caller must use positive frequencies for a relation with an omega=0
        pole. Actual nonfinite roots, residual errors and degeneracies remain
        reported rather than being suppressed.
        '''
        # Verify the normalized even quadratic at every sampled point.
        constant = np.empty((len(self.omega), len(self.kh)), dtype=np.complex128)
        for row, omega in enumerate(self.omega):
            for col, kh in enumerate(self.kh):
                coefficients = self.kz_coefficients(omega, kh)
                if len(coefficients) != 3 or coefficients[0] != 1 or coefficients[1] != 0:
                    raise ValueError("The analytic solver requires k_z² + c0 = 0.")
                constant[row, col] = coefficients[2]
        # Keep explicit positive and negative square-root sheets without numerical
        # matching.
        positive = np.sqrt(-constant + 0j)
        roots = np.stack((positive, -positive), axis=2)
        if not return_diagnostics:
            return roots
        # Measure finite-root status, residual errors, and sheet coalescence when
        # requested.
        residuals, _ = self.kz_root_quality(roots)
        finite = self._finite_complex_mask(roots).all(axis=2)
        max_residual = np.max(residuals, axis=2)
        unstable = finite & (~np.isfinite(max_residual) | (max_residual > 1e-8))
        degenerate = finite & (np.abs(2*positive) <= 1e-10 + 1e-8*np.abs(positive))
        return roots, {
            'solve_failed_mask': ~finite,
            'solver_instability_mask': unstable,
            'plot_suspicious_mask': ~finite | unstable,
            'uncertain_mask': degenerate,
            'branch_degenerate_mask': degenerate,
            'max_root_residual': max_residual,
        }

    def _classify_magnetoacoustic_roots(self, tracked_roots, diagnostics):

        '''
        Purpose
        -------
        Classify tracked roots as fast or slow and compute propagation diagnostics.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict
            Tracking and solver maps returned by kz_solve_grid(return_diagnostics=True).

        Outputs
        -------
        labels: np.array, int
            Per-root fast (+1), slow (-1), or unresolved (0) classification with the
            input shape.

        branch_family, branch_confidence: np.array
            Family labels and winning-label fractions with shape (n_branch,).
            A family requires at least 80 percent agreement among classified
            cells only; unresolved labels are excluded from the denominator.

        mode_ambiguous: np.array, bool
            Per-cell flag for an unresolved root or disagreement with its
            whole-sheet family.

        mode_diagnostics: dict
            Classification arrays, propagation masks, vertical group velocities
            (km/s), normalized dispersion/frequency residuals, and tracking
            confidence.
        '''

        # Allocate per-root mode labels, residuals, and energy-direction diagnostics.
        n_omega, n_kh, branch_count = tracked_roots.shape
        labels = np.zeros(tracked_roots.shape, dtype=np.int8)
        residual_fast = np.full(tracked_roots.shape, np.nan)
        residual_slow = np.full(tracked_roots.shape, np.nan)
        dispersion_residual = np.full(tracked_roots.shape, np.nan)
        group_velocity_z = np.full(tracked_roots.shape, np.nan)
        propagating = np.zeros(tracked_roots.shape, dtype=bool)
        finite = self._finite_complex_mask(tracked_roots)
        a2, c2 = self.a**2, self.cs**2
        degeneracy_mask = np.asarray(
            diagnostics.get('branch_degenerate_mask', np.zeros((n_omega, n_kh), dtype=bool)),
            dtype=bool
        )

        # Evaluate each finite root against the homogeneous magnetoacoustic relation.
        for i, omega in enumerate(self.omega):
            omega2 = omega**2
            for j, kh in enumerate(self.kh):
                for branch in np.flatnonzero(finite[i, j]):
                    kz = tracked_roots[i, j, branch]
                    # Treat only roots with negligible imaginary parts as propagating.
                    scale_kz = max(abs(kz.real), abs(kh), 1e-12)
                    is_real = abs(kz.imag) <= 1e-9*scale_kz
                    propagating[i, j, branch] = is_real
                    k2 = kh**2 + kz**2
                    k_parallel = kh*sin(self.theta)*cos(self.phi) + kz*cos(self.theta)
                    # Normalize the full dispersion residual by the absolute term sum.
                    d_value = (
                        omega**4 - (a2 + c2)*omega2*k2
                        + a2*c2*k2*k_parallel**2
                    )
                    d_scale = (
                        abs(omega**4) + abs((a2 + c2)*omega2*k2)
                        + abs(a2*c2*k2*k_parallel**2)
                    )
                    dispersion_residual[i, j, branch] = abs(d_value)/max(d_scale, 1e-300)
                    if not is_real:
                        continue

                    # Compare the real root with the fast and slow frequency
                    # eigenvalues.
                    kz_real = kz.real
                    k2_real = kh**2 + kz_real**2
                    if k2_real <= 1e-24:
                        continue
                    k_parallel_real = kh*sin(self.theta)*cos(self.phi) + kz_real*cos(self.theta)
                    discriminant = (a2 + c2)**2 - 4.0*a2*c2*k_parallel_real**2/k2_real
                    discriminant_scale = max((a2 + c2)**2, 1e-300)
                    if discriminant < -1e-12*discriminant_scale:
                        continue
                    split = np.sqrt(max(discriminant, 0.0))
                    omega2_fast = 0.5*k2_real*(a2 + c2 + split)
                    omega2_slow = 0.5*k2_real*(a2 + c2 - split)
                    scale_fast = max(abs(omega2), abs(omega2_fast), 1e-300)
                    scale_slow = max(abs(omega2), abs(omega2_slow), 1e-300)
                    residual_fast[i, j, branch] = abs(omega2 - omega2_fast)/scale_fast
                    residual_slow[i, j, branch] = abs(omega2 - omega2_slow)/scale_slow
                    # Leave coalescing or inaccurate eigenvalue matches unclassified.
                    eigenvalue_gap = abs(omega2_fast - omega2_slow)/max(
                        abs(omega2_fast), abs(omega2_slow), abs(omega2), 1e-300
                    )
                    best = min(residual_fast[i, j, branch], residual_slow[i, j, branch])
                    if best <= 1e-7 and eigenvalue_gap > 1e-8:
                        labels[i, j, branch] = (
                            1 if residual_fast[i, j, branch] < residual_slow[i, j, branch] else -1
                        )

                    # Obtain the vertical group velocity from implicit dispersion
                    # derivatives.
                    d_omega = 4.0*omega**3 - 2.0*(a2 + c2)*omega*k2_real
                    d_kz = (
                        -2.0*(a2 + c2)*omega2*kz_real
                        + 2.0*a2*c2*kz_real*k_parallel_real**2
                        + 2.0*a2*c2*k2_real*k_parallel_real*cos(self.theta)
                    )
                    derivative_scale = max(
                        abs(4.0*omega**3),
                        abs(2.0*(a2 + c2)*omega*k2_real),
                        1e-300
                    )
                    if abs(d_omega) > 1e-12*derivative_scale:
                        group_velocity_z[i, j, branch] = -d_kz/d_omega

        # Continue classifications only through adjacent, nondegenerate cells.
        # Conflicting neighboring labels remain ambiguous instead of being forced.
        changed = True
        while changed:
            changed = False
            for i in range(n_omega):
                for j in range(n_kh):
                    if degeneracy_mask[i, j]:
                        continue
                    for branch in np.flatnonzero(finite[i, j] & ~propagating[i, j] & (labels[i, j] == 0)):
                        neighbor_labels = []
                        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                            if 0 <= ni < n_omega and 0 <= nj < n_kh and not degeneracy_mask[ni, nj]:
                                value = labels[ni, nj, branch]
                                if value != 0:
                                    neighbor_labels.append(value)
                        if neighbor_labels and all(value == neighbor_labels[0] for value in neighbor_labels):
                            labels[i, j, branch] = neighbor_labels[0]
                            changed = True

        # Summarize each sheet when 80 percent of its classified cells agree.
        branch_family = np.zeros(branch_count, dtype=np.int8)
        branch_confidence = np.zeros(branch_count, dtype=float)
        for branch in range(branch_count):
            branch_labels = labels[:, :, branch]
            fast_count = int(np.count_nonzero(branch_labels == 1))
            slow_count = int(np.count_nonzero(branch_labels == -1))
            total = fast_count + slow_count
            if total:
                winner, count = (1, fast_count) if fast_count > slow_count else (-1, slow_count)
                branch_confidence[branch] = count/total
                if branch_confidence[branch] >= 0.8:
                    branch_family[branch] = winner

        # Retain unresolved roots and disagreements with the whole-sheet family.
        mode_ambiguous = np.any(labels == 0, axis=2) | np.any(
            (labels != 0) & (branch_family[None, None, :] != labels), axis=2
        )
        return labels, branch_family, branch_confidence, mode_ambiguous, {
            'mode_classification': labels,
            'mode_branch_family': branch_family,
            'mode_branch_confidence': branch_confidence,
            'mode_ambiguous_mask': mode_ambiguous,
            'propagating_root_mask': propagating,
            'evanescent_root_mask': finite & ~propagating,
            'vertical_group_velocity_km_s': group_velocity_z,
            'normalized_fast_frequency_residual': residual_fast,
            'normalized_slow_frequency_residual': residual_slow,
            'normalized_magnetoacoustic_residual': dispersion_residual,
            'branch_tracking_confidence': diagnostics.get('assignment_confidence'),
        }

    def _select_magnetoacoustic_mode(self, tracked_roots, diagnostics):

        '''
        Purpose
        -------
        Select the requested fast or slow pair and order it by energy direction.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict
            Tracking and solver maps returned by kz_solve_grid(return_diagnostics=True).

        Outputs
        -------
        output: np.array, complex
            Requested mode roots, shape (n_omega, n_kh, 2), in km^-1. Positive
            vertical energy direction precedes negative, then undefined direction;
            missing roots remain NaN. Update diagnostics in place with selected
            indices, masks, group velocities, residuals, and the complete root grid.
        '''

        # Classify the complete root set before selecting the requested physical mode.
        labels, family, _, ambiguous, mode_diagnostics = (
            self._classify_magnetoacoustic_roots(tracked_roots, diagnostics)
        )
        # Allocate two output slots and retain their original tracked-root indices.
        requested = 1 if self.model == 'fast_mode' else -1
        output = np.full((*tracked_roots.shape[:2], 2), np.nan + 0.0j, dtype=np.complex128)
        selected_indices = np.full(output.shape, -1, dtype=int)
        selected_ids = set()
        # Select at most two local roots carrying the requested fast or slow label.
        for i, j in np.ndindex(tracked_roots.shape[:2]):
            candidates = np.flatnonzero(labels[i, j] == requested)
            if candidates.size == 0:
                continue
            if candidates.size > 2:
                ambiguous[i, j] = True
                continue
            # Place upward energy before downward energy, then undefined direction.
            velocities = mode_diagnostics['vertical_group_velocity_km_s'][i, j, candidates]
            direction_order = np.where(velocities > 0.0, 0, np.where(velocities < 0.0, 1, 2))
            order = np.lexsort((candidates, direction_order))
            for output_branch, candidate_index in enumerate(candidates[order][:2]):
                output[i, j, output_branch] = tracked_roots[i, j, candidate_index]
                selected_indices[i, j, output_branch] = candidate_index
                selected_ids.add(int(candidate_index))
        # Collect global candidate sheets and their median energy directions.
        selected_ids = sorted(selected_ids)
        family_indices = np.flatnonzero(family == requested)
        ordered = []
        for branch in family_indices:
            velocities = mode_diagnostics['vertical_group_velocity_km_s'][:, :, branch]
            velocities = velocities[np.isfinite(velocities)]
            median_velocity = float(np.median(velocities)) if velocities.size else np.nan
            if median_velocity > 0:
                direction_order = 0
            elif median_velocity < 0:
                direction_order = 1
            else:
                direction_order = 2
            ordered.append((direction_order, int(branch)))
        # Preserve complete-root evidence alongside diagnostics for the selected pair.
        mode_diagnostics.update({
            'selected_mode': self.model,
            'mode_branch_ids': np.asarray(selected_ids, dtype=int),
            'mode_branch_candidate_ids': np.asarray(family_indices, dtype=int),
            'mode_branch_labels': tuple(
                ('upward energy' if order == 0 else 'downward energy' if order == 1 else 'direction undefined')
                for order, _ in sorted(ordered)
            ),
            # Family confidence describes whole tracked sheets, which can
            # exchange fast/slow identity across the grid. Ambiguity in an
            # unselected sheet must not hide a locally identified mode pair.
            # Preserve that full-root diagnostic separately for inspection.
            'all_roots_mode_ambiguous_mask': ambiguous,
            'mode_ambiguous_mask': np.count_nonzero(np.isfinite(output), axis=2) != 2,
            'tracked_all_roots': tracked_roots,
            'selected_root_indices': selected_indices,
            'selected_propagating_root_mask': np.take_along_axis(
                mode_diagnostics['propagating_root_mask'],
                np.maximum(selected_indices, 0), axis=2
            ) & (selected_indices >= 0),
            'selected_vertical_group_velocity_km_s': np.take_along_axis(
                mode_diagnostics['vertical_group_velocity_km_s'],
                np.maximum(selected_indices, 0), axis=2
            ),
            'selected_normalized_dispersion_residual': np.take_along_axis(
                mode_diagnostics['normalized_magnetoacoustic_residual'],
                np.maximum(selected_indices, 0), axis=2
            ),
            'mode_classification_tolerance': 1e-7,
        })
        # Clear gathered diagnostics for missing selected slots and publish them in
        # place.
        mode_diagnostics['selected_vertical_group_velocity_km_s'][selected_indices < 0] = np.nan
        mode_diagnostics['selected_normalized_dispersion_residual'][selected_indices < 0] = np.nan
        diagnostics.update(mode_diagnostics)
        return output

    def kz_solve_grid(
        self,
        return_diagnostics=False,
        branch_swap_correction=True,
        branch_swap_iterations=3,
        branch_swap_ratio=2.0,
        branch_identity_refinement=True,
        branch_identity_iterations=2,
        branch_identity_ratio=1.2,
        branch_identity_real_weight=2.0,
        branch_identity_improvement_rtol=1e-3,
        high_precision_validation=False,
        high_precision_digits=60,
        **tracking_kwargs
    ):

        '''
        Purpose
        -------
        Solve kz over the full omega-kh grid and track continuous branches.

        Inputs
        ------
        return_diagnostics: bool
            Return the diagnostic dictionary alongside the roots when True.

        branch_swap_correction: bool
            Run pair-swap synchronization when the initial tracking has excessive jumps.

        branch_swap_iterations: int
            Maximum synchronization passes through all branch pairs.

        branch_swap_ratio: float
            Minimum competing-cost ratio used for synchronization constraints.

        branch_identity_refinement: bool
            Test coherent region swaps against the global branch-identity objective.

        branch_identity_iterations: int
            Maximum global identity-refinement passes.

        branch_identity_ratio: float
            Minimum competing-cost ratio used to construct identity-refinement
            candidates.

        branch_identity_real_weight: float
            Additional real-part weight used to construct identity candidates and
            the swap-gain map.

        branch_identity_improvement_rtol: float
            Minimum fractional global-objective reduction for accepting an identity
            candidate.

        high_precision_validation: bool
            Independently check unstable, ill-conditioned, or unresolved cells with
            mpmath.

        high_precision_digits: int
            Decimal precision of the optional high-precision check.

        **tracking_kwargs: dict
            Additional keyword arguments forwarded to track_kz_branches(), such as
            seed and degeneracy tolerances.

        Outputs
        -------
        kz: np.array, complex
            Tracked root grid in km^-1 with shape (n_omega, n_kh, kz_order);
            slow_mode and fast_mode return the selected two-root pair.

        diagnostics: dict, optional
            Returned only when requested: raw/pre-correction roots, solver
            degree/failure maps, tracking and swap statistics, residual/condition
            estimates, topology and ambiguity masks, and available model-specific
            propagation diagnostics.
        '''

        # Solve every polynomial and label its roots across the full sampled grid.
        raw_roots, solve_info = self._kz_solve_raw_grid(return_info=True)
        tracked_roots, diagnostics = self.track_kz_branches(
            raw_roots, return_diagnostics=True, **tracking_kwargs
        )
        # Keep coefficient, degree-reduction, and missing-root diagnostics separate.
        diagnostics['coefficient_degree'] = solve_info['coefficient_degree']
        diagnostics['effective_degree'] = solve_info['effective_degree']
        diagnostics['degree_reduced_mask'] = solve_info['degree_reduced']
        diagnostics['zero_polynomial_mask'] = solve_info['zero_polynomial']
        diagnostics['nonfinite_coefficient_mask'] = solve_info['nonfinite_coefficients']
        diagnostics['root_scale'] = solve_info['root_scale']
        diagnostics['solve_failed_mask'] = solve_info['solver_failed']
        diagnostics['missing_root_mask'] = (
            diagnostics['finite_root_count'] < self._raw_kz_root_count()
        )
        # Save the initial labeling and measure jumps beyond the raw-root motion.
        pre_swap_tracked_roots = tracked_roots.copy()
        pre_swap_tracked_jump, pre_swap_ratio, pre_swap_excess_mask = self._branch_jump_excess_mask(
            pre_swap_tracked_roots,
            diagnostics['root_set_jump'],
            diagnostics['uncertainty_threshold']
        )

        # Run parity synchronization only when the initial labels show excess jumps.
        if branch_swap_correction and np.any(pre_swap_excess_mask):
            tracked_roots, local_index_by_branch, branch_swap_stats = self._synchronize_branch_swaps(
                raw_roots,
                diagnostics['local_index_by_branch'],
                max_iterations=branch_swap_iterations,
                ratio_threshold=branch_swap_ratio
            )
            diagnostics['local_index_by_branch'] = local_index_by_branch
        else:
            branch_swap_stats = {
                'enabled': bool(branch_swap_correction),
                'skipped_no_branch_excess': bool(branch_swap_correction),
                'iterations': 0,
                'total_changed_pixels': 0,
                'ratio_threshold': float(branch_swap_ratio),
                'pair_stats': []
            }

        # Test global identity candidates independently of the excess-jump trigger.
        if branch_identity_refinement:
            tracked_roots, local_index_by_branch, identity_stats = self._refine_branch_identity(
                raw_roots,
                diagnostics['local_index_by_branch'],
                max_iterations=branch_identity_iterations,
                ratio_threshold=branch_identity_ratio,
                real_weight=branch_identity_real_weight,
                objective_improvement_rtol=branch_identity_improvement_rtol
            )
            diagnostics['local_index_by_branch'] = local_index_by_branch
        else:
            identity_stats = {
                'enabled': False,
                'iterations': 0,
                'total_changed_pixels': 0,
                'ratio_threshold': float(branch_identity_ratio),
                'complex_weight': 1.0,
                'real_weight': float(branch_identity_real_weight),
                'objective_improvement_rtol': float(branch_identity_improvement_rtol),
                'pair_stats': []
            }

        # Retain raw and pre-correction arrays for before/after comparisons.
        diagnostics['raw_roots'] = raw_roots
        diagnostics['pre_swap_tracked_roots'] = pre_swap_tracked_roots
        diagnostics['pre_swap_tracked_branch_jump'] = pre_swap_tracked_jump
        diagnostics['pre_swap_branch_jump_ratio'] = pre_swap_ratio
        diagnostics['pre_swap_branch_jump_excess_mask'] = pre_swap_excess_mask
        diagnostics['branch_swap_correction'] = branch_swap_stats
        diagnostics['branch_identity_refinement'] = identity_stats
        diagnostics['smoothness'] = self.compare_kz_root_ordering(raw_roots, tracked_roots)
        diagnostics['raw_order_jump'] = self.kz_smoothness_map(raw_roots)
        (
            diagnostics['tracked_branch_jump'],
            diagnostics['branch_jump_ratio'],
            diagnostics['branch_jump_excess_mask']
        ) = self._branch_jump_excess_mask(
            tracked_roots,
            diagnostics['root_set_jump'],
            diagnostics['uncertainty_threshold']
        )
        diagnostics['branch_swap_correction']['pre_branch_jump_excess_pixels'] = int(
            np.count_nonzero(diagnostics['pre_swap_branch_jump_excess_mask'])
        )
        diagnostics['branch_swap_correction']['post_branch_jump_excess_pixels'] = int(
            np.count_nonzero(diagnostics['branch_jump_excess_mask'])
        )
        # Measure final pair separation, remaining swap gain, and root-sheet topology.
        branch_pair_geometry = self._branch_pair_geometry_maps(
            tracked_roots,
            degenerate_threshold=diagnostics['degenerate_threshold']
        )
        diagnostics.update(branch_pair_geometry)
        diagnostics['pair_swap_gain_map'] = self._pair_swap_gain_map(
            tracked_roots,
            complex_weight=1.0,
            real_weight=branch_identity_real_weight
        )
        final_edge_maps = self._edge_match_maps(raw_roots)
        diagnostics.update(
            self._branch_topology_maps(
                final_edge_maps,
                diagnostics['local_index_by_branch']
            )
        )
        diagnostics['uncertain_mask'] |= diagnostics['branch_jump_excess_mask']
        diagnostics['uncertain_mask'] |= diagnostics['branch_degenerate_mask']
        # Evaluate polynomial quality independently of branch-label continuity.
        (diagnostics['root_residuals'],
         diagnostics['root_condition_estimates']) = self.kz_root_quality(raw_roots)
        diagnostics['max_root_residual'] = np.full(raw_roots.shape[:2], np.nan)
        diagnostics['max_root_condition'] = np.full(raw_roots.shape[:2], np.nan)
        residual_valid = np.any(np.isfinite(diagnostics['root_residuals']), axis=2)
        diagnostics['max_root_residual'][residual_valid] = np.nanmax(
            diagnostics['root_residuals'][residual_valid], axis=1
        )
        condition_valid = np.any(~np.isnan(diagnostics['root_condition_estimates']), axis=2)
        diagnostics['max_root_condition'][condition_valid] = np.nanmax(
            diagnostics['root_condition_estimates'][condition_valid], axis=1
        )
        # Separate residual instability and ill-conditioning from topological label
        # cuts.
        diagnostics['solver_instability_mask'] = (
            ~np.isnan(diagnostics['max_root_residual'])
            & (diagnostics['max_root_residual'] > 1e-8)
        )
        diagnostics['solver_instability_mask'] |= (
            np.any(
                self._finite_complex_mask(raw_roots)
                & ~np.isfinite(diagnostics['root_residuals']), axis=2
            )
        )
        diagnostics['ill_conditioned_mask'] = (
            diagnostics['max_root_condition'] > 1e8
        )
        diagnostics['topological_branch_cut_mask'] = (
            diagnostics['branch_jump_excess_mask']
            & diagnostics['branch_cut_mask']
        )
        diagnostics['unresolved_branch_jump_mask'] = (
            diagnostics['branch_jump_excess_mask']
            & ~diagnostics['branch_degenerate_mask']
            & ~diagnostics['topological_branch_cut_mask']
        )
        # Combine failures and unresolved numerical artifacts into the plotting mask.
        diagnostics['plot_suspicious_mask'] = (
            diagnostics['solve_failed_mask']
            | diagnostics['unresolved_branch_jump_mask']
            | diagnostics['solver_instability_mask']
        )
        diagnostics['uncertain_mask'] |= diagnostics['degree_reduced_mask']
        diagnostics['uncertain_mask'] |= diagnostics['ill_conditioned_mask']
        diagnostics['uncertain_mask'] |= diagnostics['nonfinite_coefficient_mask']
        # Independently check selected sensitive cells without replacing any root.
        if high_precision_validation:
            validation_mask = (
                diagnostics['solver_instability_mask']
                | diagnostics['ill_conditioned_mask']
                | diagnostics['unresolved_branch_jump_mask']
            )
            diagnostics['high_precision_validation'] = self.kz_validate_high_precision(
                raw_roots, validation_mask, digits=high_precision_digits
            )
            high_precision = diagnostics['high_precision_validation']
            diagnostics['high_precision_disagreement_mask'] = (
                high_precision['checked_mask']
                & (high_precision['max_relative_difference'] > 1e-8)
            )
            diagnostics['uncertain_mask'] |= high_precision['failed_mask']
            diagnostics['uncertain_mask'] |= diagnostics['high_precision_disagreement_mask']
            diagnostics['plot_suspicious_mask'] |= (
                diagnostics['high_precision_disagreement_mask']
            )

        # Attach model-specific classification, propagation, and energy-direction
        # results.
        if self.model in ('slow_mode', 'fast_mode'):
            tracked_roots = self._select_magnetoacoustic_mode(tracked_roots, diagnostics)
        # Evaluate the Alfvén dispersion residual and vertical group velocity directly.
        elif self.model == 'alfven_mode':
            finite = self._finite_complex_mask(tracked_roots)
            propagating = np.zeros(tracked_roots.shape, dtype=bool)
            group_velocity = np.full(tracked_roots.shape, np.nan)
            normalized_residual = np.full(tracked_roots.shape, np.nan)
            mu = sin(self.theta)*cos(self.phi)
            nu = cos(self.theta)
            for i, omega in enumerate(self.omega):
                for j, kh in enumerate(self.kh):
                    for branch in np.flatnonzero(finite[i, j]):
                        kz = tracked_roots[i, j, branch]
                        scale = max(abs(kz.real), abs(kh), 1e-12)
                        is_real = abs(kz.imag) <= 1e-9*scale
                        propagating[i, j, branch] = is_real
                        k_parallel = kh*mu + kz*nu
                        residual_scale = abs(omega**2) + abs(self.a**2*k_parallel**2)
                        normalized_residual[i, j, branch] = abs(
                            omega**2 - self.a**2*k_parallel**2
                        )/max(residual_scale, 1e-300)
                        if is_real and abs(omega) > 1e-12:
                            group_velocity[i, j, branch] = (
                                self.a**2*k_parallel.real*nu/omega
                            )
            diagnostics['propagating_root_mask'] = propagating
            diagnostics['evanescent_root_mask'] = finite & ~propagating
            diagnostics['vertical_group_velocity_km_s'] = group_velocity
            diagnostics['normalized_alfven_residual'] = normalized_residual
            diagnostics['mode_ambiguous_mask'] = (
                np.any(~finite, axis=2) | diagnostics['branch_degenerate_mask']
            )

        # Return the requested root grid, optionally accompanied by all diagnostic maps.
        if return_diagnostics:
            return tracked_roots, diagnostics

        return tracked_roots

    def kz_root_quality(self, roots):

        '''
        Purpose
        -------
        Calculate term-scaled residuals and local relative root condition.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        residuals: np.array, float
            Dimensionless polynomial residual divided by the sum of absolute
            polynomial terms, with roots.shape.

        conditions: np.array, float
            Dimensionless local relative root-condition estimates with roots.shape.
            Unavailable coefficients or roots leave NaN entries.

        Notes
        -----
        These residuals diagnose the polynomial solve independently of branch
        tracking. Large branch jumps with small residuals point to assignment
        problems; large residuals point to an ill-conditioned or failed solve.
        '''

        # Allocate one residual and condition estimate for every supplied root.
        roots = np.asarray(roots, dtype=np.complex128)
        residuals = np.full(roots.shape, np.nan)
        conditions = np.full(roots.shape, np.nan)
        powers = None

        # Reconstruct the local polynomial, skipping invalid or constant coefficients.
        for i, omega_i in enumerate(self.omega):
            for j, kh_j in enumerate(self.kh):
                try:
                    coefficients = self.kz_coefficients(omega_i, kh_j)
                    coefficients = self._trim_polynomial_coefficients(coefficients, rtol=0.0)
                except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
                    continue

                if coefficients is None or coefficients.size <= 1:
                    continue

                # Match the power vector to the local polynomial's effective degree.
                if powers is None or powers.size != coefficients.size:
                    powers = np.arange(coefficients.size - 1, -1, -1)

                point_roots = roots[i, j, :]
                valid = self._finite_complex_mask(point_roots)
                if not np.any(valid):
                    continue

                # Evaluate the polynomial, absolute term sum, and derivative at finite
                # roots.
                selected = point_roots[valid]
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    values = np.polyval(coefficients, selected)
                    scale = np.sum(
                        np.abs(coefficients)[None, :]
                        * np.abs(selected)[:, None]**powers[None, :], axis=1
                    )
                    slopes = np.polyval(np.polyder(coefficients), selected)
                    residual = np.abs(values)/np.maximum(scale, 1e-300)
                    condition = scale/np.maximum(
                        np.abs(selected*slopes), 1e-300
                    )
                # Store dimensionless residuals and relative root-condition estimates.
                residuals[i, j, valid] = residual
                conditions[i, j, valid] = condition

        return residuals, conditions

    def kz_root_residuals(self, roots):
        '''
        Purpose
        -------
        Return term-scaled residuals, preserving the original public API.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        residuals: np.array, float
            Term-scaled polynomial residuals with roots.shape; unavailable values
            remain NaN.
        '''

        # Expose only the residual component of the shared quality calculation.
        return self.kz_root_quality(roots)[0]

    def kz_validate_high_precision(self, roots, point_mask, digits=60):
        '''
        Purpose
        -------
        Independently resolve selected root sets with mpmath precision.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        point_mask: np.array, bool
            Cells to check, with shape (n_omega, n_kh).

        digits: int
            Decimal precision used by the optional mpmath root calculation.

        Outputs
        -------
        validation: dict
            checked_mask, failed_mask, maximum matched relative root difference, and
            decimal precision. Recompute roots of the existing floating-point
            coefficients; do not replace roots or labels.

        Notes
        -----
        This is an opt-in check for numerically sensitive grid points. It does
        not silently substitute a different root or branch label. A failed
        high-precision solve is reported rather than interpreted as agreement.
        '''

        # Load the optional high-precision dependency only when validation is requested.
        try:
            import mpmath as mp
        except ImportError as exc:
            raise ImportError(
                'High-precision validation requires mpmath.'
            ) from exc
        # Require the selected-cell mask to match the supplied root grid.
        roots = np.asarray(roots, dtype=np.complex128)
        point_mask = np.asarray(point_mask, dtype=bool)
        if roots.shape[:2] != point_mask.shape:
            raise ValueError('point_mask must match the omega-kh grid.')
        # Track successful checks separately from failed or unselected cells.
        difference = np.full(point_mask.shape, np.nan)
        failed = np.zeros(point_mask.shape, dtype=bool)
        checked = np.zeros(point_mask.shape, dtype=bool)
        # Re-solve the existing coefficients at the requested decimal precision.
        with mp.workdps(int(digits)):
            for i, j in np.argwhere(point_mask):
                coefficients = self.kz_coefficients(self.omega[i], self.kh[j])
                scaled, radius, degree = self._scaled_kz_coefficients(coefficients)
                finite = roots[i, j, self._finite_complex_mask(roots[i, j])]
                if scaled is None or finite.size != degree:
                    failed[i, j] = True
                    continue
                try:
                    # Apply the same root-radius scaling before the high-precision
                    # solve.
                    mp_radius = mp.mpf(str(radius))
                    raw = coefficients[np.flatnonzero(coefficients != 0)[0]:]
                    mp_coefficients = [
                        mp.mpc(str(value.real), str(value.imag))
                        * mp_radius**power
                        for power, value in zip(range(degree, -1, -1), raw)
                    ]
                    scale = max(abs(value) for value in mp_coefficients)
                    reference = np.asarray([
                        complex(value*mp_radius)
                        for value in mp.polyroots(
                            [value/scale for value in mp_coefficients],
                            maxsteps=200, extraprec=digits
                        )
                    ], dtype=np.complex128)
                    # Match the two complete root sets before measuring relative
                    # disagreement.
                    rows, cols = optimize.linear_sum_assignment(
                        np.abs(finite[:, None] - reference[None, :])
                    )
                    difference[i, j] = float(np.max(
                        np.abs(finite[rows] - reference[cols])
                        / np.maximum(np.abs(reference[cols]), radius*1e-12)
                    ))
                    checked[i, j] = True
                except (ArithmeticError, ValueError, OverflowError):
                    failed[i, j] = True
        # Report verification evidence without replacing the input root values.
        return {'checked_mask': checked, 'failed_mask': failed,
                'max_relative_difference': difference, 'digits': int(digits)}

    def kz_branch_jump_maps(self, roots):

        '''
        Purpose
        -------
        Map the largest adjacent-grid root jump touching each branch pixel.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        jumps: np.array, float
            Largest finite adjacent-grid root displacement touching each branch
            pixel, with roots.shape, in km^-1; cells with no finite neighbors remain
            NaN.
        '''

        # Initialize a separate neighboring-jump map for each branch.
        roots = np.asarray(roots, dtype=np.complex128)
        jumps = np.full(roots.shape, np.nan)

        for branch in range(roots.shape[2]):
            # Measure finite frequency-edge jumps and update both adjacent pixels.
            omega_a = roots[1:, :, branch]
            omega_b = roots[:-1, :, branch]
            omega_valid = self._finite_complex_mask(omega_a) & self._finite_complex_mask(omega_b)
            omega_jump = np.full(omega_a.shape, np.nan)
            omega_jump[omega_valid] = np.abs(omega_a[omega_valid] - omega_b[omega_valid])
            self._update_max_map(jumps[1:, :, branch], omega_jump)
            self._update_max_map(jumps[:-1, :, branch], omega_jump)

            # Repeat the maximum-jump update for horizontal-wavenumber edges.
            kh_a = roots[:, 1:, branch]
            kh_b = roots[:, :-1, branch]
            kh_valid = self._finite_complex_mask(kh_a) & self._finite_complex_mask(kh_b)
            kh_jump = np.full(kh_a.shape, np.nan)
            kh_jump[kh_valid] = np.abs(kh_a[kh_valid] - kh_b[kh_valid])
            self._update_max_map(jumps[:, 1:, branch], kh_jump)
            self._update_max_map(jumps[:, :-1, branch], kh_jump)

        return jumps

    def kz_smoothness_map(self, roots):

        '''
        Purpose
        -------
        Map the largest finite neighbor-to-neighbor change across all branches.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        smoothness: np.array, float
            Maximum finite neighboring-root displacement over all branches at each
            cell, shape (n_omega, n_kh), in km^-1; no finite edges gives NaN.
        '''

        # Allocate a shared maximum-jump map over all branch values.
        roots = np.asarray(roots, dtype=np.complex128)
        smoothness = np.full(roots.shape[:2], np.nan)

        def update(target, values):
            '''
            Purpose
            -------
            Merge adjacent-grid changes into the shared maximum smoothness map.

            Inputs
            ------
            target: np.array, float
                Writable view of the smoothness map to update in place.

            values: np.array, float
                Finite edge changes to merge into the target view (km^-1).

            Outputs
            -------
            None
                Merge finite edge changes into target in place, retaining the
                largest available change.
            '''

            # Update the existing map only with finite, larger edge changes.
            self._update_max_map(target, values)

        # Merge finite edge changes from every branch and both grid directions.
        for branch in range(roots.shape[2]):
            # Compare adjacent frequency rows and update both ends of each edge.
            omega_a = roots[1:, :, branch]
            omega_b = roots[:-1, :, branch]
            omega_valid = self._finite_complex_mask(omega_a) & self._finite_complex_mask(omega_b)
            omega_jump = np.full(omega_a.shape, np.nan)
            omega_jump[omega_valid] = np.abs(omega_a[omega_valid] - omega_b[omega_valid])
            update(smoothness[1:, :], omega_jump)
            update(smoothness[:-1, :], omega_jump)

            # Compare adjacent horizontal-wavenumber columns in the same way.
            kh_a = roots[:, 1:, branch]
            kh_b = roots[:, :-1, branch]
            kh_valid = self._finite_complex_mask(kh_a) & self._finite_complex_mask(kh_b)
            kh_jump = np.full(kh_a.shape, np.nan)
            kh_jump[kh_valid] = np.abs(kh_a[kh_valid] - kh_b[kh_valid])
            update(smoothness[:, 1:], kh_jump)
            update(smoothness[:, :-1], kh_jump)

        return smoothness

    def kz_smoothness_summary(self, roots):

        '''
        Purpose
        -------
        Summarize finite adjacent-root jumps globally and by branch.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        summary: dict
            Mean, median, 95th percentile, maximum, and per-branch 95th percentile
            of finite edge jumps (km^-1); empty samples give NaN.
        '''

        # Collect finite adjacent-root changes globally and for each branch.
        roots = np.asarray(roots, dtype=np.complex128)
        jumps = []
        per_branch_p95 = []

        # Use both grid axes when calculating branch jump percentiles.
        for branch in range(roots.shape[2]):
            branch_jumps = []
            for axis in (0, 1):
                ahead = np.take(roots[:, :, branch], indices=range(1, roots.shape[axis]), axis=axis)
                behind = np.take(roots[:, :, branch], indices=range(0, roots.shape[axis] - 1), axis=axis)
                valid = self._finite_complex_mask(ahead) & self._finite_complex_mask(behind)
                diff = np.abs(ahead[valid] - behind[valid])
                if diff.size > 0:
                    branch_jumps.append(diff)
                    jumps.append(diff)
            if len(branch_jumps) == 0:
                per_branch_p95.append(np.nan)
            else:
                per_branch_p95.append(float(np.percentile(np.concatenate(branch_jumps), 95)))

        # Combine edge samples, retaining an empty array when none are finite.
        if len(jumps) == 0:
            all_jumps = np.asarray([])
        else:
            all_jumps = np.concatenate(jumps)

        # Return NaN statistics when the grid has no finite adjacent-root samples.
        if all_jumps.size == 0:
            return {
                'mean': np.nan,
                'median': np.nan,
                'p95': np.nan,
                'max': np.nan,
                'per_branch_p95': np.asarray(per_branch_p95)
            }

        # Summarize the available edge distribution in the original wavenumber units.
        return {
            'mean': float(np.mean(all_jumps)),
            'median': float(np.median(all_jumps)),
            'p95': float(np.percentile(all_jumps, 95)),
            'max': float(np.max(all_jumps)),
            'per_branch_p95': np.asarray(per_branch_p95)
        }

    def compare_kz_root_ordering(self, raw_roots, tracked_roots):

        '''
        Purpose
        -------
        Compare raw and tracked smoothness and report the 95th-percentile reduction.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        Outputs
        -------
        comparison: dict
            Raw/tracked jump summaries and raw-p95/tracked-p95 reduction factor; a
            missing or nonpositive tracked percentile gives NaN.
        '''

        # Compute the same jump statistics for raw and continuity-tracked slots.
        raw_summary = self.kz_smoothness_summary(raw_roots)
        tracked_summary = self.kz_smoothness_summary(tracked_roots)
        raw_p95 = raw_summary['p95']
        tracked_p95 = tracked_summary['p95']
        # Form the reduction factor only for finite, positive tracked jump scales.
        if np.isfinite(raw_p95) and np.isfinite(tracked_p95) and tracked_p95 > 0:
            p95_reduction_factor = raw_p95 / tracked_p95
        else:
            p95_reduction_factor = np.nan

        return {
            'raw': raw_summary,
            'tracked': tracked_summary,
            'p95_jump_reduction_factor': p95_reduction_factor
        }

    def plot_kz_branches(
        self,
        roots,
        component='real',
        kh_scale=1e3,
        omega_scale=(2*pi)**(-1)*1e3,
        cmap='coolwarm',
        robust=True,
        diagnostics=None,
        mask=None,
        mask_suspicious=False,
        vmin=None,
        vmax=None,
        bad_color='0.65',
        shared_colorbar=True,
        colorbar_label=None
    ):

        '''
        Purpose
        -------
        Plot each kz branch over the omega-kh grid with shared scaling options.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        component: str
            Root component: 'real', 'imag', 'abs', or 'phase'; aliases accepted by
            _grid_component() also apply. Phase is in radians; other components are
            in km^-1.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        cmap: str or matplotlib.colors.Colormap
            Colormap used to display the selected root component.

        robust: bool
            Use finite-value percentile limits instead of the full finite range.

        diagnostics: dict or None
            Optional maps used to hide flagged cells when mask_suspicious=True.

        mask: np.array, bool or None
            Explicit cells to hide; shape (n_omega, n_kh) or (n_omega, n_kh,
            n_branch). When combining with a two-dimensional diagnostic mask,
            supply a two-dimensional explicit mask for compatible broadcasting.

        mask_suspicious: bool
            Also hide diagnostics['plot_suspicious_mask'], falling back to
            uncertain_mask when needed.

        vmin: float or None
            Lower color limit; calculate it from visible values when omitted.

        vmax: float or None
            Upper color limit; calculate it from visible values when omitted.

        bad_color: color
            Color assigned to masked or nonfinite root values.

        shared_colorbar: bool
            Use one colorbar for all branch panels when True; otherwise add one to
            each panel.

        colorbar_label: str or None
            Optional text applied to the root-value colorbar.

        Outputs
        -------
        fig, axes: tuple
            Matplotlib figure and two-dimensional axes array. Unused panels are
            hidden; the figure is returned without calling show().
        '''

        # Import plotting support only when a figure is requested.
        import matplotlib.pyplot as plt

        # Extract the requested component and apply any explicit diagnostic mask.
        values = self.kz_plot_values(
            roots,
            component=component,
            diagnostics=diagnostics,
            mask=mask,
            mask_suspicious=mask_suspicious
        )
        n_branch = roots.shape[2]
        n_cols = min(3, n_branch)
        n_rows = int(np.ceil(n_branch/n_cols))

        # Allocate up to three branch panels per row and convert the grid coordinates.
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5*n_cols, 3.8*n_rows),
            squeeze=False,
            constrained_layout=shared_colorbar
        )
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        # Derive any unspecified color limits from the same visible root values.
        if vmin is None or vmax is None:
            computed_vmin, computed_vmax = self.kz_color_limits(
                roots,
                component=component,
                diagnostics=diagnostics,
                mask=mask,
                mask_suspicious=mask_suspicious,
                robust=robust
            )
            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Copy the colormap so masked-value colors do not alter global settings.
        cmap_plot = plt.get_cmap(cmap).copy()
        cmap_plot.set_bad(bad_color)
        mesh = None
        plotted_axes = []

        # Draw each available branch and hide the unused subplot slots.
        for branch in range(n_rows*n_cols):
            ax = axes.flat[branch]
            if branch >= n_branch:
                ax.axis('off')
                continue
            mesh = ax.pcolormesh(
                kh_plot, omega_plot, values[:, :, branch], shading='auto',
                cmap=cmap_plot, vmin=vmin, vmax=vmax, rasterized=True
            )
            ax.set_title(f'kz branch {branch + 1} ({component})')
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
            ax.set_ylabel(r'$\nu\,[\mathrm{mHz}]$')
            plotted_axes.append(ax)
            if not shared_colorbar:
                branch_colorbar = fig.colorbar(mesh, ax=ax)
                if colorbar_label is not None:
                    branch_colorbar.set_label(colorbar_label)

        # Attach the shared colorbar after all branch panels have been drawn.
        if shared_colorbar and mesh is not None:
            colorbar = fig.colorbar(mesh, ax=plotted_axes)
            if colorbar_label is not None:
                colorbar.set_label(colorbar_label)

        if not shared_colorbar:
            fig.tight_layout()
        return fig, axes

    def kz_plot_values(self, roots, component='real', diagnostics=None, mask=None, mask_suspicious=False):

        '''
        Purpose
        -------
        Extract a root component and mask explicitly selected or suspicious cells.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        component: str
            Root component: 'real', 'imag', 'abs', or 'phase'; aliases accepted by
            _grid_component() also apply. Phase is in radians; other components are
            in km^-1.

        diagnostics: dict or None
            Optional plot-suspicion or uncertainty masks to apply when
            mask_suspicious=True.

        mask: np.array, bool or None
            Explicit cells to hide; shape (n_omega, n_kh) or (n_omega, n_kh,
            n_branch). When combining with a two-dimensional diagnostic mask,
            supply a two-dimensional explicit mask for compatible broadcasting.

        mask_suspicious: bool
            Also hide diagnostics['plot_suspicious_mask'], falling back to
            uncertain_mask when needed.

        Outputs
        -------
        values: np.array, float
            Independent component array with roots.shape; selected cells are
            replaced by NaN without modifying roots.
        '''

        # Copy the requested root component before applying any visibility masks.
        roots = np.asarray(roots, dtype=np.complex128)
        values = self._grid_component(roots, component).copy()

        # Combine the caller's mask with optional diagnostic exclusions.
        plot_mask = None
        if mask is not None:
            plot_mask = np.asarray(mask, dtype=bool)
        if mask_suspicious and diagnostics is not None:
            # Prefer the plot-specific mask: pixels with failed solves, high
            # residuals, or branch jumps larger than the local raw root-set
            # motion. Broader degeneracy/uncertainty regions remain available
            # in diagnostics['uncertain_mask'] without being hidden by default.
            if 'plot_suspicious_mask' in diagnostics:
                suspicious = np.asarray(diagnostics['plot_suspicious_mask'], dtype=bool)
            elif 'uncertain_mask' in diagnostics:
                suspicious = np.asarray(diagnostics['uncertain_mask'], dtype=bool)
            else:
                suspicious = None
            if suspicious is not None:
                plot_mask = suspicious if plot_mask is None else (plot_mask | suspicious)

        # Apply whole-cell or per-branch exclusions according to the mask shape.
        if plot_mask is not None:
            if plot_mask.ndim == 2:
                values[plot_mask, :] = np.nan
            elif plot_mask.ndim == 3:
                values[plot_mask] = np.nan
            else:
                raise ValueError("mask must have shape (omega, kh) or (omega, kh, branch).")

        return values

    def kz_color_limits(self, roots, component='real', diagnostics=None, mask=None, mask_suspicious=False, robust=False, percentiles=(2, 98), symmetric=False):

        '''
        Purpose
        -------
        Compute finite plot limits, optionally using robust percentiles or symmetry.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        component: str
            Root component: 'real', 'imag', 'abs', or 'phase'; aliases accepted by
            _grid_component() also apply. Phase is in radians; other components are
            in km^-1.

        diagnostics: dict or None
            Optional maps supplying the mask used for visible color-limit samples.

        mask: np.array, bool or None
            Explicit cells to hide; shape (n_omega, n_kh) or (n_omega, n_kh,
            n_branch). When combining with a two-dimensional diagnostic mask,
            supply a two-dimensional explicit mask for compatible broadcasting.

        mask_suspicious: bool
            Also hide diagnostics['plot_suspicious_mask'], falling back to
            uncertain_mask when needed.

        robust: bool
            Use finite-value percentile limits instead of the full finite range.

        percentiles: tuple
            Lower and upper percentiles used when robust=True.

        symmetric: bool
            Expand the finite color range to equal positive and negative limits.

        Outputs
        -------
        vmin, vmax: tuple
            Finite limits in the selected component's units; return (None, None)
            when no visible finite values remain.
        '''

        # Use exactly the values and masks that the root plot will display.
        values = self.kz_plot_values(
            roots,
            component=component,
            diagnostics=diagnostics,
            mask=mask,
            mask_suspicious=mask_suspicious
        )
        # Exclude nonfinite entries and leave limits unspecified for an empty plot.
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None, None

        # Use percentile limits unless the resulting range collapses.
        if robust:
            vmin, vmax = np.percentile(finite_values, percentiles)
            if np.isclose(vmin, vmax):
                vmin, vmax = np.min(finite_values), np.max(finite_values)
        else:
            vmin, vmax = np.min(finite_values), np.max(finite_values)

        # Expand both sides to the larger absolute color limit when requested.
        if symmetric:
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit

        return vmin, vmax

    def plot_kz_tracking_diagnostics(self, raw_roots, tracked_roots, diagnostics=None, kh_scale=1e3, omega_scale=(2*pi)**(-1)*1e3):

        '''
        Purpose
        -------
        Plot root jumps, separation, assignment confidence, residuals, and flagged
        cells.

        Inputs
        ------
        raw_roots: np.array, complex
            Unordered polynomial roots with shape (n_omega, n_kh, n_branch), in
            km^-1; unavailable roots are NaN.

        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict or None
            Optional solver/tracking maps. When omitted, recompute branch-tracking
            diagnostics from raw_roots.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        Outputs
        -------
        fig, axes: tuple
            Matplotlib figure and (3, 3) axes array comparing jumps, separation,
            confidence, residuals, and masks; no show() call.
        '''

        # Load plotting support and obtain tracking maps when none were supplied.
        import matplotlib.pyplot as plt

        if diagnostics is None:
            _, diagnostics = self.track_kz_branches(raw_roots, return_diagnostics=True)

        # Use solver diagnostics when available and calculate fallback jump maps.
        raw_jump = diagnostics.get('raw_order_jump', self.kz_smoothness_map(raw_roots))
        tracked_jump = diagnostics.get('tracked_branch_jump', self.kz_smoothness_map(tracked_roots))
        root_set_jump = diagnostics.get('root_set_jump', self.kz_smoothness_map(raw_roots))
        max_residual = diagnostics.get('max_root_residual', np.full(raw_jump.shape, np.nan))

        def log10_positive(values):
            '''
            Purpose
            -------
            Apply log10 only to positive entries and leave other cells as NaN.

            Inputs
            ------
            values: np.array, float
                Values to transform; entries that are not positive are left as NaN.

            Outputs
            -------
            logged: np.array, float
                Base-10 logarithm with the input shape; zero, negative, and NaN
                entries remain NaN.
            '''

            # Transform positive values only, leaving undefined logarithms as NaN.
            values = np.asarray(values)
            logged = np.full(values.shape, np.nan, dtype=float)
            valid = values > 0
            logged[valid] = np.log10(values[valid])
            return logged

        # Build the nine-panel grid using the requested coordinate conversions.
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), squeeze=False)
        plot_suspicious_mask = diagnostics.get('plot_suspicious_mask', diagnostics['uncertain_mask'])
        # Compare root motion, assignment quality, polynomial error, and flagged cells.
        panels = [
            (axes[0, 0], log10_positive(raw_jump), 'raw solver-order log10 jump', 'viridis', None, None),
            (axes[0, 1], log10_positive(tracked_jump), 'tracked-branch log10 jump', 'viridis', None, None),
            (axes[0, 2], log10_positive(root_set_jump), 'root-set optimal log10 jump', 'viridis', None, None),
            (axes[1, 0], log10_positive(diagnostics['min_root_separation']), 'log10 min root separation', 'magma', None, None),
            (axes[1, 1], log10_positive(diagnostics['max_assignment_cost']), 'log10 assignment cost', 'magma', None, None),
            (axes[1, 2], diagnostics['assignment_confidence'], 'assignment confidence', 'cividis', 0, 1),
            (axes[2, 0], log10_positive(max_residual), 'log10 polynomial residual', 'plasma', None, None),
            (axes[2, 1], diagnostics['degenerate_mask'].astype(float), 'near-degenerate mask', 'gray_r', 0, 1),
            (axes[2, 2], np.asarray(plot_suspicious_mask, dtype=float), 'flagged plot-artifact mask', 'gray_r', 0, 1)
        ]

        # Draw each diagnostic with its own scale, labels, and colorbar.
        for ax, values, title, cmap, vmin, vmax in panels:
            mesh = ax.pcolormesh(kh_plot, omega_plot, values, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_title(title)
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
            ax.set_ylabel(r'$\nu\,[\mathrm{mHz}]$')
            fig.colorbar(mesh, ax=ax)

        fig.tight_layout()
        return fig, axes

    def plot_branch_swap_diagnostics(
        self,
        tracked_roots,
        diagnostics,
        branches=None,
        component='real',
        dz=None,
        kh_scale=1e3,
        omega_scale=(2*pi)**(-1)*1e3,
        kz_cmap='hot',
        phase_cmap='coolwarm',
        phase_vlim=80
    ):

        '''
        Purpose
        -------
        Plot before/after maps for the branch-swap synchronization pass.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict
            Tracking and solver maps returned by kz_solve_grid(return_diagnostics=True).

        branches: iterable or None
            Zero-based branch indices to display; None selects all branches.

        component: str
            Root component: 'real', 'imag', 'abs', or 'phase'; aliases accepted by
            _grid_component() also apply. Phase is in radians; other components are
            in km^-1.

        dz: float or None
            Height separation (km). When supplied, include phase differences in degrees.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        kz_cmap: str or matplotlib.colors.Colormap
            Colormap used for the vertical-wavenumber panels.

        phase_cmap: str or matplotlib.colors.Colormap
            Colormap used for phase or paired branch-value panels.

        phase_vlim: float
            Absolute limit of the symmetric phase color scale (degrees).

        Outputs
        -------
        fig, axes: tuple
            Matplotlib figure and (2, n_branches) axes array, or (4, n_branches)
            when phase panels are included; no show() call.

        Notes
        -----
        This diagnostic is intended to verify that the correction relabels
        roots across a suspected branch-cut boundary instead of masking the
        affected pixels. If dz is supplied, the corresponding phase-difference
        maps are plotted below the kz maps.
        '''

        # Load plotting support and require the saved pre-correction root grid.
        import matplotlib.pyplot as plt

        if 'pre_swap_tracked_roots' not in diagnostics:
            raise ValueError("diagnostics must include 'pre_swap_tracked_roots'. Run kz_solve_grid(return_diagnostics=True).")

        # Select the zero-based branch indices for the before/after comparison.
        pre_roots = np.asarray(diagnostics['pre_swap_tracked_roots'], dtype=np.complex128)
        post_roots = np.asarray(tracked_roots, dtype=np.complex128)
        if branches is None:
            branches = range(post_roots.shape[2])
        branches = list(branches)

        # Use one finite wavenumber color range for both assignments.
        kz_pre = self._grid_component(pre_roots, component)
        kz_post = self._grid_component(post_roots, component)
        kz_finite = np.concatenate((kz_pre[np.isfinite(kz_pre)], kz_post[np.isfinite(kz_post)]))
        kz_vmin = np.min(kz_finite) if kz_finite.size > 0 else None
        kz_vmax = np.max(kz_finite) if kz_finite.size > 0 else None

        # Reserve two additional rows when a height separation is supplied.
        plot_phase = dz is not None
        n_rows = 4 if plot_phase else 2
        fig, axes = plt.subplots(
            n_rows,
            len(branches),
            figsize=(4.1*len(branches), 3.5*n_rows),
            squeeze=False,
            constrained_layout=True
        )
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        kz_cmap_plot = plt.get_cmap(kz_cmap).copy()
        kz_cmap_plot.set_bad('0.65')

        # Compare the same branch slots before and after label synchronization.
        for col, branch in enumerate(branches):
            mesh = axes[0, col].pcolormesh(
                kh_plot,
                omega_plot,
                kz_pre[:, :, branch],
                shading='auto',
                cmap=kz_cmap_plot,
                vmin=kz_vmin,
                vmax=kz_vmax,
                rasterized=True
            )
            axes[0, col].set_title(f'pre-swap kz {branch + 1}')
            axes[1, col].pcolormesh(
                kh_plot,
                omega_plot,
                kz_post[:, :, branch],
                shading='auto',
                cmap=kz_cmap_plot,
                vmin=kz_vmin,
                vmax=kz_vmax,
                rasterized=True
            )
            axes[1, col].set_title(f'post-swap kz {branch + 1}')
            for row in range(2):
                axes[row, col].set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
                axes[row, col].set_ylabel(r'$\nu\,[\mathrm{mHz}]$')

        # Attach one wavenumber colorbar to the before/after rows.
        colorbar = fig.colorbar(mesh, ax=axes[:2, :].ravel())
        colorbar.set_label(r'$k_z\,[\mathrm{km}^{-1}]$')

        # Convert both root assignments to phase differences for the supplied height
        # gap.
        if plot_phase:
            _, omega_grid = np.meshgrid(self.kh, self.omega)
            phase_pre = []
            phase_post = []
            for branch in branches:
                pre_speed = self.phase_speed(omega_grid, pre_roots[:, :, branch])
                post_speed = self.phase_speed(omega_grid, post_roots[:, :, branch])
                phase_pre.append(
                    np.real(np.nan_to_num(self.phase_difference(omega_grid, pre_speed, dz)*180*pi**(-1), nan=0.0, posinf=0.0, neginf=0.0))
                )
                phase_post.append(
                    np.real(np.nan_to_num(self.phase_difference(omega_grid, post_speed, dz)*180*pi**(-1), nan=0.0, posinf=0.0, neginf=0.0))
                )

            # Draw phase panels with one symmetric degree scale.
            phase_mesh = None
            for col, branch in enumerate(branches):
                phase_mesh = axes[2, col].pcolormesh(
                    kh_plot,
                    omega_plot,
                    phase_pre[col],
                    shading='auto',
                    cmap=phase_cmap,
                    vmin=-phase_vlim,
                    vmax=phase_vlim,
                    rasterized=True
                )
                axes[2, col].set_title(f'pre-swap Delta phi {branch + 1}')
                axes[3, col].pcolormesh(
                    kh_plot,
                    omega_plot,
                    phase_post[col],
                    shading='auto',
                    cmap=phase_cmap,
                    vmin=-phase_vlim,
                    vmax=phase_vlim,
                    rasterized=True
                )
                axes[3, col].set_title(f'post-swap Delta phi {branch + 1}')
                for row in range(2, 4):
                    axes[row, col].set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
                    axes[row, col].set_ylabel(r'$\nu\,[\mathrm{mHz}]$')

            # Label the shared phase-difference colorbar in degrees.
            phase_colorbar = fig.colorbar(phase_mesh, ax=axes[2:, :].ravel())
            phase_colorbar.set_label(r'$\Delta\phi\,[\mathrm{deg}]$')

        return fig, axes

    def plot_branch_identity_diagnostics(
        self,
        tracked_roots,
        diagnostics,
        branch_pair=None,
        dz=None,
        kh_scale=1e3,
        omega_scale=(2*pi)**(-1)*1e3,
        phase_cmap='coolwarm',
        scalar_cmap='viridis',
        phase_vlim=80,
        pair_close_rtol=0.25
    ):

        '''
        Purpose
        -------
        Inspect whether a suspected large-scale boundary is a branch swap.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict
            Tracking and solver maps returned by kz_solve_grid(return_diagnostics=True).

        branch_pair: tuple or None
            Two one-based branch numbers. Automatic selection first uses the most
            frequent close pair, then the largest refinement candidate, then the
            first two branches.

        dz: float or None
            Height separation (km). When supplied, include phase differences in degrees.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        phase_cmap: str or matplotlib.colors.Colormap
            Colormap used for phase or paired branch-value panels.

        scalar_cmap: str or matplotlib.colors.Colormap
            Colormap used for scalar diagnostic maps.

        phase_vlim: float
            Absolute limit of the symmetric phase color scale (degrees).

        pair_close_rtol: float
            Relative separation threshold for the selected pair's close-root mask.

        Outputs
        -------
        fig, axes: tuple
            Matplotlib figure and (3, 4) axes array comparing the current/candidate
            pair, separation, swap gain, and tracking masks. The candidate is
            displayed without changing tracked_roots.

        Notes
        -----
        The plot compares the current assignment with the candidate obtained
        by swapping the selected pair over the graph-parity region. During
        kz_solve_grid, enabled identity refinement applies a candidate only
        when its fractional objective improvement exceeds the configured
        threshold. This plot displays a candidate without applying it. A
        rejected candidate alone does not establish a true root degeneracy.
        '''

        # Load plotting support and read the recorded identity-refinement evidence.
        import matplotlib.pyplot as plt

        tracked_roots = np.asarray(tracked_roots, dtype=np.complex128)
        n_branch = tracked_roots.shape[2]
        identity_stats = diagnostics.get('branch_identity_refinement', {})
        pair_stats = identity_stats.get('pair_stats', [])

        # Prefer the most frequently close pair, then the largest refinement candidate.
        if branch_pair is None:
            close_mask = diagnostics.get('branch_close_mask')
            nearest_a = diagnostics.get('nearest_branch_a')
            nearest_b = diagnostics.get('nearest_branch_b')
            if close_mask is not None and nearest_a is not None and nearest_b is not None and np.any(close_mask):
                pair_counts = {}
                for pair_a, pair_b in zip(nearest_a[close_mask], nearest_b[close_mask]):
                    pair = (int(pair_a), int(pair_b))
                    if pair[0] >= 0 and pair[1] >= 0:
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
                branch_a, branch_b = max(pair_counts.items(), key=lambda item: item[1])[0]
            else:
                candidate_stats = [
                    stat for stat in pair_stats
                    if stat.get('candidate_pixels', stat.get('swap_pixels', 0)) > 0
                ]
                if len(candidate_stats) > 0:
                    selected = max(
                        candidate_stats,
                        key=lambda stat: (
                            stat.get('candidate_pixels', stat.get('swap_pixels', 0)),
                            stat.get('objective_improvement_rtol', -np.inf)
                        )
                    )
                    branch_a, branch_b = selected['pair']
                else:
                    branch_a = 0
                    branch_b = 1 if n_branch > 1 else 0
        else:
            if len(branch_pair) != 2:
                raise ValueError("branch_pair must contain two one-based branch numbers.")
            branch_a = int(branch_pair[0]) - 1
            branch_b = int(branch_pair[1]) - 1

        # Validate and normalize the selected one-based branch pair.
        if branch_a < 0 or branch_b < 0 or branch_a >= n_branch or branch_b >= n_branch or branch_a == branch_b:
            raise ValueError("branch_pair must refer to two valid one-based branches.")
        if branch_a > branch_b:
            branch_a, branch_b = branch_b, branch_a

        # Reconstruct the parity candidate using the recorded refinement settings.
        raw_roots = diagnostics.get('raw_roots')
        local_index_by_branch = diagnostics.get('local_index_by_branch')
        branch_values = tracked_roots
        real_weight = identity_stats.get('real_weight', 2.0)
        ratio_threshold = identity_stats.get('ratio_threshold', 1.2)
        parity, parity_stats = self._pair_swap_parity(
            branch_values,
            branch_a,
            branch_b,
            ratio_threshold,
            complex_weight=1.0,
            real_weight=real_weight
        )

        # Use raw-root indices when available, otherwise swap a root-grid copy.
        if raw_roots is not None and local_index_by_branch is not None:
            candidate_indices, _ = self._apply_pair_swap(
                local_index_by_branch,
                branch_a,
                branch_b,
                parity
            )
            candidate_roots = self._values_from_local_indices(raw_roots, candidate_indices)
        else:
            candidate_roots = tracked_roots.copy()
            candidate_a_view = candidate_roots[:, :, branch_a]
            candidate_b_view = candidate_roots[:, :, branch_b]
            candidate_a = candidate_a_view[parity].copy()
            candidate_a_view[parity] = candidate_b_view[parity]
            candidate_b_view[parity] = candidate_a

        # Compare the global identity objective without committing the candidate.
        before_objective = self._pair_identity_objective(branch_values, branch_a, branch_b)
        after_objective = self._pair_identity_objective(candidate_roots, branch_a, branch_b)
        improvement = before_objective['total'] - after_objective['total']
        improvement_rtol = improvement / max(abs(before_objective['total']), 1e-300)

        # Choose real wavenumber values or height-scaled phase values for comparison.
        if dz is None:
            current_values = np.real(tracked_roots)
            candidate_values = np.real(candidate_roots)
            color_label = r'$k_z\,[\mathrm{km}^{-1}]$'
            finite = np.concatenate((
                current_values[:, :, [branch_a, branch_b]][np.isfinite(current_values[:, :, [branch_a, branch_b]])],
                candidate_values[:, :, [branch_a, branch_b]][np.isfinite(candidate_values[:, :, [branch_a, branch_b]])]
            ))
            vmin = np.min(finite) if finite.size > 0 else None
            vmax = np.max(finite) if finite.size > 0 else None
        else:
            current_values = np.real(tracked_roots)*dz*180*pi**(-1)
            candidate_values = np.real(candidate_roots)*dz*180*pi**(-1)
            color_label = r'$\Delta\phi\,[\mathrm{deg}]$'
            vmin = -phase_vlim
            vmax = phase_vlim

        # Measure the selected pair's separation and its relative close-root region.
        separation = np.abs(tracked_roots[:, :, branch_a] - tracked_roots[:, :, branch_b])
        abs_pair = np.stack((
            np.abs(tracked_roots[:, :, branch_a]),
            np.abs(tracked_roots[:, :, branch_b])
        ), axis=2)
        pair_scale = np.nanmedian(np.where(np.isfinite(abs_pair), abs_pair, np.nan), axis=2)
        pair_scale[~np.isfinite(pair_scale)] = 1.0
        close_mask = np.isfinite(separation) & (separation <= pair_close_rtol*np.maximum(pair_scale, 1e-300))
        log_separation = np.full(separation.shape, np.nan)
        valid_separation = separation > 0
        log_separation[valid_separation] = np.log10(separation[valid_separation])

        # Map positive local swap gains and display their logarithms.
        gain_map = self._pair_swap_gain_map(
            branch_values,
            branch_a=branch_a,
            branch_b=branch_b,
            complex_weight=1.0,
            real_weight=real_weight
        )
        log_gain = np.full(gain_map.shape, np.nan)
        valid_gain = gain_map > 0
        log_gain[valid_gain] = np.log10(gain_map[valid_gain])

        # Build comparison panels and diagnostic panels on a common coordinate grid.
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), squeeze=False, constrained_layout=True)
        scalar_cmap_plot = plt.get_cmap(scalar_cmap).copy()
        scalar_cmap_plot.set_bad('0.65')

        panels = [
            (axes[0, 0], current_values[:, :, branch_a], f'current branch {branch_a + 1}', phase_cmap, vmin, vmax),
            (axes[0, 1], current_values[:, :, branch_b], f'current branch {branch_b + 1}', phase_cmap, vmin, vmax),
            (axes[0, 2], candidate_values[:, :, branch_a], f'candidate branch {branch_a + 1}', phase_cmap, vmin, vmax),
            (axes[0, 3], candidate_values[:, :, branch_b], f'candidate branch {branch_b + 1}', phase_cmap, vmin, vmax),
            (axes[1, 0], log_separation, 'log10 pair separation', scalar_cmap_plot, None, None),
            (axes[1, 1], close_mask.astype(float), 'close-root zone', 'gray_r', 0, 1),
            (axes[1, 2], log_gain, 'log10 positive swap gain', scalar_cmap_plot, None, None),
            (axes[1, 3], parity.astype(float), 'candidate swap region', 'gray_r', 0, 1),
            (axes[2, 0], diagnostics.get('tracked_branch_jump', self.kz_smoothness_map(tracked_roots)), 'tracked jump', scalar_cmap_plot, None, None),
            (axes[2, 1], diagnostics.get('root_set_jump', self.kz_smoothness_map(tracked_roots)), 'root-set jump floor', scalar_cmap_plot, None, None),
            (axes[2, 2], diagnostics.get('branch_degenerate_mask', np.zeros(separation.shape)).astype(float), 'true-degeneracy mask', 'gray_r', 0, 1),
            (axes[2, 3], diagnostics.get('plot_suspicious_mask', np.zeros(separation.shape)).astype(float), 'unresolved artifact mask', 'gray_r', 0, 1)
        ]

        # Share the first row's value scale and give scalar diagnostics individual
        # colorbars.
        main_mesh = None
        for panel_index, (ax, values, title, cmap, panel_vmin, panel_vmax) in enumerate(panels):
            mesh = ax.pcolormesh(
                kh_plot,
                omega_plot,
                values,
                shading='auto',
                cmap=cmap,
                vmin=panel_vmin,
                vmax=panel_vmax,
                rasterized=True
            )
            ax.set_title(title)
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
            ax.set_ylabel(r'$\nu\,[\mathrm{mHz}]$')
            if panel_index < 4:
                main_mesh = mesh
            else:
                fig.colorbar(mesh, ax=ax)

        if main_mesh is not None:
            colorbar = fig.colorbar(main_mesh, ax=axes[0, :].ravel())
            colorbar.set_label(color_label)

        # Report the proposed region size and fractional objective improvement.
        fig.suptitle(
            f'branch pair {branch_a + 1}-{branch_b + 1}: '
            f'candidate pixels={parity_stats["swap_pixels"]}, '
            f'objective improvement={improvement_rtol:.3e}'
        )
        return fig, axes

    def plot_branch_topology_diagnostics(
        self,
        tracked_roots,
        diagnostics,
        branch=1,
        dz=None,
        kh_scale=1e3,
        omega_scale=(2*pi)**(-1)*1e3,
        value_cmap='coolwarm',
        scalar_cmap='viridis',
        phase_vlim=80
    ):

        '''
        Purpose
        -------
        Show whether sharp branch features are numerical artifacts or branch
        cuts associated with nontrivial root-sheet topology.

        Inputs
        ------
        tracked_roots: np.array, complex
            Continuity-labeled roots with shape (n_omega, n_kh, n_branch), in km^-1.

        diagnostics: dict
            Tracking and solver maps returned by kz_solve_grid(return_diagnostics=True).

        branch: int
            One-based branch number to inspect.

        dz: float or None
            Height separation (km). When supplied, include phase differences in degrees.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        value_cmap: str or matplotlib.colors.Colormap
            Colormap used for the selected branch-value panel.

        scalar_cmap: str or matplotlib.colors.Colormap
            Colormap used for scalar diagnostic maps.

        phase_vlim: float
            Absolute limit of the symmetric phase color scale (degrees).

        Outputs
        -------
        fig, axes: tuple
            Matplotlib figure and (3, 3) axes array of the selected branch,
            jump/separation maps, and topology/uncertainty masks; no show() call.

        Notes
        -----
        `branch` is one-based. When `dz` is supplied the branch panel shows
        Delta phi in degrees; otherwise it shows Re(k_z). The branch-cut panel
        marks edges where the current global labels cannot follow the local raw
        root continuation. The monodromy panel marks plaquettes where the
        sampled Hungarian root permutations around a closed loop are
        non-identity. This discrete diagnostic does not by itself establish
        an analytic root-sheet branch point or coalescence.
        '''

        # Load plotting support and validate the requested one-based branch number.
        import matplotlib.pyplot as plt

        tracked_roots = np.asarray(tracked_roots, dtype=np.complex128)
        branch_index = int(branch) - 1
        if branch_index < 0 or branch_index >= tracked_roots.shape[2]:
            raise ValueError("branch must be a valid one-based branch number.")

        # Display real wavenumber or height-scaled phase for the selected branch.
        if dz is None:
            branch_values = np.real(tracked_roots[:, :, branch_index])
            branch_title = rf'branch {branch} Re($k_z$)'
            branch_label = r'$k_z\,[\mathrm{km}^{-1}]$'
            finite = branch_values[np.isfinite(branch_values)]
            if finite.size > 0:
                branch_vmin = np.min(finite)
                branch_vmax = np.max(finite)
            else:
                branch_vmin = None
                branch_vmax = None
        else:
            branch_values = np.real(tracked_roots[:, :, branch_index])*dz*180*pi**(-1)
            branch_title = rf'branch {branch} $\Delta\phi$'
            branch_label = r'$\Delta\phi\,[\mathrm{deg}]$'
            branch_vmin = -phase_vlim
            branch_vmax = phase_vlim

        def log10_positive(values):
            '''
            Purpose
            -------
            Apply log10 only to positive entries and leave other cells as NaN.

            Inputs
            ------
            values: np.array, float
                Values to transform; entries that are not positive are left as NaN.

            Outputs
            -------
            logged: np.array, float
                Base-10 logarithm with the input shape; zero, negative, and NaN
                entries remain NaN.
            '''

            # Transform positive values only, leaving undefined logarithms as NaN.
            values = np.asarray(values)
            logged = np.full(values.shape, np.nan, dtype=float)
            valid = values > 0
            logged[valid] = np.log10(values[valid])
            return logged

        # Prepare a nine-panel layout and independent colormaps for missing values.
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), squeeze=False, constrained_layout=True)
        branch_cmap = plt.get_cmap(value_cmap).copy()
        scalar_cmap_plot = plt.get_cmap(scalar_cmap).copy()
        branch_cmap.set_bad('0.65')
        scalar_cmap_plot.set_bad('0.65')

        # Compare branch structure with jumps, cuts, monodromy, and numerical flags.
        panels = [
            (axes[0, 0], branch_values, branch_title, branch_cmap, branch_vmin, branch_vmax, branch_label),
            (axes[0, 1], log10_positive(diagnostics.get('tracked_branch_jump', self.kz_smoothness_map(tracked_roots))), 'log10 tracked jump', scalar_cmap_plot, None, None, None),
            (axes[0, 2], log10_positive(diagnostics.get('root_set_jump', self.kz_smoothness_map(tracked_roots))), 'log10 root-set jump floor', scalar_cmap_plot, None, None, None),
            (axes[1, 0], diagnostics.get('branch_cut_mask', np.zeros(tracked_roots.shape[:2])).astype(float), 'label branch-cut mask', 'gray_r', 0, 1, None),
            (axes[1, 1], diagnostics.get('topological_branch_cut_mask', np.zeros(tracked_roots.shape[:2])).astype(float), 'topological jump/cut mask', 'gray_r', 0, 1, None),
            (axes[1, 2], diagnostics.get('monodromy_mask', np.zeros(tracked_roots.shape[:2])).astype(float), 'root-sheet monodromy mask', 'gray_r', 0, 1, None),
            (axes[2, 0], log10_positive(diagnostics.get('branch_min_separation', diagnostics.get('min_root_separation', np.nan))), 'log10 nearest branch separation', scalar_cmap_plot, None, None, None),
            (axes[2, 1], diagnostics.get('branch_close_mask', np.zeros(tracked_roots.shape[:2])).astype(float), 'close branch-pair zone', 'gray_r', 0, 1, None),
            (axes[2, 2], diagnostics.get('plot_suspicious_mask', np.zeros(tracked_roots.shape[:2])).astype(float), 'unresolved numerical artifact mask', 'gray_r', 0, 1, None)
        ]

        # Draw each map and label its coordinates and color scale.
        for ax, values, title, cmap, vmin, vmax, colorbar_label in panels:
            mesh = ax.pcolormesh(
                kh_plot,
                omega_plot,
                values,
                shading='auto',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True
            )
            ax.set_title(title)
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
            ax.set_ylabel(r'$\nu\,[\mathrm{mHz}]$')
            colorbar = fig.colorbar(mesh, ax=ax)
            if colorbar_label is not None:
                colorbar.set_label(colorbar_label)

        return fig, axes

    def plot_kz_line_cut(self, roots, axis='omega', index=None, component='real', branches=None, kh_scale=1e3, omega_scale=(2*pi)**(-1)*1e3):

        '''
        Purpose
        -------
        Plot one-dimensional cuts through tracked or raw kz grids.

        Inputs
        ------
        roots: np.array, complex
            Vertical wavenumbers with shape (n_omega, n_kh, n_branch), in km^-1.

        axis: str
            Cut direction: 'omega', 'nu', or 'frequency' fixes kh; 'kh' or
            'wavenumber' fixes omega.

        index: int or None
            Index in the fixed grid dimension; defaults to its middle sample.

        component: str
            Root component: 'real', 'imag', 'abs', or 'phase'; aliases accepted by
            _grid_component() also apply. Phase is in radians; other components are
            in km^-1.

        branches: iterable or None
            Zero-based branch indices to display; None selects all branches.

        kh_scale: float
            Multiplier converting stored horizontal wavenumbers to plot coordinates;
            1e3 gives Mm^-1.

        omega_scale: float
            Multiplier converting stored angular frequencies to plot coordinates;
            1e3/(2*pi) gives mHz.

        Outputs
        -------
        fig, ax: tuple
            Matplotlib figure and axes containing the requested branch cuts; no
            show() call.
        '''

        # Load plotting support and extract the requested component and branches.
        import matplotlib.pyplot as plt

        roots = np.asarray(roots, dtype=np.complex128)
        values = self._grid_component(roots, component)
        if branches is None:
            branches = range(roots.shape[2])

        # Normalize the cut direction and allocate the line-plot axes.
        axis = axis.lower()
        fig, ax = plt.subplots(figsize=(7, 4))
        # Plot frequency variation at a fixed horizontal-wavenumber sample.
        if axis in ('omega', 'nu', 'frequency'):
            if index is None:
                index = len(self.kh)//2
            x = self.omega*omega_scale
            for branch in branches:
                ax.plot(x, values[:, index, branch], label=f'branch {branch + 1}')
            ax.set_xlabel(r'$\nu\,[\mathrm{mHz}]$')
            ax.set_title(rf'$k_h={self.kh[index]*kh_scale:.3g}\,\mathrm{{Mm}}^{{-1}}$')
        # Plot horizontal-wavenumber variation at a fixed frequency sample.
        elif axis in ('kh', 'wavenumber'):
            if index is None:
                index = len(self.omega)//2
            x = self.kh*kh_scale
            for branch in branches:
                ax.plot(x, values[index, :, branch], label=f'branch {branch + 1}')
            ax.set_xlabel(r'$k_h\,[\mathrm{Mm}^{-1}]$')
            ax.set_title(rf'$\nu={self.omega[index]*omega_scale:.3g}\,\mathrm{{mHz}}$')
        else:
            raise ValueError("axis must be 'omega' or 'kh'.")

        # Label the selected component and identify each plotted branch.
        ax.set_ylabel(f'kz {component}')
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def phase_speed(self, omega, kz):

        '''
        Purpose
        -------
        Calculate the phase speed of the wave.

        Inputs
        ------
        omega: np.array or scalar, float
            Angular frequency (rad/s), broadcastable with kz.

        kz: np.array or scalar, real or complex
            Vertical wavenumber (km^-1), broadcastable with omega.

        Outputs
        -------
        v_phase: np.array or scalar, real or complex
            Vertical phase speed omega/kz (km/s), with NumPy broadcasting.
            Floating or complex NumPy inputs retain inf/NaN results at
            singularities; a zero Python scalar kz raises ZeroDivisionError.
            Integer arrays reject the negative power used for the reciprocal.

        Author(s)
        ---------
        Julio M. Morales, June 03, 2024
        '''

        # Calculate the phase speed
        with np.errstate(divide='ignore', invalid='ignore'):
            v_phase = omega*kz**(-1) # km/s

        # Return the broadcast phase speed, including any complex component.
        return v_phase
    
    def phase_difference(self, omega, v_phase, dz):

        '''
        Purpose
        -------
        Calculate the phase difference between two points.

        Inputs
        ------
        omega: np.array or scalar, float
            Angular frequency (rad/s), broadcastable with v_phase and dz.

        v_phase: np.array or scalar, real or complex
            Vertical phase speed (km/s), broadcastable with omega and dz.

        dz: np.array or scalar, float
            Height separation (km), broadcastable with omega and v_phase.

        Outputs
        -------
        delta_phi: np.array or scalar, real or complex
            Unwrapped phase difference omega*dz/v_phase (rad), with NumPy
            broadcasting. Floating or complex NumPy inputs retain inf/NaN
            results at singularities; a zero Python scalar v_phase raises
            ZeroDivisionError. Integer arrays reject the reciprocal power.

        Author(s)
        ---------
        Julio M. Morales, June 03, 2024
        '''

        # Calculate the phase difference
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_phi = omega*dz*v_phase**(-1) # rad

        # Return the unwrapped phase in radians, including any complex component.
        return delta_phi

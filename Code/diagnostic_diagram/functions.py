# Import the relevant libraries
import heapq
import numpy as np
from numpy import sin, cos, sqrt, pi
from scipy import optimize

class DiagnosticDiagram:

    def __init__(self, kh_grid, omega_grid, params):

        # Assign attributes and global variables
        self.kh = kh_grid[0]  # km^-1
        self.omega  = omega_grid[:, 0]  # rad/s
        self.params = params
        self.model  = self.params['model']
        self.cs = self.params['cs']  # km/s
        self.g  = self.params['g']   # km/s^2

        # Initialize parameters according to the model
        if self.model == 'sf1966':
            self.N, self.wac = self.params['N'], self.params['wac']
        elif self.model == 'mt1981':
            self.H, self.N = self.params['H'], self.params['N']
        elif self.model == 'mt1982':
            self.gamma, self.N, self.H, self.tau = self.params['gamma'], self.params['N'], self.params['H'], self.params['tau']
        elif self.model == 'bunte1993':
            self.a, self.epsilon, self.gamma, self.H, self.tau, self.wac = self.params['a'], self.params['epsilon'], self.params['gamma'], self.params['H'], self.params['tau'], self.params['wac']
        elif self.model == 'nc2009':
            self.a, self.ax, self.ay, self.az, self.theta, self.phi, self.N, self.wac = self.params['a'], self.params['ax'], self.params['ay'], self.params['az'], self.params['theta'], self.params['phi'], self.params['N'], self.params['wac']
        else:
            raise ValueError("Model not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009'.")

    @staticmethod
    def _finite_complex_mask(values):

        values = np.asarray(values)
        return np.isfinite(values.real) & np.isfinite(values.imag)

    @staticmethod
    def _sorted_finite_indices(values):

        values = np.asarray(values)
        finite = DiagnosticDiagram._finite_complex_mask(values)
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size == 0:
            return finite_indices

        finite_values = values[finite_indices]
        sort_order = np.lexsort((finite_values.imag, finite_values.real))
        return finite_indices[sort_order]

    @staticmethod
    def _min_pairwise_distance(values):

        values = np.asarray(values)
        finite_values = values[DiagnosticDiagram._finite_complex_mask(values)]
        if finite_values.size < 2:
            return np.nan

        distances = np.abs(finite_values[:, None] - finite_values[None, :])
        distances[np.diag_indices_from(distances)] = np.inf
        return np.min(distances)

    @staticmethod
    def _trim_polynomial_coefficients(coefficients, rtol=1e-12, atol=0.0):

        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.size == 0:
            return coefficients
        if not np.all(DiagnosticDiagram._finite_complex_mask(coefficients)):
            return None

        scale = np.max(np.abs(coefficients))
        tolerance = max(atol, rtol*scale)
        first = 0
        while first < coefficients.size - 1 and np.abs(coefficients[first]) <= tolerance:
            first += 1

        trimmed = coefficients[first:]
        if np.all(np.abs(trimmed) <= tolerance):
            return np.asarray([0.0 + 0.0j])

        return trimmed

    @staticmethod
    def _grid_component(values, component):

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

        valid = np.isfinite(values)
        current_valid = np.isfinite(target)
        target[valid & current_valid] = np.maximum(target[valid & current_valid], values[valid & current_valid])
        target[valid & ~current_valid] = values[valid & ~current_valid]

    def fmode_dispersion(self):

        '''
        Purpose
        -------
        Calculate the theoretical frequency of the f-mode as a function of wavenumber (k).

        Inputs
        ------
        None

        Outputs
        -------
        omega: np.array, float:
            Theoretical angular frequency for input wavenumber (rad/s).

        Author(s)
        ---------
        Julio M. Morales, August 30, 2022
        '''

        # Calculate omega
        omega = sqrt(self.g*np.array(self.kh)) # rad/s

        return omega
    
    def omega_poly(self, omega, kh):

        '''
        Purpose
        -------
        Calculate the polynomial in omega.

        Inputs
        ------
        omega: float
            Angular frequency (rad/s)

        kh: float
            Horizontal wavenumber (km^-1)

        Outputs
        -------
        poly: np.array, float
            Polynomial in omega for input parameters.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # Determine the model
        if (self.model == 'sf1966'):
            c4 = self.cs**(-2)
            c3 = 0
            c2 = -(kh**2 + self.wac**2*(self.cs)**(-2))
            c1 = 0
            c0 = self.N**2*kh**2
            poly = c4*omega**4 + c3*omega**3 + c2*omega**2 + c1*omega + c0
        elif (self.model == 'mt1981'):
            c4 = self.cs**(-2)
            c3 = 0
            c2 = -(kh**2 + (4*self.H**2)**(-1))
            c1 = 0
            c0 = self.N**2*kh**2
            poly = c4*omega**4 + c3*omega**3 + c2*omega**2 + c1*omega + c0
        elif (self.model == 'bunte1993') or (self.model == 'mt1982'):
            if (self.model == 'mt1982'):
                self.a = 0
            c8 = 4*self.H**2*self.a**2*self.gamma**2*self.tau**2 + 4*self.H**2*self.cs**2*self.gamma**2*self.tau**2
            c7 = 0
            c6 = -4*self.H**2*self.a**4*self.gamma**2*kh**2*self.tau**2 - 12*self.H**2*self.a**2*self.cs**2*self.gamma**2*kh**2*self.tau**2 + 4*self.H**2*self.a**2*self.gamma**2 - 4*self.H**2*self.cs**4*self.gamma**2*kh**2*self.tau**2 + 4*self.H**2*self.cs**2*self.gamma - self.a**4*self.gamma**2*self.tau**2 - 2*self.a**2*self.cs**2*self.gamma**2*self.tau**2 - self.cs**4*self.gamma**2*self.tau**2
            c5 = 0
            c4 = 8*self.H**2*self.a**4*self.cs**2*self.gamma**2*kh**4*self.tau**2 - 4*self.H**2*self.a**4*self.gamma**2*kh**2 + 8*self.H**2*self.a**2*self.cs**4*self.gamma**2*kh**4*self.tau**2 - 12*self.H**2*self.a**2*self.cs**2*self.gamma*kh**2 - 4*self.H**2*self.a**2*self.g**2*self.gamma**2*kh**2*self.tau**2 - 4*self.H**2*self.cs**4*kh**2 - 4*self.H**2*self.cs**2*self.g**2*self.gamma**2*kh**2*self.tau**2 + 4*self.H*self.a**2*self.cs**2*self.g*self.gamma**2*kh**2*self.tau**2 + 4*self.H*self.cs**4*self.g*self.gamma**2*kh**2*self.tau**2 + 2*self.a**4*self.cs**2*self.gamma**2*kh**2*self.tau**2 - self.a**4*self.gamma**2 + 2*self.a**2*self.cs**4*self.gamma**2*kh**2*self.tau**2 - 2*self.a**2*self.cs**2*self.gamma - self.cs**4
            c3 = 0
            c2 = -4*self.H**2*self.a**4*self.cs**4*self.gamma**2*kh**6*self.tau**2 + 8*self.H**2*self.a**4*self.cs**2*self.gamma*kh**4 + 8*self.H**2*self.a**2*self.cs**4*kh**4 + 4*self.H**2*self.a**2*self.cs**2*self.g**2*self.gamma**2*kh**4*self.tau**2 - 4*self.H**2*self.a**2*self.g**2*self.gamma**2*kh**2 - 4*self.H**2*self.cs**2*self.g**2*self.gamma*kh**2 - 4*self.H*self.a**2*self.cs**4*self.g*self.gamma**2*kh**4*self.tau**2 + 4*self.H*self.a**2*self.cs**2*self.g*self.gamma*kh**2 + 4*self.H*self.cs**4*self.g*kh**2 - self.a**4*self.cs**4*self.gamma**2*kh**4*self.tau**2 + 2*self.a**4*self.cs**2*self.gamma*kh**2 + 2*self.a**2*self.cs**4*kh**2
            c1 = 0
            c0 = -4*self.H**2*self.a**4*self.cs**4*kh**6 + 4*self.H**2*self.a**2*self.cs**2*self.g**2*self.gamma*kh**4 - 4*self.H*self.a**2*self.cs**4*self.g*kh**4 - self.a**4*self.cs**4*kh**4
            poly = c8*omega**8 + c7*omega**7 + c6*omega**6 + c5*omega**5 + c4*omega**4 + c3*omega**3 + c2*omega**2 + c1*omega + c0
        elif (self.model == 'nc2009'):
            c6 = 1
            c5 = 0
            c4 = -self.ax**2*kh**2 - self.a**2*kh**2 - self.cs**2*kh**2 - self.wac**2
            c3 = 0
            c2 = self.N**2*self.cs**2*kh**2 + self.ax**4*kh**4 + 2*self.ax**2*self.cs**2*kh**4 + self.ay**2*kh**2*self.wac**2 + self.ax**2*kh**2*self.wac**2 + self.az**2*kh**2*self.wac**2
            c1 = 0
            c0 = -self.N**2*self.ax**2*self.cs**2*kh**4 - self.ax**4*self.cs**2*kh**6 - self.ax**2*self.az**2*kh**4*self.wac**2
            poly = c6*omega**6 + c5*omega**5 + c4*omega**4 + c3*omega**3 + c2*omega**2 + c1*omega + c0
        else:
            raise ValueError("Model for kz not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009'.")

        return poly
        
    def kz_coefficients(self, omega, kh):

        '''
        Purpose
        -------
        Calculate the polynomial coefficients in descending powers of kz.

        Inputs
        ------
        omega: float
            Angular frequency (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        coefficients: np.array, complex
            Polynomial coefficients ordered for np.roots / np.polyval.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # Determine the model
        if (self.model == 'sf1966'):
            c2 = 1
            c1 = 0
            c0 = kh**2*(omega**2 - self.N**2)*omega**(-2) - (omega**2 - self.wac**2)*self.cs**(-2)
            coefficients = [c2, c1, c0]
        elif (self.model == 'mt1981'):
            c2 = 1
            c1 = 0
            c0 = kh**2 + (4*self.H**2)**(-1) - self.N**2*kh**2*omega**(-2) - omega**2*self.cs**(-2)
            coefficients = [c2, c1, c0]
        elif (self.model == 'mt1982'):
            c2 = 1
            c1 = 0
            c0 = -0.5*(self.N**2*kh**2*omega**(-2) - kh**2 + (omega**2*self.tau**2*(-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))**2*(omega**2*self.tau**2 + 1)**(-2) + (self.N**2*kh**2*omega**(-2) - kh**2 + (-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))**2)**(0.5) + (-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))
            coefficients = [c2, c1, c0]
        elif (self.model == 'bunte1993'):
            a0 = self.a#self.B*(4*pi*self.rho)**(-0.5)
            gamma_hat = (1 - 1j*omega*self.tau*self.gamma)*(1 - 1j*omega*self.tau)**(-1)
            c2_hat = self.cs**2*gamma_hat
            n2_hat = (gamma_hat**2 - 1)*self.g*(gamma_hat*self.H)**(-1)
            omega2_hat = c2_hat*(2*self.H)**(-1)
            p1 = (self.cs**2 + a0**2)*omega**2 - self.cs**2*a0**2*kh**2
            p_gamma = (self.cs**2 + self.gamma*a0**2)*omega**2 - self.cs**2*a0**2*kh**2
            g1 = omega**4 + self.g**2*(self.cs**2*(self.g*self.H)**(-1) - 1)*kh**2
            g_gamma = self.gamma*omega**4 + self.gamma*self.g**2*(self.cs**2*(self.gamma*self.g*self.H)**(-1) - 1)*kh**2
            if (self.a == 0):
                c2 = 1
                c1 = 0
                c0 = -(omega**2 - omega2_hat)*(c2_hat)**(-1) - kh**2*(n2_hat*omega**(-2) - 1)
            else:
                c2 = 1
                c1 = 0
                c0 = (4*self.H**2)**(-1) + kh**2 - (p_gamma*g_gamma + (omega*self.tau*self.gamma)**2*p1*g1) * (p_gamma**2 + (omega*self.tau*self.gamma)**2*p1**2)**(-1)
            coefficients = [c2, c1, c0]
        elif (self.model == 'nc2009'):
            c6 = -self.a**4*self.cs**2*cos(self.theta)**4
            c5 = -4*kh*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta)**3
            c4 = -6*kh**2*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos(self.theta)**2 - kh**2*self.a**4*self.cs**2*cos(self.theta)**4 + omega**2*self.a**4*cos(self.theta)**2 + 2*omega**2*self.a**2*self.cs**2*cos(self.theta)**2 - self.a**2*self.az**2*self.wac**2*cos(self.theta)**2
            c3 = -4*kh**3*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos(self.theta) - 4*kh**3*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta)**3 + 2*kh*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos(self.theta) + 4*kh*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta)
            c2 = -kh**4*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 - 6*kh**4*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos(self.theta)**2 + kh**2*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.a**4*cos(self.theta)**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*cos(self.theta)**2 - kh**2*self.N**2*self.a**2*self.cs**2*cos(self.theta)**2 - kh**2*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*self.a**2*self.az**2*self.wac**2*cos(self.theta)**2 - omega**4*self.a**2*cos(self.theta)**2 - omega**4*self.a**2 - omega**4*self.cs**2 + omega**2*self.a**2*self.wac**2*cos(self.theta)**2 + omega**2*self.az**2*self.wac**2
            c1 = -4*kh**5*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos(self.theta) + 2*kh**3*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos(self.theta) + 4*kh**3*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh**3*self.N**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh**3*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh*omega**4*self.a**2*sin(self.theta)*cos(self.phi)*cos(self.theta) + 2*kh*omega**2*self.a**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta)
            c0 = -kh**6*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 + kh**4*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**4*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.N**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2 - kh**2*omega**4*self.cs**2 + kh**2*omega**2*self.N**2*self.cs**2 + kh**2*omega**2*self.a**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.ay**2*self.wac**2 + kh**2*omega**2*self.az**2*self.wac**2 + omega**6 - omega**4*self.wac**2
            coefficients = [c6, c5, c4, c3, c2, c1, c0]
        else:
            raise ValueError("Model for kz not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009'.")

        return np.asarray(coefficients, dtype=np.complex128)

    def kz_poly(self, kz, omega, kh):

        '''
        Purpose
        -------
        Calculate the polynomial in kz.

        Inputs
        ------
        kz: float
            Vertical wavenumber (km^-1).

        omega: float
            Angular frequency (rad/s).

        kh: float
            Horizontal wavenumber (km^-1).

        Outputs
        -------
        poly: np.array, complex
            Value of the polynomial in kz for input parameters.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        coefficients = self.kz_coefficients(omega, kh)
        poly = np.polyval(coefficients, kz)

        return poly
    
    def omega_solve(self, kh):

        '''
        Purpose
        -------
        Calculate the boundaries of the dispersion relation.

        Inputs
        ------
        kh: float
            i-th horizontal wavenumber (km^-1).

        Outputs
        -------
        omega_bounds: np.array, float:
            Boundaries of the dispersion relation for input parameters (rad/s).

        Author(s)
        ---------
        Julio M. Morales, July 30th, 2025
        '''

        # Define a fine grid for omega
        fine_grid = np.linspace(0, max(self.omega), 5_000)  # rad/s

        # List to store roots
        roots = np.zeros(self.params['omega_order'])

        # Compute the polynomial being solved on a fine grid and find sign changes
        poly = self.omega_poly(fine_grid, kh)
        sign_changes = np.nonzero(np.diff(np.sign(poly)))[0]
        
        # Loop through the sign changes to find roots by brentq method
        for ind, idx in enumerate(sign_changes):
            A0, B0 = fine_grid[idx], fine_grid[idx + 1]
            try:
                root = optimize.brentq(self.omega_poly, A0, B0, args = (kh))
                roots[ind] = root
            except ValueError:
                root = np.nan
                roots[ind] = np.nan

        return np.nan_to_num(roots, nan=0.0)
    
    def _solve_kz_roots_at_point(self, omega, kh, coefficient_rtol=1e-12):

        root_count = int(self.params['kz_order'])
        roots = np.full(root_count, np.nan + 0.0j, dtype=np.complex128)

        try:
            coefficients = self.kz_coefficients(omega, kh)
            coefficients = self._trim_polynomial_coefficients(coefficients, rtol=coefficient_rtol)
        except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
            return roots

        if coefficients is None or coefficients.size <= 1:
            return roots

        try:
            coefficient_scale = np.max(np.abs(coefficients))
            if coefficient_scale > 0:
                coefficients = coefficients / coefficient_scale
            point_roots = np.roots(coefficients)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return roots

        point_roots = point_roots[self._finite_complex_mask(point_roots)]
        roots[:min(root_count, point_roots.size)] = point_roots[:root_count]
        return roots

    def _kz_solve_column_raw(self, kh):

        '''
        Purpose
        -------
        Solve raw, untracked kz roots for one kh column.

        Inputs
        ------
        kh: float
            i-th horizontal wavenumber (km^-1).

        Outputs
        -------
        roots: np.array, complex
            Raw roots of the kz polynomial as a function of angular frequency.
            These roots are only used internally before branch tracking.

        Author(s)
        ---------
        Julio M. Morales, August 07th, 2025
        '''

        roots = np.full((len(self.omega), self.params['kz_order']), np.nan + 0.0j, dtype=np.complex128)

        # Loop through frequency
        for j, omega_j in enumerate(self.omega):
            roots[j, :] = self._solve_kz_roots_at_point(omega_j, kh)

        return roots

    def kz_solve(self, *args, return_diagnostics=False, **tracking_kwargs):

        '''
        Purpose
        -------
        Solve for continuity-tracked vertical-wavenumber branches over the full
        omega-kh grid.

        Notes
        -----
        The previous public use of kz_solve(kh) returned one untracked column
        and encouraged downstream stacking/manual branch fixes. That path is
        intentionally disabled so kz_solve() and kz_solve_grid() both use the
        same global branch-tracking workflow.
        '''

        if len(args) > 0:
            raise TypeError("kz_solve() now solves the full grid with branch tracking. Use kz_solve_grid(return_diagnostics=True) instead of passing one kh value.")

        return self.kz_solve_grid(return_diagnostics=return_diagnostics, **tracking_kwargs)

    def _kz_solve_raw_grid(self):

        '''
        Purpose
        -------
        Solve the kz polynomial at every omega-kh grid point before branch
        tracking. Raw roots are retained only for diagnostics.

        Outputs
        -------
        roots: np.array, complex
            Raw solver roots with shape (omega, kh, kz_order).
        '''

        roots = np.full((len(self.omega), len(self.kh), self.params['kz_order']), np.nan + 0.0j, dtype=np.complex128)
        for i, kh_i in enumerate(self.kh):
            roots[:, i, :] = self._kz_solve_column_raw(kh_i)

        return roots

    def _root_geometry_maps(self, raw_roots, degenerate_rtol, degenerate_atol):

        n_omega, n_kh, _ = raw_roots.shape
        finite_count = np.zeros((n_omega, n_kh), dtype=int)
        min_separation = np.full((n_omega, n_kh), np.nan)
        local_scale = np.full((n_omega, n_kh), degenerate_atol)

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

        threshold = degenerate_atol + degenerate_rtol*local_scale
        degenerate_mask = (finite_count > 1) & np.isfinite(min_separation) & (min_separation <= threshold)

        return finite_count, min_separation, local_scale, degenerate_mask

    def _select_tracking_seed(self, raw_roots, assigned, finite_count, min_separation):

        available = (~assigned) & (finite_count > 0)
        if not np.any(available):
            return None

        max_count = np.max(finite_count[available])
        candidates = np.argwhere(available & (finite_count == max_count))
        center = 0.5*np.asarray(available.shape)

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

        values = np.full(branch_count, np.nan + 0.0j, dtype=np.complex128)
        sorted_indices = self._sorted_finite_indices(roots)
        for branch, root_index in enumerate(sorted_indices[:branch_count]):
            values[branch] = roots[root_index]

        return values

    def _match_root_sets(self, source_roots, target_roots):

        branch_count = source_roots.size
        permutation = np.full(branch_count, -1, dtype=int)
        stats = {
            'matched_count': 0,
            'assignment_cost': np.nan,
            'assignment_margin': np.nan,
            'max_assignment_cost': np.nan
        }

        source_indices = self._sorted_finite_indices(source_roots)
        target_indices = self._sorted_finite_indices(target_roots)
        if source_indices.size == 0 or target_indices.size == 0:
            return permutation, stats

        source_values = source_roots[source_indices]
        target_values = target_roots[target_indices]
        cost = np.abs(source_values[:, None] - target_values[None, :])

        scale_values = np.concatenate((source_values, target_values))
        scale_values = scale_values[self._finite_complex_mask(scale_values)]
        cost_scale = max(np.median(np.abs(scale_values)), self._min_pairwise_distance(target_values), 1e-30)
        tie_break = np.finfo(float).eps*cost_scale*(
            source_indices[:, None]*(branch_count + 1) + target_indices[None, :]
        )
        rows, cols = optimize.linear_sum_assignment(cost + tie_break)

        for row, col in zip(rows, cols):
            permutation[source_indices[row]] = target_indices[col]

        matched_costs = cost[rows, cols]
        if matched_costs.size > 0:
            stats['matched_count'] = int(matched_costs.size)
            stats['assignment_cost'] = float(np.mean(matched_costs))
            stats['max_assignment_cost'] = float(np.max(matched_costs))

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

        inverse = np.full_like(permutation, -1)
        for source_index, target_index in enumerate(permutation):
            if target_index >= 0:
                inverse[target_index] = source_index

        return inverse

    def _edge_confidence(self, cost, margin):

        if not np.isfinite(cost):
            return 0.0
        if not np.isfinite(margin):
            margin = 0.0

        margin = max(0.0, margin)
        return margin / (margin + cost + 1e-30)

    def _edge_match_maps(self, raw_roots):

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

        for i in range(n_omega):
            for j in range(n_kh - 1):
                permutation, stats = self._match_root_sets(raw_roots[i, j, :], raw_roots[i, j + 1, :])
                right_perm[i, j, :] = permutation
                right_cost[i, j] = stats['assignment_cost']
                right_max_cost[i, j] = stats['max_assignment_cost']
                right_margin[i, j] = stats['assignment_margin']
                right_matched[i, j] = stats['matched_count']

        for i in range(n_omega - 1):
            for j in range(n_kh):
                permutation, stats = self._match_root_sets(raw_roots[i, j, :], raw_roots[i + 1, j, :])
                down_perm[i, j, :] = permutation
                down_cost[i, j] = stats['assignment_cost']
                down_max_cost[i, j] = stats['max_assignment_cost']
                down_margin[i, j] = stats['assignment_margin']
                down_matched[i, j] = stats['matched_count']

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

        pixel_map = np.full(shape, np.nan)
        if right_values.size > 0:
            self._update_max_map(pixel_map[:, :-1], right_values)
            self._update_max_map(pixel_map[:, 1:], right_values)
        if down_values.size > 0:
            self._update_max_map(pixel_map[:-1, :], down_values)
            self._update_max_map(pixel_map[1:, :], down_values)

        return pixel_map

    def _edge_min_to_pixel_map(self, right_values, down_values, shape):

        pixel_map = np.full(shape, np.nan)

        def update_min(target, values):
            valid = np.isfinite(values)
            current_valid = np.isfinite(target)
            target[valid & current_valid] = np.minimum(target[valid & current_valid], values[valid & current_valid])
            target[valid & ~current_valid] = values[valid & ~current_valid]

        if right_values.size > 0:
            update_min(pixel_map[:, :-1], right_values)
            update_min(pixel_map[:, 1:], right_values)
        if down_values.size > 0:
            update_min(pixel_map[:-1, :], down_values)
            update_min(pixel_map[1:, :], down_values)

        return pixel_map

    def _values_from_local_indices(self, raw_roots, local_index_by_branch):

        values = np.full(raw_roots.shape, np.nan + 0.0j, dtype=np.complex128)
        for i, j in np.ndindex(raw_roots.shape[:2]):
            for branch, root_index in enumerate(local_index_by_branch[i, j, :]):
                if root_index >= 0:
                    values[i, j, branch] = raw_roots[i, j, root_index]

        return values

    class _ParityDisjointSet:

        def __init__(self, size):

            self.parent = np.arange(size, dtype=int)
            self.rank = np.zeros(size, dtype=int)
            self.parity_to_parent = np.zeros(size, dtype=int)

        def find(self, node):

            parent = self.parent[node]
            if parent != node:
                root, parity_to_root = self.find(parent)
                self.parity_to_parent[node] ^= parity_to_root
                self.parent[node] = root

            return self.parent[node], self.parity_to_parent[node]

        def union(self, node_a, node_b, parity):

            root_a, parity_a = self.find(node_a)
            root_b, parity_b = self.find(node_b)

            if root_a == root_b:
                return (parity_a ^ parity_b) == parity

            if self.rank[root_a] < self.rank[root_b]:
                root_a, root_b = root_b, root_a
                parity_a, parity_b = parity_b, parity_a

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

        n_omega, n_kh, _ = branch_values.shape
        constraints = []

        def append_constraints(source_values, target_values, source_nodes, target_nodes):
            valid = (
                self._finite_complex_mask(source_values[:, branch_a])
                & self._finite_complex_mask(source_values[:, branch_b])
                & self._finite_complex_mask(target_values[:, branch_a])
                & self._finite_complex_mask(target_values[:, branch_b])
            )
            if not np.any(valid):
                return

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
            min_cost = np.minimum(same_cost, swap_cost)
            max_cost = np.maximum(same_cost, swap_cost)
            separated = max_cost > ratio_threshold*np.maximum(min_cost, 1e-300)
            use = valid & separated & np.isfinite(min_cost) & np.isfinite(max_cost)
            if not np.any(use):
                return

            gain = max_cost[use] - min_cost[use]
            confidence = gain / np.maximum(max_cost[use] + min_cost[use], 1e-300)
            parity = (swap_cost[use] < same_cost[use]).astype(int)

            for item in zip(confidence, gain, source_nodes[use], target_nodes[use], parity):
                constraints.append(item)

        if n_kh > 1:
            source = branch_values[:, :-1, :].reshape(-1, branch_values.shape[2])
            target = branch_values[:, 1:, :].reshape(-1, branch_values.shape[2])
            source_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:, :-1].ravel()
            target_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:, 1:].ravel()
            append_constraints(source, target, source_nodes, target_nodes)

        if n_omega > 1:
            source = branch_values[:-1, :, :].reshape(-1, branch_values.shape[2])
            target = branch_values[1:, :, :].reshape(-1, branch_values.shape[2])
            source_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[:-1, :].ravel()
            target_nodes = np.arange(n_omega*n_kh, dtype=int).reshape(n_omega, n_kh)[1:, :].ravel()
            append_constraints(source, target, source_nodes, target_nodes)

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

        n_omega, n_kh, _ = branch_values.shape
        constraints = self._pair_swap_constraints(
            branch_values,
            branch_a,
            branch_b,
            ratio_threshold,
            complex_weight=complex_weight,
            real_weight=real_weight
        )
        disjoint_set = self._ParityDisjointSet(n_omega*n_kh)
        accepted = 0
        rejected = 0
        accepted_flips = 0

        for _, _, source_node, target_node, parity in constraints:
            if disjoint_set.union(int(source_node), int(target_node), int(parity)):
                accepted += 1
                accepted_flips += int(parity)
            else:
                rejected += 1

        parity = np.zeros(n_omega*n_kh, dtype=bool)
        for node in range(n_omega*n_kh):
            _, node_parity = disjoint_set.find(node)
            parity[node] = bool(node_parity)

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

        corrected_indices = local_index_by_branch.copy()
        swap_mask = np.asarray(parity, dtype=bool)
        if not np.any(swap_mask):
            return corrected_indices, 0

        valid_swap = (
            swap_mask
            & (corrected_indices[:, :, branch_a] >= 0)
            & (corrected_indices[:, :, branch_b] >= 0)
        )
        if not np.any(valid_swap):
            return corrected_indices, 0

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

        values = np.asarray(branch_values[:, :, [branch_a, branch_b]], dtype=np.complex128)

        def diff_energy(array, order):
            total = 0.0
            count = 0
            for axis in (0, 1):
                if array.shape[axis] <= order:
                    continue
                diff = np.diff(array, n=order, axis=axis)
                valid = np.isfinite(diff)
                if not np.any(valid):
                    continue
                total += float(np.sum(np.abs(diff[valid])**2))
                count += int(np.count_nonzero(valid))
            if count == 0:
                return np.nan
            return total / count

        complex_jump = diff_energy(values, 1)
        real_jump = diff_energy(values.real, 1)
        imag_jump = diff_energy(values.imag, 1)
        real_curvature = diff_energy(values.real, 2)
        imag_curvature = diff_energy(values.imag, 2)

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

        corrected_indices = local_index_by_branch.copy()
        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
        branch_count = raw_roots.shape[2]
        pair_stats = []
        total_changed_pixels = 0
        completed_iterations = 0

        for iteration in range(max_iterations):
            iteration_changed_pixels = 0
            for branch_a in range(branch_count):
                for branch_b in range(branch_a + 1, branch_count):
                    parity, stats = self._pair_swap_parity(
                        branch_values,
                        branch_a,
                        branch_b,
                        ratio_threshold,
                        complex_weight=complex_weight,
                        real_weight=real_weight
                    )
                    if stats['accepted_flip_constraints'] == 0 or stats['swap_pixels'] == 0:
                        pair_stats.append({
                            **stats,
                            'iteration': int(iteration),
                            'changed_pixels': 0,
                            'accepted': False,
                            'rejected_reason': 'no coherent swap candidate'
                        })
                        continue

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

                    before_objective = self._pair_identity_objective(branch_values, branch_a, branch_b)
                    candidate_values = self._values_from_local_indices(raw_roots, updated_indices)
                    after_objective = self._pair_identity_objective(candidate_values, branch_a, branch_b)
                    improvement = before_objective['total'] - after_objective['total']
                    improvement_rtol = improvement / max(abs(before_objective['total']), 1e-300)
                    accepted = (
                        np.isfinite(improvement_rtol)
                        and improvement_rtol > objective_improvement_rtol
                    )

                    if accepted:
                        corrected_indices = updated_indices
                        branch_values = candidate_values
                        iteration_changed_pixels += changed_pixels
                        total_changed_pixels += changed_pixels

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

        branch_values = np.asarray(branch_values, dtype=np.complex128)
        n_omega, n_kh, branch_count = branch_values.shape
        min_separation = np.full((n_omega, n_kh), np.nan)
        nearest_branch_a = np.full((n_omega, n_kh), -1, dtype=int)
        nearest_branch_b = np.full((n_omega, n_kh), -1, dtype=int)

        abs_values = np.where(self._finite_complex_mask(branch_values), np.abs(branch_values), np.nan)
        local_scale = np.nanmedian(abs_values, axis=2)
        local_scale[~np.isfinite(local_scale)] = 1.0
        if degenerate_threshold is None:
            degenerate_threshold = 1e-10 + 1e-3*local_scale
        close_threshold = np.maximum(degenerate_threshold, close_atol + close_rtol*local_scale)

        for branch_a in range(branch_count):
            for branch_b in range(branch_a + 1, branch_count):
                separation = np.abs(branch_values[:, :, branch_a] - branch_values[:, :, branch_b])
                valid = np.isfinite(separation)
                update = valid & (~np.isfinite(min_separation) | (separation < min_separation))
                min_separation[update] = separation[update]
                nearest_branch_a[update] = branch_a
                nearest_branch_b[update] = branch_b

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

        branch_values = np.asarray(branch_values, dtype=np.complex128)
        n_omega, n_kh, branch_count = branch_values.shape
        gain_map = np.full((n_omega, n_kh), np.nan)

        if branch_a is None or branch_b is None:
            pairs = [
                (a, b)
                for a in range(branch_count)
                for b in range(a + 1, branch_count)
            ]
        else:
            pairs = [(branch_a, branch_b)]

        def edge_gain(source_values, target_values):
            valid = (
                self._finite_complex_mask(source_values[:, branch_a])
                & self._finite_complex_mask(source_values[:, branch_b])
                & self._finite_complex_mask(target_values[:, branch_a])
                & self._finite_complex_mask(target_values[:, branch_b])
            )
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
            gain = np.full(source_values.shape[0], np.nan)
            gain[valid] = same_cost[valid] - swap_cost[valid]
            return gain

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

        Notes
        -----
        For each adjacent grid edge, the raw root sets define a local analytic
        continuation through the Hungarian permutation. If the current global
        labels do not follow that local permutation, the edge lies on a label
        branch cut. Around a true root coalescence/exceptional point, the
        product of local permutations around a plaquette can be non-identity;
        no single global labeling can satisfy all such local continuations.
        These maps distinguish that topology from polynomial-solver failures.
        '''

        local_index_by_branch = np.asarray(local_index_by_branch, dtype=int)
        n_omega, n_kh, branch_count = local_index_by_branch.shape
        right_mismatch_count = np.zeros((n_omega, max(n_kh - 1, 0)), dtype=int)
        down_mismatch_count = np.zeros((max(n_omega - 1, 0), n_kh), dtype=int)

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

        branch_cut_strength = self._edge_max_to_pixel_map(
            np.where(right_mismatch_count > 0, right_mismatch_count, np.nan),
            np.where(down_mismatch_count > 0, down_mismatch_count, np.nan),
            (n_omega, n_kh)
        )
        branch_cut_mask = np.isfinite(branch_cut_strength)

        def invert_permutation(permutation):
            inverse = np.full_like(permutation, -1)
            for source_index, target_index in enumerate(permutation):
                if target_index >= 0:
                    inverse[target_index] = source_index
            return inverse

        def compose_permutations(first, second):
            composed = np.full_like(first, -1)
            for source_index, intermediate_index in enumerate(first):
                if intermediate_index >= 0 and second[intermediate_index] >= 0:
                    composed[source_index] = second[intermediate_index]
            return composed

        monodromy_cell_mask = np.zeros((max(n_omega - 1, 0), max(n_kh - 1, 0)), dtype=bool)
        monodromy_moved_count = np.zeros_like(monodromy_cell_mask, dtype=int)
        expected = np.arange(branch_count, dtype=int)
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
                valid = loop >= 0
                if np.all(valid):
                    moved = loop != expected
                    if np.any(moved):
                        monodromy_cell_mask[i, j] = True
                        monodromy_moved_count[i, j] = int(np.count_nonzero(moved))

        monodromy_mask = np.zeros((n_omega, n_kh), dtype=bool)
        if monodromy_cell_mask.size > 0:
            monodromy_mask[:-1, :-1] |= monodromy_cell_mask
            monodromy_mask[1:, :-1] |= monodromy_cell_mask
            monodromy_mask[:-1, 1:] |= monodromy_cell_mask
            monodromy_mask[1:, 1:] |= monodromy_cell_mask

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

        corrected_indices = local_index_by_branch.copy()
        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
        branch_count = raw_roots.shape[2]
        pair_stats = []
        total_changed_pixels = 0
        completed_iterations = 0

        for iteration in range(max_iterations):
            iteration_changed_pixels = 0
            for branch_a in range(branch_count):
                for branch_b in range(branch_a + 1, branch_count):
                    parity, stats = self._pair_swap_parity(
                        branch_values,
                        branch_a,
                        branch_b,
                        ratio_threshold
                    )
                    if stats['accepted_flip_constraints'] == 0 or stats['swap_pixels'] == 0:
                        pair_stats.append({**stats, 'iteration': int(iteration), 'changed_pixels': 0})
                        continue

                    updated_indices, changed_pixels = self._apply_pair_swap(
                        corrected_indices,
                        branch_a,
                        branch_b,
                        parity
                    )
                    if changed_pixels > 0:
                        corrected_indices = updated_indices
                        branch_values = self._values_from_local_indices(raw_roots, corrected_indices)
                        iteration_changed_pixels += changed_pixels
                        total_changed_pixels += changed_pixels

                    pair_stats.append({**stats, 'iteration': int(iteration), 'changed_pixels': int(changed_pixels)})

            completed_iterations = iteration + 1
            if iteration_changed_pixels == 0:
                break

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

        raw_roots = np.asarray(raw_roots, dtype=np.complex128)
        if raw_roots.ndim != 3:
            raise ValueError("raw_roots must have shape (omega, kh, branch).")

        n_omega, n_kh, branch_count = raw_roots.shape
        tracked_roots = np.full_like(raw_roots, np.nan + 0.0j)
        local_index_by_branch = np.full(raw_roots.shape, -1, dtype=int)
        assigned = np.zeros((n_omega, n_kh), dtype=bool)
        component_id = np.full((n_omega, n_kh), -1, dtype=int)
        parent_index = np.full((n_omega, n_kh, 2), -1, dtype=int)

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
            root_indices = self._sorted_finite_indices(raw_roots[i, j, :])
            for branch, root_index in enumerate(root_indices[:branch_count]):
                local_index_by_branch[i, j, branch] = root_index
                tracked_roots[i, j, branch] = raw_roots[i, j, root_index]
            assigned[i, j] = True
            component_id[i, j] = component

        def edge_from_to(ui, uj, vi, vj):
            if ui == vi and vj == uj + 1:
                return (
                    edge_maps['right_perm'][ui, uj, :],
                    edge_maps['right_cost'][ui, uj],
                    edge_maps['right_max_cost'][ui, uj],
                    edge_maps['right_margin'][ui, uj],
                    edge_maps['right_matched'][ui, uj]
                )
            if ui == vi and vj == uj - 1:
                return (
                    self._invert_permutation(edge_maps['right_perm'][vi, vj, :]),
                    edge_maps['right_cost'][vi, vj],
                    edge_maps['right_max_cost'][vi, vj],
                    edge_maps['right_margin'][vi, vj],
                    edge_maps['right_matched'][vi, vj]
                )
            if vi == ui + 1 and uj == vj:
                return (
                    edge_maps['down_perm'][ui, uj, :],
                    edge_maps['down_cost'][ui, uj],
                    edge_maps['down_max_cost'][ui, uj],
                    edge_maps['down_margin'][ui, uj],
                    edge_maps['down_matched'][ui, uj]
                )
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
            for branch in range(branch_count):
                source_index = local_index_by_branch[ui, uj, branch]
                if source_index < 0:
                    continue
                target_index = permutation[source_index]
                if target_index < 0:
                    continue
                local_index_by_branch[vi, vj, branch] = target_index
                tracked_roots[vi, vj, branch] = raw_roots[vi, vj, target_index]

            assigned[vi, vj] = True
            component_id[vi, vj] = component
            parent_index[vi, vj, :] = [ui, uj]
            matched_count[vi, vj] = edge_matched
            assignment_cost[vi, vj] = cost
            max_assignment_cost[vi, vj] = edge_max_cost
            assignment_margin[vi, vj] = margin
            assignment_confidence[vi, vj] = self._edge_confidence(edge_max_cost, margin)

        def push_neighbors(heap, i, j):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= n_omega or nj < 0 or nj >= n_kh:
                    continue
                if assigned[ni, nj] or finite_count[ni, nj] == 0:
                    continue
                _, cost, edge_max_cost, margin, edge_matched = edge_from_to(i, j, ni, nj)
                complete = int(edge_matched == branch_count)
                confidence = self._edge_confidence(edge_max_cost, margin)
                margin_priority = max(0.0, margin) if np.isfinite(margin) else 0.0
                cost_priority = edge_max_cost if np.isfinite(edge_max_cost) else np.inf
                heapq.heappush(heap, (-complete, -confidence, -margin_priority, cost_priority, i, j, ni, nj))

        requested_seed = seed_index
        component = 0
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

            seeds.append(seed)
            assign_seed(seed[0], seed[1], component)

            heap = []
            push_neighbors(heap, seed[0], seed[1])

            while heap:
                _, _, _, _, ui, uj, vi, vj = heapq.heappop(heap)
                if assigned[vi, vj] or not assigned[ui, uj]:
                    continue

                permutation, cost, edge_max_cost, margin, edge_matched = edge_from_to(ui, uj, vi, vj)
                assign_from_edge(ui, uj, vi, vj, component, permutation, cost, edge_max_cost, margin, edge_matched)
                push_neighbors(heap, vi, vj)

            component += 1

        root_set_jump_map = self._edge_max_to_pixel_map(
            edge_maps['right_max_cost'], edge_maps['down_max_cost'], (n_omega, n_kh)
        )
        root_set_margin_map = self._edge_min_to_pixel_map(
            edge_maps['right_margin'], edge_maps['down_margin'], (n_omega, n_kh)
        )
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
        **tracking_kwargs
    ):

        '''
        Purpose
        -------
        Solve kz over the full omega-kh grid and track continuous branches.

        Outputs
        -------
        kz: np.array, complex
            Continuity-tracked branches with shape (omega, kh, kz_order).

        diagnostics: dict, optional
            Returned when return_diagnostics=True. Includes raw, untracked
            polynomial roots for comparison diagnostics, smoothness summaries,
            degeneracy flags, and uncertainty masks.
        '''

        raw_roots = self._kz_solve_raw_grid()
        tracked_roots, diagnostics = self.track_kz_branches(
            raw_roots, return_diagnostics=True, **tracking_kwargs
        )
        pre_swap_tracked_roots = tracked_roots.copy()
        pre_swap_tracked_jump, pre_swap_ratio, pre_swap_excess_mask = self._branch_jump_excess_mask(
            pre_swap_tracked_roots,
            diagnostics['root_set_jump'],
            diagnostics['uncertainty_threshold']
        )

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
        diagnostics['root_residuals'] = self.kz_root_residuals(raw_roots)
        diagnostics['max_root_residual'] = np.full(raw_roots.shape[:2], np.nan)
        residual_valid = np.any(np.isfinite(diagnostics['root_residuals']), axis=2)
        diagnostics['max_root_residual'][residual_valid] = np.nanmax(
            diagnostics['root_residuals'][residual_valid], axis=1
        )
        diagnostics['solver_instability_mask'] = (
            np.isfinite(diagnostics['max_root_residual'])
            & (diagnostics['max_root_residual'] > 1e-8)
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
        diagnostics['plot_suspicious_mask'] = (
            diagnostics['solve_failed_mask']
            | diagnostics['unresolved_branch_jump_mask']
            | diagnostics['solver_instability_mask']
        )

        if return_diagnostics:
            return tracked_roots, diagnostics

        return tracked_roots

    def kz_root_residuals(self, roots):

        '''
        Purpose
        -------
        Calculate normalized polynomial residuals for kz roots.

        Notes
        -----
        These residuals diagnose the polynomial solve independently of branch
        tracking. Large branch jumps with small residuals point to assignment
        problems; large residuals point to an ill-conditioned or failed solve.
        '''

        roots = np.asarray(roots, dtype=np.complex128)
        residuals = np.full(roots.shape, np.nan)
        powers = None

        for i, omega_i in enumerate(self.omega):
            for j, kh_j in enumerate(self.kh):
                try:
                    coefficients = self.kz_coefficients(omega_i, kh_j)
                    coefficients = self._trim_polynomial_coefficients(coefficients)
                except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
                    continue

                if coefficients is None or coefficients.size <= 1:
                    continue

                if powers is None or powers.size != coefficients.size:
                    powers = np.arange(coefficients.size - 1, -1, -1)

                point_roots = roots[i, j, :]
                valid = self._finite_complex_mask(point_roots)
                if not np.any(valid):
                    continue

                values = np.polyval(coefficients, point_roots[valid])
                scale = np.sum(
                    np.abs(coefficients)[None, :] * np.maximum(1.0, np.abs(point_roots[valid]))[:, None]**powers[None, :],
                    axis=1
                )
                residuals[i, j, valid] = np.abs(values) / np.maximum(scale, 1e-300)

        return residuals

    def kz_branch_jump_maps(self, roots):

        roots = np.asarray(roots, dtype=np.complex128)
        jumps = np.full(roots.shape, np.nan)

        for branch in range(roots.shape[2]):
            omega_a = roots[1:, :, branch]
            omega_b = roots[:-1, :, branch]
            omega_valid = self._finite_complex_mask(omega_a) & self._finite_complex_mask(omega_b)
            omega_jump = np.full(omega_a.shape, np.nan)
            omega_jump[omega_valid] = np.abs(omega_a[omega_valid] - omega_b[omega_valid])
            self._update_max_map(jumps[1:, :, branch], omega_jump)
            self._update_max_map(jumps[:-1, :, branch], omega_jump)

            kh_a = roots[:, 1:, branch]
            kh_b = roots[:, :-1, branch]
            kh_valid = self._finite_complex_mask(kh_a) & self._finite_complex_mask(kh_b)
            kh_jump = np.full(kh_a.shape, np.nan)
            kh_jump[kh_valid] = np.abs(kh_a[kh_valid] - kh_b[kh_valid])
            self._update_max_map(jumps[:, 1:, branch], kh_jump)
            self._update_max_map(jumps[:, :-1, branch], kh_jump)

        return jumps

    def kz_smoothness_map(self, roots):

        roots = np.asarray(roots, dtype=np.complex128)
        smoothness = np.full(roots.shape[:2], np.nan)

        def update(target, values):
            self._update_max_map(target, values)

        for branch in range(roots.shape[2]):
            omega_a = roots[1:, :, branch]
            omega_b = roots[:-1, :, branch]
            omega_valid = self._finite_complex_mask(omega_a) & self._finite_complex_mask(omega_b)
            omega_jump = np.full(omega_a.shape, np.nan)
            omega_jump[omega_valid] = np.abs(omega_a[omega_valid] - omega_b[omega_valid])
            update(smoothness[1:, :], omega_jump)
            update(smoothness[:-1, :], omega_jump)

            kh_a = roots[:, 1:, branch]
            kh_b = roots[:, :-1, branch]
            kh_valid = self._finite_complex_mask(kh_a) & self._finite_complex_mask(kh_b)
            kh_jump = np.full(kh_a.shape, np.nan)
            kh_jump[kh_valid] = np.abs(kh_a[kh_valid] - kh_b[kh_valid])
            update(smoothness[:, 1:], kh_jump)
            update(smoothness[:, :-1], kh_jump)

        return smoothness

    def kz_smoothness_summary(self, roots):

        roots = np.asarray(roots, dtype=np.complex128)
        jumps = []
        per_branch_p95 = []

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

        if len(jumps) == 0:
            all_jumps = np.asarray([])
        else:
            all_jumps = np.concatenate(jumps)

        if all_jumps.size == 0:
            return {
                'mean': np.nan,
                'median': np.nan,
                'p95': np.nan,
                'max': np.nan,
                'per_branch_p95': np.asarray(per_branch_p95)
            }

        return {
            'mean': float(np.mean(all_jumps)),
            'median': float(np.median(all_jumps)),
            'p95': float(np.percentile(all_jumps, 95)),
            'max': float(np.max(all_jumps)),
            'per_branch_p95': np.asarray(per_branch_p95)
        }

    def compare_kz_root_ordering(self, raw_roots, tracked_roots):

        raw_summary = self.kz_smoothness_summary(raw_roots)
        tracked_summary = self.kz_smoothness_summary(tracked_roots)
        raw_p95 = raw_summary['p95']
        tracked_p95 = tracked_summary['p95']
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

        import matplotlib.pyplot as plt

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

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5*n_cols, 3.8*n_rows),
            squeeze=False,
            constrained_layout=shared_colorbar
        )
        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
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

        cmap_plot = plt.get_cmap(cmap).copy()
        cmap_plot.set_bad(bad_color)
        mesh = None
        plotted_axes = []

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

        if shared_colorbar and mesh is not None:
            colorbar = fig.colorbar(mesh, ax=plotted_axes)
            if colorbar_label is not None:
                colorbar.set_label(colorbar_label)

        if not shared_colorbar:
            fig.tight_layout()
        return fig, axes

    def kz_plot_values(self, roots, component='real', diagnostics=None, mask=None, mask_suspicious=False):

        roots = np.asarray(roots, dtype=np.complex128)
        values = self._grid_component(roots, component).copy()

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

        if plot_mask is not None:
            if plot_mask.ndim == 2:
                values[plot_mask, :] = np.nan
            elif plot_mask.ndim == 3:
                values[plot_mask] = np.nan
            else:
                raise ValueError("mask must have shape (omega, kh) or (omega, kh, branch).")

        return values

    def kz_color_limits(self, roots, component='real', diagnostics=None, mask=None, mask_suspicious=False, robust=False, percentiles=(2, 98), symmetric=False):

        values = self.kz_plot_values(
            roots,
            component=component,
            diagnostics=diagnostics,
            mask=mask,
            mask_suspicious=mask_suspicious
        )
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None, None

        if robust:
            vmin, vmax = np.percentile(finite_values, percentiles)
            if np.isclose(vmin, vmax):
                vmin, vmax = np.min(finite_values), np.max(finite_values)
        else:
            vmin, vmax = np.min(finite_values), np.max(finite_values)

        if symmetric:
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit

        return vmin, vmax

    def plot_kz_tracking_diagnostics(self, raw_roots, tracked_roots, diagnostics=None, kh_scale=1e3, omega_scale=(2*pi)**(-1)*1e3):

        import matplotlib.pyplot as plt

        if diagnostics is None:
            _, diagnostics = self.track_kz_branches(raw_roots, return_diagnostics=True)

        raw_jump = diagnostics.get('raw_order_jump', self.kz_smoothness_map(raw_roots))
        tracked_jump = diagnostics.get('tracked_branch_jump', self.kz_smoothness_map(tracked_roots))
        root_set_jump = diagnostics.get('root_set_jump', self.kz_smoothness_map(raw_roots))
        max_residual = diagnostics.get('max_root_residual', np.full(raw_jump.shape, np.nan))

        def log10_positive(values):
            values = np.asarray(values)
            logged = np.full(values.shape, np.nan, dtype=float)
            valid = values > 0
            logged[valid] = np.log10(values[valid])
            return logged

        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), squeeze=False)
        plot_suspicious_mask = diagnostics.get('plot_suspicious_mask', diagnostics['uncertain_mask'])
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

        Notes
        -----
        This diagnostic is intended to verify that the correction relabels
        roots across a suspected branch-cut boundary instead of masking the
        affected pixels. If dz is supplied, the corresponding phase-difference
        maps are plotted below the kz maps.
        '''

        import matplotlib.pyplot as plt

        if 'pre_swap_tracked_roots' not in diagnostics:
            raise ValueError("diagnostics must include 'pre_swap_tracked_roots'. Run kz_solve_grid(return_diagnostics=True).")

        pre_roots = np.asarray(diagnostics['pre_swap_tracked_roots'], dtype=np.complex128)
        post_roots = np.asarray(tracked_roots, dtype=np.complex128)
        if branches is None:
            branches = range(post_roots.shape[2])
        branches = list(branches)

        kz_pre = self._grid_component(pre_roots, component)
        kz_post = self._grid_component(post_roots, component)
        kz_finite = np.concatenate((kz_pre[np.isfinite(kz_pre)], kz_post[np.isfinite(kz_post)]))
        kz_vmin = np.min(kz_finite) if kz_finite.size > 0 else None
        kz_vmax = np.max(kz_finite) if kz_finite.size > 0 else None

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

        colorbar = fig.colorbar(mesh, ax=axes[:2, :].ravel())
        colorbar.set_label(r'$k_z\,[\mathrm{km}^{-1}]$')

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
        branch_pair: tuple, optional
            One-based branch numbers, e.g. (4, 5). If omitted, the pair with
            the largest identity-refinement candidate is selected.

        Notes
        -----
        The plot compares the current assignment with the candidate obtained
        by swapping the selected pair over the graph-parity region. If the
        candidate lowers the global identity objective, kz_solve_grid applies
        it automatically. If not, the boundary should be interpreted through
        the separation/close-root panels as a true near-degenerate or crossing
        zone, not as a hidden plotting mask.
        '''

        import matplotlib.pyplot as plt

        tracked_roots = np.asarray(tracked_roots, dtype=np.complex128)
        n_branch = tracked_roots.shape[2]
        identity_stats = diagnostics.get('branch_identity_refinement', {})
        pair_stats = identity_stats.get('pair_stats', [])

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

        if branch_a < 0 or branch_b < 0 or branch_a >= n_branch or branch_b >= n_branch or branch_a == branch_b:
            raise ValueError("branch_pair must refer to two valid one-based branches.")
        if branch_a > branch_b:
            branch_a, branch_b = branch_b, branch_a

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

        before_objective = self._pair_identity_objective(branch_values, branch_a, branch_b)
        after_objective = self._pair_identity_objective(candidate_roots, branch_a, branch_b)
        improvement = before_objective['total'] - after_objective['total']
        improvement_rtol = improvement / max(abs(before_objective['total']), 1e-300)

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

        Notes
        -----
        `branch` is one-based. When `dz` is supplied the branch panel shows
        Delta phi in degrees; otherwise it shows Re(k_z). The branch-cut panel
        marks edges where the current global labels cannot follow the local raw
        root continuation. The monodromy panel marks plaquettes where the raw
        root permutations around a closed loop are non-identity, indicating a
        true root-sheet branch point or coalescence in the sampled domain.
        '''

        import matplotlib.pyplot as plt

        tracked_roots = np.asarray(tracked_roots, dtype=np.complex128)
        branch_index = int(branch) - 1
        if branch_index < 0 or branch_index >= tracked_roots.shape[2]:
            raise ValueError("branch must be a valid one-based branch number.")

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
            values = np.asarray(values)
            logged = np.full(values.shape, np.nan, dtype=float)
            valid = values > 0
            logged[valid] = np.log10(values[valid])
            return logged

        kh_plot, omega_plot = np.meshgrid(self.kh*kh_scale, self.omega*omega_scale)
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), squeeze=False, constrained_layout=True)
        branch_cmap = plt.get_cmap(value_cmap).copy()
        scalar_cmap_plot = plt.get_cmap(scalar_cmap).copy()
        branch_cmap.set_bad('0.65')
        scalar_cmap_plot.set_bad('0.65')

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
            Root grid with shape (omega, kh, branch).

        axis: str
            Direction to plot along. Use 'omega' for a fixed kh cut or 'kh'
            for a fixed omega cut.

        index: int, optional
            Fixed-grid index. Defaults to the middle of the fixed dimension.
        '''

        import matplotlib.pyplot as plt

        roots = np.asarray(roots, dtype=np.complex128)
        values = self._grid_component(roots, component)
        if branches is None:
            branches = range(roots.shape[2])

        axis = axis.lower()
        fig, ax = plt.subplots(figsize=(7, 4))
        if axis in ('omega', 'nu', 'frequency'):
            if index is None:
                index = len(self.kh)//2
            x = self.omega*omega_scale
            for branch in branches:
                ax.plot(x, values[:, index, branch], label=f'branch {branch + 1}')
            ax.set_xlabel(r'$\nu\,[\mathrm{mHz}]$')
            ax.set_title(rf'$k_h={self.kh[index]*kh_scale:.3g}\,\mathrm{{Mm}}^{{-1}}$')
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
        omega: np.array, float
            Angular frequency (rad/s).

        kz: float
            Vertical wavenumber (km^-1).

        Outputs
        -------
        v_phase: np.array, float
            Phase speed of the wave (km/s).

        Author(s)
        ---------
        Julio M. Morales, June 03, 2024
        '''

        # Calculate the phase speed
        with np.errstate(divide='ignore', invalid='ignore'):
            v_phase = omega*kz**(-1) # km/s

        return v_phase
    
    def phase_difference(self, omega, v_phase, dz):

        '''
        Purpose
        -------
        Calculate the phase difference between two points.

        Inputs
        ------
        omega; np.array, float:
            Frequency of the wave (rad/s).

        v_phase; np.array, float:
            Phase speed of the wave (km/s).

        dz; float:
            Distance between two points (km).

        Outputs
        -------
        delta_phi; np.array, float:
            Phase difference between two points (rad).

        Author(s)
        ---------
        Julio M. Morales, June 03, 2024
        '''

        # Calculate the phase difference
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_phi = omega*dz*v_phase**(-1) # rad

        return delta_phi

# Import the relevant libraries
import numpy as np
from numpy import sin, cos, sqrt, pi
from scipy import optimize

class DiagnosticDiagram:

    def __init__(self, kh_grid, omega_grid, params):

        # Assign attributes and global variables
        self.kh = kh_grid[0]  # km^-1
        self.omega = omega_grid[:, 0]  # rad/s
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
        poly: np.array, float
            Value of the polynomial in kz for input parameters.

        Author(s)
        ---------
        Julio M. Morales, August 08th, 2025
        '''

        # Determine the model
        if (self.model == 'sf1966'):
            c2 = 1
            c1 = 0
            c0 = kh**2*(omega**2 - self.N**2)*omega**(-2) - (omega**2 - self.wac**2)*self.cs**(-2)
            poly = c2*kz**2 + c1*kz + c0
        elif (self.model == 'mt1981'):
            c2 = 1
            c1 = 0
            c0 = kh**2 + (4*self.H**2)**(-1) - self.N**2*kh**2*omega**(-2) - omega**2*self.cs**(-2)
            poly = c2*kz**2 + c1*kz + c0
        elif (self.model == 'mt1982'):
            c2 = 1
            c1 = 0
            c0 = -0.5*(self.N**2*kh**2*omega**(-2) - kh**2 + (omega**2*self.tau**2*(-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))**2*(omega**2*self.tau**2 + 1)**(-2) + (self.N**2*kh**2*omega**(-2) - kh**2 + (-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))**2)**(0.5) + (-self.N**2*kh**2*omega**(-2) + omega**2*(self.gamma - 1)*self.cs**(-2))*(omega**2*self.tau**2 + 1)**(-1) + omega**2*self.cs**(-2) - 1*(4*self.H**2)**(-1))
            poly = c2*kz**2 + c1*kz + c0
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
            poly = c2*kz**2 + c1*kz + c0
        elif (self.model == 'nc2009'):
            c6 = -self.a**4*self.cs**2*cos(self.theta)**4
            c5 = -4*kh*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta)**3
            c4 = -6*kh**2*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos(self.theta)**2 - kh**2*self.a**4*self.cs**2*cos(self.theta)**4 + omega**2*self.a**4*cos(self.theta)**2 + 2*omega**2*self.a**2*self.cs**2*cos(self.theta)**2 - self.a**2*self.az**2*self.wac**2*cos(self.theta)**2
            c3 = -4*kh**3*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos(self.theta) - 4*kh**3*self.a**4*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta)**3 + 2*kh*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos(self.theta) + 4*kh*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta)
            c2 = -kh**4*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 - 6*kh**4*self.a**4*self.cs**2*sin(self.theta)**2*cos(self.phi)**2*cos(self.theta)**2 + kh**2*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.a**4*cos(self.theta)**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**2*omega**2*self.a**2*self.cs**2*cos(self.theta)**2 - kh**2*self.N**2*self.a**2*self.cs**2*cos(self.theta)**2 - kh**2*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*self.a**2*self.az**2*self.wac**2*cos(self.theta)**2 - omega**4*self.a**2*cos(self.theta)**2 - omega**4*self.a**2 - omega**4*self.cs**2 + omega**2*self.a**2*self.wac**2*cos(self.theta)**2 + omega**2*self.az**2*self.wac**2
            c1 = -4*kh**5*self.a**4*self.cs**2*sin(self.theta)**3*cos(self.phi)**3*cos(self.theta) + 2*kh**3*omega**2*self.a**4*sin(self.theta)*cos(self.phi)*cos(self.theta) + 4*kh**3*omega**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh**3*self.N**2*self.a**2*self.cs**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh**3*self.a**2*self.az**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta) - 2*kh*omega**4*self.a**2*sin(self.theta)*cos(self.phi)*cos(self.theta) + 2*kh*omega**2*self.a**2*self.wac**2*sin(self.theta)*cos(self.phi)*cos(self.theta)
            c0 = -kh**6*self.a**4*self.cs**2*sin(self.theta)**4*cos(self.phi)**4 + kh**4*omega**2*self.a**4*sin(self.theta)**2*cos(self.phi)**2 + 2*kh**4*omega**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.N**2*self.a**2*self.cs**2*sin(self.theta)**2*cos(self.phi)**2 - kh**4*self.a**2*self.az**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2*sin(self.theta)**2*cos(self.phi)**2 - kh**2*omega**4*self.a**2 - kh**2*omega**4*self.cs**2 + kh**2*omega**2*self.N**2*self.cs**2 + kh**2*omega**2*self.a**2*self.wac**2*sin(self.theta)**2*cos(self.phi)**2 + kh**2*omega**2*self.ay**2*self.wac**2 + kh**2*omega**2*self.az**2*self.wac**2 + omega**6 - omega**4*self.wac**2
            poly = c6*kz**6 + c5*kz**5 + c4*kz**4 + c3*kz**3 + c2*kz**2 + c1*kz + c0
        else:
            raise ValueError("Model for kz not recognized. Please choose from 'sf1966', 'mt1981', 'mt1982', 'bunte1993', 'nc2009'.")

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
    
    def kz_solve(self, kh):

        '''
        Purpose
        -------
        Solve for the vertical wavenumber kz in parallel.

        Inputs
        ------
        kh: float
            i-th horizontal wavenumber (km^-1).

        Outputs
        -------
        roots: np.array, float
            Array of roots of the polynomial as a function of horizontal wavenumber (km^-1) and angular frequency (rad/s).

        Author(s)
        ---------
        Julio M. Morales, August 07th, 2025
        '''

        # Define a fine grid of kz values to search for roots
        fine_grid = np.linspace(0, 1, 5_000)  # km^-1

        # List to store roots
        roots = np.full((len(self.omega), self.params['kz_order']), np.nan)

        # Loop through frequency
        for j, omega_j in enumerate(self.omega):

            # Compute the polynomial being solved on a fine grid and find sign changes
            poly = self.kz_poly(fine_grid, omega_j, kh)
            sign_changes = np.nonzero(np.diff(np.sign(poly)))[0]
            if len(sign_changes) > self.params['kz_order']:
                roots[j, :] = np.nan
                continue
            # Loop through the sign changes to find roots by brentq method
            for ind, idx in enumerate(sign_changes):
                A0, B0 = fine_grid[idx], fine_grid[idx + 1]
                try:
                    root = optimize.brentq(self.kz_poly, A0, B0, args = (omega_j, kh))
                    roots[j, ind] = root
                except ValueError:
                    roots[j, ind] = np.nan
        
        return np.nan_to_num(roots, nan=0.0)

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
        delta_phi = omega*dz*v_phase**(-1) # rad

        return delta_phi
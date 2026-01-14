from .main import list_propagating_modes
from scipy.integrate import solve_bvp
import scipy.constants as cst
import numpy as np
from functools import lru_cache


# Auxiliary functions
def epsilon(n: int) -> int:
    return 1 if n == 0 else 2  # From eq. I.20

def integral_1(m_p, m_q, n, a, b) -> float:
    return \
        - 4 * a * m_p * m_q * (2 * a**2 * n**2 + b**2 * (m_p**2 + m_q**2)) / \
        (np.pi**2 * (m_p**2 - m_q**2)**2 * (a**2 * n**2 + b**2 * m_p**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_q**2)**0.5)

def integral_2(m_p, m_q, n, a, b) -> float:
    return \
        - 8 * a**3 * b**2 * m_p * m_q / \
        (np.pi**4 * (m_p**2 - m_q**2)**2 * (a**2 * n**2 + b**2 * m_p**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_q**2)**0.5)

def integral_3(m_p, m_q, n, a, b) -> float:
    return \
        2 * a**2 * b * n * m_p * (epsilon(n) * epsilon(m_q))**0.5 / \
        (np.pi**2 * (m_p**2 - m_q**2) * (a**2 * n**2 + b**2 * m_p**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_q**2)**0.5)

def integral_4(m_p, m_q, n, a, b) -> float:
    return \
        2 * a**2 * b * n * m_q * (epsilon(n) * epsilon(m_p))**0.5 / \
        (np.pi**2 * (m_q**2 - m_p**2) * (a**2 * n**2 + b**2 * m_q**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_p**2)**0.5)

def integral_5(m_p, m_q, n, a, b) -> float:
    # if m_p and m_q and n:
    return \
        - a * (a**2 * n**2 * m_q**2 + m_p**2 * (a**2 * n**2 + 2 * b**2 * m_q**2)) * \
        epsilon(n) * (epsilon(m_p) * epsilon(m_q))**0.5 / \
        (np.pi**2 * (m_p**2 - m_q**2)**2 * (a**2 * n**2 + b**2 * m_p**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_q**2)**0.5) * \
        (3 - epsilon(n))  # Factor for n = 0 
    # elif m_p and m_q and not n:
    #     return \
    #         - 4 * a * m_p * m_q * epsilon(n) * (epsilon(m_p) * epsilon(m_q))**0.5 / \
    #         (np.pi**2 * (m_p**2 - m_q**2)**2)

def integral_6(m_p, m_q, n, a, b) -> float:
    # if m_p and m_q and n:
    return \
        - a**3 * b**2 * (m_p**2 + m_q**2) * epsilon(n) * (epsilon(m_p) * epsilon(m_q))**0.5 / \
        (np.pi**4 * (m_p**2 - m_q**2)**2 * (a**2 * n**2 + b**2 * m_p**2)**0.5 * \
        (a**2 * n**2 + b**2 * m_q**2)**0.5) * \
        (3 - epsilon(n))  # Factor for n = 0 
    # elif m_p and m_q and not n:
    #     return \
    #         -2 * a**3 * (m_p**2 + m_q**2) * epsilon(n) * (epsilon(m_p) * epsilon(m_q))**0.5 / \
    #         (np.pi**4 * m_p * m_q * (m_p**2 - m_q**2)**2)
        

@lru_cache(maxsize=1000)
def coupling_coefficient(mode_p, mode_q, sign, f, a, b):

    m_p = int(mode_p[1])
    n_p = int(mode_p[2])
    m_q = int(mode_q[1])
    n_q = int(mode_q[2])



    # Early termination if the modes are not coupled
    if ((m_p+m_q)%2 == 0) or (n_p != n_q):
        # print(f"Modes {mode_j} and {mode_m} are not coupled")
        return 0.0j
    
    else:
        n = n_p  # n_p = n_q = n

    # Propagation constants. Most books use beta but this book uses h
    # This book also uses the alpha constant eq. I.22

    chi_p = cst.pi * ((m_p / a) ** 2 + (n / b) ** 2) ** 0.5
    chi_q = cst.pi * ((m_q / a) ** 2 + (n / b) ** 2) ** 0.5

    k = 2 * cst.pi * f / cst.c

    beta_p = (k ** 2 - chi_p ** 2) ** 0.5
    beta_q = (k ** 2 - chi_q ** 2) ** 0.5

    if mode_p[0][1] == 'M' and mode_q[0][1] == 'M':
        C = \
            1j / 2 / (beta_p * beta_q)**0.5 * ((beta_p * beta_q + sign * k**2) * \
            integral_1(m_p, m_q, n, a, b) - sign * chi_p**2 * chi_q**2 * \
            integral_2(m_p, m_q, n, a, b))

    elif mode_p[0][1] == 'M' and mode_q[0][1] == 'E':
        C = \
            1j * k * (beta_p + sign * beta_q) / 2 / (beta_p * beta_q)**0.5 * \
            integral_3(m_p, m_q, n, a, b)
        
    elif mode_p[0][1] == 'E' and mode_q[0][1] == 'M':
        C = \
            1j * k * (beta_q + sign * beta_p) / 2 / (beta_p * beta_q)**0.5 * \
            integral_4(m_p, m_q, n, a, b)
        
    elif mode_p[0][1] == 'E' and mode_q[0][1] == 'E':
        C = \
            1j / 2 / (beta_p * beta_q)**0.5 * ((k**2 + sign * beta_p * beta_q) * \
            integral_5(m_p, m_q, n, a, b) - chi_p**2 * chi_q**2 * integral_6(m_p, m_q, n, a, b))

    return C



def solve_mode_coupling(f, a, b, mode_in, length, curvature, resolution=1000, coupled_modes=None, maximum_modes=None, verbose=False, **kwargs):
    """"
    Solve the mode coupling problem in a rectangular waveguide with arbitrary curvature.
    
    Parameters
    ----------
    f: float
        Frequency (Hz).
    a: float
        First dimension of the rectangular waveguide (meters).
    b: float
        Second dimension of the rectangular waveguide (meters). The curvature is applied perpendicular to this dimension.
    mode_in: tuple (str, int, int)
        Input mode in the format (type, m, n). For example, ('TE', 1, 0) for TE10.
    length: float
        Length of the waveguide (meters).
    curvature: function(ndarray) -> ndarray
        Curvature function that returns the local curvature (in meters^-1) at a given positions in the waveguide. The function is evaluated between 0 and `length`.
    resolution: int or str, optional
        Number of points to evaluate the curvature function and solve the boundary-value problem. Default is 1000. If 'auto', the number of points is automatically determined based on frequency.
    coupled_modes: list, optional
        List of tuples representing the coupled modes. If None, all propagating modes are considered. Default is None.
    maximum_modes: int, optional
        Maximum number of modes to consider. Default is all propagating modes.
    verbose: bool, optional
        Print additional information. Default is False.
    kwargs: dict
        Additional arguments to pass to scipy.integrate.solve_bvp.

    Returns
    -------
    modes: list (N,)
        List of tuples representing the propagating modes used in the calculation.
    results: object
        Object containing the solution of the boundary-value problem. See scipy.integrate.solve_bvp https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html
        The attribute results.y contains the complex amplitudes of the forward and backward modes at each point in the waveguide.
    """


    modes = list_propagating_modes(f, a, b)

    if verbose:
        print(f"Propagating modes below {f/1e6:.0f} MHz:")
        for mode_name, m, n, cutoff in modes:
            print(f"{mode_name}: {cutoff/1e6:.0f} MHz")


    mode_in = list(mode_in)
    mode_in[0] = mode_in[0]+str(mode_in[1])+','+str(mode_in[2])
    m_in = int(mode_in[1])
    n_in = int(mode_in[2])

    if coupled_modes is None:

        coupled_modes = []
        for mode_name, m, n, cutoff in modes:
            if n_in == n:
                coupled_modes.append((mode_name, m, n, cutoff))

        if verbose:
            modes_string = '\n'.join(str(mode) for mode in coupled_modes)
            print(f"Modes with same n as {mode_in[0]}:\n{modes_string}")

    if maximum_modes is not None:
        if len(coupled_modes) > maximum_modes:
            coupled_modes = coupled_modes[:maximum_modes]

    excited_mode_index = [mode[0] for mode in coupled_modes].index(mode_in[0])

    def c(z):
        return curvature(z)
    

    def dP_j_dl(z, P):

        result = []
        for ii, mode_j in enumerate(coupled_modes):

            m_j = int(mode_j[1])
            n_j = int(mode_j[2])

            # Propagation constants. Most books use beta but this book uses h
            # This book also uses the alpha constant eq. I.22

            chi_j = cst.pi * ((m_j / a) ** 2 + (n_j / b) ** 2) ** 0.5

            k = 2 * cst.pi * f / cst.c

            beta_j = (k ** 2 - chi_j ** 2) ** 0.5
            # Accumulate the derivatives of forward modes
            summation = -1j * beta_j * P[ii]

            # Sum over forward modes
            for i, mode_m in enumerate(coupled_modes):
                summation -= c(z) * coupling_coefficient(mode_j, mode_m, 1, f, a, b) * P[i]

            # Sum over backward modes
            for i, mode_m in enumerate(coupled_modes):
                summation -= c(z) * coupling_coefficient(mode_j, mode_m, -1, f, a, b) * P[len(coupled_modes) + i]

            result.append(summation)

        for ii, mode_j in enumerate(coupled_modes):

            m_j = int(mode_j[1])
            n_j = int(mode_j[2])

            # Propagation constants. Most books use beta but this book uses h
            # This book also uses the alpha constant eq. I.22

            chi_j = cst.pi * ((m_j / a) ** 2 + (n_j / b) ** 2) ** 0.5

            k = 2 * cst.pi * f / cst.c

            beta_j = (k ** 2 - chi_j ** 2) ** 0.5
            # Accumulate the derivatives of backward modes
            summation = 1j * beta_j * P[len(coupled_modes) + ii]

            # Sum over forward modes
            for i, mode_m in enumerate(coupled_modes):
                summation += c(z) * coupling_coefficient(mode_j, mode_m, -1, f, a, b) * P[i]

            # Sum over backward modes
            for i, mode_m in enumerate(coupled_modes):
                summation += c(z) * coupling_coefficient(mode_j, mode_m, 1, f, a, b) * P[len(coupled_modes) + i]

            result.append(summation)

        return result


    # Boundary conditions are set on the "left" side for the forward modes and on the "right" side for the backward modes
    # All forward modes are zero on the left side except for the excited mode
    # All backward modes are zero on the right side
    def boundary_conditions(P_a, P_b):
        return_values = np.concatenate((P_a[:len(P_a)//2], P_b[len(P_b)//2:]))
        return_values[excited_mode_index] -= 1
        return return_values

    chi_in = cst.pi * ((m_in / a) ** 2 + (n_in / b) ** 2) ** 0.5

    k = 2 * cst.pi * f / cst.c

    beta_in = (k ** 2 - chi_in ** 2) ** 0.5

    lowest_chi = cst.pi / max(a, b)
    highest_beta = (k ** 2 - lowest_chi ** 2) ** 0.5
    lowest_lambda = 2 * cst.pi / highest_beta

    if resolution == 'auto':
        resolution = max(10, int(np.ceil(10 * length / lowest_lambda)))
        print(f"Resolution set to {resolution} points based on the lowest wavelength {lowest_lambda:.2e} m.")

    n_points = resolution
    z = np.linspace(0, length, n_points)

    initial_guess = np.zeros((len(coupled_modes) * 2, n_points))
    initial_guess = initial_guess.astype(complex)

    # Initial guess is just the excited mode propagating alone with no coupling
    # initial_guess[excited_mode_index] = np.exp(-1j * beta_in * z) 
    initial_guess[excited_mode_index] = np.ones(n_points) * (1+0.j)

    result = solve_bvp(
        dP_j_dl,
        boundary_conditions,
        x = z,
        y = initial_guess,
        **kwargs,
    )

    if verbose:
        print(result.message)

    return coupled_modes, result



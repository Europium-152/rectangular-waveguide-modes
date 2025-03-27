from .main import list_propagating_modes
from scipy.integrate import solve_bvp, solve_ivp
import scipy.constants as cst
import numpy as np
from functools import lru_cache
from .mode_coupling import coupling_coefficient
from scipy.interpolate import interp1d


def solution_adder(x1, y1, x2, y2):
    """Add two solutions together. x1 and x2 might have different numbers of points but must refer to the same domain.
    
    Parameters
    ----------
    x1: ndarray
        Domain of the first solution.
    y1: ndarray
        First solution.
    x2: ndarray
        Domain of the second solution.
    y2: ndarray
        Second solution.
    c1: float, optional
        Multiplicative factor for the first solution. Default is 1.
    c2: float, optional
        Multiplicative factor for the second solution. Default is 1.
    """
    # Determine which domain has more points
    if len(x1) > len(x2):
        x = x1
        y = y1
        x_other = x2
        y_other = y2
    else:
        x = x2
        y = y2
        x_other = x1
        y_other = y1

    # Interpolate the other solution to the domain of the first solution
    y_other = interp1d(x_other, y_other)(x)

    return x, y + y_other



def solve_initial_value_problem(f, a, b, mode_in, length, curvature_radius, resolution=1000, coupled_modes=None, maximum_modes=None, verbose=False):
    """"
    Solve the mode coupling problem in a rectangular waveguide with arbitrary curvature by converting the problem to an initial-value problem.
    
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
    curvature_radius: function(ndarray) -> ndarray
        Curvature function that returns the curvature radius (in meters) at a given positions in the waveguide. The function is evaluated between 0 and `length`.
    resolution: int, optional
        Number of points to evaluate the curvature function and solve the boundary-value problem. Default is 1000.
    coupled_modes: list, optional
        List of tuples representing the coupled modes. If None, all propagating modes are considered. Default is None.
    maximum_modes: int, optional
        Maximum number of modes to consider. Default is all propagating modes.
    verbose: bool, optional
        Print additional information. Default is False.

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
            print(f"Modes with same n as {mode_in[0]}:\n{'\n'.join(str(mode) for mode in coupled_modes)}")

    if maximum_modes is not None:
        if len(coupled_modes) > maximum_modes:
            coupled_modes = coupled_modes[:maximum_modes]

    excited_mode_index = [mode[0] for mode in coupled_modes].index(mode_in[0])

    def c(z):
        return 1/curvature_radius(z)
    

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
    

    # Solve everything using the initial value solver. Must find 2N solutions for the N modes and then invert matrix
    solutions = []

    # First N solution correspond to having 1 single forward mode at the output
    for i in range(len(coupled_modes)):

        y0 = np.zeros(len(coupled_modes) * 2).astype(complex)
        y0[i] = 1 + 0.j


        result = solve_ivp(dP_j_dl, [length, 0], y0)

        solutions.append(result)

    # Last N solutions correspond to having 1 single backward mode at the input
    for i in range(len(coupled_modes)):

        y0 = np.zeros(len(coupled_modes) * 2).astype(complex)
        y0[len(coupled_modes) + i] = 1 + 0.j

        result = solve_ivp(dP_j_dl, [0, length], y0)

        solutions.append(result)


    # Construct matrix to invert
    N = len(coupled_modes)

    matrix = [[solutions[j].y[i, -1] for j in range(2*N)] for i in range(2*N)]

    matrix = np.array(matrix)

    inv_matrix = np.linalg.inv(matrix)

    coefficients = np.dot(inv_matrix, np.array([1 if i == excited_mode_index else 0 for i in range(2*N)]))
    # Should be equivalent to:
    # coefficients = inv_matrix[:, excited_mode_index]

    final_x = solutions[0].t[::-1]
    final_y = solutions[0].y[..., ::-1] * coefficients[0]

    for i in range(1, N):
        final_x, final_y = solution_adder(final_x, final_y, solutions[i].t[::-1], solutions[i].y[..., ::-1] * coefficients[i])

    for i in range(N, 2*N):
        final_x, final_y = solution_adder(final_x, final_y, solutions[i].t, solutions[i].y * coefficients[i])


    return coupled_modes, final_x, final_y
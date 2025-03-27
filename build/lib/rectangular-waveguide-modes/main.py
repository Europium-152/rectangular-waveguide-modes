import math
from scipy.constants import mu_0, epsilon_0
import typer

def calculate_te_cutoff(m: int, n: int, a: float, b: float, mu: float, epsilon: float) -> float:
    """
    Calculate the cutoff frequency for TE modes in a rectangular waveguide.

    Args:
        m (int): Mode number in the x-direction.
        n (int): Mode number in the y-direction.
        a (float): Long dimension of the rectangular waveguide (meters).
        b (float): Short dimension of the rectangular waveguide (meters).
        mu (float): Permeability of the medium (H/m).
        epsilon (float): Permittivity of the medium (F/m).

    Returns:
        float: The cutoff frequency (Hz).
    """
    if m == 0 and n == 0:
        return float('inf')  # No TE00 mode exists
    term1 = (m * math.pi / a) ** 2
    term2 = (n * math.pi / b) ** 2
    cutoff_frequency = (1 / (2 * math.pi * math.sqrt(mu * epsilon))) * math.sqrt(term1 + term2)
    return cutoff_frequency

def calculate_tm_cutoff(m: int, n: int, a: float, b: float, mu: float, epsilon: float) -> float:
    """
    Calculate the cutoff frequency for TM modes in a rectangular waveguide.

    Args:
        m (int): Mode number in the x-direction.
        n (int): Mode number in the y-direction.
        a (float): Long dimension of the rectangular waveguide (meters).
        b (float): Short dimension of the rectangular waveguide (meters).
        mu (float): Permeability of the medium (H/m).
        epsilon (float): Permittivity of the medium (F/m).

    Returns:
        float: The cutoff frequency (Hz).
    """
    if m == 0 or n == 0:
        return float('inf')  # No TM modes with m=0 or n=0
    return calculate_te_cutoff(m, n, a, b, mu, epsilon)

def list_propagating_modes(f_max: float, a: float, b: float, mu: float, epsilon: float):
    """
    Lists all propagating modes (TE and TM) in a rectangular waveguide below the frequency f_max.

    Args:
        f_max (float): Maximum frequency to consider.
        a (float): Long dimension of the rectangular waveguide (meters).
        b (float): Short dimension of the rectangular waveguide (meters).
        mu (float): Permeability of the medium (H/m).
        epsilon (float): Permittivity of the medium (F/m).

    Returns:
        list: A list of tuples representing propagating modes (type, m, n, cutoff_frequency).
    """
    modes = []

    # Search TE modes
    m = 0
    while True:
        found_mode = False
        n = 1 if m == 0 else 0  # TE00 mode does not exist
        while True:
            cutoff = calculate_te_cutoff(m, n, a, b, mu, epsilon)
            if cutoff < f_max:
                modes.append((f"TE{m}{n}", m, n, cutoff))
                found_mode = True
            else:
                break
            n += 1
        if not found_mode and m > 0:  # Stop if no new modes are found for m > 0
            break
        m += 1

    # Search TM modes
    m = 1
    while True:
        found_mode = False
        n = 1
        while True:
            cutoff = calculate_tm_cutoff(m, n, a, b, mu, epsilon)
            if cutoff < f_max:
                modes.append((f"TM{m}{n}", m, n, cutoff))
                found_mode = True
            else:
                break
            n += 1
        if not found_mode:  # Stop if no new modes are found
            break
        m += 1

    return sorted(modes, key=lambda x: x[3])  # Sort by cutoff frequency

def main(f_max: float, a: float, b: float, mu: float = mu_0, epsilon: float = epsilon_0):
    """
    CLI to list propagating modes in a rectangular waveguide below a given maximum frequency.

    Args:
        f_max (float): Maximum frequency to consider (Hz).
        a (float): Long dimension of the rectangular waveguide (meters).
        b (float): Short dimension of the rectangular waveguide (meters).
        mu (float): Permeability of the medium (H/m). Default is vacuum permeability.
        epsilon (float): Permittivity of the medium (F/m). Default is vacuum permittivity.
    """
    modes = list_propagating_modes(f_max, a, b, mu, epsilon)
    typer.echo("\nPropagating Modes Below f_max:")
    for mode in modes:
        typer.echo(f"{mode[0]}: Cutoff Frequency = {mode[3]*1e-9:.3f} GHz")

if __name__ == "__main__":
    typer.run(main)

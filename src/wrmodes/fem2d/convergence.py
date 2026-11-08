"""Convergence study of the 2D solver: mesh refinement and number of evanescent port modes."""
import numpy as np

__all__ = ["convergence_study"]


def convergence_study(component, f, mesh_densities=(10, 15, 20, 30), n_evanescent=(2, 4, 6, 10),
                      basis1=None, basis2=None, verbose=False):
    """Change of every |S| entry of a 2D-solver component under mesh refinement and under the number M of
    evanescent port modes.

    Parameters
    ----------
    component : Coplanar2D (or Miter)
    f : float
        Frequency (Hz).
    mesh_densities : sequence
        Elements per transverse wavelength, from coarse to fine (M at the component's setting).
    n_evanescent : sequence
        Values of M, from small to large (mesh at the component's setting).
    basis1, basis2 : ModeBasis, optional
        Port bases (default: the component's default basis).

    Returns
    -------
    dict with
        'mesh': list of (density, ndof, max_abs_change, gsm), where max_abs_change is max |(|S| - |S_finest|)| over
                all entries, and also the complex change max |S - S_finest| (key 'mesh_complex');
        'modes': the same for M (keys 'modes', 'modes_complex');
        'unitarity': list of (label, unitarity defect) for every run;
        'reciprocity': the largest reciprocity defect of all runs.
    """
    if basis1 is None or basis2 is None:
        d1, d2 = component.default_basis(f)
        basis1, basis2 = basis1 or d1, basis2 or d2
    out = {"unitarity": [], "reciprocity": 0.0}

    def run(label, **kw):
        g = component.with_options(**kw).gsm(f, basis1, basis2)
        out["unitarity"].append((label, g.unitarity_defect()))
        out["reciprocity"] = max(out["reciprocity"], g.reciprocity_defect() / np.max(np.abs(g.S)))
        if verbose:
            print(f"{label}: ndof {g.fem_info['ndof']}, unitarity {g.unitarity_defect():.2e}")
        return g

    for key, param, values in (("mesh", "mesh_density", mesh_densities), ("modes", "n_evanescent", n_evanescent)):
        gs = [run(f"{param}={v}", **{param: v}) for v in values]
        ref = gs[-1].S
        out[key] = [(v, sum(n for _, n in g.fem_info["ndof"]), float(np.max(np.abs(np.abs(g.S) - np.abs(ref)))), g)
                    for v, g in zip(values, gs)]
        out[key + "_complex"] = [(v, float(np.max(np.abs(g.S - ref)))) for v, g in zip(values, gs)]
    return out

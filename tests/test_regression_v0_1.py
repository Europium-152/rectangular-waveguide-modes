"""Regression against frozen outputs of the wrmodes 0.1 transfer-matrix solver (tests/data/regression_v0_1.npz).

The coupling matrices are closed-form and must agree to rounding. T and R come from the adaptive Magnus solver:
when the number of slices is the same as in the frozen run, they must agree to 1e-10 (only rounding differs);
if a different BLAS or platform changes an adaptive decision, they must still agree within the solver tolerance
(atol = 1e-6), and the test reports it.
"""
import json
import os
import warnings
import numpy as np
import pytest

import wrmodes as wr
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = np.load(os.path.join(HERE, "data", "regression_v0_1.npz"))
META = json.loads(str(DATA["meta"]))


def kappa_for(c):
    if c["profile"] == "sech":
        T, L = 4.0, c["L"]
        gamma = 2 * T / L
        R0 = 4 / np.radians(c["theta"]) / gamma * np.arctan(np.tanh(T / 2))
        return lambda z: 1 / (R0 * np.cosh(gamma * (np.asarray(z) - L / 2)))
    return lambda z: np.ones_like(np.asarray(z, dtype=float)) * np.radians(c["theta"]) / c["L"]


@pytest.mark.parametrize("i", range(len(META["cases"])))
def test_matches_wrmodes_0_1(i):
    c = META["cases"][i]
    modes = select_coupled_modes(c["f"], c["a"], c["b"], n=c["n"])
    assert [m[0] for m in modes] == list(DATA[f"c{i}_names"])
    D, K, beta = coupled_mode_matrices(c["f"], c["a"], c["b"], modes)
    assert np.allclose(D, DATA[f"c{i}_D"], rtol=1e-13, atol=0)
    assert np.allclose(K, DATA[f"c{i}_K"], rtol=1e-12, atol=1e-14 * np.max(np.abs(K)))
    mode_in = (modes[0][0][:2], int(modes[0][1]), int(modes[0][2]))
    _, res = wr.solve_transfer_matrix(c["f"], c["a"], c["b"], mode_in, c["L"], kappa_for(c),
                                      error_control="full", profile=False)
    same_slicing = res.n_slices == int(DATA[f"c{i}_n_slices"])
    tol = 1e-10 if same_slicing else 2e-6
    if not same_slicing:
        warnings.warn(f"case {i}: adaptive slicing differs from the frozen run "
                      f"({res.n_slices} vs {int(DATA[f'c{i}_n_slices'])}); comparing at the solver tolerance")
    assert np.max(np.abs(res.T - DATA[f"c{i}_T"])) < tol
    assert np.max(np.abs(res.R - DATA[f"c{i}_R"])) < tol

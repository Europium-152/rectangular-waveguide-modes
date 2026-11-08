"""Mode matching: junctions between rectangular guides (CONVENTIONS.md sections 4-6).

* `overlap_matrix`: closed-form overlaps of the normalized mode fields of two guides, one inside the other.
* `junction_gsm`: symmetric GSM of a junction from its overlap matrix.
* `Step`: the step junction component; `convergence_study` for any component with evanescent modes.
* `marcuvitz`: single-mode equivalent circuits of steps from Marcuvitz, Waveguide Handbook (1951), for comparison.
"""
from .overlaps import overlap_matrix
from .junction import junction_gsm
from .step import Step, convergence_study
from . import marcuvitz

__all__ = ["overlap_matrix", "junction_gsm", "Step", "convergence_study", "marcuvitz"]

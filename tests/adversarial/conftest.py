"""WP7 adversarial tests: put the WP7 references (benchmarks/validation_v1) on the path.

Tests of a card's component skip with a clear message until that card's interface is merged
(ref_interfaces.require).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.path.join(HERE, "..", "..", "benchmarks", "validation_v1"))
if REF not in sys.path:
    sys.path.insert(0, REF)

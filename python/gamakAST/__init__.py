"""gamakAST — egglog-based symbolic rewriting for the SR engine.

Phase 1.5 surface: the `denoise` mutation operator. Takes an expression in
egglog `Math` surface syntax plus training-data rows, and returns a (possibly
smaller) equivalent form that still reproduces the input's behaviour on the
data — deterministic, real-domain, no sympy.

Example
-------
>>> from gamakAST import denoise
>>> rows = [{"x": 1.0, "y": 5.0}, {"x": 2.0, "y": -3.0}]
>>> denoise('(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))', rows)
{'expr': '(Var "x")', 'cost': ..., 'changed': True}
"""

from ._gamakast import (
    denoise,
    denoise_karva,
    denoise_karva_candidates,
    eclass_variants,
    eclass_extract_hff,
    eclass_extract_hff_instrumented,
    master_pset,
    master_constants,
    master_lattice,
    physics_mutate,
    physics_mutate_karva,
    proves_equal,
    snap_karva,
    concretize_karva,
)
from .sympy_bridge import to_math, from_math, equals

__all__ = [
    "denoise",
    "denoise_karva",
    "denoise_karva_candidates",
    "eclass_variants",
    "eclass_extract_hff",
    "eclass_extract_hff_instrumented",
    "master_pset",
    "master_constants",
    "master_lattice",
    "physics_mutate",
    "physics_mutate_karva",
    "proves_equal",
    "snap_karva",
    "concretize_karva",
    "to_math",
    "from_math",
    "equals",
]
__version__ = "0.1.0"

"""The SymPy side of the srbench_equiv agreement measurement, run OFFLINE (sympy never in the scoring loop).

The claim "fuller's srbench_equiv agrees with SymPy on 11 of 12" (2026-09-28) was measured by
phylu/logs/score_all.py and score_all_b.py, which printed their table to stdout and saved nothing. This script
re-measures it and SAVES it: for the twelve laws of phylu/logs/SIX_SOLVED and SIX_SOLVED_B, the tidied model (the
`model:` line of each run.log, what SRBench scores), the truth from hff's FEYNMAN_REGISTRY, both as fuller Math
s-expressions (fuller-native Pow2/Pow3/Sub, the encoder of phylu 4e8bf0e4), and SRBench's own verdict
(hff/notebooks/_srbench_check.srbench_solution, which imports SRBench's round_floats), with a 120 s bound per law.

It writes tests/fixtures/srbench_equiv_sympy.tsv; `srbench_equiv::tests::agreement_with_sympy_on_the_twelve_laws`
runs fuller on the same pairs and asserts the agreement.

    python3 parity/gen_srbench_equiv_fixture.py
"""
import gzip
import os
import re
import signal
import sys

HFF = "/Users/andrewmorgan/Dev/gamakon/hff/notebooks"
sys.path.insert(0, os.path.join(HFF, "_ledgers", "srbench_repo", "experiment"))
sys.path.insert(0, HFF)
import _srbench_check as C  # noqa: E402
from feynman_problems import FEYNMAN_REGISTRY  # noqa: E402
import sympy as sp  # noqa: E402
from sympy.parsing.sympy_parser import parse_expr  # noqa: E402

DS = os.path.join(HFF, "_ledgers", "pmlb_repo", "datasets")
LOGS = "/Users/andrewmorgan/Dev/gamakon/phylu/logs"
RUNS = {
    "SIX_SOLVED": ["feynman_I_11_19", "feynman_I_12_4", "feynman_I_18_14", "feynman_II_13_17", "feynman_III_19_51", "feynman_test_9"],
    "SIX_SOLVED_B": ["feynman_III_12_43", "feynman_II_35_21", "feynman_II_37_1", "feynman_I_12_1", "feynman_I_8_14", "feynman_I_30_3"],
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "srbench_equiv_sympy.tsv")
SECONDS = 120


def columns(law):
    with gzip.open(f"{DS}/{law}/{law}.tsv.gz", "rt") as f:
        return [c for c in f.readline().rstrip("\n").split("\t") if c != "target"]


def neg_of(term):
    if term.is_Mul:
        coeff = term.args[0]
        if coeff.is_Number and coeff < 0:
            rest = sp.Mul(*([sp.Integer(-1) * coeff] + list(term.args[1:]))) if coeff != -1 else sp.Mul(*term.args[1:])
            return to_sexpr(rest)
    if term.is_Number and term < 0:
        return f"(Num {float(-term)})"
    return None


def to_sexpr(e):
    if e.is_Symbol:
        return f'(Var "{e.name}")'
    if e.is_Number:
        return f"(Num {float(e)})"
    if e.is_Add:
        pos = [t for t in e.args if neg_of(t) is None]
        neg = [t for t in e.args if neg_of(t) is not None]
        if pos and neg:
            s = to_sexpr(pos[0])
            for t in pos[1:]:
                s = f"(Add {s} {to_sexpr(t)})"
            for t in neg:
                s = f"(Sub {s} {neg_of(t)})"
            return s
        s = to_sexpr(e.args[0])
        for t in e.args[1:]:
            s = f"(Add {s} {to_sexpr(t)})"
        return s
    if e.is_Mul:
        s = to_sexpr(e.args[0])
        for t in e.args[1:]:
            s = f"(Mul {s} {to_sexpr(t)})"
        return s
    if e.is_Pow:
        b, ex = e.args
        if ex == 2:
            return f"(Pow2 {to_sexpr(b)})"
        if ex == 3:
            return f"(Pow3 {to_sexpr(b)})"
        if ex.is_Integer and ex > 0:
            s = to_sexpr(b)
            for _ in range(int(ex) - 1):
                s = f"(Mul {s} {to_sexpr(b)})"
            return s
        if ex.is_Integer and ex < 0:
            pos = to_sexpr(b)
            for _ in range(-int(ex) - 1):
                pos = f"(Mul {pos} {to_sexpr(b)})"
            return f"(ProtectedInv {pos})"
        if ex == sp.Rational(1, 2):
            return f"(ProtectedSqrt {to_sexpr(b)})"
        if ex == sp.Rational(-1, 2):
            return f"(ProtectedInv (ProtectedSqrt {to_sexpr(b)}))"
        return f"(ProtectedExp (Mul {to_sexpr(ex)} (ProtectedLog {to_sexpr(b)})))"
    fns = {sp.sin: "Sin", sp.cos: "Cos", sp.exp: "ProtectedExp", sp.log: "ProtectedLog", sp.Abs: "Abs",
           sp.tanh: "Tanh", sp.asin: "ProtectedAsin", sp.acos: "ProtectedAcos", sp.tan: "Tan"}
    for k, v in fns.items():
        if e.func == k:
            return f"({v} {to_sexpr(e.args[0])})"
    raise ValueError(f"unhandled {e.func}")


def alarm(_s, _f):
    raise TimeoutError()


def sympy_verdict(model, truth, xnames):
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(SECONDS)
    try:
        ok, _why = C.srbench_solution(model, truth, xnames, 1.0)
        signal.alarm(0)
        return "pass" if ok else "fail"
    except TimeoutError:
        signal.alarm(0)
        return "undecided"


def main():
    rows = ["law\tvariables\tsympy\ttruth_infix\tmodel_infix\ttruth_sexpr\tmodel_sexpr"]
    for run, laws in RUNS.items():
        for law in laws:
            reg = FEYNMAN_REGISTRY[law.replace("feynman_", "")]
            names = columns(law)
            xnames = [f"x_{i}" for i in range(len(names))]
            truth = str(reg.truth_expr)
            for i, v in sorted(enumerate(names), key=lambda t: -len(t[1])):
                truth = re.sub(rf"\b{re.escape(v)}\b", f"x_{i}", truth)
            truth = truth.replace("pi", "3.141592653589793")
            local = {n: sp.Symbol(n) for n in xnames}
            truth_sexpr = to_sexpr(sp.sympify(truth, locals=local))
            model = next(line[6:].strip() for line in open(f"{LOGS}/{run}/{law}/run.log") if line.startswith("model:"))
            model_sexpr = to_sexpr(parse_expr(model.replace("^", "**"), local_dict=local))
            verdict = sympy_verdict(model, truth, xnames)
            print(f"{law}\t{verdict}", flush=True)
            rows.append("\t".join([law, ",".join(xnames), verdict, truth, model, truth_sexpr, model_sexpr]))
    with open(OUT, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

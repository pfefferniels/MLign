"""Pin the TISMIR 2023 §3.5 evaluation semantics (worked example, Table 1).

Prediction:   m(sn1,pn1), m(sn2,pn2)
Ground truth: d(sn1), i(pn1), m(sn2,pn2)
→ match TP {m(sn2,pn2)}, FP {m(sn1,pn1)} ⇒ match P = 1/2.
The paper's headline pools FN over ALL classes (R = 1/3); our per-class match
recall is 1/2 with the deletion/insertion misses surfacing as del-R = ins-R = 0.
Both views agree on what was right and wrong; we assert the per-class view and
derive the paper's pooled view in `pooled_prf` for exact comparability.

Run: .venv/bin/python -m pytest eval/test_metrics.py -q  (or plain python)
"""

from metrics import AlignmentScore, pooled_prf
from nasap import Alignment


def test_tismir_worked_example():
    pred = Alignment(
        matches=frozenset({("sn1", "pn1"), ("sn2", "pn2")}),
        insertions=frozenset(),
        deletions=frozenset(),
    )
    truth = Alignment(
        matches=frozenset({("sn2", "pn2")}),
        insertions=frozenset({"pn1"}),
        deletions=frozenset({"sn1"}),
    )
    s = AlignmentScore.compare(truth, pred)
    assert s.match.precision == 0.5
    assert s.match.recall == 1.0  # per-class: the 1 GT match was found
    assert s.deletion.recall == 0.0
    assert s.insertion.recall == 0.0

    p, r, f = pooled_prf(truth, pred)
    assert p == 0.5  # 1 TP / 2 predictions — paper's precision
    assert abs(r - 1 / 3) < 1e-12  # 1 TP / 3 GT labels — paper's recall


def test_empty_both_sides_is_perfect():
    empty = Alignment(frozenset(), frozenset(), frozenset())
    s = AlignmentScore.compare(empty, empty)
    assert s.match.fscore == 1.0
    assert s.insertion.fscore == 1.0
    assert s.deletion.fscore == 1.0


if __name__ == "__main__":
    test_tismir_worked_example()
    test_empty_both_sides_is_perfect()
    print("metrics tests OK")

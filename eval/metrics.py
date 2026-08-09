"""Alignment evaluation metrics, parangonar-compatible.

The parangonar papers (Peter et al.) score note alignments as F-measure over
the three label classes: a predicted (score_id, perf_id) match pair is correct
iff the ground truth contains exactly that pair; insertions/deletions are
correct iff the GT labels that note the same way. We report per-class P/R/F
and the match-class F as the headline number, macro-averaged over pieces the
way parangonar does (mean over performances).
"""

from __future__ import annotations

from dataclasses import dataclass

from nasap import Alignment


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    fscore: float
    n_true: int
    n_pred: int

    @classmethod
    def from_sets(cls, true: frozenset, pred: frozenset) -> "PRF":
        tp = len(true & pred)
        p = tp / len(pred) if pred else (1.0 if not true else 0.0)
        r = tp / len(true) if true else (1.0 if not pred else 0.0)
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return cls(p, r, f, len(true), len(pred))


@dataclass(frozen=True)
class AlignmentScore:
    match: PRF
    insertion: PRF
    deletion: PRF

    @classmethod
    def compare(cls, truth: Alignment, pred: Alignment) -> "AlignmentScore":
        return cls(
            match=PRF.from_sets(truth.matches, pred.matches),
            insertion=PRF.from_sets(truth.insertions, pred.insertions),
            deletion=PRF.from_sets(truth.deletions, pred.deletions),
        )

    def row(self) -> str:
        return (
            f"match F={self.match.fscore:.4f} (P={self.match.precision:.4f} "
            f"R={self.match.recall:.4f}, n={self.match.n_true}) | "
            f"ins F={self.insertion.fscore:.3f} (n={self.insertion.n_true}) | "
            f"del F={self.deletion.fscore:.3f} (n={self.deletion.n_true})"
        )


def pooled_prf(truth: Alignment, pred: Alignment) -> tuple[float, float, float]:
    """TISMIR's pooled counting (their Table 1): every predicted label is one
    prediction, every GT label one target, TP = exact label agreement."""
    tp = (
        len(truth.matches & pred.matches)
        + len(truth.insertions & pred.insertions)
        + len(truth.deletions & pred.deletions)
    )
    n_pred = len(pred.matches) + len(pred.insertions) + len(pred.deletions)
    n_true = len(truth.matches) + len(truth.insertions) + len(truth.deletions)
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_true if n_true else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def asynchrony(truth: Alignment, pred: Alignment, perf_onset_sec: dict) -> dict:
    """Timing agreement (parangonar's second metric family, note-level form):
    for every score note matched in BOTH truth and prediction, the absolute
    difference between the performed onsets of the two chosen performance
    notes. Perfect alignment → all zeros. Reports median + fraction within
    25/50/100 ms over the common score notes."""
    t_map = {s: p for (s, p) in truth.matches}
    p_map = {s: p for (s, p) in pred.matches}
    diffs = []
    for s, tp in t_map.items():
        pp = p_map.get(s)
        if pp is None or tp not in perf_onset_sec or pp not in perf_onset_sec:
            continue
        diffs.append(abs(perf_onset_sec[tp] - perf_onset_sec[pp]))
    if not diffs:
        return {"n": 0, "median_ms": None, "lt25": None, "lt50": None, "lt100": None}
    diffs.sort()
    n = len(diffs)
    frac = lambda t: sum(1 for d in diffs if d < t) / n  # noqa: E731
    return {
        "n": n,
        "median_ms": round(diffs[n // 2] * 1000, 2),
        "lt25": round(frac(0.025), 4),
        "lt50": round(frac(0.05), 4),
        "lt100": round(frac(0.1), 4),
    }


def macro_average(scores: list[AlignmentScore]) -> dict:
    """Mean-over-performances of each class F (parangonar's headline numbers)."""

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "match_f": mean([s.match.fscore for s in scores]),
        "match_p": mean([s.match.precision for s in scores]),
        "match_r": mean([s.match.recall for s in scores]),
        "insertion_f": mean([s.insertion.fscore for s in scores]),
        "deletion_f": mean([s.deletion.fscore for s in scores]),
        "n": len(scores),
    }

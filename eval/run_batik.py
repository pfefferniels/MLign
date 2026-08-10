"""Batik-plays-Mozart benchmark: 36 Mozart sonata movements, matchfile v1.0.0 GT.

Same protocol as run_eval (nASAP): load_match GT, aligner fed the MusicXML
score (merged+maximally unfolded) and the performance MIDI.

Usage:
  .venv/bin/python eval/run_batik.py --aligner baseline
  .venv/bin/python eval/run_batik.py --ckpt runs/v0-syn/best.pt --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import AlignmentScore, macro_average  # noqa: E402
from nasap import load_match  # noqa: E402
from run_eval import get_aligner, triples_to_alignment  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402

BATIK = ROOT / "data/benchmarks/batik_plays_mozart"


def perf_from_match(match_path: Path) -> PerfTable:
    import numpy as np

    from nasap import perf_notes_from_match

    records = perf_notes_from_match(match_path)
    arr = np.empty(
        len(records),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U32")],
    )
    for i, r in enumerate(records):
        arr[i] = (r["onset"], r["duration"], r["pitch"], r["velocity"], r["id"])
    return PerfTable(arr[np.lexsort((arr["pitch"], arr["onset"]))])


def score_from_match(match_path: Path) -> ScoreTable:
    import numpy as np

    from nasap import score_notes_from_match

    records = score_notes_from_match(match_path)
    out = np.empty(
        len(records),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
    )
    for i, r in enumerate(records):
        out[i] = (r["onset"], r["duration"], r["pitch"], r["voice"], r["id"])
    return ScoreTable(out[np.lexsort((out["pitch"], out["onset"]))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="baseline")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    name = f"model:{args.ckpt}" if args.ckpt else args.aligner
    if args.aligner in ("dualdtw", "automatic", "gluenote"):
        # Table-based adapter: parangonar must see the SAME match-file-derived
        # tables as the model (its default adapter re-parses MusicXML/MIDI and
        # lands in a different id space → scores 0 here).
        import contextlib
        import io
        import warnings

        import numpy as np
        import parangonar as pa

        matcher = {
            "dualdtw": pa.DualDTWNoteMatcher,
            "automatic": pa.AutomaticNoteMatcher,
            "gluenote": pa.TheGlueNoteMatcher,
        }[args.aligner]()

        def aligner(entry, score, perf, _m=matcher):
            sna = np.empty(len(score.notes), dtype=[
                ("onset_beat", "f8"), ("duration_beat", "f8"), ("pitch", "i4"),
                ("voice", "i4"), ("id", "U32"), ("is_grace", "b"),
            ])
            for dst, src in [("onset_beat", "onset"), ("duration_beat", "duration"),
                             ("pitch", "pitch"), ("voice", "voice"), ("id", "id")]:
                sna[dst] = score.notes[src]
            sna["is_grace"] = 0
            pna = np.empty(len(perf.notes), dtype=[
                ("onset_sec", "f8"), ("duration_sec", "f8"), ("pitch", "i4"),
                ("velocity", "i4"), ("id", "U32"),
            ])
            for dst, src in [("onset_sec", "onset"), ("duration_sec", "duration"),
                             ("pitch", "pitch"), ("velocity", "velocity"), ("id", "id")]:
                pna[dst] = perf.notes[src]
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                pred = _m(sna, pna)
            out = []
            for d in pred:
                if d["label"] == "match":
                    out.append({"label": "match", "score_id": d["score_id"], "perf_id": d["performance_id"]})
                elif d["label"] == "deletion":
                    out.append({"label": "deletion", "score_id": d["score_id"]})
                elif d["label"] in ("insertion", "ornament"):
                    out.append({"label": "insertion", "perf_id": d["performance_id"]})
            return out
    else:
        aligner = get_aligner(name)

    matches = sorted((BATIK / "match").glob("*.match"))
    if args.limit:
        matches = matches[: args.limit]

    scores, rows = [], []
    t0 = time.time()
    for mf in matches:
        stem = mf.stem
        entry = {
            "score": BATIK / "scores" / f"{stem}.musicxml",
            "midi": BATIK / "midi" / f"{stem}.mid",
        }
        if not entry["score"].exists() or not entry["midi"].exists():
            print(f"SKIP {stem}: missing files", file=sys.stderr)
            continue
        try:
            # BOTH sides from the match file: score = GT's performed unfolding;
            # perf = the note( lines (GT numbers notes NON-sequentially on many
            # movements — MIDI-parse remapping is wrong beyond kv279).
            score = score_from_match(mf)
            perf = perf_from_match(mf)
            triples = aligner(entry, score, perf)
            pred = triples_to_alignment(triples)
            truth = load_match(mf)
            s = AlignmentScore.compare(truth, pred)
            scores.append(s)
            rows.append({"piece": stem, "match_f": s.match.fscore})
            print(f"{stem}: {s.row()}", flush=True)
        except Exception as err:
            print(f"FAIL {stem}: {err}", file=sys.stderr)

    agg = macro_average(scores)
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

"""The MLign command-line product.

    mlign align SCORE PERFORMANCE.mid [-o OUT] [--format match|json|jsonl]
                [--ckpt CKPT]

SCORE: .mei (parsed via espressivo → exact xml:ids) or .musicxml/.xml
(partitura). Output formats:
  json   — internal representation (docs/DESIGN.md §2), default
  match  — partitura match file (parangonar-compatible)
  jsonl  — mpmify row schema mirror

Model checkpoint resolution: --ckpt, else $MLIGN_CKPT, else models/mlign-v3.pt
(the released model) relative to the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

ESPRESSIVO_DIST = "/Users/nielspfeffer/Projects/meico-ts/dist/index.js"

_MEI_BRIDGE = """
import { readFileSync } from 'node:fs';
import * as esp from %(dist)s;
const mei = readFileSync(process.argv[2], 'utf8');
const keep = console.log; console.log = () => {}; console.error = () => {};
const movements = esp.convertMeiToMsm(mei);
const { msm } = movements[%(mdiv)d];
const data = esp.extractScoreData ? esp.extractScoreData(msm) : null;
console.log = keep;
// Fallback: parse the MSM ourselves via performMsmToData-like path is heavy;
// use the score renderer's note discovery through renderMidi? Simplest robust
// path: regex over the MSM score elements (attribute order is fixed by the
// serializer).
const notes = [];
const partRe = /<part\\s[^>]*name="([^"]*)"[^>]*number="(\\d+)"[^>]*>([\\s\\S]*?)<\\/part>/g;
let pm;
while ((pm = partRe.exec(msm)) !== null) {
  const noteRe = /<note\\s([^>]*)\\/>/g;
  let nm;
  while ((nm = noteRe.exec(pm[3])) !== null) {
    const attrs = {};
    for (const m of nm[1].matchAll(/([\\w.:]+)="([^"]*)"/g)) attrs[m[1]] = m[2];
    if (attrs['midi.pitch'] === undefined) continue;
    notes.push({
      id: attrs['xml:id'] ?? null,
      date: Number(attrs['date']),
      duration: Number(attrs['duration']),
      pitch: Math.round(Number(attrs['midi.pitch'])),
      part: Number(pm[2]) - 1,
    });
  }
}
const ppqm = /pulsesPerQuarter="(\\d+)"/.exec(msm);
process.stdout.write(JSON.stringify({ ppq: ppqm ? Number(ppqm[1]) : 720, notes }));
"""


def load_score_table(path: Path, mdiv: int = 0):
    from .tables import ScoreTable

    if path.suffix.lower() == ".mei":
        script = _MEI_BRIDGE % {"dist": json.dumps(ESPRESSIVO_DIST), "mdiv": mdiv}
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
            fh.write(script)
            bridge = fh.name
        try:
            out = subprocess.run(
                ["node", bridge, str(path)], capture_output=True, text=True, check=True
            ).stdout
        finally:
            os.unlink(bridge)
        payload = json.loads(out[out.index("{") :])
        records = [n for n in payload["notes"]]
        for i, n in enumerate(records):
            if n["id"] is None:
                n["id"] = f"mlign_noid_{i}"
        return ScoreTable.from_records(records, ppq=payload["ppq"])
    return ScoreTable.from_musicxml(path)


def write_match(path: Path, triples: list[dict], score, perf) -> None:
    """Minimal matchfile v1.0.0 writer (ids only; timing from the tables)."""
    s_by_id = {str(n["id"]): n for n in score.notes}
    p_by_id = {str(n["id"]): n for n in perf.notes}
    lines = ["info(matchFileVersion,1.0.0)."]
    for t in triples:
        if t["label"] == "match":
            s = s_by_id[t["score_id"]]
            p = p_by_id[t["perf_id"]]
            lines.append(
                f"snote({t['score_id']},[-,-],0,0:0,0,0,{s['onset']:.4f},"
                f"{s['onset'] + s['duration']:.4f},[])-note({t['perf_id']},"
                f"{int(p['pitch'])},{int(p['onset'] * 1000)},"
                f"{int((p['onset'] + p['duration']) * 1000)},{int(p['velocity'])},0,0)."
            )
    for t in triples:
        if t["label"] == "deletion":
            s = s_by_id[t["score_id"]]
            lines.append(
                f"snote({t['score_id']},[-,-],0,0:0,0,0,{s['onset']:.4f},"
                f"{s['onset'] + s['duration']:.4f},[])-deletion."
            )
    for t in triples:
        if t["label"] == "insertion":
            p = p_by_id[t["perf_id"]]
            lines.append(
                f"insertion-note({t['perf_id']},{int(p['pitch'])},"
                f"{int(p['onset'] * 1000)},{int((p['onset'] + p['duration']) * 1000)},"
                f"{int(p['velocity'])},0,0)."
            )
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mlign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    al = sub.add_parser("align")
    al.add_argument("score")
    al.add_argument("performance")
    al.add_argument("-o", "--out", default="-")
    al.add_argument("--format", choices=["json", "match", "jsonl"], default="json")
    al.add_argument("--ckpt", default=os.environ.get("MLIGN_CKPT", str(ROOT / "models/mlign-v3.pt")))
    al.add_argument("--mdiv", type=int, default=0)
    al.add_argument("--engine", choices=["model", "baseline"], default="model")
    args = ap.parse_args(argv)

    from .tables import PerfTable

    score = load_score_table(Path(args.score), args.mdiv)
    perf = PerfTable.from_midi(args.performance)

    if args.engine == "baseline" or not Path(args.ckpt).exists():
        if args.engine == "model":
            print(f"checkpoint {args.ckpt} not found; falling back to baseline", file=sys.stderr)
        from .baseline import align_baseline

        triples = align_baseline(score, perf)
    else:
        import torch

        from .infer import align_with_model
        from .model import NoteAligner, config_from_ckpt

        device = "cpu"  # inference is light; CPU avoids MPS allocator quirks
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = NoteAligner(config_from_ckpt(cfg, ckpt["model"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        triples = align_with_model(model, score, perf, device)

    out = Path(args.out) if args.out != "-" else None
    if args.format == "json":
        text = json.dumps({"alignment": triples}, indent=2)
    elif args.format == "jsonl":
        text = "\n".join(json.dumps(t) for t in triples)
    else:
        if out is None:
            raise SystemExit("--format match requires -o FILE")
        write_match(out, triples, score, perf)
        print(f"wrote {out}")
        return
    if out is None:
        print(text)
    else:
        out.write_text(text)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

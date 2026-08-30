"""`mlign serve` — a tiny local HTTP server for the browser test page.

    PYTHONPATH=src python -m mlign.serve [--port 8765] [--ckpt models/mlign-v4.pt]

Serves web/index.html at / and accepts POST /align (multipart: `score` = .mei
or .musicxml file, `performance` = .mid file). Returns JSON:

    {
      "score":  [{"id", "onset", "duration", "pitch", "voice"}, ...],   # quarters
      "perf":   [{"id", "onset", "duration", "pitch", "velocity"}, ...], # seconds
      "alignment": [DESIGN §2 records with confidence],
      "stats": {"matches", "insertions", "deletions", "seconds"}
    }

Standard library only (http.server + cgi-free multipart parsing) so it runs in
the same venv as the CLI with no extra dependencies. Single-threaded on
purpose: the model is loaded once and inference is CPU-bound.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WEB = ROOT / "web"


def _parse_multipart(headers, body: bytes) -> dict[str, tuple[str, bytes]]:
    """→ {field: (filename, bytes)} using the stdlib email parser."""
    ctype = headers.get("Content-Type", "")
    raw = b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    msg = BytesParser(policy=email_default).parsebytes(raw)
    out: dict[str, tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        out[name] = (part.get_filename() or "", part.get_payload(decode=True) or b"")
    return out


class Aligner:
    def __init__(self, ckpt: str):
        import torch

        from .infer import align_with_model
        from .model import NoteAligner, config_from_ckpt

        self._align = align_with_model
        self.device = "cpu"
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = ck.get("config", {})
        self.model = NoteAligner(config_from_ckpt(cfg, ck["model"])).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ckpt = ckpt

    def run(self, score_path: Path, midi_path: Path) -> dict:
        from .cli import load_score_table
        from .tables import PerfTable

        t0 = time.time()
        score = load_score_table(score_path)
        perf = PerfTable.from_midi(midi_path)
        triples = self._align(self.model, score, perf, self.device)
        return {
            "score": [
                {"id": str(n["id"]), "onset": float(n["onset"]), "duration": float(n["duration"]),
                 "pitch": int(n["pitch"]), "voice": int(n["voice"])}
                for n in score.notes
            ],
            "perf": [
                {"id": str(n["id"]), "onset": float(n["onset"]), "duration": float(n["duration"]),
                 "pitch": int(n["pitch"]), "velocity": int(n["velocity"])}
                for n in perf.notes
            ],
            "alignment": triples,
            "stats": {
                "matches": sum(1 for t in triples if t["label"] == "match"),
                "insertions": sum(1 for t in triples if t["label"] == "insertion"),
                "deletions": sum(1 for t in triples if t["label"] == "deletion"),
                "seconds": round(time.time() - t0, 2),
                "checkpoint": os.path.basename(self.ckpt),
            },
        }


ALIGNER: Aligner | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = (WEB / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if path == "/health":
            return self._send(200, json.dumps({"ok": True, "checkpoint": ALIGNER.ckpt}).encode())
        if path.startswith("/demo/"):
            f = (WEB / "demo" / Path(path).name)
            if f.exists() and f.parent == WEB / "demo":
                return self._send(200, f.read_bytes(), "application/octet-stream")
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/align":
            return self._send(404, b'{"error":"not found"}')
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            parts = _parse_multipart(self.headers, body)
            if "score" not in parts or "performance" not in parts:
                return self._send(400, b'{"error":"need multipart fields score and performance"}')
            with tempfile.TemporaryDirectory() as td:
                s_name, s_bytes = parts["score"]
                p_name, p_bytes = parts["performance"]
                s_ext = Path(s_name).suffix.lower() or ".musicxml"
                score_path = Path(td) / f"score{s_ext}"
                midi_path = Path(td) / "performance.mid"
                score_path.write_bytes(s_bytes)
                midi_path.write_bytes(p_bytes)
                result = ALIGNER.run(score_path, midi_path)
            return self._send(200, json.dumps(result).encode())
        except Exception as err:  # surface to the page
            import traceback

            traceback.print_exc()
            return self._send(500, json.dumps({"error": f"{type(err).__name__}: {err}"}).encode())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mlign serve")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt", default=os.environ.get("MLIGN_CKPT", str(ROOT / "models/mlign-v4.pt")))
    args = ap.parse_args(argv)
    global ALIGNER
    ALIGNER = Aligner(args.ckpt)
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MLign test page: http://127.0.0.1:{args.port}/  (model {os.path.basename(args.ckpt)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

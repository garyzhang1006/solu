#!/usr/bin/env python3
"""
Solu AI tutor backend -- a thin proxy that keeps the Anthropic API key off the
browser and streams Claude Sonnet 4.6 explanations to the page.

The in-browser ML predictor (RDKit-JS + ONNX) is unchanged and still runs 100%
client-side. This server only adds the optional tutor:

  GET  /                serves web/ (the app, with the tutor card wired in)
  POST /api/tutor       {smiles, name, logS, klass, mgl, descriptors[], neighbors[],
                         question?} -> Server-Sent-Events stream of the explanation

Run:
    pip install anthropic        # (also in requirements.txt)
    export ANTHROPIC_API_KEY=sk-ant-...
    python tutor/server.py       # http://localhost:8000

The model is fixed to claude-sonnet-4-6 (the user's choice). Thinking is disabled
and effort is low so the tutor streams back fast -- it is explaining known
structure-property chemistry over data we hand it, not solving a hard problem.
"""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MODEL = "claude-sonnet-4-6"  # user-specified
PORT = int(os.environ.get("PORT", "8000"))

SYSTEM = """You are the tutor inside Solu, a tool that teaches WHY molecules dissolve in water.

A separate machine-learning model has already predicted this molecule's water solubility (logS). Your job is to explain the structure-property chemistry behind that number so a curious student understands it -- never to re-predict it.

Ground every claim in the data you are given:
- Tie the explanation to the molecule's actual descriptor values (LogP, polar surface area, H-bond donors/acceptors, weight, rings, etc.).
- Cite at least one of the provided "nearest measured molecules" -- these are real, published lab measurements of structurally similar compounds. Use them as evidence ("a close structural analogue was measured at logS ...").
- Use the "like dissolves like" idea: polar / H-bonding groups pull a molecule into water; greasy, large, flat-aromatic structure pushes it out.

Style: warm, concise, plain prose (no markdown headings, no bullet lists). Around 110-160 words unless the user's question needs more. These are educational estimates, not laboratory values -- say so if a student would over-trust them. If asked something off-topic, gently steer back to this molecule's solubility."""


def build_user_message(p):
    """Turn the prediction context the browser sends into one grounded prompt."""
    name = p.get("name") or "this molecule"
    lines = [
        f"Molecule: {name}",
        f"SMILES: {p.get('smiles', '?')}",
        f"Model prediction: logS = {p.get('logS')} ({p.get('klass', '')}), "
        f"roughly {p.get('mgl', '?')} in water.",
        "",
        "Computed descriptors (value -- where it sits across the dataset):",
    ]
    for d in p.get("descriptors", []):
        pct = d.get("percentile")
        where = f"{pct}th percentile" if pct is not None else "n/a"
        lines.append(f"  - {d.get('label')}: {d.get('value')}{d.get('unit', '')}  ({where})")

    neighbors = p.get("neighbors", [])
    if neighbors:
        lines.append("")
        lines.append("Nearest measured molecules (real published logS, your evidence):")
        for n in neighbors:
            lines.append(
                f"  - SMILES {n.get('smiles')}  measured logS {n.get('logS')}  [{n.get('source')}]"
            )

    q = (p.get("question") or "").strip()
    lines.append("")
    if q:
        lines.append(f"Student's question: {q}")
    else:
        lines.append(
            "Explain why this molecule has this predicted solubility -- the structure-property "
            "reasons -- and back it up with the nearest measured molecules above."
        )
    return "\n".join(lines)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, format, *args):  # quieter logs (signature matches base class)
        sys.stderr.write("  %s\n" % (format % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path.rstrip("/") != "/api/tutor":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self._cors()
            self.end_headers()
            self._sse({"error": "ANTHROPIC_API_KEY not set on the server. "
                                "export it and restart tutor/server.py."})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        try:
            client = anthropic.Anthropic()
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM,
                thinking={"type": "disabled"},        # snappy tutor; no deep reasoning needed
                output_config={"effort": "low"},       # concise, low-latency answers
                messages=[{"role": "user", "content": build_user_message(payload)}],
            ) as stream:
                for text in stream.text_stream:
                    self._sse({"text": text})
            self._sse({"done": True})
        except anthropic.APIError as e:
            self._sse({"error": f"Anthropic API error: {getattr(e, 'message', str(e))}"})
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream
        except Exception as e:  # noqa: BLE001 -- surface anything else to the page
            self._sse({"error": f"{type(e).__name__}: {e}"})


def main():
    if not WEB.exists():
        sys.exit(f"web/ not found at {WEB}")
    key = "set" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING (tutor will error)"
    print(f"Solu tutor on http://localhost:{PORT}  model={MODEL}  ANTHROPIC_API_KEY={key}")
    if not (WEB / "grounding.json").exists():
        print("  note: web/grounding.json missing -- run `python tutor/build_grounding.py` "
              "so the tutor can cite real molecules.")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

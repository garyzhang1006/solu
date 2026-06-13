"""
Bundle web/index.html + artifacts.json + model.onnx into one portable HTML file
(dist/solu.html). The two fetch() calls are replaced with inlined data so the file
runs from file:// with no local server (only the RDKit-JS / ORT CDN scripts load
remotely). Useful as a robust submission artifact.
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

html = (WEB / "index.html").read_text()
artifacts = (WEB / "artifacts.json").read_text()
model_b64 = base64.b64encode((WEB / "model.onnx").read_bytes()).decode()

data_script = (
    "<script>\n"
    "window.__ARTIFACTS__ = " + artifacts + ";\n"
    'window.__MODEL_B64__ = "' + model_b64 + '";\n'
    "</script>\n"
)

# inject data just before the app script, and swap the two fetch() calls for inlined data
marker = "<script>\nconst FEATURE_FALLBACK_LIB"
assert marker in html, "app script marker not found - did index.html change?"
html = html.replace(marker, data_script + "<script>\nconst FEATURE_FALLBACK_LIB", 1)

a = 'this.art = await (await fetch("./artifacts.json")).json();'
b = "this.art = window.__ARTIFACTS__;"
assert a in html, "artifacts fetch line not found"
html = html.replace(a, b, 1)

c = 'const buf = await (await fetch("./model.onnx")).arrayBuffer();'
d = "const buf = Uint8Array.from(atob(window.__MODEL_B64__), ch=>ch.charCodeAt(0)).buffer;"
assert c in html, "model fetch line not found"
html = html.replace(c, d, 1)

out = DIST / "solu.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, model {len(model_b64)/1024:.0f} KB base64)")

"""
Bundle web/index.html + app.css + app.js + artifacts.json + model.onnx into one
portable HTML file (dist/solu.html). External CSS/JS are inlined and the two
fetch() calls are replaced with inlined data, so the file runs from file:// with
no local server (only the RDKit-JS / ORT CDN scripts load remotely). Useful as a
robust submission artifact.

(The AI tutor's fetch("/api/tutor") still needs tutor/server.py + an API key; the
standalone runs prediction + "why" + quiz, and the tutor card degrades gracefully.)
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

html = (WEB / "index.html").read_text()
css = (WEB / "app.css").read_text()
js = (WEB / "app.js").read_text()
artifacts = (WEB / "artifacts.json").read_text()
model_b64 = base64.b64encode((WEB / "model.onnx").read_bytes()).decode()

# inline the two fetch() calls in the app JS before embedding it
a = 'this.art = await (await fetch("./artifacts.json")).json();'
b = "this.art = window.__ARTIFACTS__;"
assert a in js, "artifacts fetch line not found in app.js"
js = js.replace(a, b, 1)

c = 'const buf = await (await fetch("./model.onnx")).arrayBuffer();'
d = "const buf = Uint8Array.from(atob(window.__MODEL_B64__), ch=>ch.charCodeAt(0)).buffer;"
assert c in js, "model fetch line not found in app.js"
js = js.replace(c, d, 1)

# inline external CSS: <link ...app.css...> -> <style>...</style>
link = '<link rel="stylesheet" href="./app.css" />'
assert link in html, "app.css link not found in index.html"
html = html.replace(link, "<style>\n" + css + "\n</style>", 1)

# inline external JS (+ data) : <script src="./app.js"></script> -> data + inline app script
data_script = (
    "<script>\n"
    "window.__ARTIFACTS__ = " + artifacts + ";\n"
    'window.__MODEL_B64__ = "' + model_b64 + '";\n'
    "</script>\n"
)
script_tag = '<script src="./app.js"></script>'
assert script_tag in html, "app.js script tag not found in index.html"
html = html.replace(script_tag, data_script + "<script>\n" + js + "\n</script>", 1)

out = DIST / "solu.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, model {len(model_b64)/1024:.0f} KB base64)")

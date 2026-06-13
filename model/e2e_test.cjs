// End-to-end gate: the EXACT browser inference chain, run in Node.
// RDKit-JS computes the 9 descriptors -> onnxruntime-web runs model.onnx ->
// compare logS to the Python-side onnx_logS in the fixture. If this passes,
// the standalone web app will produce identical predictions to the trained model.
const fs = require("fs");
const path = require("path");
const initRDKitModule = require("@rdkit/rdkit");
const ort = require("onnxruntime-web");

const WEB = path.join(__dirname, "..", "web");
const FEAT = ["CrippenClogP", "amw", "tpsa", "NumHBD", "NumHBA",
  "NumRotatableBonds", "NumAromaticRings", "FractionCSP3", "NumRings"];

(async () => {
  const RDKit = await initRDKitModule();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = path.join(__dirname, "..", "node_modules", "onnxruntime-web", "dist") + path.sep;
  const modelBuf = fs.readFileSync(path.join(WEB, "model.onnx"));
  const session = await ort.InferenceSession.create(modelBuf);
  const inName = session.inputNames[0];
  const fixture = JSON.parse(fs.readFileSync(path.join(WEB, "parity_fixture.json"), "utf8"));

  let worst = 0;
  console.log("smiles                          ort-web   python    diff");
  for (const rec of fixture) {
    const mol = RDKit.get_mol(rec.smiles);
    const d = JSON.parse(mol.get_descriptors());
    mol.delete();
    const x = Float32Array.from(FEAT.map((k) => d[k]));
    const tensor = new ort.Tensor("float32", x, [1, FEAT.length]);
    const out = await session.run({ [inName]: tensor });
    const pred = out[session.outputNames[0]].data[0];
    const diff = Math.abs(pred - rec.onnx_logS);
    worst = Math.max(worst, diff);
    console.log(`${rec.smiles.slice(0, 30).padEnd(31)} ${pred.toFixed(4).padStart(8)} ${rec.onnx_logS.toFixed(4).padStart(8)} ${diff.toExponential(2).padStart(9)}`);
  }
  const PASS = worst < 1e-3;
  console.log(`\nworst |ort-web - python onnx| = ${worst.toExponential(3)} -> ${PASS ? "PASS" : "FAIL"}`);
  process.exit(PASS ? 0 : 1);
})().catch((e) => { console.error("e2e error:", e); process.exit(2); });

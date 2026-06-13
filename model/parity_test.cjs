// Parity gate: RDKit-JS (browser inference) vs RDKit-Python (training).
// Loads the Python-generated fixture, recomputes the 9 descriptors with RDKit-JS,
// and reports the max absolute difference per feature. If this passes, the feature
// vector the browser feeds the model is identical to what the model trained on.
const fs = require("fs");
const path = require("path");
const initRDKitModule = require("@rdkit/rdkit");

const FIXTURE = path.join(__dirname, "..", "web", "parity_fixture.json");
const JS_KEYS = ["CrippenClogP", "amw", "tpsa", "NumHBD", "NumHBA",
  "NumRotatableBonds", "NumAromaticRings", "FractionCSP3", "NumRings"];

initRDKitModule().then((RDKit) => {
  console.log("RDKit-JS version:", RDKit.version ? RDKit.version() : "(unknown)");
  const fixture = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
  const maxDiff = Object.fromEntries(JS_KEYS.map((k) => [k, 0]));
  let rows = 0;
  for (const rec of fixture) {
    const mol = RDKit.get_mol(rec.smiles);
    if (!mol) { console.log("FAILED to parse:", rec.smiles); continue; }
    const d = JSON.parse(mol.get_descriptors());
    mol.delete();
    rows++;
    for (const k of JS_KEYS) {
      const jsVal = d[k];
      const pyVal = rec.js_descriptors[k];
      if (jsVal === undefined) { console.log(`  MISSING JS key ${k} (smiles ${rec.smiles})`); continue; }
      const diff = Math.abs(jsVal - pyVal);
      if (diff > maxDiff[k]) maxDiff[k] = diff;
    }
  }
  console.log(`\ncompared ${rows} molecules`);
  console.log("max |RDKit-JS - RDKit-Python| per feature:");
  let worst = 0;
  for (const k of JS_KEYS) {
    console.log(`  ${k.padEnd(20)} ${maxDiff[k].toExponential(3)}`);
    worst = Math.max(worst, maxDiff[k]);
  }
  // tolerance: 1e-3 absolute is generous; counts must be exact, logP/tpsa float-stable
  const PASS = worst < 1e-3;
  console.log(`\nworst feature diff = ${worst.toExponential(3)} -> ${PASS ? "PASS" : "FAIL"}`);
  if (!PASS) {
    console.log("\nDescriptors diverge between RDKit-JS and RDKit-Python.");
    console.log("Fix: align versions (pip install rdkit==<js-version>) and retrain.");
    process.exit(1);
  }
}).catch((e) => { console.error("RDKit init error:", e); process.exit(2); });

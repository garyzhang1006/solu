# Solu — Demo Video Script (~75 seconds)

Goal: hit all four judging criteria — Idea, Implementation, Design, Presentation.
Tip: record at 1280×900, light mode, browser already loaded so there's no wait.

---

## [0:00–0:10] Hook — the problem
**Screen:** Hero section.
**Narration:** "Students memorize that 'like dissolves like' — but never *see* why. Solu turns water solubility into something you can explore. And it's not a chatbot: it's a real machine-learning model trained on published lab data, running entirely in your browser."

## [0:10–0:25] Core demo — predict
**Screen:** Click the **Aspirin** chip, then **Caffeine**, then type a custom SMILES.
**Narration:** "Pick any molecule. RDKit draws the structure and computes its chemical descriptors live, and a gradient-boosting model predicts how well it dissolves — here, log-solubility, and a real-world milligrams-per-liter value." 
**Show:** the verdict + gauge animating.

## [0:25–0:45] The teaching moment — "Why?"
**Screen:** Scroll to the **Why?** contribution bars; hover a couple of properties.
**Narration:** "This is the part that teaches. Every prediction is broken down: LogP — lipophilicity — pushes solubility down; polar surface area and hydrogen-bond donors push it up. Hover any property to learn what it means. Students see the *reasoning*, not just an answer."

## [0:45–0:58] Active learning — quiz + contrast
**Screen:** Pick **DDT** (insoluble), use **Test yourself** to guess, then **Table sugar** (soluble).
**Narration:** "Quiz mode makes you commit before the reveal. Compare greasy DDT — practically insoluble — with sugar, loaded with hydroxyl groups and highly soluble. The model's reasons line up with the chemistry."

## [0:58–1:10] Implementation credibility
**Screen:** Scroll to **How it works** + metrics (R² 0.88, RMSE 0.75, 1,128 molecules).
**Narration:** "Under the hood: trained on the ESOL dataset from chemistry researchers, R-squared 0.88 — beating the classic textbook equation. The model is exported to ONNX and verified to predict bit-for-bit identically in the browser. No server, no API, no LLM."

## [1:10–1:15] Close
**Screen:** Back to hero.
**Narration:** "Solu — see why molecules dissolve."

---

## One-liner for the submission
"Solu is an in-browser AI chemistry tutor: a gradient-boosting model trained on the published ESOL solubility dataset predicts any molecule's water solubility and teaches the structure–property reasoning — deterministically, with no LLM, verified identical between training and browser."

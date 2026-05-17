const scorer = require("./scorer.cjs");
const evidenceRules = scorer.EVIDENCE_RULES;
const labelSelection = require("./label_selection.cjs");

module.exports = {
  ...scorer,
  ...evidenceRules,
  ...labelSelection,
};
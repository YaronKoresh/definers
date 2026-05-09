const scorer = require("./scorer.cjs");
const evidenceRules = require("./evidence_rules.cjs");
const labelSelection = require("./label_selection.cjs");
const semver = require("./semver.cjs");

module.exports = {
  ...scorer,
  ...evidenceRules,
  ...labelSelection,
  ...semver
};
const { deriveSemverDecision, SEMVER_CHANNEL_DEFINITIONS, SEMVER_THRESHOLDS } = require("./scorer.cjs");

module.exports = {
  deriveSemverDecision,
  SEMVER_CHANNEL_DEFINITIONS,
  SEMVER_THRESHOLDS
};
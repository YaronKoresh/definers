const { createNode } = require("./helpers.cjs");

const ROLLOUT_TREE = createNode("rollout", { color: "c2e0c6", description: "Feature rollout changes.", legacyMatches: ["feature-flag"], releaseRelevant: true }, [
  createNode("feature toggle", { color: "c2e0c6", description: "Feature toggle changes.", legacyMatches: ["feature-flag"], releaseRelevant: true }, [
    createNode("kill switch", { color: "c2e0c6", description: "Kill switch changes.", legacyMatches: ["feature-flag"], releaseRelevant: true }),
    createNode("default state", { color: "c2e0c6", description: "Default flag state changes.", legacyMatches: ["feature-flag"], releaseRelevant: true }),
    createNode("cohort rule", { color: "c2e0c6", description: "Cohort or rollout targeting changes.", legacyMatches: ["feature-flag"], releaseRelevant: true })
  ])
]);

module.exports = {
  ROLLOUT_TREE
};
function normalizeSuiteKey(value) {
  return String(value || "").trim().toUpperCase();
}

function normalizeScenarioRegressionFixture(input) {
  const suiteKey = normalizeSuiteKey(input.suiteKey);
  return {
    fixtureId: String(input.fixtureId || `${suiteKey.toLowerCase()}-${String(input.kind || "scenario")}`),
    kind: String(input.kind || ""),
    materialization: {
      count: Number(input.materialization?.count || 1),
      seed: Number(input.materialization?.seed || 1)
    },
    reason: String(input.reason || ""),
    replayKey: input.replayKey ? String(input.replayKey) : null,
    severity: input.severity ? String(input.severity) : null,
    suiteKey
  };
}

const COMMITTED_REGRESSION_FIXTURES = Object.freeze([
  {
    fixtureId: "autobot-patch-unavailable-feature",
    kind: "patch-unavailable-feature",
    materialization: { count: 13, seed: 20260413 },
    reason: "Preserves additive classification when GitHub omits raw patches.",
    suiteKey: "AUTOBOT"
  },
  {
    fixtureId: "autobot-adversarial-empty-payload",
    kind: "adversarial-empty-payload",
    materialization: { count: 17, seed: 20260413 },
    reason: "Guards mandatory maintenance-label fallback when GitHub omits patch payloads.",
    suiteKey: "AUTOBOT"
  },
  {
    fixtureId: "autobot-pipeline-label-race-recovery",
    kind: "label-race-recovery",
    materialization: { count: 13, seed: 20260413 },
    reason: "Replays the transient label provisioning race against the live pipeline path.",
    suiteKey: "AUTOBOT_PIPELINE"
  },
  {
    fixtureId: "unified-pr-untrusted-fork",
    kind: "unified-pr-untrusted-fork",
    materialization: { count: 7, seed: 20260413 },
    reason: "Keeps smart-link suppressed for untrusted forks while Autobot continues to run.",
    suiteKey: "UNIFIED_AUTOMATION"
  },
  {
    fixtureId: "unified-issue-stale-reopened",
    kind: "unified-issue-stale-reopened",
    materialization: { count: 7, seed: 20260413 },
    reason: "Suppresses stale explicit targets after a reopened issue is reactivated.",
    suiteKey: "UNIFIED_AUTOMATION"
  }
].map((entry) => Object.freeze(normalizeScenarioRegressionFixture(entry))));

function getCommittedRegressionFixtures(suiteKey) {
  const normalizedSuiteKey = normalizeSuiteKey(suiteKey);
  return COMMITTED_REGRESSION_FIXTURES
    .filter((entry) => entry.suiteKey === normalizedSuiteKey)
    .map((entry) => normalizeScenarioRegressionFixture(entry));
}

function formatRegressionFixtureEntry(entry) {
  const normalized = normalizeScenarioRegressionFixture(entry);
  const lines = [
    "{",
    `  fixtureId: "${normalized.fixtureId}",`,
    `  kind: "${normalized.kind}",`,
    `  materialization: { count: ${normalized.materialization.count}, seed: ${normalized.materialization.seed} },`,
    `  reason: "${normalized.reason.replace(/\\/g, "\\\\").replace(/"/g, "\\\"")}",`,
    `  suiteKey: "${normalized.suiteKey}"`
  ];
  if (normalized.replayKey) {
    lines.splice(lines.length - 1, 0, `  replayKey: "${normalized.replayKey}",`);
  }
  if (normalized.severity) {
    lines.splice(lines.length - 1, 0, `  severity: "${normalized.severity}",`);
  }
  lines.push("}");
  return lines.join("\n");
}

module.exports = {
  COMMITTED_REGRESSION_FIXTURES,
  formatRegressionFixtureEntry,
  getCommittedRegressionFixtures,
  normalizeScenarioRegressionFixture,
  normalizeSuiteKey
};
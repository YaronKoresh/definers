const assert = require("node:assert/strict");
const { test } = require("./javascript_test_harness.cjs");
const {
  COMMITTED_REGRESSION_FIXTURES,
  formatRegressionFixtureEntry
} = require("./deterministic_scenario_regression_fixtures.cjs");
const {
  buildRegressionMaterializationRuns,
  buildScenarioCaptureCandidate,
  deriveRotatingRegressionSeed,
  deriveRotatingScenarioSeed,
  evaluateScenarioLane,
  normalizeScenarioLane,
  resolveScenarioLanePlan
} = require("./deterministic_scenario_lanes.cjs");

test("scenario lane normalization defaults to deterministic", () => {
  assert.equal(normalizeScenarioLane(), "deterministic");
  assert.equal(normalizeScenarioLane(" exploratory "), "exploratory");
  assert.equal(normalizeScenarioLane("unknown"), "deterministic");
});

test("exploratory seeds rotate deterministically from the rotation key", () => {
  const first = deriveRotatingScenarioSeed("AUTOBOT", "2026-04-13");
  const second = deriveRotatingScenarioSeed("AUTOBOT", "2026-04-13");
  const third = deriveRotatingScenarioSeed("AUTOBOT", "2026-04-14");

  assert.equal(first, second);
  assert.notEqual(first, third);
});

test("committed regression fixtures are grouped by suite seed and count", () => {
  const runs = buildRegressionMaterializationRuns(COMMITTED_REGRESSION_FIXTURES);
  const unifiedRun = runs.find((entry) => entry.suiteKey === "UNIFIED_AUTOMATION");

  assert.ok(unifiedRun);
  assert.equal(unifiedRun.config.seed, 20260413);
  assert.equal(unifiedRun.config.count, 7);
  assert.equal(unifiedRun.fixtures.length, 2);
});

test("exploratory regression fixtures stay active and rotate on exploratory seeds", () => {
  const runs = buildRegressionMaterializationRuns(COMMITTED_REGRESSION_FIXTURES, {
    lane: "exploratory",
    rotationKey: "2026-04-13"
  });
  const autobotRun = runs.find((entry) => entry.suiteKey === "AUTOBOT");

  assert.ok(autobotRun);
  assert.equal(autobotRun.config.seed, deriveRotatingRegressionSeed("AUTOBOT", "2026-04-13"));
  assert.notEqual(autobotRun.config.seed, COMMITTED_REGRESSION_FIXTURES[0].materialization.seed);
  assert.ok(autobotRun.config.count >= COMMITTED_REGRESSION_FIXTURES[0].materialization.count);
});

test("capture candidates preserve promotion-ready materialization data", () => {
  const candidate = buildScenarioCaptureCandidate({
    evaluation: {
      acceptedOutcome: false,
      acceptedReasons: [],
      actualResult: {
        labels: ["runtime"],
        semverDecision: "major"
      },
      deviations: [],
      kind: "runtime-drop",
      name: "autobot:runtime-drop:1:seed-20260412",
      optimalResult: {
        requiredLabels: ["breaking-change", "runtime"],
        semverDecision: "major"
      },
      scenario: {
        replayKey: "AUTOBOT:seed-20260412:scenario-1:runtime-drop"
      },
      warnings: [{ code: "unexpected-critical-label", message: "unexpected critical label runtime present outside the scenario contract" }]
    },
    lane: "exploratory",
    suiteConfig: { count: 36, seed: 20260412 },
    suiteKey: "AUTOBOT"
  });

  assert.equal(candidate.materialization.seed, 20260412);
  assert.equal(candidate.materialization.count, 36);
  assert.equal(candidate.severity, "warning");
  assert.equal(candidate.suiteKey, "AUTOBOT");
  assert.ok(candidate.fixtureId.includes("runtime-drop"));
});

test("fixture formatter emits a ready-to-paste registry object", () => {
  const text = formatRegressionFixtureEntry(COMMITTED_REGRESSION_FIXTURES[0]);

  assert.ok(text.includes('fixtureId: "autobot-patch-unavailable-feature"'));
  assert.ok(text.includes('suiteKey: "AUTOBOT"'));
});

test("lane planning keeps deterministic suites fixed and exploratory suites rotating", () => {
  const deterministicPlan = resolveScenarioLanePlan({
    includeBaselineSuites: true,
    includeRegressionFixtures: false,
    lane: "deterministic",
    suiteKeys: ["AUTOBOT"]
  });
  const exploratoryPlan = resolveScenarioLanePlan({
    includeBaselineSuites: true,
    includeRegressionFixtures: false,
    lane: "exploratory",
    rotationKey: "2026-04-13",
    suiteKeys: ["AUTOBOT"]
  });

  assert.equal(deterministicPlan.baselineRuns[0].config.seed, 20260412);
  assert.notEqual(exploratoryPlan.baselineRuns[0].config.seed, deterministicPlan.baselineRuns[0].config.seed);
});

test("exploratory lane keeps regression fixtures enabled by default", () => {
  const plan = resolveScenarioLanePlan({
    lane: "exploratory",
    rotationKey: "2026-04-13",
    suiteKeys: ["AUTOBOT", "UNIFIED_AUTOMATION"]
  });

  assert.equal(plan.includeRegressionFixtures, true);
  assert.ok(plan.regressionRuns.length >= 2);
  assert.ok(plan.regressionRuns.every((run) => run.config.seed !== 20260413));
});

test("deterministic lane replays committed regression fixtures cleanly", async () => {
  const result = await evaluateScenarioLane({
    includeBaselineSuites: false,
    lane: "deterministic",
    suiteKeys: ["AUTOBOT_PIPELINE", "UNIFIED_AUTOMATION"]
  });

  assert.equal(result.failed, false, result.reportText);
  assert.equal(result.summary.deviationCount, 0, result.reportText);
  assert.equal(result.summary.warningCount, 0, result.reportText);
  assert.ok(result.regressionRuns.length >= 2);
});
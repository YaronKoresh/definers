const { analyzePullRequestSnapshotData } = require("../.github/scripts/autobot/pr_analysis.cjs");
const { analyzeCandidatePair } = require("../.github/scripts/autobot/smart_link/core.cjs");
const {
  evaluateAutobotPipelineScenarioSuite,
  evaluateAutobotScenarioSuite,
  formatScenarioEvaluationBlock,
  evaluateSmartLinkScenarioSuite,
  evaluateUnifiedAutomationScenarioSuite,
  summarizeScenarioEvaluations
} = require("./deterministic_scenario_engine.cjs");
const {
  getCommittedRegressionFixtures,
  normalizeScenarioRegressionFixture,
  normalizeSuiteKey
} = require("./deterministic_scenario_regression_fixtures.cjs");

const DEFAULT_SCENARIO_LANE = "deterministic";
const SCENARIO_LANE_KEYS = Object.freeze(["deterministic", "exploratory", "capture"]);
const SUITE_KEYS = Object.freeze(["AUTOBOT", "AUTOBOT_PIPELINE", "SMART_LINK", "UNIFIED_AUTOMATION"]);
const DETERMINISTIC_LANE_SUITE_CONFIGS = Object.freeze({
  AUTOBOT: Object.freeze({ count: 36, seed: 20260412 }),
  AUTOBOT_PIPELINE: Object.freeze({ count: 18, seed: 20260412 }),
  SMART_LINK: Object.freeze({ count: 30, seed: 20260412 }),
  UNIFIED_AUTOMATION: Object.freeze({ count: 50, seed: 20260412 })
});
const EXPLORATORY_LANE_COUNTS = Object.freeze({
  AUTOBOT: 48,
  AUTOBOT_PIPELINE: 24,
  SMART_LINK: 40,
  UNIFIED_AUTOMATION: 64
});

function cloneJson(value) {
  if (value === undefined) {
    return undefined;
  }
  return JSON.parse(JSON.stringify(value));
}

function parseBooleanFlag(value, fallback) {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function uniqueStrings(values) {
  const result = [];
  const seen = new Set();
  for (const entry of Array.isArray(values) ? values : [values]) {
    const normalized = String(entry || "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function normalizeScenarioLane(value) {
  const normalized = String(value || DEFAULT_SCENARIO_LANE).trim().toLowerCase();
  return SCENARIO_LANE_KEYS.includes(normalized) ? normalized : DEFAULT_SCENARIO_LANE;
}

function hashToSeed(value) {
  let hash = 2166136261;
  for (const symbol of String(value || "")) {
    hash ^= symbol.charCodeAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash || 1;
}

function buildAutomaticRotationKey(input = {}) {
  const nowMs = Number.isFinite(Number(input.nowMs)) ? Number(input.nowMs) : Date.now();
  const pid = Number.isFinite(Number(input.pid)) ? Number(input.pid) : process.pid;
  const hrtime = String(
    input.hrtime
      || (typeof process.hrtime?.bigint === "function" ? process.hrtime.bigint() : nowMs)
  );
  return `run:${nowMs}:${pid}:${hrtime}`;
}

function normalizeRotationKey(value, input = {}) {
  const normalized = String(value || "").trim();
  return normalized || buildAutomaticRotationKey(input);
}

function deriveRotatingScenarioSeed(suiteKey, rotationKey) {
  return hashToSeed(`${normalizeSuiteKey(suiteKey)}:${normalizeRotationKey(rotationKey)}`);
}

function deriveRotatingRegressionSeed(suiteKey, rotationKey) {
  return hashToSeed(`${normalizeSuiteKey(suiteKey)}:regression:${normalizeRotationKey(rotationKey)}`);
}

function resolveIncludeRegressionFixtures(input, lane, env = process.env) {
  if (input !== undefined) {
    return parseBooleanFlag(input, true);
  }
  const envValue = env.SCENARIO_INCLUDE_REGRESSION_FIXTURES;
  if (String(envValue || "").trim()) {
    return parseBooleanFlag(envValue, true);
  }
  return true;
}

function getDefaultSuiteConfig(suiteKey) {
  return DETERMINISTIC_LANE_SUITE_CONFIGS[normalizeSuiteKey(suiteKey)] || { count: 12, seed: 20260412 };
}

function resolveSuiteOverride(suiteOverrides, suiteKey) {
  return suiteOverrides?.[suiteKey]
    || suiteOverrides?.[suiteKey.toLowerCase()]
    || {};
}

function resolveLaneSuiteConfig(suiteKey, lane, input = {}) {
  const normalizedSuiteKey = normalizeSuiteKey(suiteKey);
  const baseConfig = getDefaultSuiteConfig(normalizedSuiteKey);
  const override = resolveSuiteOverride(input.suiteOverrides, normalizedSuiteKey);
  if (lane === "exploratory") {
    return {
      count: Number(override.count || EXPLORATORY_LANE_COUNTS[normalizedSuiteKey] || baseConfig.count),
      seed: Number(override.seed || deriveRotatingScenarioSeed(normalizedSuiteKey, input.rotationKey))
    };
  }
  return {
    count: Number(override.count || baseConfig.count),
    seed: Number(override.seed || baseConfig.seed)
  };
}

function buildRegressionMaterializationRuns(fixtures, options = {}) {
  const lane = normalizeScenarioLane(options.lane || "deterministic");
  if (lane === "exploratory") {
    const runsBySuite = new Map();
    for (const rawFixture of fixtures || []) {
      const fixture = normalizeScenarioRegressionFixture(rawFixture);
      if (!runsBySuite.has(fixture.suiteKey)) {
        const override = resolveSuiteOverride(options.suiteOverrides, fixture.suiteKey);
        const fixtureCount = Math.max(...(fixtures || [])
          .map((entry) => normalizeScenarioRegressionFixture(entry))
          .filter((entry) => entry.suiteKey === fixture.suiteKey)
          .map((entry) => entry.materialization.count), 1);
        runsBySuite.set(fixture.suiteKey, {
          config: {
            count: Number(override.count || Math.max(EXPLORATORY_LANE_COUNTS[fixture.suiteKey] || fixtureCount, fixtureCount)),
            seed: Number(override.seed || deriveRotatingRegressionSeed(fixture.suiteKey, options.rotationKey))
          },
          fixtures: [],
          suiteKey: fixture.suiteKey
        });
      }
      runsBySuite.get(fixture.suiteKey).fixtures.push(fixture);
    }
    return [...runsBySuite.values()];
  }
  const groups = new Map();
  for (const rawFixture of fixtures || []) {
    const fixture = normalizeScenarioRegressionFixture(rawFixture);
    const groupKey = [fixture.suiteKey, fixture.materialization.seed, fixture.materialization.count].join(":");
    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        config: {
          count: fixture.materialization.count,
          seed: fixture.materialization.seed
        },
        fixtures: [],
        suiteKey: fixture.suiteKey
      });
    }
    groups.get(groupKey).fixtures.push(fixture);
  }
  return [...groups.values()];
}

function resolveScenarioLanePlan(input = {}) {
  const lane = normalizeScenarioLane(input.lane || process.env.SCENARIO_LANE);
  const includeBaselineSuites = input.includeBaselineSuites !== false;
  const includeRegressionFixtures = resolveIncludeRegressionFixtures(input.includeRegressionFixtures, lane, input.env || process.env);
  const explicitRotationKey = String(input.rotationKey || process.env.SCENARIO_ROTATION_KEY || "").trim();
  const rotationKey = lane === "exploratory"
    ? normalizeRotationKey(explicitRotationKey, input.rotationContext)
    : null;
  const rotationKeyMode = lane === "exploratory"
    ? explicitRotationKey ? "explicit" : "automatic"
    : "fixed";
  const selectedSuiteKeys = uniqueStrings(input.suiteKeys && input.suiteKeys.length > 0 ? input.suiteKeys : SUITE_KEYS).map((entry) => normalizeSuiteKey(entry));
  const baselineRuns = includeBaselineSuites
    ? selectedSuiteKeys.map((suiteKey) => ({
        config: resolveLaneSuiteConfig(suiteKey, lane, {
          rotationKey,
          suiteOverrides: input.suiteOverrides || {}
        }),
        mode: "baseline",
        suiteKey
      }))
    : [];
  const regressionFixtures = includeRegressionFixtures
    ? selectedSuiteKeys.flatMap((suiteKey) => getCommittedRegressionFixtures(suiteKey))
    : [];
  return {
    baselineRuns,
    includeBaselineSuites,
    includeRegressionFixtures,
    lane,
    regressionRuns: buildRegressionMaterializationRuns(regressionFixtures, {
      lane,
      rotationKey,
      suiteOverrides: input.suiteOverrides || {}
    }),
    rotationKey,
    rotationKeyMode,
    strict: lane === "deterministic",
    suiteKeys: selectedSuiteKeys
  };
}

async function evaluateScenarioSuiteByKey(suiteKey, config) {
  const normalizedSuiteKey = normalizeSuiteKey(suiteKey);
  if (normalizedSuiteKey === "AUTOBOT") {
    return evaluateAutobotScenarioSuite({
      analyzeScenario: analyzePullRequestSnapshotData,
      config
    });
  }
  if (normalizedSuiteKey === "AUTOBOT_PIPELINE") {
    return evaluateAutobotPipelineScenarioSuite({ config });
  }
  if (normalizedSuiteKey === "SMART_LINK") {
    return evaluateSmartLinkScenarioSuite({
      analyzeScenario: analyzeCandidatePair,
      config
    });
  }
  if (normalizedSuiteKey === "UNIFIED_AUTOMATION") {
    return evaluateUnifiedAutomationScenarioSuite({ config });
  }
  throw new Error(`Unsupported suite key: ${suiteKey}`);
}

function summarizeSelectedEvaluations(suiteKey, evaluations) {
  return summarizeScenarioEvaluations(evaluations, { suiteKey: normalizeSuiteKey(suiteKey) });
}

function extractCaptureSeverity(evaluation) {
  if ((evaluation.deviations || []).length > 0) {
    return "deviation";
  }
  if ((evaluation.warnings || []).length > 0) {
    return "warning";
  }
  if (evaluation.acceptedOutcome) {
    return "accepted";
  }
  return "pass";
}

function buildScenarioCaptureCandidate(input) {
  const severity = extractCaptureSeverity(input.evaluation);
  const messages = severity === "deviation"
    ? input.evaluation.deviations.slice()
    : severity === "warning"
      ? (input.evaluation.warnings || []).map((warning) => warning.message)
      : (input.evaluation.acceptedReasons || []).slice();
  const replayKey = input.evaluation.scenario?.replayKey || null;
  return {
    actualResult: cloneJson(input.evaluation.actualResult),
    captureVersion: 1,
    fixtureId: `${normalizeSuiteKey(input.suiteKey).toLowerCase()}-${input.evaluation.kind}-${input.suiteConfig.seed}-${severity}`.replace(/[^a-z0-9-]+/g, "-"),
    kind: input.evaluation.kind,
    lane: normalizeScenarioLane(input.lane),
    materialization: {
      count: Number(input.suiteConfig.count),
      seed: Number(input.suiteConfig.seed)
    },
    messages,
    name: input.evaluation.name,
    optimalResult: cloneJson(input.evaluation.optimalResult),
    reason: messages[0] || `${severity} captured from ${input.evaluation.kind}`,
    replayKey,
    scenario: cloneJson(input.evaluation.scenario),
    severity,
    suiteKey: normalizeSuiteKey(input.suiteKey)
  };
}

function collectScenarioCaptureCandidates(input) {
  const captureSeverities = new Set(input.captureSeverities || ["deviation", "warning"]);
  return (input.evaluations || [])
    .filter((evaluation) => captureSeverities.has(extractCaptureSeverity(evaluation)))
    .map((evaluation) => buildScenarioCaptureCandidate({
      evaluation,
      lane: input.lane,
      suiteConfig: input.suiteConfig,
      suiteKey: input.suiteKey
    }));
}

async function evaluateRegressionRun(run, lane) {
  const suite = await evaluateScenarioSuiteByKey(run.suiteKey, run.config);
  const selectedEvaluations = [];
  const materializationIssues = [];
  for (const fixture of run.fixtures) {
    const evaluation = suite.evaluations.find((entry) => entry.kind === fixture.kind);
    if (!evaluation) {
      materializationIssues.push(`missing regression fixture ${fixture.fixtureId} (${fixture.kind}) for seed ${run.config.seed} and count ${run.config.count}`);
      continue;
    }
    selectedEvaluations.push(evaluation);
  }
  return {
    config: run.config,
    evaluations: selectedEvaluations,
    fixtures: run.fixtures,
    lane,
    materializationIssues,
    mode: "regression",
    suiteKey: run.suiteKey,
    summary: summarizeSelectedEvaluations(run.suiteKey, selectedEvaluations)
  };
}

function summarizeLaneRuns(baseRuns, regressionRuns) {
  return [...baseRuns, ...regressionRuns].reduce((summary, run) => {
    summary.baseRunCount += run.mode === "baseline" ? 1 : 0;
    summary.regressionRunCount += run.mode === "regression" ? 1 : 0;
    summary.totalScenarios += run.summary.totalScenarios;
    summary.deviationCount += run.summary.deviationCount;
    summary.deviationScenarioCount += run.summary.deviationScenarioCount;
    summary.warningCount += run.summary.warningCount;
    summary.warningScenarioCount += run.summary.warningScenarioCount;
    summary.acceptedOutcomeCount += run.summary.acceptedOutcomeCount;
    summary.falseGreenCount += run.summary.falseGreenCount || 0;
    summary.materializationIssueCount += (run.materializationIssues || []).length;
    return summary;
  }, {
    acceptedOutcomeCount: 0,
    baseRunCount: 0,
    deviationCount: 0,
    deviationScenarioCount: 0,
    falseGreenCount: 0,
    materializationIssueCount: 0,
    regressionRunCount: 0,
    totalScenarios: 0,
    warningCount: 0,
    warningScenarioCount: 0
  });
}

function formatRunHeader(run) {
  if (run.mode === "baseline") {
    return `${run.mode.toUpperCase()} ${run.suiteKey} seed=${run.config.seed} count=${run.config.count}`;
  }
  return `${run.mode.toUpperCase()} ${run.suiteKey} seed=${run.config.seed} count=${run.config.count} fixtures=${run.fixtures.length}`;
}

function collectProblemEvaluations(run) {
  return (run.evaluations || []).filter((evaluation) => (evaluation.deviations || []).length > 0 || (evaluation.warnings || []).length > 0);
}

function formatScenarioLaneReport(input) {
  const lines = [
    `Scenario lane: ${input.lane}`,
    `Rotation key: ${input.rotationKey || "(fixed)"}`,
    `Rotation mode: ${input.rotationKeyMode || "fixed"}`,
    `Baseline runs: ${input.summary.baseRunCount}`,
    `Regression runs: ${input.summary.regressionRunCount}`,
    `Regression fixture replay: ${input.includeRegressionFixtures ? "enabled" : "disabled"}`,
    `Scenarios audited: ${input.summary.totalScenarios}`,
    `Deviation items: ${input.summary.deviationCount}`,
    `Warning items: ${input.summary.warningCount}`,
    `Accepted outcomes: ${input.summary.acceptedOutcomeCount}`,
    `False greens: ${input.summary.falseGreenCount}`,
    `Fixture materialization issues: ${input.summary.materializationIssueCount}`,
    `Capture candidates: ${input.captureCandidates.length}`,
    ""
  ];
  for (const run of [...input.baseRuns, ...input.regressionRuns]) {
    const problemEvaluations = collectProblemEvaluations(run);
    lines.push(formatRunHeader(run));
    lines.push(`Scenarios: ${run.summary.totalScenarios}`);
    lines.push(`Deviation items: ${run.summary.deviationCount}`);
    lines.push(`Warning items: ${run.summary.warningCount}`);
    if ((run.materializationIssues || []).length > 0) {
      lines.push("Materialization issues:");
      for (const issue of run.materializationIssues) {
        lines.push(`- ${issue}`);
      }
    }
    if (run.summary.deviationCount === 0 && run.summary.warningCount === 0 && (run.materializationIssues || []).length === 0) {
      lines.push("Status: clean");
    }
    if (problemEvaluations.length > 0) {
      lines.push("Problem scenarios:");
      for (const evaluation of problemEvaluations) {
        lines.push(...formatScenarioEvaluationBlock(evaluation));
        lines.push("");
      }
    }
    lines.push("");
  }
  if (input.captureCandidates.length > 0) {
    lines.push("Promotion-ready capture candidates:");
    for (const candidate of input.captureCandidates.slice(0, 12)) {
      lines.push(`- ${candidate.fixtureId} (${candidate.suiteKey}/${candidate.kind})`);
      lines.push(`  replay=${candidate.replayKey || "(unavailable)"}`);
      lines.push(`  materialization seed=${candidate.materialization.seed} count=${candidate.materialization.count}`);
      lines.push(`  ${candidate.reason}`);
    }
    if (input.captureCandidates.length > 12) {
      lines.push(`- ... ${input.captureCandidates.length - 12} more candidates omitted`);
    }
    lines.push("");
  }
  lines.push(`Gate result: ${input.failed ? "failed" : "passed"}`);
  return lines.join("\n");
}

async function evaluateScenarioLane(input = {}) {
  const plan = resolveScenarioLanePlan(input);
  const baseRuns = [];
  for (const run of plan.baselineRuns) {
    const suite = await evaluateScenarioSuiteByKey(run.suiteKey, run.config);
    baseRuns.push({
      config: run.config,
      evaluations: suite.evaluations,
      lane: plan.lane,
      mode: "baseline",
      suiteKey: run.suiteKey,
      summary: suite.summary
    });
  }
  const regressionRuns = [];
  for (const run of plan.regressionRuns) {
    regressionRuns.push(await evaluateRegressionRun(run, plan.lane));
  }
  const captureCandidates = [
    ...baseRuns.flatMap((run) => collectScenarioCaptureCandidates({
      evaluations: run.evaluations,
      lane: plan.lane,
      suiteConfig: run.config,
      suiteKey: run.suiteKey
    })),
    ...regressionRuns.flatMap((run) => collectScenarioCaptureCandidates({
      evaluations: run.evaluations,
      lane: plan.lane,
      suiteConfig: run.config,
      suiteKey: run.suiteKey
    }))
  ];
  const summary = summarizeLaneRuns(baseRuns, regressionRuns);
  const failed = plan.strict && (summary.deviationCount > 0 || summary.warningCount > 0 || summary.materializationIssueCount > 0);
  return {
    ...plan,
    baseRuns,
    captureCandidates,
    failed,
    regressionRuns,
    reportText: formatScenarioLaneReport({
      baseRuns,
      captureCandidates,
      failed,
      lane: plan.lane,
      includeRegressionFixtures: plan.includeRegressionFixtures,
      regressionRuns,
      rotationKey: plan.rotationKey,
      rotationKeyMode: plan.rotationKeyMode,
      summary
    }),
    summary
  };
}

module.exports = {
  buildRegressionMaterializationRuns,
  buildScenarioCaptureCandidate,
  collectScenarioCaptureCandidates,
  buildAutomaticRotationKey,
  deriveRotatingScenarioSeed,
  deriveRotatingRegressionSeed,
  evaluateScenarioLane,
  formatScenarioLaneReport,
  getDefaultSuiteConfig,
  normalizeScenarioLane,
  resolveScenarioLanePlan
};
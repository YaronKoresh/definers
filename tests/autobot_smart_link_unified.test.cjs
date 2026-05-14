const assert = require("node:assert/strict");
const { test } = require("./javascript_test_harness.cjs");
const { evaluateUnifiedAutomationScenarioSuite } = require("./deterministic_scenario_engine.cjs");

const {
  buildCombinedAutobotLabels,
  deriveLinkedAutobotLabels,
  shouldAnalyzeSmartLinks,
  shouldPrepareAutobotProjectState
} = require("../.github/scripts/autobot/unified.cjs");

test("linked close-family issues enrich PR labels deterministically", () => {
  const labels = deriveLinkedAutobotLabels({
    emittedResults: [
      {
        candidate: {
          labels: ["documentation", "bug"],
          number: 41
        },
        relationKind: "closes"
      },
      {
        candidate: {
          labels: ["workflow"],
          number: 77
        },
        relationKind: "connects"
      }
    ],
    source: {
      kind: "pull_request"
    }
  });

  assert.deepEqual(labels, ["bug", "documentation"]);
});

test("advisory-fix links propagate security even without candidate labels", () => {
  const labels = deriveLinkedAutobotLabels({
    emittedResults: [
      {
        candidate: {
          labels: [],
          number: 18
        },
        relationKind: "advisory_fix"
      }
    ],
    source: {
      kind: "pull_request"
    }
  });

  assert.deepEqual(labels, ["security"]);
});

test("weak connect-only links do not backfill labels by themselves", () => {
  const labels = deriveLinkedAutobotLabels({
    emittedResults: [
      {
        candidate: {
          labels: ["workflow"],
          number: 99
        },
        relationKind: "connects"
      }
    ],
    source: {
      kind: "pull_request"
    }
  });

  assert.deepEqual(labels, []);
});

test("combined autobot labels keep priority order across direct and linked evidence", () => {
  const labels = buildCombinedAutobotLabels(["workflow", "documentation"], ["bug", "workflow", "security"]);

  assert.deepEqual(labels, ["security", "bug", "documentation", "workflow"]);
});

test("unified gating keeps smart-link broad and autobot project sync scoped", () => {
  const featureBranchPr = {
    eventName: "pull_request",
    payload: {
      action: "opened",
      pull_request: {
        base: {
          ref: "feature/next"
        }
      }
    }
  };

  const issueEvent = {
    eventName: "issues",
    payload: {
      action: "opened"
    }
  };

  assert.equal(shouldAnalyzeSmartLinks(featureBranchPr), true);
  assert.equal(shouldPrepareAutobotProjectState(featureBranchPr), false);
  assert.equal(shouldAnalyzeSmartLinks(issueEvent), true);
  assert.equal(shouldPrepareAutobotProjectState(issueEvent), true);
});

test("unified automation scenario engine keeps 50 end-to-end runs aligned with optimal expectations", async () => {
  const suite = await evaluateUnifiedAutomationScenarioSuite();

  console.log(suite.reportText);
  assert.equal(suite.summary.totalScenarios, 50, suite.reportText);
  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.deviationScenarioCount, 0, suite.reportText);
  assert.ok(suite.summary.metrics.semverConfusion.patch.patch >= 1, suite.reportText);
  assert.equal(typeof suite.summary.falseGreenRate, "number");
});

test("untrusted fork scenarios keep autobot active while suppressing smart-link analysis", async () => {
  const suite = await evaluateUnifiedAutomationScenarioSuite({
    config: {
      count: 7,
      seed: 20260413
    }
  });
  const evaluation = suite.evaluations.find((entry) => entry.kind === "unified-pr-untrusted-fork");

  assert.ok(evaluation, suite.reportText);
  assert.equal(evaluation.actualResult.autobotReady, true);
  assert.equal(evaluation.actualResult.smartLinkReady, false);
  assert.equal(evaluation.actualResult.graphCommentPresent, false);
  assert.equal(evaluation.actualResult.smartLinkCommentPresent, false);
});

test("reopened stale issue scenarios suppress historic smart-link targets while keeping active dependencies", async () => {
  const suite = await evaluateUnifiedAutomationScenarioSuite({
    config: {
      count: 7,
      seed: 20260413
    }
  });
  const evaluation = suite.evaluations.find((entry) => entry.kind === "unified-issue-stale-reopened");

  assert.ok(evaluation, suite.reportText);
  assert.equal(evaluation.actualResult.autobotReady, true);
  assert.equal(evaluation.actualResult.smartLinkReady, true);
  assert.equal(evaluation.actualResult.smartLinkCommentPresent, true);
  assert.ok(evaluation.rawResult.smartLinkCommentBody.includes("Depends On"));
  assert.ok(!evaluation.rawResult.smartLinkCommentBody.includes(evaluation.scenario.expected.smartLinkCommentExcludes[0]));
  assert.equal(evaluation.passed, true, suite.reportText);
});
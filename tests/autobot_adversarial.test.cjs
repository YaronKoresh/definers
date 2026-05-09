const assert = require("node:assert/strict");
const { test } = require("./javascript_test_harness.cjs");
const {
  analyzePullRequestSnapshotData
} = require("../.github/scripts/autobot/pr_analysis.cjs");
const {
  createAdversarialConflictingEvidenceSnapshot,
  createAdversarialCrossDomainFalsePositiveSnapshot,
  createAdversarialEmptyPayloadSnapshot,
  createAdversarialMaxBoundarySnapshot
} = require("./deterministic_scenario_engine.cjs");
const {
  AutobotLabelRegistry,
  normalizeLabelName
} = require("../.github/scripts/autobot/labels.cjs");

function hasExpectedLabel(labels, expectedLabel) {
  return (labels || []).some((label) => AutobotLabelRegistry.matchesExpectedLabel(label, expectedLabel));
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return function nextRandom() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function assertNoForbiddenLabels(labels, forbidden, scenarioName) {
  for (const label of forbidden) {
    const normalizedForbidden = normalizeLabelName(label);
    const found = labels.some((emitted) => normalizeLabelName(emitted) === normalizedForbidden);
    assert.equal(found, false, `${scenarioName}: forbidden label "${label}" was emitted in [${labels.join(", ")}]`);
  }
}

function assertRequiredAnyLabels(labels, requiredAnyGroups, scenarioName) {
  for (const group of requiredAnyGroups) {
    const found = group.some((expected) => hasExpectedLabel(labels, expected));
    assert.ok(found, `${scenarioName}: none of [${group.join(", ")}] found in [${labels.join(", ")}]`);
  }
}

test("adversarial: empty payload does not produce API-family labels", () => {
  const random = seededRandom(20260414);
  const scenario = createAdversarialEmptyPayloadSnapshot(random, 0, 20260414);
  const outputs = analyzePullRequestSnapshotData(scenario.input);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

  assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  assertRequiredAnyLabels(deterministicLabels, scenario.expected.requiredAnyLabels, scenario.name);
});

test("adversarial: cross-domain false positive rejects API labels from workflow files containing github.rest", () => {
  const random = seededRandom(20260414);
  const scenario = createAdversarialCrossDomainFalsePositiveSnapshot(random, 0, 20260414);
  const outputs = analyzePullRequestSnapshotData(scenario.input);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

  assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  assertRequiredAnyLabels(deterministicLabels, scenario.expected.requiredAnyLabels, scenario.name);
  assert.equal(outputs.release_relevant, "false", "workflow-only PR should not be release-relevant");
});

test("adversarial: conflicting evidence in autobot infrastructure does not leak API labels", () => {
  const random = seededRandom(20260414);
  const scenario = createAdversarialConflictingEvidenceSnapshot(random, 0, 20260414);
  const outputs = analyzePullRequestSnapshotData(scenario.input);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

  assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  assertRequiredAnyLabels(deterministicLabels, scenario.expected.requiredAnyLabels, scenario.name);
});

test("adversarial: max boundary workflow patch does not emit API labels", () => {
  const random = seededRandom(20260414);
  const scenario = createAdversarialMaxBoundarySnapshot(random, 0, 20260414);
  const outputs = analyzePullRequestSnapshotData(scenario.input);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

  assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  assertRequiredAnyLabels(deterministicLabels, scenario.expected.requiredAnyLabels, scenario.name);
});

test("adversarial: 50 seeded cross-domain false positive scenarios all reject API labels", () => {
  for (let scenarioIndex = 0; scenarioIndex < 50; scenarioIndex += 1) {
    const seed = 20260414 + scenarioIndex;
    const random = seededRandom(seed);
    const scenario = createAdversarialCrossDomainFalsePositiveSnapshot(random, scenarioIndex, seed);
    const outputs = analyzePullRequestSnapshotData(scenario.input);
    const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

    assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  }
});

test("adversarial: 50 seeded conflicting evidence scenarios all reject API labels", () => {
  for (let scenarioIndex = 0; scenarioIndex < 50; scenarioIndex += 1) {
    const seed = 20260414 + scenarioIndex;
    const random = seededRandom(seed);
    const scenario = createAdversarialConflictingEvidenceSnapshot(random, scenarioIndex, seed);
    const outputs = analyzePullRequestSnapshotData(scenario.input);
    const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

    assertNoForbiddenLabels(deterministicLabels, scenario.expected.forbiddenLabels, scenario.name);
  }
});

test("adversarial: original pr-size.yml false positive scenario is now rejected", () => {
  const snapshot = {
    pullRequest: {
      number: 42,
      title: "Overhaul autobot workflow system",
      body: "",
      headRef: "autobot/overhaul"
    },
    totals: {
      filesChanged: 3,
      additions: 120,
      deletions: 60,
      totalChanges: 180
    },
    files: [
      {
        filename: ".github/workflows/pr-size.yml",
        status: "modified",
        additions: 40,
        deletions: 20,
        patch: [
          "+    uses: actions/github-script@v9",
          "+    with:",
          "+      script: |",
          "+        const { owner, repo } = context.repo;",
          "+        await github.rest.issues.createComment({",
          "+          owner,",
          "+          repo,",
          "+          issue_number: context.issue.number,",
          "+          body: `PR size warning: ${totalChanges} changes`",
          "+        });"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 50,
        deletions: 25,
        patch: [
          "+      - name: Run scenario coverage",
          "+        run: node --test tests/autobot_scripts.test.cjs",
          "+      - uses: actions/github-script@v9",
          "+        with:",
          "+          script: |",
          "+            const { analyzeUnifiedAutomationState } = require(helperPath);"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: ".github/scripts/autobot/pr_analysis.cjs",
        status: "modified",
        additions: 30,
        deletions: 15,
        patch: [
          "+function deriveTechnicalLabels(filesWithContext) {",
          "+  const labels = new Map();",
          "+  return labels;",
          "+}"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const outputs = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("rest"), `"rest" must not appear in [${deterministicLabels.join(", ")}]`);
  assert.ok(!deterministicLabels.includes("api"), `"api" must not appear in [${deterministicLabels.join(", ")}]`);
  assert.ok(!deterministicLabels.includes("route"), `"route" must not appear in [${deterministicLabels.join(", ")}]`);
  assert.ok(
    hasExpectedLabel(deterministicLabels, "github") || hasExpectedLabel(deterministicLabels, "workflow"),
    `expected github or workflow label in [${deterministicLabels.join(", ")}]`
  );
});

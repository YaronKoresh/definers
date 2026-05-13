const assert = require("node:assert/strict");
const { test } = require("./javascript_test_harness.cjs");
const {
  evaluateSmartLinkScenarioSuite
} = require("./deterministic_scenario_engine.cjs");

const {
  analyzeCandidatePair,
  buildDirectiveSummary,
  buildMermaidGraphLines,
  extractExplicitReferenceSignals,
  normalizeSmartLinkEntity,
  selectSmartLinkResults
} = require("../.github/scripts/autobot/smart_link/core.cjs");
const { stripManagedBlock } = require("../.github/scripts/autobot/smart_link/renderer.cjs");

function createEntity(overrides = {}) {
  return normalizeSmartLinkEntity({
    body: "",
    changedFiles: [],
    kind: "issue",
    labels: [],
    number: 1,
    repoFullName: "octo/example",
    state: "open",
    title: "",
    updatedAt: "2026-04-10T00:00:00Z",
    ...overrides
  });
}

test("explicit reference extraction treats implements as a close-family signal", () => {
  const result = extractExplicitReferenceSignals({
    body: "Implements #42 and connects to #43.",
    branch: "feature/42-smart-link",
    repoFullName: "octo/example",
    title: "Implements #42"
  });

  assert.deepEqual(result.implementIds, [42]);
  assert.deepEqual(result.connectIds, [43]);
  assert.ok(result.branchIds.includes(42));
});

test("explicit implements on an open issue emits a closes relation above threshold 80", () => {
  const source = createEntity({
    body: "Implements #42.",
    changedFiles: ["src/repo/runtime.py"],
    kind: "pull_request",
    labels: ["enhancement"],
    number: 7,
    title: "Implement runtime support"
  });
  const candidate = createEntity({
    body: "Runtime support feature request",
    labels: ["enhancement"],
    number: 42,
    title: "Add runtime support"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.relationKind, "closes");
  assert.ok(result.emittedScore >= 80);
  assert.deepEqual(result.suppressionReasons, []);
});

test("close-family references to pull requests are downgraded to connects", () => {
  const source = createEntity({
    body: "Implements #51.",
    kind: "pull_request",
    number: 8,
    title: "Implement shared refactor"
  });
  const candidate = createEntity({
    body: "Shared refactor work",
    kind: "pull_request",
    number: 51,
    title: "Refactor shared flow"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.relationKind, "connects");
  assert.ok(result.emittedScore >= 80);
});

test("lexical-only similarity is suppressed even when words overlap", () => {
  const source = createEntity({
    body: "Waveform alignment tonal contour balancing.",
    kind: "issue",
    number: 10,
    title: "Waveform alignment tonal contour balancing"
  });
  const candidate = createEntity({
    body: "Contour balancing waveform alignment improvements.",
    kind: "issue",
    number: 11,
    title: "Waveform contour alignment improvements"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.thresholdPassed, false);
  assert.ok(result.suppressionReasons.includes("lexical-only"));
});

test("alert identifier and remediation reference emit advisory fix above threshold 80", () => {
  const source = createEntity({
    alertIdentifiers: ["CVE-2026-1234"],
    body: "CVE-2026-1234 affects package:repo-core.",
    ecosystemSignals: ["python"],
    kind: "security_alert",
    packageSignals: ["package:repo-core"],
    remediationReferences: [77],
    title: "Security advisory for repo-core"
  });
  const candidate = createEntity({
    alertIdentifiers: ["CVE-2026-1234"],
    body: "Remediates CVE-2026-1234 in repo-core.",
    ecosystemSignals: ["python"],
    kind: "pull_request",
    number: 77,
    packageSignals: ["package:repo-core"],
    title: "Patch CVE-2026-1234"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.relationKind, "advisory_fix");
  assert.ok(result.emittedScore >= 80);
  assert.deepEqual(result.suppressionReasons, []);
});

test("close-family references to closed issues are suppressed", () => {
  const source = createEntity({
    body: "Closes #19.",
    kind: "pull_request",
    number: 9,
    title: "Close stale issue"
  });
  const candidate = createEntity({
    body: "Already fixed.",
    number: 19,
    state: "closed",
    title: "Issue already resolved"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.thresholdPassed, false);
  assert.ok(result.suppressionReasons.includes("close-target-not-open"));
});

test("stale explicit connect targets are suppressed even with matching structure", () => {
  const source = createEntity({
    body: "Connects to #29.",
    kind: "issue",
    labels: ["bug"],
    milestone: { number: 5, title: "Tracking runtime" },
    number: 15,
    title: "Runtime bug reopened",
    updatedAt: "2026-04-12T00:00:00Z"
  });
  const candidate = createEntity({
    body: "References #15.",
    kind: "pull_request",
    labels: ["bug"],
    milestone: { number: 5, title: "Tracking runtime" },
    number: 29,
    title: "Historic runtime workaround",
    updatedAt: "2024-01-10T00:00:00Z"
  });

  const result = analyzeCandidatePair({ candidate, source, threshold: 80 });

  assert.equal(result.thresholdPassed, false);
  assert.ok(result.suppressionReasons.includes("stale-target"));
});

test("selection keeps deterministic ordering for tied scores", () => {
  const source = createEntity({
    body: "References #20 and #21.",
    kind: "issue",
    number: 5,
    title: "Track linked work"
  });
  const candidate20 = createEntity({
    body: "References #5.",
    number: 20,
    title: "Track linked work"
  });
  const candidate21 = createEntity({
    body: "References #5.",
    number: 21,
    title: "Track linked work"
  });

  const emitted = selectSmartLinkResults({
    candidateResults: [
      analyzeCandidatePair({ candidate: candidate21, source, threshold: 80 }),
      analyzeCandidatePair({ candidate: candidate20, source, threshold: 80 })
    ],
    threshold: 80
  });

  assert.deepEqual(emitted.map((result) => result.candidate.number), [20, 21]);
});

test("directive summary only emits issue close directives", () => {
  const source = createEntity({
    body: "Implements #40 and implements #41.",
    kind: "pull_request",
    number: 12,
    title: "Implement linked work"
  });
  const issueCandidate = createEntity({
    body: "Implementation target",
    kind: "issue",
    number: 40,
    title: "Issue target"
  });
  const pullCandidate = createEntity({
    body: "Implementation target",
    kind: "pull_request",
    number: 41,
    title: "PR target"
  });
  const emitted = selectSmartLinkResults({
    candidateResults: [
      analyzeCandidatePair({ candidate: issueCandidate, source, threshold: 80 }),
      analyzeCandidatePair({ candidate: pullCandidate, source, threshold: 80 })
    ],
    threshold: 80
  });

  const summary = buildDirectiveSummary(emitted);

  assert.deepEqual(summary.closeIds, [40]);
  assert.deepEqual(summary.connectIds, [41]);
});

test("mermaid graph lines are deterministic for emitted results", () => {
  const source = createEntity({
    body: "References #30.",
    kind: "issue",
    number: 3,
    title: "Source issue"
  });
  const candidate = createEntity({
    body: "References #3.",
    kind: "pull_request",
    number: 30,
    title: "Target PR"
  });
  const emitted = selectSmartLinkResults({
    candidateResults: [analyzeCandidatePair({ candidate, source, threshold: 80 })],
    threshold: 80
  });

  const lines = buildMermaidGraphLines(source, emitted);

  assert.ok(lines[0].includes("graph TD"));
  assert.ok(lines.some((line) => line.includes("ISSUE3")));
  assert.ok(lines.some((line) => line.includes("PR30")));
});

test("managed body stripping removes only the smart-link block", () => {
  const body = [
    "User content above.",
    "",
    "<!-- smart-autolinker:start -->",
    "### Smart Link Intelligence",
    "Managed block",
    "<!-- smart-autolinker:end -->",
    "",
    "User content below."
  ].join("\n");

  assert.equal(stripManagedBlock(body), "User content above.\n\nUser content below.");
});

test("string alert identifiers are preserved in normalized entities", () => {
  const entity = createEntity({
    id: "GHSA-abcd-efgh-ijkl",
    kind: "security_alert",
    title: "GHSA alert"
  });

  assert.equal(entity.id, "GHSA-abcd-efgh-ijkl");
});

test("smart-link scenario engine keeps actual output aligned with optimal expectations", () => {
  const suite = evaluateSmartLinkScenarioSuite({
    analyzeScenario: analyzeCandidatePair
  });

  console.log(suite.reportText);
  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.deviationScenarioCount, 0, suite.reportText);
  assert.ok(suite.summary.metrics.relationConfusion.closes.closes >= 1, suite.reportText);
});
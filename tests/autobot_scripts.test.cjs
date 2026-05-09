const assert = require("node:assert/strict");
const { test } = require("./javascript_test_harness.cjs");
const {
  evaluateAutobotPipelineScenarioSuite,
  evaluateAutobotScenarioSuite,
  evaluateUnifiedAutomationScenarioSuite,
  summarizeScenarioEvaluations,
  validateAutobotScenarioContract
} = require("./deterministic_scenario_engine.cjs");
const { createAutobotPipelineGithubMock } = require("./autobot_pipeline_test_support.cjs");

const {
  AutobotLabelRegistry,
  MAX_AUTOBOT_LABELS,
  normalizeLabelName,
  trimLowSignalLabels
} = require("../.github/scripts/autobot/labels.cjs");
const {
  AutobotDeterministicScorer,
  scoreDeterministicEvidence
} = require("../.github/scripts/autobot/measurement/scorer.cjs");
const {
  AutobotIssueClassifier,
  analyzeIssueIntake
} = require("../.github/scripts/autobot/issue_intake.cjs");
const {
  AutobotPromptBuilder,
  buildIssueSummaryArtifacts
} = require("../.github/scripts/autobot/prompts.cjs");
const {
  AutobotPullRequestAnalyzer,
  analyzePullRequestSnapshotData
} = require("../.github/scripts/autobot/pr_analysis.cjs");
const {
  AutobotProjectManager,
  resolvePrLabelDelta
} = require("../.github/scripts/autobot/project_manager.cjs");
const {
  analyzeCurrentGitDiff,
  buildLocalGitDiffSnapshot,
  formatCaptureFixtureFromPayload,
  formatLocalDiffAnalysis
} = require("../scripts/print_autobot_regression_fixture.cjs");

function hasExpectedLabel(labels, expectedLabel) {
  return (labels || []).some((label) => AutobotLabelRegistry.matchesExpectedLabel(label, expectedLabel));
}

function lineIncludesAny(source, fragments) {
  return (fragments || []).some((fragment) => String(source || "").includes(fragment));
}

test("label registry normalizes canonical separators", () => {
  assert.equal(normalizeLabelName("breaking_change"), "breaking-change");
  assert.equal(AutobotLabelRegistry.normalizeLabelName("Breaking Change"), "breaking-change");
});

test("label registry trims low-signal labels with version-sensitive priority", () => {
  const labels = trimLowSignalLabels([
    "documentation",
    "config",
    "bug",
    "workflow",
    "enhancement",
    "cleanup",
    "runtime",
    "security",
    "test",
    "schema",
    "api",
    "dependencies",
    "breaking_change",
    "tooling"
  ]);

  assert.ok(labels.length <= MAX_AUTOBOT_LABELS);
  assert.equal(labels[0], "breaking-change");
  assert.ok(labels.includes("security"));
  assert.ok(labels.includes("api"));
  assert.ok(!labels.includes("breaking_change"));
});

test("deterministic scorer promotes destructive public export removal to major", () => {
  const result = scoreDeterministicEvidence([{ ruleId: "removed-public-export" }]);

  assert.equal(result.semver.decision, "major");
  assert.ok(result.labelScores["breaking-change"].retained);
  assert.ok(result.emittedLabels.some((entry) => entry.label === "breaking-change"));
  assert.equal(AutobotDeterministicScorer.scoreDeterministicEvidence([{ ruleId: "removed-public-export" }]).semver.decision, "major");
});

test("deterministic scorer recognizes technical security posture rules", () => {
  const result = scoreDeterministicEvidence([
    { ruleId: "security-vulnerability" },
    { ruleId: "security-compliance" },
    { ruleId: "security-hardening" },
    { ruleId: "security-pen-test" }
  ]);

  assert.ok(result.emittedLabels.some((entry) => entry.label === "vulnerability"));
  assert.ok(result.emittedLabels.some((entry) => entry.label === "compliance"));
  assert.ok(result.emittedLabels.some((entry) => entry.label === "hardening"));
  assert.ok(result.emittedLabels.some((entry) => entry.label === "pen-test"));
  assert.ok(hasExpectedLabel(result.emittedLabels.map((entry) => entry.label), "security"));
});

test("issue classifier detects documentation issues from the template", () => {
  const issue = {
    title: "Documentation issue report: missing README section",
    body: [
      "Documentation issue report",
      "Link(s) to the affected documentation",
      "Detailed description of the problem",
      "The section is outdated"
    ].join("\n")
  };

  const result = analyzeIssueIntake(issue);
  const classResult = AutobotIssueClassifier.analyzeIssueIntake(issue);

  assert.deepEqual(result.labels, ["readme"]);
  assert.equal(result.releaseRelevant, false);
  assert.ok(result.evidenceItems.some((item) => item.ruleId === "issue-documentation-report"));
  assert.deepEqual(classResult.labels, result.labels);
});

test("project manager keeps broad issue classifications alongside technical labels", () => {
  const labels = AutobotProjectManager.inferIssueLabels({
    title: "Bug: runtime failure on Windows",
    body: [
      "Thank you for helping us squash this bug",
      "Detailed steps to reproduce",
      "Potential causes / workarounds / related issues (optional)",
      "Custom modifications / configuration"
    ].join("\n")
  });

  assert.ok(hasExpectedLabel(labels, "windows"));
  assert.ok(!labels.includes("bug"));
});

test("issue classifier emits technical security posture labels", () => {
  const issue = {
    title: "Security advisory: token hardening after pen-test for PCI compliance",
    body: [
      "Security advisory",
      "The pen-test found a vulnerability in the token flow.",
      "The remediation adds hardening for PCI compliance.",
      "This closes the advisory follow-up."
    ].join("\n")
  };

  const result = analyzeIssueIntake(issue);

  assert.ok(result.labels.includes("vulnerability"));
  assert.ok(result.labels.includes("hardening"));
  assert.ok(result.labels.includes("compliance"));
  assert.ok(result.labels.includes("pen-test"));
  assert.ok(!result.labels.includes("security"));
});

test("prompt builder emits deterministic fallback summary", () => {
  const issue = {
    number: 14,
    title: "Bug: runtime failure on Windows",
    body: ""
  };

  const result = buildIssueSummaryArtifacts({ issue });
  const classResult = AutobotPromptBuilder.buildIssueSummaryArtifacts({ issue });

  assert.equal(result.ready, "true");
  assert.ok(result.fallbackSummary.includes("Issue #14"));
  assert.ok(result.fallbackSummary.includes("The issue body is empty"));
  assert.equal(classResult.fallbackSummary, result.fallbackSummary);
});

test("local diff snapshot builder materializes tracked and untracked files for preview analysis", () => {
  const snapshot = buildLocalGitDiffSnapshot({
    diffText: [
      "diff --git a/src/definers/security/token_guard.py b/src/definers/security/token_guard.py",
      "index 1111111..2222222 100644",
      "--- a/src/definers/security/token_guard.py",
      "+++ b/src/definers/security/token_guard.py",
      "@@ -1 +1,4 @@",
      "-def sanitize_token(value):",
      "+def sanitize_token(value):",
      "+    if not value:",
      "+        raise ValueError(\"missing credential\")",
      "+    return value"
    ].join("\n"),
    headRef: "feature/local-preview",
    readFileText: (filename) => {
      assert.equal(filename, "src/definers/api/preview_endpoint.py");
      return [
        "@router.get(\"/preview\")",
        "def preview_endpoint():",
        "    return {\"status\": \"ok\"}"
      ].join("\n");
    },
    untrackedFiles: ["src/definers/api/preview_endpoint.py"]
  });

  assert.equal(snapshot.pullRequest.headRef, "feature/local-preview");
  assert.equal(snapshot.totals.filesChanged, 2);
  assert.equal(snapshot.totals.additions, 7);
  assert.equal(snapshot.totals.deletions, 1);
  assert.equal(snapshot.totals.totalChanges, 8);
  assert.deepEqual(snapshot.files.map((file) => ({ filename: file.filename, status: file.status })), [
    { filename: "src/definers/security/token_guard.py", status: "modified" },
    { filename: "src/definers/api/preview_endpoint.py", status: "added" }
  ]);
  assert.ok(snapshot.files[1].patch.includes("preview_endpoint"));
});

test("local diff preview analyzes the current git diff and renders a human-readable report", () => {
  const diffText = [
    "diff --git a/src/definers/security/token_guard.py b/src/definers/security/token_guard.py",
    "index 1111111..2222222 100644",
    "--- a/src/definers/security/token_guard.py",
    "+++ b/src/definers/security/token_guard.py",
    "@@ -1 +1,4 @@",
    "-def sanitize_token(value):",
    "+def sanitize_token(value):",
    "+    if not value:",
    "+        raise ValueError(\"missing credential\")",
    "+    return value"
  ].join("\n");
  const preview = analyzeCurrentGitDiff({
    cwd: "C:/repo",
    execFileSync: (_command, args) => {
      const key = args.join(" ");
      if (key === "rev-parse --show-toplevel") {
        return "C:/repo\n";
      }
      if (key === "diff --find-renames --no-ext-diff --no-color --unified=3 HEAD") {
        return diffText;
      }
      if (key === "ls-files --others --exclude-standard -z") {
        return "src/definers/api/preview_endpoint.py\u0000";
      }
      if (key === "rev-parse --abbrev-ref HEAD") {
        return "feature/local-preview\n";
      }
      throw new Error(`Unexpected git command: ${key}`);
    },
    readFileSync: (filePath) => {
      assert.ok(filePath.replace(/\\/g, "/").endsWith("/src/definers/api/preview_endpoint.py"));
      return [
        "@router.get(\"/preview\")",
        "def preview_endpoint():",
        "    return {\"status\": \"ok\"}"
      ].join("\n");
    }
  });
  const deterministicLabels = JSON.parse(preview.outputs.deterministic_labels_json);
  const candidateLabels = JSON.parse(preview.outputs.candidate_labels_json);
  const report = formatLocalDiffAnalysis(preview);

  assert.equal(preview.baseRef, "HEAD");
  assert.equal(preview.snapshot.pullRequest.headRef, "feature/local-preview");
  assert.equal(preview.outputs.release_relevant, "true");
  assert.ok(hasExpectedLabel(deterministicLabels, "security"));
  assert.ok(!deterministicLabels.includes("security"));
  assert.ok(hasExpectedLabel(candidateLabels, "api") || hasExpectedLabel(deterministicLabels, "api"));
  assert.ok(report.includes("## Local Autobot Preview"));
  assert.ok(report.includes("Final emitted technical labels:"));
  assert.ok(!report.includes("Held back context labels:"));
  assert.ok(report.includes("### Decision"));
  assert.ok(report.includes("### Label Rationale"));
  assert.ok(report.includes("Confidence"));
  assert.ok(report.includes("### Key Evidence"));
  assert.ok(report.includes("## Autobot Summary"));
});

test("local preview script preserves legacy capture bundle formatting", () => {
  const fixtureText = formatCaptureFixtureFromPayload({
    captureCandidates: [{
      fixtureId: "autobot-local-preview",
      kind: "additive-feature",
      materialization: { count: 13, seed: 20260413 },
      reason: "Promotes a real local preview failure into the regression registry.",
      replayKey: "AUTOBOT:feature-local-preview",
      severity: "warning",
      suiteKey: "AUTOBOT"
    }]
  });

  assert.ok(fixtureText.includes("fixtureId: \"autobot-local-preview\""));
  assert.ok(fixtureText.includes("severity: \"warning\""));
  assert.ok(fixtureText.includes("suiteKey: \"AUTOBOT\""));
});

test("local preview renders semver none for maintenance-only labels", () => {
  const snapshot = {
    pullRequest: {
      number: 18,
      title: "Refine autobot workflow coverage",
      body: "",
      headRef: "autobot/coverage-refresh"
    },
    totals: {
      filesChanged: 2,
      additions: 40,
      deletions: 12,
      totalChanges: 52
    },
    files: [
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 12,
        deletions: 4,
        patch: [
          "+on:",
          "+  workflow_dispatch:",
          "+jobs:",
          "+  autobot:",
          "+    uses: actions/github-script@v9"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "tests/autobot_scripts.test.cjs",
        status: "modified",
        additions: 28,
        deletions: 8,
        patch: [
          "+test(\"workflow maintenance coverage\", () => {",
          "+  assert.equal(true, true);",
          "+});"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const outputs = analyzePullRequestSnapshotData(snapshot);
  const report = formatLocalDiffAnalysis({
    baseRef: "HEAD",
    outputs,
    repoRoot: "C:/repo",
    snapshot
  });

  assert.equal(outputs.release_relevant, "false");
  assert.ok(report.includes("Semver: none."));
  assert.ok(!report.includes("Semver: patch."));
});

test("PR analyzer does not emit api for internal router wording alone", () => {
  const snapshot = {
    pullRequest: {
      number: 21,
      title: "Refine internal router wiring",
      body: "",
      headRef: "router/internal-wiring"
    },
    totals: {
      filesChanged: 1,
      additions: 8,
      deletions: 2,
      totalChanges: 10
    },
    files: [
      {
        filename: "src/definers/internal_router.py",
        status: "modified",
        additions: 8,
        deletions: 2,
        patch: [
          "-def configure_routes(router):",
          "+def configure_routes(router):",
          "+    router.get(\"/health\")",
          "+    router.get(\"/status\")",
          "+    return router"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const outputs = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);
  const candidateLabels = JSON.parse(outputs.candidate_labels_json);

  assert.ok(!deterministicLabels.includes("api"));
  assert.ok(!candidateLabels.includes("api"));
});

test("PR analyzer does not infer route params from static routes and dict literals", () => {
  const snapshot = {
    pullRequest: {
      number: 23,
      title: "Add static health endpoint",
      body: "",
      headRef: "feature/static-health"
    },
    totals: {
      filesChanged: 1,
      additions: 6,
      deletions: 0,
      totalChanges: 6
    },
    files: [
      {
        filename: "src/definers/api/health.py",
        status: "added",
        additions: 6,
        deletions: 0,
        patch: [
          "+@router.get(\"/health\")",
          "+def healthcheck():",
          "+    return {\"status\": \"ok\"}"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "api"));
  assert.ok(!deterministicLabels.includes("route param"));
});

test("PR analyzer emits route param for dynamic path segments", () => {
  const snapshot = {
    pullRequest: {
      number: 24,
      title: "Add item detail endpoint",
      body: "",
      headRef: "feature/item-detail"
    },
    totals: {
      filesChanged: 1,
      additions: 6,
      deletions: 0,
      totalChanges: 6
    },
    files: [
      {
        filename: "src/definers/api/items.py",
        status: "added",
        additions: 6,
        deletions: 0,
        patch: [
          "+@router.get(\"/items/{item_id}\")",
          "+def get_item(item_id: str):",
          "+    return {\"item_id\": item_id}"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(deterministicLabels.includes("route param"));
});

test("PR analyzer does not promote generic auth changes to compliance", () => {
  const snapshot = {
    pullRequest: {
      number: 25,
      title: "Harden token guard",
      body: "",
      headRef: "security/token-guard"
    },
    totals: {
      filesChanged: 1,
      additions: 4,
      deletions: 1,
      totalChanges: 5
    },
    files: [
      {
        filename: "src/definers/security/token_guard.py",
        status: "modified",
        additions: 4,
        deletions: 1,
        patch: [
          "-def sanitize_token(value):",
          "+def sanitize_token(value):",
          "+    if not value:",
          "+        raise ValueError(\"missing credential\")",
          "+    return value"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "security"));
  assert.ok(!deterministicLabels.includes("compliance"));
});

test("PR analyzer does not infer heap usage from generic memory helpers", () => {
  const snapshot = {
    pullRequest: {
      number: 26,
      title: "Refine transfer buffering helpers",
      body: "",
      headRef: "runtime/transfer-buffering"
    },
    totals: {
      filesChanged: 1,
      additions: 4,
      deletions: 0,
      totalChanges: 4
    },
    files: [
      {
        filename: "src/definers/media/web_transfer.py",
        status: "modified",
        additions: 4,
        deletions: 0,
        patch: [
          "+memory_map = {}",
          "+buffer = io.BytesIO()",
          "+buffer.write(b\"ok\")",
          "+return buffer.getvalue()"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("heap usage"));
});

test("PR analyzer does not infer import time from Python import statements", () => {
  const snapshot = {
    pullRequest: {
      number: 27,
      title: "Split database helpers",
      body: "",
      headRef: "refactor/database-split"
    },
    totals: {
      filesChanged: 1,
      additions: 2,
      deletions: 0,
      totalChanges: 2
    },
    files: [
      {
        filename: "src/definers/database/__init__.py",
        status: "modified",
        additions: 2,
        deletions: 0,
        patch: [
          "+from time import time",
          "+from definers.database.core import connect"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("import time"));
});

test("PR analyzer does not infer filesystem from pathlib imports alone", () => {
  const snapshot = {
    pullRequest: {
      number: 28,
      title: "Refine model installation module split",
      body: "",
      headRef: "refactor/model-installation-split"
    },
    totals: {
      filesChanged: 1,
      additions: 2,
      deletions: 0,
      totalChanges: 2
    },
    files: [
      {
        filename: "src/definers/model_installation/__init__.py",
        status: "modified",
        additions: 2,
        deletions: 0,
        patch: [
          "+from pathlib import Path",
          "+from definers.model_installation.core import install_model"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("filesystem"));
});

test("PR analyzer replaces broad runtime labels with specific descendants", () => {
  const snapshot = {
    pullRequest: {
      number: 31,
      title: "Harden Windows subprocess execution paths",
      body: "",
      headRef: "runtime/windows-subprocess-hardening"
    },
    totals: {
      filesChanged: 1,
      additions: 9,
      deletions: 2,
      totalChanges: 11
    },
    files: [
      {
        filename: "src/definers/command_runner.py",
        status: "modified",
        additions: 9,
        deletions: 2,
        patch: [
          "-def run_command(command):",
          "+def run_command(command):",
          "+    if platform.system() == \"Windows\":",
          "+        normalized = os.path.normpath(command)",
          "+        return subprocess.run([\"cmd\", \"/c\", normalized], capture_output=True, text=True)",
          "+    return subprocess.run(command, capture_output=True, text=True)"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(deterministicLabels.includes("subprocess io"));
  assert.ok(deterministicLabels.includes("path normalization") || deterministicLabels.includes("path separator") || deterministicLabels.includes("shell command"));
  assert.ok(!deterministicLabels.includes("windows"));
  assert.ok(!deterministicLabels.includes("filesystem"));
  assert.ok(!deterministicLabels.includes("process"));
  assert.ok(deterministicLabels.every((label) => label.replace(/[_-]+/g, " ").split(/\s+/).filter(Boolean).length <= 2));
});

test("PR analyzer does not infer compliance from audio sox references", () => {
  const snapshot = {
    pullRequest: {
      number: 29,
      title: "Split dataset value helpers",
      body: "",
      headRef: "refactor/dataset-value-split"
    },
    totals: {
      filesChanged: 1,
      additions: 1,
      deletions: 0,
      totalChanges: 1
    },
    files: [
      {
        filename: "src/definers/data/datasets/value.py",
        status: "modified",
        additions: 1,
        deletions: 0,
        patch: "+transformer = definers.sox.Transformer()",
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("compliance"));
});

test("PR analyzer does not infer shell from task cmd keys and markdown fences", () => {
  const snapshot = {
    pullRequest: {
      number: 30,
      title: "Refine autobot contributor workflow docs",
      body: "",
      headRef: "docs/autobot-workflow"
    },
    totals: {
      filesChanged: 2,
      additions: 3,
      deletions: 0,
      totalChanges: 3
    },
    files: [
      {
        filename: "pyproject.toml",
        status: "modified",
        additions: 2,
        deletions: 0,
        patch: [
          "+[tool.poe.tasks.autobot-preview]",
          "+cmd = \"node scripts/print_autobot_regression_fixture.cjs\""
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "CONTRIBUTING.md",
        status: "modified",
        additions: 1,
        deletions: 0,
        patch: "+```bash",
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("shell"));
  assert.ok(!deterministicLabels.includes("shell command"));
});

test("PR analyzer collapses workflow overlap on small maintenance PRs", () => {
  const snapshot = {
    pullRequest: {
      number: 22,
      title: "Refresh autobot workflow maintenance coverage",
      body: "",
      headRef: "autobot/workflow-maintenance"
    },
    totals: {
      filesChanged: 2,
      additions: 40,
      deletions: 12,
      totalChanges: 52
    },
    files: [
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 12,
        deletions: 4,
        patch: [
          "+on:",
          "+  workflow_dispatch:",
          "+jobs:",
          "+  autobot:",
          "+    uses: actions/github-script@v9"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "tests/autobot_scripts.test.cjs",
        status: "modified",
        additions: 28,
        deletions: 8,
        patch: [
          "+test(\"workflow maintenance coverage\", () => {",
          "+  assert.equal(true, true);",
          "+});"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const outputs = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(outputs.deterministic_labels_json);
  const candidateLabels = JSON.parse(outputs.candidate_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "workflow"));
  assert.ok(hasExpectedLabel(deterministicLabels, "test"));
  assert.ok(!deterministicLabels.includes("workflow"));
  assert.ok(!deterministicLabels.includes("test"));
  assert.ok(!deterministicLabels.includes("ci"));
  assert.ok(!deterministicLabels.includes("automation"));
  assert.ok(!deterministicLabels.includes("github"));
  assert.ok(deterministicLabels.length <= 4);
  assert.ok(hasExpectedLabel(candidateLabels, "workflow"));
  assert.ok(hasExpectedLabel(candidateLabels, "test"));
  assert.ok(!candidateLabels.includes("workflow"));
  assert.ok(!candidateLabels.includes("test"));
  assert.ok(!candidateLabels.includes("ci"));
  assert.ok(!candidateLabels.includes("automation"));
  assert.ok(!candidateLabels.includes("github"));
  assert.ok(candidateLabels.length <= 4);
});

test("PR analyzer avoids feature-flag self-classification for autobot infrastructure", () => {
  const snapshot = {
    pullRequest: {
      number: 7,
      title: "Refine autobot release heuristics",
      body: "",
      headRef: "autobot/refactor"
    },
    totals: {
      filesChanged: 1,
      additions: 12,
      deletions: 2,
      totalChanges: 14
    },
    files: [
      {
        filename: ".github/scripts/autobot/project_manager.cjs",
        status: "modified",
        additions: 12,
        deletions: 2,
        patch: [
          "+ const rollout = \"feature-flag\";",
          "+ const note = \"release automation\";"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const classResult = AutobotPullRequestAnalyzer.analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(!deterministicLabels.includes("feature-flag"));
  assert.equal(classResult.deterministic_labels_json, result.deterministic_labels_json);
});

test("project manager resolves PR label deltas from the shared registry rules", () => {
  const result = resolvePrLabelDelta({
    action: "opened",
    previousBotLabels: ["bug", "workflow"],
    currentPrLabels: ["bug", "workflow", "documentation"],
    autobotLabelsRaw: JSON.stringify(["bug", "documentation"])
  });

  const classResult = AutobotProjectManager.resolvePrLabelDelta({
    action: "opened",
    previousBotLabels: ["bug", "workflow"],
    currentPrLabels: ["bug", "workflow", "documentation"],
    autobotLabelsRaw: JSON.stringify(["bug", "documentation"])
  });

  assert.deepEqual(result.labelsToAdd, []);
  assert.deepEqual(result.labelsToRemove, ["workflow"]);
  assert.deepEqual(result.nextAutobotLabels, ["bug", "documentation"]);
  assert.deepEqual(classResult, result);
});

test("PR analyzer keeps dependency and workflow maintenance at patch and preserves dependencies label", () => {
  const snapshot = {
    pullRequest: {
      number: 12,
      title: "Patch vulnerable dependencies and consolidate autobot workflows",
      body: "",
      headRef: "security/patch-autobot"
    },
    totals: {
      filesChanged: 5,
      additions: 160,
      deletions: 90,
      totalChanges: 250
    },
    files: [
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 18,
        deletions: 14,
        patch: [
          "-    runs-on: windows-latest",
          "-    runs-on: ubuntu-22.04",
          "+    uses: actions/checkout@v6",
          "+    uses: actions/github-script@v9"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "package-lock.json",
        status: "modified",
        additions: 32,
        deletions: 28,
        patch: [
          "-      \"os\": [\"linux\", \"win32\"]",
          "+      \"integrity\": \"sha512-updated\"",
          "+      \"version\": \"1.2.3\""
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "src/server/vendor_patch.py",
        status: "added",
        additions: 40,
        deletions: 0,
        patch: "+def vendor_patch():\n+    return \"patched\"",
        rawPatchAvailable: true
      },
      {
        filename: "src/balancer/vendor_patch.py",
        status: "added",
        additions: 35,
        deletions: 0,
        patch: "+def vendor_balance_patch():\n+    return True",
        rawPatchAvailable: true
      },
      {
        filename: "src/shared/vendor_patch.py",
        status: "added",
        additions: 35,
        deletions: 0,
        patch: "+def shared_vendor_patch():\n+    return 1",
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);
  const semver = JSON.parse(result.deterministic_semver_json);

  assert.equal(semver.decision, "patch");
  assert.deepEqual(semver.hardSignals, []);
  assert.ok(hasExpectedLabel(deterministicLabels, "dependencies"));
  assert.ok(!deterministicLabels.includes("enhancement"));
  assert.ok(!deterministicLabels.includes("dependencies"));
});

test("PR analyzer does not treat pyproject CLI task changes as dependency updates", () => {
  const snapshot = {
    pullRequest: {
      number: 12,
      title: "Refine local autobot preview command",
      body: "",
      headRef: "tooling/autobot-preview"
    },
    totals: {
      filesChanged: 1,
      additions: 1,
      deletions: 1,
      totalChanges: 2
    },
    files: [{
      filename: "pyproject.toml",
      status: "modified",
      additions: 1,
      deletions: 1,
      patch: [
        "[tool.poe.tasks.autobot-preview]",
        "-cmd = \"node scripts/print_autobot_regression_fixture.cjs --old-mode\"",
        "+cmd = \"node scripts/print_autobot_regression_fixture.cjs\""
      ].join("\n"),
      rawPatchAvailable: true
    }]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);
  const candidateLabels = JSON.parse(result.candidate_labels_json);
  const dependencySpecificLabels = ["dependencies", "dependency group", "lockfile", "package lock", "poetry lock"];

  assert.ok(hasExpectedLabel(deterministicLabels, "config"));
  assert.ok(dependencySpecificLabels.every((label) => !deterministicLabels.includes(label)));
  assert.ok(dependencySpecificLabels.every((label) => !candidateLabels.includes(label)));
  assert.ok(!result.deterministic_summary.includes("Dependency declarations or lockfiles changed."));
});

test("PR analyzer treats pyproject optional dependency changes as dependencies", () => {
  const snapshot = {
    pullRequest: {
      number: 14,
      title: "Adjust optional audio dependency versions",
      body: "",
      headRef: "deps/audio-refresh"
    },
    totals: {
      filesChanged: 1,
      additions: 1,
      deletions: 1,
      totalChanges: 2
    },
    files: [{
      filename: "pyproject.toml",
      status: "modified",
      additions: 1,
      deletions: 1,
      patch: [
        "[project.optional-dependencies]",
        "audio = [",
        "-    \"librosa>=0.10.0\",",
        "+    \"librosa>=0.11.0\",",
        "]"
      ].join("\n"),
      rawPatchAvailable: true
    }]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "dependencies"));
  assert.ok(result.deterministic_summary.includes("Final emitted technical labels:"));
  assert.ok(!result.deterministic_summary.includes("Held back context labels:"));
  assert.ok(lineIncludesAny(result.deterministic_summary, ["dependency group", "pyproject"]));
});

test("PR analyzer does not treat package.json script changes as dependency updates", () => {
  const snapshot = {
    pullRequest: {
      number: 15,
      title: "Refine package script entry points",
      body: "",
      headRef: "tooling/package-scripts"
    },
    totals: {
      filesChanged: 1,
      additions: 1,
      deletions: 1,
      totalChanges: 2
    },
    files: [{
      filename: "package.json",
      status: "modified",
      additions: 1,
      deletions: 1,
      patch: [
        "  \"scripts\": {",
        "-    \"test\": \"vitest\"",
        "+    \"test\": \"node --test tests/autobot_scripts.test.cjs\"",
        "  }"
      ].join("\n"),
      rawPatchAvailable: true
    }]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "config"));
  assert.ok(!deterministicLabels.includes("config"));
  assert.ok(!deterministicLabels.includes("dependencies"));
});

test("PR analyzer treats package.json dependency block changes as dependencies", () => {
  const snapshot = {
    pullRequest: {
      number: 16,
      title: "Update package dependency versions",
      body: "",
      headRef: "deps/package-json"
    },
    totals: {
      filesChanged: 1,
      additions: 1,
      deletions: 1,
      totalChanges: 2
    },
    files: [{
      filename: "package.json",
      status: "modified",
      additions: 1,
      deletions: 1,
      patch: [
        "  \"dependencies\": {",
        "-    \"vite\": \"^5.0.0\",",
        "+    \"vite\": \"^5.1.0\",",
        "  }"
      ].join("\n"),
      rawPatchAvailable: true
    }]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);

  assert.ok(hasExpectedLabel(deterministicLabels, "dependencies"));
  assert.ok(!deterministicLabels.includes("dependencies"));
});

test("PR analyzer still treats explicit runtime support drops in runtime contracts as major", () => {
  const snapshot = {
    pullRequest: {
      number: 13,
      title: "Drop Python 3.9 runtime support",
      body: "",
      headRef: "breaking/runtime-policy"
    },
    totals: {
      filesChanged: 1,
      additions: 0,
      deletions: 1,
      totalChanges: 1
    },
    files: [
      {
        filename: "pyproject.toml",
        status: "modified",
        additions: 0,
        deletions: 1,
        patch: "-requires-python = \">=3.9\"",
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const semver = JSON.parse(result.deterministic_semver_json);

  assert.equal(semver.decision, "major");
  assert.equal(semver.hardRule, true);
  assert.deepEqual(semver.hardSignals, ["runtime-support-dropped"]);
});

test("PR analyzer ignores version-critical vocabulary inside autobot maintenance infrastructure", () => {
  const snapshot = {
    pullRequest: {
      number: 20,
      title: "Patch vulnerable dependencies and refine autobot maintenance coverage",
      body: "",
      headRef: "autobot/maintenance-refresh"
    },
    totals: {
      filesChanged: 5,
      additions: 220,
      deletions: 75,
      totalChanges: 295
    },
    files: [
      {
        filename: ".github/scripts/autobot/measurement/scorer.cjs",
        status: "modified",
        additions: 80,
        deletions: 20,
        patch: [
          "+  \"additive-api-contract\": { hardSemver: \"minor\" },",
          "+  \"runtime-support-added\": { hardSemver: \"minor\" },",
          "+  \"compatibility-shim\": { labelBoosts: { compatibility: 0.85 } },"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "test/autobot_scripts.test.cjs",
        status: "modified",
        additions: 55,
        deletions: 12,
        patch: [
          "+test(\"security runtime compatibility api\", () => {",
          "+  assert.equal(\"additive-api-contract\", \"additive-api-contract\");",
          "+});"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "test/deterministic_scenario_engine.cjs",
        status: "modified",
        additions: 40,
        deletions: 10,
        patch: [
          "+semverDecision: \"minor\"",
          "+hardSignals: [\"additive-api-contract\"]",
          "+headRef: \"security/runtime-compat\""
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 15,
        deletions: 6,
        patch: [
          "+on:",
          "+  workflow_dispatch:",
          "+jobs:",
          "+  autobot:",
          "+    uses: actions/github-script@v9"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "package-lock.json",
        status: "modified",
        additions: 30,
        deletions: 27,
        patch: [
          "+      \"version\": \"1.2.3\"",
          "+      \"integrity\": \"sha512-hardened\""
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);
  const semver = JSON.parse(result.deterministic_semver_json);

  assert.equal(semver.decision, "patch");
  assert.ok(!deterministicLabels.includes("api"));
  assert.ok(!deterministicLabels.includes("compatibility"));
  assert.ok(!deterministicLabels.includes("runtime"));
  assert.ok(!hasExpectedLabel(deterministicLabels, "security"));
  assert.ok(hasExpectedLabel(deterministicLabels, "workflow"));
  assert.ok(hasExpectedLabel(deterministicLabels, "test"));
  assert.ok(hasExpectedLabel(deterministicLabels, "dependencies"));
  assert.ok(!deterministicLabels.includes("automation"));
  assert.ok(!deterministicLabels.includes("ci"));
  assert.ok(!deterministicLabels.includes("github"));
  assert.ok(deterministicLabels.length <= 6);
  assert.ok(result.deterministic_summary.includes("Final emitted technical labels:"));
  assert.ok(!result.deterministic_summary.includes("Held back context labels:"));
  assert.ok(lineIncludesAny(result.deterministic_summary, ["lockfile", "package lock"]));
});

test("PR analyzer preserves additive API classification when raw patches are unavailable", () => {
  const snapshot = {
    pullRequest: {
      number: 22,
      title: "Feature: add preview endpoint with oversized diff",
      body: "GitHub omitted the raw patch because the file is too large.",
      headRef: "feature/oversized-preview"
    },
    totals: {
      filesChanged: 2,
      additions: 148,
      deletions: 4,
      totalChanges: 152
    },
    files: [
      {
        filename: "src/definers/api/preview_endpoint.py",
        status: "added",
        additions: 124,
        deletions: 0,
        patch: "",
        rawPatchAvailable: false
      },
      {
        filename: "src/definers/__init__.py",
        status: "modified",
        additions: 24,
        deletions: 4,
        patch: "",
        rawPatchAvailable: false
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);
  const semver = JSON.parse(result.deterministic_semver_json);

  assert.equal(semver.decision, "minor");
  assert.ok(hasExpectedLabel(deterministicLabels, "api"));
  assert.ok(!deterministicLabels.includes("security"));
  assert.ok(!deterministicLabels.includes("breaking-change"));
});

test("PR analyzer maps compatibility shims to technical release labels", () => {
  const snapshot = {
    pullRequest: {
      number: 23,
      title: "Add backward compatibility shim for preview callers",
      body: "",
      headRef: "compat/preview-shim"
    },
    totals: {
      filesChanged: 3,
      additions: 64,
      deletions: 4,
      totalChanges: 68
    },
    files: [
      {
        filename: "src/definers/compat/preview_shim.py",
        status: "added",
        additions: 36,
        deletions: 0,
        patch: [
          "+def preview_shim(payload):",
          "+    return payload"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "tests/test_preview_shim.py",
        status: "added",
        additions: 16,
        deletions: 0,
        patch: "+def test_preview_shim():\n+    assert True",
        rawPatchAvailable: true
      },
      {
        filename: "docs/reference/preview_shim.md",
        status: "added",
        additions: 12,
        deletions: 0,
        patch: "+This shim preserves backward compatibility for preview callers.",
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const deterministicLabels = JSON.parse(result.deterministic_labels_json);
  const semver = JSON.parse(result.deterministic_semver_json);

  assert.equal(semver.decision, "patch");
  assert.ok(deterministicLabels.includes("shim module"));
  assert.ok(hasExpectedLabel(deterministicLabels, "compatibility"));
  assert.ok(!deterministicLabels.includes("compatibility"));
  assert.ok(result.deterministic_summary.includes("shim module"));
});

test("PR summary limits release relevance to version-impact signals", () => {
  const snapshot = {
    pullRequest: {
      number: 21,
      title: "Harden auth flow and add runtime endpoint support",
      body: "",
      headRef: "feature/release-signal-filter"
    },
    totals: {
      filesChanged: 4,
      additions: 126,
      deletions: 18,
      totalChanges: 144
    },
    files: [
      {
        filename: ".github/workflows/autobot.yml",
        status: "modified",
        additions: 12,
        deletions: 4,
        patch: [
          "+on:",
          "+  workflow_dispatch:",
          "+jobs:",
          "+  autobot:",
          "+    uses: actions/github-script@v9"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "pyproject.toml",
        status: "modified",
        additions: 1,
        deletions: 0,
        patch: "+requires-python = \">=3.12\"",
        rawPatchAvailable: true
      },
      {
        filename: "src/definers/api/router.py",
        status: "added",
        additions: 88,
        deletions: 0,
        patch: [
          "+@router.get(\"/health\")",
          "+def healthcheck():",
          "+    return {\"status\": \"ok\"}"
        ].join("\n"),
        rawPatchAvailable: true
      },
      {
        filename: "src/definers/security/token_guard.py",
        status: "modified",
        additions: 25,
        deletions: 14,
        patch: [
          "+def sanitize_token(value):",
          "+    if not value:",
          "+        raise ValueError(\"missing credential\")",
          "+    return value"
        ].join("\n"),
        rawPatchAvailable: true
      }
    ]
  };

  const result = analyzePullRequestSnapshotData(snapshot);
  const decisionSectionMatch = result.deterministic_summary.match(/### Decision\n([\s\S]*?)\n\n### Label Rationale/);

  assert.ok(decisionSectionMatch);
  const releaseSignalLine = decisionSectionMatch[1]
    .split("\n")
    .find((line) => line.includes("Final emitted technical labels:"));
  assert.ok(releaseSignalLine);
  assert.ok(lineIncludesAny(releaseSignalLine, ["auth", "token", "jwt", "authz"]));
  assert.ok(lineIncludesAny(releaseSignalLine, ["api", "route", "route param", "request body", "openapi spec"]));
  assert.ok(lineIncludesAny(releaseSignalLine, ["runtime", "support matrix", "python version", "dockerfile", "image tag"]));
  assert.ok(!releaseSignalLine.includes("workflow"));
  assert.ok(!releaseSignalLine.includes("ci"));
  assert.ok(!releaseSignalLine.includes("automation"));
  assert.ok(!releaseSignalLine.includes("github"));
});

test("project manager recreates the next patch milestone after demilestoning when a draft release already reserves the current version", async () => {
  const github = createAutobotPipelineGithubMock({
    issueLabels: ["security"],
    releases: [
      { draft: true, id: 2, prerelease: false, tag_name: "v0.0.2" },
      { draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }
    ]
  });

  await AutobotProjectManager.syncProjectMilestone({
    context: {
      eventName: "pull_request",
      payload: {
        action: "demilestoned"
      }
    },
    github,
    issueNumber: 12,
    owner: "octo",
    repo: "definers"
  });

  assert.ok(github.state.createdMilestones.includes("v0.0.3"));
  assert.deepEqual(github.state.createdMilestones, ["v0.0.3"]);
  assert.equal(github.state.issueUpdates.at(-1).milestone, "v0.0.3");
});

test("github mock failure injection replays a transient label failure once", async () => {
  const github = createAutobotPipelineGithubMock({
    failures: {
      "issues.addLabels": { message: "invalid label", status: 422, times: 1 }
    }
  });

  await assert.rejects(
    github.rest.issues.addLabels({ issue_number: 12, labels: ["workflow"] }),
    /invalid label/
  );

  await github.rest.issues.addLabels({ issue_number: 12, labels: ["workflow"] });

  assert.deepEqual(github.state.failureLog.map((entry) => entry.operationName), ["issues.addLabels"]);
  assert.deepEqual(github.state.issuesByNumber.get(12).labels.map((label) => label.name), ["workflow"]);
});

test("scenario engine emits replayable realism metadata under weighted selection", () => {
  const suite = evaluateAutobotScenarioSuite({
    analyzeScenario: analyzePullRequestSnapshotData,
    config: {
      count: 13,
      seed: 20260413
    }
  });

  assert.equal(new Set(suite.scenarios.map((scenario) => scenario.kind)).size, suite.scenarios.length);
  for (const scenario of suite.scenarios) {
    assert.ok(String(scenario.replayKey || "").includes(suite.key));
    assert.equal(scenario.realismProfile.weightedSelection, true);
    assert.ok(scenario.realismProfile.authorAssociation);
    assert.ok(scenario.realismProfile.sourceUpdatedAt);
    assert.ok(Object.prototype.hasOwnProperty.call(scenario.realismProfile, "patchBudgetNoiseMode"));
  }
  assert.ok(suite.summary.metrics.semverConfusion.minor.minor >= 1, suite.reportText);
  assert.ok(Object.prototype.hasOwnProperty.call(suite.summary.metrics.criticalLabelMetrics, "breaking-change"));
  assert.equal(typeof suite.summary.falseGreenRate, "number");
  assert.equal(typeof suite.summary.exactPassRate, "number");
});

test("autobot contract validator accepts configured alternative semver and label outcomes", () => {
  const validation = validateAutobotScenarioContract({
    expected: {
      acceptableLabelSets: [["api", "enhancement"]],
      acceptableSemverDecisions: ["minor"],
      requiredLabels: ["breaking-change"],
      semverDecision: "major"
    }
  }, {
    deterministic_labels_json: JSON.stringify(["api", "enhancement"]),
    deterministic_semver_json: JSON.stringify({
      decision: "minor",
      hardSignals: []
    })
  });

  assert.deepEqual(validation.deviations, []);
  assert.equal(validation.acceptedOutcome, true);
  assert.ok(validation.acceptedReasons.some((reason) => reason.includes("acceptable")));
});

test("autobot contract validator warns on unexpected critical labels outside the scenario contract", () => {
  const scenario = {
    expected: {
      requiredLabels: ["workflow"],
      semverDecision: "patch"
    }
  };
  const validation = validateAutobotScenarioContract(scenario, {
    deterministic_labels_json: JSON.stringify(["workflow", "security"]),
    deterministic_semver_json: JSON.stringify({
      decision: "patch",
      hardSignals: []
    })
  });
  const summary = summarizeScenarioEvaluations([{
    acceptedOutcome: validation.acceptedOutcome,
    actualResult: {
      labels: ["workflow", "security"],
      semverDecision: "patch"
    },
    contract: validation.contract,
    deviations: validation.deviations,
    passed: validation.deviations.length === 0,
    scenario,
    warnings: validation.warnings
  }], { suiteKey: "AUTOBOT" });

  assert.deepEqual(validation.deviations, []);
  assert.ok(validation.warnings.some((warning) => warning.code === "unexpected-critical-label"));
  assert.equal(summary.falseGreenCount, 1);
  assert.equal(summary.warningScenarioCount, 1);
});

test("pipeline scenario suite recovers from transient label provisioning races", async () => {
  const suite = await evaluateAutobotPipelineScenarioSuite({
    config: {
      count: 13,
      seed: 20260413
    }
  });
  const evaluation = suite.evaluations.find((entry) => entry.kind === "label-race-recovery");

  assert.ok(evaluation, suite.reportText);
  assert.deepEqual(evaluation.actualResult.failureOperations, ["issues.addLabels"]);
  assert.equal(evaluation.actualResult.milestoneTitle, null);
  assert.equal(evaluation.passed, true, suite.reportText);
});

test("PR analyzer scenario engine keeps actual output aligned with optimal expectations", () => {
  const suite = evaluateAutobotScenarioSuite({
    analyzeScenario: analyzePullRequestSnapshotData
  });

  console.log(suite.reportText);
  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.deviationScenarioCount, 0, suite.reportText);
});

test("PR analyzer exploratory seed stays free of warning scenarios", () => {
  const suite = evaluateAutobotScenarioSuite({
    analyzeScenario: analyzePullRequestSnapshotData,
    config: {
      count: 48,
      seed: 3104803941
    }
  });

  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.warningScenarioCount, 0, suite.reportText);
  assert.equal(suite.summary.falseGreenCount, 0, suite.reportText);
});

test("Autobot pipeline scenario engine keeps end-to-end output aligned with optimal expectations", async () => {
  const suite = await evaluateAutobotPipelineScenarioSuite();

  console.log(suite.reportText);
  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.deviationScenarioCount, 0, suite.reportText);
});

test("unified automation exploratory seed stays free of warnings and deviations", async () => {
  const suite = await evaluateUnifiedAutomationScenarioSuite({
    config: {
      count: 64,
      seed: 2203428311
    }
  });

  assert.equal(suite.summary.deviationCount, 0, suite.reportText);
  assert.equal(suite.summary.warningScenarioCount, 0, suite.reportText);
  assert.equal(suite.summary.falseGreenCount, 0, suite.reportText);
});
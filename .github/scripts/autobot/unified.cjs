const fs = require("fs");

const { analyzeIssueIntake } = require("./issue_intake.cjs");
const {
  AutobotLabelRegistry,
  MAX_AUTOBOT_LABELS,
  parseAutobotLabels,
  sortLabels,
  technicalLabelsOnly,
  trimLowSignalLabels,
  uniqueValidLabels
} = require("./labels.cjs");
const { buildIssueSummaryArtifacts } = require("./prompts.cjs");
const { analyzePullRequestSnapshot, collectPullRequestSnapshot } = require("./pr_analysis.cjs");
const {
  finalizeClosedPullRequestRelease,
  prepareProjectState,
  syncPreparedProjectState,
  syncProjectMilestone
} = require("./project_manager.cjs");
const { analyzeSmartLinkSource, collectSmartLinkSource } = require("./smart_link/ingress.cjs");
const { renderSmartLinkOutputs } = require("./smart_link/renderer.cjs");

const PROJECT_STATE_FILE = "/tmp/autobot_smart_link_project_state.json";
const SMART_LINK_ANALYSIS_FILE = "/tmp/autobot_smart_link_analysis.json";
const SMART_LINK_SOURCE_FILE = "/tmp/autobot_smart_link_source.json";
const LINKED_LABEL_MIN_SCORE = 2;
const LINK_PROPAGATION_RELATION_SCORES = Object.freeze({
  advisory_fix: 3,
  closes: 3,
  connects: 1,
  depends_on: 2,
  related: 0
});
const CONNECT_PROPAGATABLE_LABELS = new Set([
  ...AutobotLabelRegistry.SECONDARY_LABELS,
  "automation",
  "ci",
  "config",
  "dependencies",
  "documentation",
  "dx",
  "github",
  "test",
  "tooling",
  "workflow"
]);
const PROPAGATABLE_LINK_LABELS = new Set([...AutobotLabelRegistry.VALID_LABELS]);

function isConnectPropagatableLabel(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  const metadata = AutobotLabelRegistry.getLabelMetadata(normalizedLabel);
  return CONNECT_PROPAGATABLE_LABELS.has(normalizedLabel) || Boolean(metadata?.secondary);
}

function isPropagatableLinkLabel(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  return PROPAGATABLE_LINK_LABELS.has(normalizedLabel);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function asString(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function resolveAuxiliaryStateFile(explicitFile, fallbackFile, stateFile, suffix) {
  const explicit = asString(explicitFile).trim();
  if (explicit) return explicit;
  const statePath = asString(stateFile).trim();
  if (statePath) return `${statePath}.${suffix}`;
  return fallbackFile;
}

function resolveUnifiedAutomationFiles(input) {
  return {
    autobotSnapshotFile: resolveAuxiliaryStateFile(input.autobotSnapshotFile, `${PROJECT_STATE_FILE}.snapshot`, input.stateFile, "autobot-snapshot.json"),
    projectStateFile: resolveAuxiliaryStateFile(input.projectStateFile, PROJECT_STATE_FILE, input.stateFile, "project.json"),
    smartLinkAnalysisFile: resolveAuxiliaryStateFile(input.smartLinkAnalysisFile, SMART_LINK_ANALYSIS_FILE, input.stateFile, "smart-link-analysis.json"),
    smartLinkSourceFile: resolveAuxiliaryStateFile(input.smartLinkSourceFile, SMART_LINK_SOURCE_FILE, input.stateFile, "smart-link-source.json")
  };
}

function normalizeAutobotLabels(value) {
  return trimLowSignalLabels(sortLabels(uniqueValidLabels(parseAutobotLabels(value))), { limit: MAX_AUTOBOT_LABELS });
}

function buildCombinedAutobotLabels(directLabels, linkedLabels) {
  return trimLowSignalLabels(sortLabels(uniqueValidLabels([...(directLabels || []), ...(linkedLabels || [])])), { limit: MAX_AUTOBOT_LABELS });
}

function shouldAnalyzeAutobotContext(context) {
  if (!context || !["issues", "pull_request"].includes(context.eventName)) return false;
  return asString(context.payload && context.payload.action).toLowerCase() !== "closed";
}

function shouldPrepareAutobotProjectState(context) {
  if (!context || !["issues", "pull_request"].includes(context.eventName)) return false;
  if (asString(context.payload && context.payload.action).toLowerCase() === "closed") return false;
  if (context.eventName === "issues") return true;
  return asString(context.payload && context.payload.pull_request && context.payload.pull_request.base && context.payload.pull_request.base.ref).toLowerCase() === "main";
}

function shouldFinalizeAutobotRelease(context) {
  if (!context || context.eventName !== "pull_request") return false;
  if (asString(context.payload && context.payload.action).toLowerCase() !== "closed") return false;
  return asString(context.payload && context.payload.pull_request && context.payload.pull_request.base && context.payload.pull_request.base.ref).toLowerCase() === "main";
}

function shouldAnalyzeSmartLinks(context) {
  if (!context) return false;
  if (!["issues", "pull_request", "repository_dispatch", "workflow_dispatch"].includes(context.eventName)) return false;
  if (["issues", "pull_request"].includes(context.eventName)) {
    return asString(context.payload && context.payload.action).toLowerCase() !== "closed";
  }
  return true;
}

function deriveLinkedAutobotLabels(analysis) {
  const source = analysis && analysis.source;
  const emittedResults = Array.isArray(analysis && analysis.emittedResults) ? analysis.emittedResults : [];
  if (!source || source.kind !== "pull_request" || emittedResults.length === 0) return [];

  const labelScores = new Map();
  for (const result of emittedResults) {
    const relationScore = LINK_PROPAGATION_RELATION_SCORES[result.relationKind] || 0;
    if (relationScore <= 0) continue;

    const candidateLabels = uniqueValidLabels(result.candidate && result.candidate.labels)
      .filter((label) => isPropagatableLinkLabel(label))
      .filter((label) => result.relationKind !== "connects" || isConnectPropagatableLabel(label));

    for (const label of candidateLabels) {
      labelScores.set(label, (labelScores.get(label) || 0) + relationScore);
    }

    if (result.relationKind === "advisory_fix") {
      labelScores.set("security", (labelScores.get("security") || 0) + relationScore);
    }
  }

  return trimLowSignalLabels(
    sortLabels(
      [...labelScores.entries()]
        .filter(([, score]) => score >= LINKED_LABEL_MIN_SCORE)
        .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
        .map(([label]) => label)
    ),
    { limit: MAX_AUTOBOT_LABELS }
  );
}

async function collectAutobotArtifacts({ context, github, owner, repo, snapshotFile }) {
  if (!shouldAnalyzeAutobotContext(context)) {
    return {
      deterministicLabels: [],
      deterministicSemverRaw: "",
      summaryText: ""
    };
  }

  if (context.eventName === "pull_request") {
    await collectPullRequestSnapshot({
      github,
      owner,
      pullRequest: context.payload.pull_request,
      repo,
      snapshotFile
    });
    const outputs = analyzePullRequestSnapshot({ snapshotFile });
    return {
      deterministicLabels: normalizeAutobotLabels(outputs.deterministic_labels_json),
      deterministicSemverRaw: asString(outputs.deterministic_semver_json),
      summaryText: asString(outputs.deterministic_summary)
    };
  }

  const issue = context.payload.issue || {};
  const issueIntake = analyzeIssueIntake(issue);
  const issueSummary = buildIssueSummaryArtifacts({ issue });
  return {
    deterministicLabels: normalizeAutobotLabels(issueIntake.labels),
    deterministicSemverRaw: "",
    summaryText: asString(issueSummary.fallbackSummary)
  };
}

async function analyzeUnifiedAutomationState({
  context,
  core,
  github,
  owner,
  repo,
  stateFile,
  thresholdInput,
  autobotSnapshotFile,
  projectStateFile,
  smartLinkAnalysisFile,
  smartLinkSourceFile
}) {
  const files = resolveUnifiedAutomationFiles({
    autobotSnapshotFile,
    projectStateFile,
    smartLinkAnalysisFile,
    smartLinkSourceFile,
    stateFile
  });
  const autobotArtifacts = await collectAutobotArtifacts({ context, github, owner, repo, snapshotFile: files.autobotSnapshotFile });
  const smartLinkEnabled = shouldAnalyzeSmartLinks(context);
  const autobotPrepareEnabled = shouldPrepareAutobotProjectState(context);
  const autobotFinalizeEnabled = shouldFinalizeAutobotRelease(context);

  let linkedAutobotLabels = [];
  let smartLinkCollectOutputs = { ready: "false" };
  let smartLinkAnalyzeOutputs = { ready: "false" };

  if (smartLinkEnabled) {
    smartLinkCollectOutputs = await collectSmartLinkSource({
      additionalLabels: autobotArtifacts.deterministicLabels,
      context,
      core,
      github,
      owner,
      repo,
      sourceFile: files.smartLinkSourceFile,
      thresholdInput
    });
    if (smartLinkCollectOutputs.ready === "true") {
      smartLinkAnalyzeOutputs = await analyzeSmartLinkSource({
        analysisFile: files.smartLinkAnalysisFile,
        core,
        github,
        owner,
        repo,
        sourceFile: files.smartLinkSourceFile
      });
      if (smartLinkAnalyzeOutputs.ready === "true") {
        linkedAutobotLabels = deriveLinkedAutobotLabels(readJson(files.smartLinkAnalysisFile));
      }
    }
  }

  const rawCombinedAutobotLabels = buildCombinedAutobotLabels(autobotArtifacts.deterministicLabels, linkedAutobotLabels);
  const combinedAutobotLabels = context?.eventName === "pull_request"
    ? technicalLabelsOnly(rawCombinedAutobotLabels, { limit: MAX_AUTOBOT_LABELS })
    : rawCombinedAutobotLabels;
  const issueNumber = context.issue && context.issue.number ? context.issue.number : null;

  if (autobotPrepareEnabled && issueNumber) {
    await prepareProjectState({
      autobotLabelsRaw: JSON.stringify(combinedAutobotLabels),
      context,
      deterministicSemverRaw: autobotArtifacts.deterministicSemverRaw,
      github,
      issueNumber,
      owner,
      repo,
      stateFile: files.projectStateFile,
      summaryText: autobotArtifacts.summaryText
    });
  }

  const state = {
    autobot: {
      combinedLabels: combinedAutobotLabels,
      directLabels: autobotArtifacts.deterministicLabels,
      finalizeReady: autobotFinalizeEnabled,
      issueNumber,
      linkedLabels: linkedAutobotLabels,
      prepareReady: autobotPrepareEnabled && issueNumber > 0,
      projectStateFile: files.projectStateFile,
      summaryText: autobotArtifacts.summaryText
    },
    smartLink: {
      analysisFile: files.smartLinkAnalysisFile,
      ready: smartLinkCollectOutputs.ready === "true" && smartLinkAnalyzeOutputs.ready === "true",
      sourceFile: files.smartLinkSourceFile
    }
  };
  writeJson(stateFile, state);

  const outputs = {
    autobot_labels_json: JSON.stringify(combinedAutobotLabels),
    autobot_ready: state.autobot.prepareReady || state.autobot.finalizeReady ? "true" : "false",
    linked_labels_json: JSON.stringify(linkedAutobotLabels),
    ready: state.smartLink.ready || state.autobot.prepareReady || state.autobot.finalizeReady ? "true" : "false",
    smart_link_ready: state.smartLink.ready ? "true" : "false"
  };
  for (const [name, value] of Object.entries(outputs)) {
    core.setOutput(name, value);
  }
  return outputs;
}

async function applyUnifiedAutomationState({ context, github, owner, repo, stateFile }) {
  const state = readJson(stateFile);
  if (state.autobot.prepareReady && state.autobot.issueNumber) {
    await syncPreparedProjectState({
      github,
      issueNumber: state.autobot.issueNumber,
      owner,
      repo,
      stateFile: state.autobot.projectStateFile
    });
  }
  if (state.smartLink.ready) {
    await renderSmartLinkOutputs({
      analysisFile: state.smartLink.analysisFile,
      github,
      owner,
      repo
    });
  }
  if (state.autobot.prepareReady && state.autobot.issueNumber) {
    await syncProjectMilestone({
      context,
      github,
      issueNumber: state.autobot.issueNumber,
      owner,
      repo
    });
  }
  if (state.autobot.finalizeReady) {
    await finalizeClosedPullRequestRelease({ github, owner, repo, context });
  }
  return state;
}

module.exports = {
  analyzeUnifiedAutomationState,
  applyUnifiedAutomationState,
  buildCombinedAutobotLabels,
  deriveLinkedAutobotLabels,
  shouldAnalyzeSmartLinks,
  shouldPrepareAutobotProjectState
};
const fs = require("fs");
const os = require("os");
const path = require("path");
const { chance, createSeededRandom, pick, pickWeighted, randomInt, randomWords, shuffle } = require("./javascript_test_harness.cjs");
const { analyzePullRequestSnapshot, collectPullRequestSnapshot } = require("../.github/scripts/autobot/pr_analysis.cjs");
const { prepareProjectState, syncPreparedProjectState, syncProjectMilestone } = require("../.github/scripts/autobot/project_manager.cjs");
const { analyzeUnifiedAutomationState, applyUnifiedAutomationState } = require("../.github/scripts/autobot/unified.cjs");
const { AutobotLabelRegistry } = require("../.github/scripts/autobot/labels.cjs");
const {
  buildDirectiveSummary,
  buildMermaidGraphLines,
  normalizeSmartLinkEntity,
  selectSmartLinkResults
} = require("../.github/scripts/autobot/smart_link/core.cjs");
const {
  createAutobotPipelineGithubMock,
  createWorkflowCoreMock,
  extractManagedBotCommentMetadata,
  findCommentContaining,
  getManagedBotComment
} = require("./autobot_pipeline_test_support.cjs");

const DEFAULT_SCENARIO_SEED = 20260412;
const DEFAULT_AUTOBOT_SCENARIO_COUNT = 36;
const DEFAULT_AUTOBOT_PIPELINE_SCENARIO_COUNT = 18;
const DEFAULT_SMART_LINK_SCENARIO_COUNT = 30;
const DEFAULT_UNIFIED_AUTOMATION_SCENARIO_COUNT = 50;
const MAX_SCENARIO_COUNT = 500;
const SEMVER_DIMENSIONS = ["none", "patch", "minor", "major"];
const SMART_LINK_RELATION_DIMENSIONS = ["none", "advisory_fix", "closes", "connects", "depends_on", "related"];
const CRITICAL_LABELS = [...new Set(AutobotLabelRegistry.RELEASE_CRITICAL_LABELS)];
const CRITICAL_LABEL_SET = new Set(CRITICAL_LABELS);
const SCENARIO_TIME_ANCHOR = Date.parse("2026-04-13T00:00:00Z");
const PULL_REQUEST_ACTION_WEIGHTS = [
  { value: "opened", weight: 4 },
  { value: "edited", weight: 2 },
  { value: "synchronize", weight: 3 },
  { value: "reopened", weight: 1 }
];
const ISSUE_ACTION_WEIGHTS = [
  { value: "opened", weight: 5 },
  { value: "edited", weight: 2 },
  { value: "reopened", weight: 1 }
];
const TRUSTED_AUTHOR_ASSOCIATION_WEIGHTS = [
  { value: "MEMBER", weight: 5 },
  { value: "COLLABORATOR", weight: 3 },
  { value: "CONTRIBUTOR", weight: 2 }
];
const NEUTRAL_REALISM_WORDS = [
  "artifact",
  "capture",
  "checkpoint",
  "client",
  "context",
  "details",
  "draft",
  "followup",
  "handoff",
  "history",
  "message",
  "notes",
  "observation",
  "operator",
  "preview",
  "reference",
  "report",
  "review",
  "sample",
  "summary",
  "thread",
  "trace",
  "window",
  "workspace"
];

function clampInteger(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function normalizeScenarioKey(value) {
  return String(value || "scenario").trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_");
}

function resolveScenarioConfig(key, defaults = {}) {
  const normalizedKey = normalizeScenarioKey(key);
  const defaultCount = clampInteger(
    defaults.count,
    normalizedKey === "SMART_LINK" ? DEFAULT_SMART_LINK_SCENARIO_COUNT : DEFAULT_AUTOBOT_SCENARIO_COUNT,
    1,
    MAX_SCENARIO_COUNT
  );
  const defaultSeed = clampInteger(defaults.seed, DEFAULT_SCENARIO_SEED, 1, 0xffffffff);
  const count = clampInteger(process.env[`${normalizedKey}_SCENARIO_COUNT`] ?? process.env.SCENARIO_COUNT, defaultCount, 1, MAX_SCENARIO_COUNT);
  const seed = clampInteger(process.env[`${normalizedKey}_SCENARIO_SEED`] ?? process.env.SCENARIO_SEED, defaultSeed, 1, 0xffffffff);
  return {
    count,
    key: normalizedKey,
    seed
  };
}

function parseJson(value) {
  return JSON.parse(String(value || "null"));
}

function stableStringify(value) {
  return JSON.stringify(value, null, 2);
}

function normalizeTempPathSegment(value) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || "scenario";
}

async function withScenarioTempWorkspace(parts, execute) {
  const prefix = `${parts.map(normalizeTempPathSegment).join("-")}-`;
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  try {
    return await execute({
      directory,
      resolveFile(fileName) {
        return path.join(directory, fileName);
      }
    });
  } finally {
    fs.rmSync(directory, { force: true, recursive: true });
  }
}

function uniqueStrings(values) {
  const items = Array.isArray(values) ? values : [values];
  const seen = new Set();
  const result = [];
  for (const item of items) {
    const normalized = String(item ?? "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function normalizeNestedStringArrays(values) {
  return (Array.isArray(values) ? values : [])
    .map((entry) => uniqueStrings(Array.isArray(entry) ? entry : [entry]))
    .filter((entry) => entry.length > 0);
}

function normalizeSemverDecision(value) {
  const normalized = String(value ?? "none").trim().toLowerCase();
  return SEMVER_DIMENSIONS.includes(normalized) ? normalized : "none";
}

function createMatrix(dimensions) {
  return Object.fromEntries(dimensions.map((row) => [
    row,
    Object.fromEntries(dimensions.map((column) => [column, 0]))
  ]));
}

function buildScenarioName(domain, kind, index, seed) {
  return `${domain}:${kind}:${index}:seed-${seed}`;
}

function toIsoTimestamp(daysBack, hourOffset = 0) {
  return new Date(SCENARIO_TIME_ANCHOR - daysBack * 86400000 - hourOffset * 3600000).toISOString();
}

function pickWeightedValue(random, values) {
  const entry = pickWeighted(random, values);
  if (entry && typeof entry === "object" && Object.prototype.hasOwnProperty.call(entry, "value")) {
    return entry.value;
  }
  return entry;
}

function buildNeutralWordSet(random, count) {
  const words = [];
  while (words.length < count) {
    const word = pick(random, NEUTRAL_REALISM_WORDS);
    if (!words.includes(word)) {
      words.push(word);
    }
  }
  return words;
}

function pickNeutralWord(random) {
  return buildNeutralWordSet(random, 1)[0];
}

function appendBodyBlock(body, block) {
  const trimmedBody = String(body || "").trim();
  if (!block) return trimmedBody;
  if (!trimmedBody) return block;
  if (trimmedBody.includes(block)) return trimmedBody;
  return `${trimmedBody}\n\n${block}`;
}

function buildRealismTimeline(random) {
  const sourceDaysBack = randomInt(random, 0, 14);
  const relatedDaysBack = sourceDaysBack + randomInt(random, 1, 30);
  const staleDaysBack = sourceDaysBack + randomInt(random, 60, 180);
  return {
    relatedUpdatedAt: toIsoTimestamp(relatedDaysBack, randomInt(random, 0, 23)),
    sourceUpdatedAt: toIsoTimestamp(sourceDaysBack, randomInt(random, 0, 23)),
    staleUpdatedAt: toIsoTimestamp(staleDaysBack, randomInt(random, 0, 23))
  };
}

function buildRealismBodyBlock(random, index, timeline) {
  const words = buildNeutralWordSet(random, 4);
  return [
    "### Context",
    "",
    `- Review window: ${timeline.sourceUpdatedAt}`,
    `- Operator trace: ${words[0]} ${words[1]} ${index}`,
    `- Notes: ${words[2]} ${words[3]}`
  ].join("\n");
}

function buildPatchBudgetNoise(random) {
  const lines = [];
  for (let index = 0; index < 32; index += 1) {
    const words = buildNeutralWordSet(random, 3);
    lines.push(`+${words[0]}_${index} = "${words[1]} ${words[2]}"`);
  }
  return lines.join("\n");
}

function isPatchBudgetNoiseCandidate(filename) {
  const normalizedPath = String(filename || "").replace(/\\/g, "/").toLowerCase();
  return /^docs\//.test(normalizedPath)
    || /^tests?\//.test(normalizedPath)
    || /^\.github\/workflows\//.test(normalizedPath)
    || /(^|\/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock)$/.test(normalizedPath)
    || /\.md$/.test(normalizedPath);
}

function applyPatchBudgetNoise(files, random) {
  const target = (files || []).find((file) => isPatchBudgetNoiseCandidate(file.filename));
  if (!target || !chance(random, 0.6)) {
    return { mode: "none", target: null };
  }
  const expandedPatch = [String(target.patch || ""), buildPatchBudgetNoise(random)]
    .filter(Boolean)
    .join("\n");
  target.patch = expandedPatch;
  target.rawPatchAvailable = Boolean(expandedPatch);
  return {
    mode: "truncated",
    target: String(target.filename || "")
  };
}

function applyPullRequestRecordRealism(pullRequest, profile) {
  if (!pullRequest) return;
  pullRequest.author_association = pullRequest.author_association || profile.authorAssociation;
  pullRequest.updated_at = pullRequest.updated_at || profile.timeline.sourceUpdatedAt;
  pullRequest.body = appendBodyBlock(pullRequest.body, profile.bodyBlock);
}

function applyIssueRecordRealism(issue, profile) {
  if (!issue) return;
  issue.author_association = issue.author_association || profile.authorAssociation;
  issue.updated_at = issue.updated_at || profile.timeline.sourceUpdatedAt;
  issue.body = appendBodyBlock(issue.body, profile.bodyBlock);
}

function buildScenarioRealismProfile(random, suiteKey, factoryName, index, seed) {
  const timeline = buildRealismTimeline(random);
  return {
    authorAssociation: pickWeightedValue(random, TRUSTED_AUTHOR_ASSOCIATION_WEIGHTS),
    bodyBlock: buildRealismBodyBlock(random, index, timeline),
    factoryName,
    issueAction: pickWeightedValue(random, ISSUE_ACTION_WEIGHTS),
    patchBudgetNoiseMode: "none",
    patchBudgetNoiseTarget: null,
    pullRequestAction: pickWeightedValue(random, PULL_REQUEST_ACTION_WEIGHTS),
    seed,
    suiteKey,
    timeline
  };
}

function applyInputScenarioRealism(scenario, random, profile) {
  if (!scenario.input || !scenario.input.pullRequest) return;
  const pullRequest = scenario.input.pullRequest;
  pullRequest.authorAssociation = pullRequest.authorAssociation || profile.authorAssociation;
  pullRequest.body = appendBodyBlock(pullRequest.body, profile.bodyBlock);
  pullRequest.eventAction = pullRequest.eventAction || profile.pullRequestAction;
  pullRequest.updatedAt = pullRequest.updatedAt || profile.timeline.sourceUpdatedAt;
  if (Array.isArray(scenario.input.files)) {
    const patchBudgetNoise = applyPatchBudgetNoise(scenario.input.files, random);
    profile.patchBudgetNoiseMode = patchBudgetNoise.mode;
    profile.patchBudgetNoiseTarget = patchBudgetNoise.target;
  }
}

function applyContextScenarioRealism(scenario, profile) {
  if (!scenario.context || !scenario.context.payload) return;
  if (scenario.context.eventName === "pull_request" && scenario.context.payload.pull_request) {
    if (String(scenario.context.payload.action || "").toLowerCase() !== "closed") {
      scenario.context.payload.action = profile.pullRequestAction;
    }
    applyPullRequestRecordRealism(scenario.context.payload.pull_request, profile);
  }
  if (scenario.context.eventName === "issues" && scenario.context.payload.issue) {
    if (String(scenario.context.payload.action || "").toLowerCase() !== "closed") {
      scenario.context.payload.action = profile.issueAction;
    }
    applyIssueRecordRealism(scenario.context.payload.issue, profile);
  }
}

function applyGithubOptionsRealism(scenario, random, profile) {
  if (!scenario.githubOptions) return;
  const githubOptions = scenario.githubOptions;
  githubOptions.issueAuthorAssociation = githubOptions.issueAuthorAssociation || profile.authorAssociation;
  githubOptions.issueBody = appendBodyBlock(githubOptions.issueBody, profile.bodyBlock);
  githubOptions.issueUpdatedAt = githubOptions.issueUpdatedAt || profile.timeline.sourceUpdatedAt;
  if (githubOptions.pullRequest) {
    applyPullRequestRecordRealism(githubOptions.pullRequest, profile);
  }
  if (!scenario.autobotScenario && Array.isArray(githubOptions.pullFiles)) {
    const patchBudgetNoise = applyPatchBudgetNoise(githubOptions.pullFiles, random);
    profile.patchBudgetNoiseMode = patchBudgetNoise.mode;
    profile.patchBudgetNoiseTarget = patchBudgetNoise.target;
  }
  for (const issue of githubOptions.additionalIssues || []) {
    issue.author_association = issue.author_association || profile.authorAssociation;
    issue.updated_at = issue.updated_at || profile.timeline.relatedUpdatedAt;
  }
  for (const pullRequest of githubOptions.additionalPullRequests || []) {
    pullRequest.author_association = pullRequest.author_association || profile.authorAssociation;
    pullRequest.updated_at = pullRequest.updated_at || profile.timeline.relatedUpdatedAt;
  }
}

function applySmartLinkScenarioRealism(scenario, profile) {
  if (scenario.source) {
    scenario.source.authorAssociation = scenario.source.authorAssociation || profile.authorAssociation;
    scenario.source.updatedAt = scenario.source.updatedAt || profile.timeline.sourceUpdatedAt;
  }
  if (scenario.candidate) {
    scenario.candidate.updatedAt = scenario.candidate.updatedAt || profile.timeline.relatedUpdatedAt;
  }
  for (const candidate of scenario.candidates || []) {
    candidate.updatedAt = candidate.updatedAt || profile.timeline.relatedUpdatedAt;
  }
}

function selectScenarioFactoryEntries(random, entries, count) {
  const baseline = shuffle(random, entries.slice());
  const selected = baseline.slice(0, Math.min(count, baseline.length));
  while (selected.length < count) {
    selected.push(pickWeighted(random, entries));
  }
  return selected;
}

function attachScenarioRealism(scenario, random, options) {
  const profile = buildScenarioRealismProfile(random, options.suiteKey, options.factoryEntry.name, options.index, options.seed);
  if (scenario.autobotScenario) {
    attachScenarioRealism(scenario.autobotScenario, random, {
      factoryEntry: { name: scenario.autobotScenario.kind },
      index: options.index,
      seed: options.seed,
      suiteKey: "AUTOBOT"
    });
  }
  applyInputScenarioRealism(scenario, random, profile);
  applyContextScenarioRealism(scenario, profile);
  applyGithubOptionsRealism(scenario, random, profile);
  if (options.suiteKey === "SMART_LINK") {
    applySmartLinkScenarioRealism(scenario, profile);
  }
  scenario.realismProfile = {
    authorAssociation: profile.authorAssociation,
    factoryName: profile.factoryName,
    issueAction: profile.issueAction,
    patchBudgetNoiseMode: profile.patchBudgetNoiseMode,
    patchBudgetNoiseTarget: profile.patchBudgetNoiseTarget,
    pullRequestAction: profile.pullRequestAction,
    relatedUpdatedAt: profile.timeline.relatedUpdatedAt,
    sourceUpdatedAt: profile.timeline.sourceUpdatedAt,
    staleUpdatedAt: profile.timeline.staleUpdatedAt,
    weightedSelection: true
  };
  scenario.replayKey = `${options.suiteKey}:seed-${options.seed}:scenario-${options.index}:${options.factoryEntry.name}`;
  return scenario;
}

function buildScenarioCases(key, config, defaultCount, factoryEntries) {
  const resolved = resolveScenarioConfig(key, {
    ...config,
    count: clampInteger(config.count, defaultCount, 1, MAX_SCENARIO_COUNT)
  });
  const random = createSeededRandom(resolved.seed);
  const selectedEntries = selectScenarioFactoryEntries(random, factoryEntries, resolved.count);
  const scenarios = selectedEntries.map((entry, index) => attachScenarioRealism(
    withOptimalResult(entry.factory(random, index, resolved.seed)),
    random,
    {
      factoryEntry: entry,
      index,
      seed: resolved.seed,
      suiteKey: resolved.key
    }
  ));
  return {
    ...resolved,
    scenarios
  };
}

function getScenarioOptimalResult(scenario) {
  if (scenario && scenario.optimalResult && typeof scenario.optimalResult === "object") {
    return scenario.optimalResult;
  }
  if (scenario && scenario.expected && typeof scenario.expected === "object") {
    return scenario.expected;
  }
  return {};
}

function withOptimalResult(scenario) {
  const optimalResult = getScenarioOptimalResult(scenario);
  return {
    ...scenario,
    optimalResult
  };
}

function buildScenarioContract(scenario) {
  const expected = getScenarioOptimalResult(scenario);
  const acceptable = {
    acceptableAutobotCommentIncludesAny: normalizeNestedStringArrays(expected.acceptableAutobotCommentIncludesAny),
    acceptableGraphCommentIncludesAny: normalizeNestedStringArrays(expected.acceptableGraphCommentIncludesAny),
    acceptableIssueLabelSets: normalizeNestedStringArrays(expected.acceptableIssueLabelSets),
    acceptableLabelSets: normalizeNestedStringArrays(expected.acceptableLabelSets),
    acceptableLinkedLabelSets: normalizeNestedStringArrays(expected.acceptableLinkedLabelSets),
    acceptablePullBodyIncludesAny: normalizeNestedStringArrays(expected.acceptablePullBodyIncludesAny),
    acceptableRelationKinds: uniqueStrings(expected.acceptableRelationKinds).map((value) => value.toLowerCase()),
    acceptableSemverDecisions: uniqueStrings(expected.acceptableSemverDecisions).map((value) => normalizeSemverDecision(value)),
    acceptableSmartLinkCommentIncludesAny: normalizeNestedStringArrays(expected.acceptableSmartLinkCommentIncludesAny),
    acceptableSuppressionReasonSets: normalizeNestedStringArrays(expected.acceptableSuppressionReasonSets)
  };
  const allowedCriticalLabels = new Set();
  const negativeCriticalLabels = new Set();
  const positiveCriticalLabels = new Set();

  function isCriticalExpectationLabel(label) {
    return CRITICAL_LABELS.some((criticalLabel) => AutobotLabelRegistry.matchesExpectedLabel(criticalLabel, label));
  }

  for (const label of uniqueStrings(expected.requiredLabels)) {
    if (isCriticalExpectationLabel(label)) {
      positiveCriticalLabels.add(label);
      allowedCriticalLabels.add(label);
    }
  }
  for (const label of uniqueStrings(expected.issueLabelsInclude)) {
    if (isCriticalExpectationLabel(label)) {
      positiveCriticalLabels.add(label);
      allowedCriticalLabels.add(label);
    }
  }
  for (const label of uniqueStrings(expected.linkedLabels)) {
    if (isCriticalExpectationLabel(label)) {
      positiveCriticalLabels.add(label);
      allowedCriticalLabels.add(label);
    }
  }
  for (const group of normalizeNestedStringArrays(expected.requiredAnyLabels)) {
    for (const label of group) {
      if (isCriticalExpectationLabel(label)) {
        allowedCriticalLabels.add(label);
      }
    }
  }
  for (const group of acceptable.acceptableLabelSets.concat(acceptable.acceptableIssueLabelSets, acceptable.acceptableLinkedLabelSets)) {
    for (const label of group) {
      if (isCriticalExpectationLabel(label)) {
        allowedCriticalLabels.add(label);
      }
    }
  }
  for (const label of uniqueStrings(expected.forbiddenLabels).concat(uniqueStrings(expected.issueLabelsExclude))) {
    if (isCriticalExpectationLabel(label)) {
      negativeCriticalLabels.add(label);
    }
  }

  return {
    acceptable,
    expected,
    metrics: {
      allowedCriticalLabels: [...allowedCriticalLabels],
      negativeCriticalLabels: [...negativeCriticalLabels],
      positiveCriticalLabels: [...positiveCriticalLabels],
      primarySemverDecision: expected.semverDecision ? normalizeSemverDecision(expected.semverDecision) : acceptable.acceptableSemverDecisions[0] || null
    }
  };
}

function createValidationResult(contract) {
  return {
    acceptedOutcome: false,
    acceptedReasons: [],
    contract,
    deviations: [],
    warnings: []
  };
}

function acceptValidation(validation, reason) {
  validation.acceptedOutcome = true;
  if (reason) {
    validation.acceptedReasons.push(String(reason));
  }
}

function appendWarning(validation, code, message) {
  if (!validation.warnings.some((warning) => warning.code === code && warning.message === message)) {
    validation.warnings.push({ code, message });
  }
}

function appendWarnings(validation, warnings) {
  for (const warning of warnings || []) {
    appendWarning(validation, warning.code, warning.message);
  }
}

function matchesAnyRequiredSet(actualValues, candidateSets) {
  return candidateSets.some((candidateSet) => candidateSet.every((value) => actualValues.some((actualValue) => AutobotLabelRegistry.matchesExpectedLabel(actualValue, value))));
}

function matchesAnyFragmentGroup(text, groups) {
  const source = String(text || "");
  return groups.some((group) => group.some((fragment) => source.includes(fragment)));
}

function validateTextFragments(validation, input) {
  const actualText = String(input.actualText || "");
  const requiredFragments = uniqueStrings(input.requiredFragments);
  const acceptableFragmentGroups = normalizeNestedStringArrays(input.acceptableFragmentGroups);
  const description = String(input.description || "text");
  const missingFragments = requiredFragments.filter((fragment) => !actualText.includes(fragment));
  if (missingFragments.length > 0) {
    if (matchesAnyFragmentGroup(actualText, acceptableFragmentGroups)) {
      acceptValidation(validation, `${description} matched an acceptable fragment alternative`);
      return;
    }
    for (const fragment of missingFragments) {
      validation.deviations.push(`expected ${description} to include ${fragment}`);
    }
    return;
  }
  if (requiredFragments.length === 0 && acceptableFragmentGroups.length > 0 && !matchesAnyFragmentGroup(actualText, acceptableFragmentGroups)) {
    validation.deviations.push(`expected ${description} to include one of the acceptable fragment alternatives`);
  }
}

function validateLabelCollection(validation, input) {
  const actualLabels = uniqueStrings(input.actualLabels);
  const requiredLabels = uniqueStrings(input.requiredLabels);
  const forbiddenLabels = uniqueStrings(input.forbiddenLabels);
  const requiredAnyLabels = normalizeNestedStringArrays(input.requiredAnyLabels);
  const acceptableLabelSets = normalizeNestedStringArrays(input.acceptableLabelSets);
  const description = String(input.description || "labels");
  const positiveDeviations = [];

  for (const label of requiredLabels) {
    if (!actualLabels.some((actualLabel) => AutobotLabelRegistry.matchesExpectedLabel(actualLabel, label))) {
      positiveDeviations.push(`expected ${description} label ${label} to be present`);
    }
  }
  for (const group of requiredAnyLabels) {
    if (!group.some((label) => actualLabels.some((actualLabel) => AutobotLabelRegistry.matchesExpectedLabel(actualLabel, label)))) {
      positiveDeviations.push(`expected one of ${description} labels ${group.join(", ")} to be present`);
    }
  }
  if (positiveDeviations.length > 0) {
    if (matchesAnyRequiredSet(actualLabels, acceptableLabelSets)) {
      acceptValidation(validation, `${description} matched an acceptable label set`);
    } else {
      validation.deviations.push(...positiveDeviations);
    }
  } else if (requiredLabels.length === 0 && requiredAnyLabels.length === 0 && acceptableLabelSets.length > 0 && !matchesAnyRequiredSet(actualLabels, acceptableLabelSets)) {
    validation.deviations.push(`expected ${description} to match one of the acceptable label sets`);
  }

  for (const label of forbiddenLabels) {
    if (actualLabels.some((actualLabel) => AutobotLabelRegistry.matchesExpectedLabel(actualLabel, label))) {
      validation.deviations.push(`expected ${description} label ${label} to be absent`);
    }
  }
}

function addUnexpectedCriticalLabelWarnings(validation, actualLabels) {
  const allowedCriticalLabels = new Set(validation.contract.metrics.allowedCriticalLabels || []);
  const negativeCriticalLabels = new Set(validation.contract.metrics.negativeCriticalLabels || []);
  for (const label of uniqueStrings(actualLabels).filter((value) => CRITICAL_LABEL_SET.has(value))) {
    if ([...negativeCriticalLabels].some((negativeLabel) => AutobotLabelRegistry.matchesExpectedLabel(label, negativeLabel))) {
      continue;
    }
    const labelMetadata = AutobotLabelRegistry.getLabelMetadata(label);
    if (![...allowedCriticalLabels].some((allowedLabel) => {
      const normalizedAllowedLabel = AutobotLabelRegistry.normalizeLabelName(allowedLabel);
      return AutobotLabelRegistry.matchesExpectedLabel(label, allowedLabel)
        || AutobotLabelRegistry.matchesExpectedLabel(allowedLabel, label)
        || AutobotLabelRegistry.isDescendantLabel(label, allowedLabel)
        || AutobotLabelRegistry.isDescendantLabel(allowedLabel, label)
        || Array.isArray(labelMetadata?.legacyMatches) && labelMetadata.legacyMatches.includes(normalizedAllowedLabel);
    })) {
      appendWarning(validation, "unexpected-critical-label", `unexpected critical label ${label} present outside the scenario contract`);
    }
  }
}

function validateAutobotScenarioContract(scenario, result) {
  const contract = buildScenarioContract(scenario);
  const validation = createValidationResult(contract);
  const labels = uniqueStrings(parseJson(result.deterministic_labels_json));
  const semver = parseJson(result.deterministic_semver_json);
  const actualDecision = normalizeSemverDecision(semver.decision);
  const acceptableSemverDecisions = contract.acceptable.acceptableSemverDecisions;

  if (contract.expected.semverDecision) {
    const expectedDecision = normalizeSemverDecision(contract.expected.semverDecision);
    if (actualDecision !== expectedDecision) {
      if (acceptableSemverDecisions.includes(actualDecision)) {
        acceptValidation(validation, `semver ${actualDecision} matched an acceptable outcome`);
      } else {
        validation.deviations.push(`expected semver ${expectedDecision} but got ${actualDecision}`);
      }
    }
  } else if (acceptableSemverDecisions.length > 0 && !acceptableSemverDecisions.includes(actualDecision)) {
    validation.deviations.push(`expected semver one of ${acceptableSemverDecisions.join(", ")} but got ${actualDecision}`);
  }

  if (Array.isArray(contract.expected.hardSignals)) {
    const actualHardSignals = uniqueStrings(Array.isArray(semver.hardSignals) ? semver.hardSignals : []).slice().sort();
    const expectedHardSignals = uniqueStrings(contract.expected.hardSignals).slice().sort();
    if (JSON.stringify(actualHardSignals) !== JSON.stringify(expectedHardSignals)) {
      validation.deviations.push(`expected hard signals ${expectedHardSignals.join(", ") || "(none)"} but got ${actualHardSignals.join(", ") || "(none)"}`);
    }
  }

  validateLabelCollection(validation, {
    actualLabels: labels,
    acceptableLabelSets: contract.acceptable.acceptableLabelSets,
    description: "scenario",
    forbiddenLabels: contract.expected.forbiddenLabels,
    requiredAnyLabels: contract.expected.requiredAnyLabels,
    requiredLabels: contract.expected.requiredLabels
  });
  addUnexpectedCriticalLabelWarnings(validation, labels);
  return validation;
}

function createSmartLinkEntity(overrides = {}) {
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

function createMilestone(number, title) {
  return { number, title };
}

function buildCodeIdentifier(random, parts = 2) {
  return buildNeutralWordSet(random, parts).join("_");
}

function buildWorkflowPatch(random, token) {
  return [
    "+on:",
    "+  workflow_dispatch:",
    "+  schedule:",
    '+    - cron: "0 5 * * *"',
    "+jobs:",
    `+  ${token}:`,
    "+    uses: actions/github-script@v9"
  ].join("\n");
}

function buildRuleVocabularyPatch(token) {
  return [
    `+  "additive-api-contract": Object.freeze({ description: "${token} api" }),`,
    `+  "runtime-support-added": Object.freeze({ description: "${token} runtime" }),`,
    `+  "compatibility-shim": Object.freeze({ description: "${token} compatibility" }),`,
    `+  "security-auth-change": Object.freeze({ description: "${token} security" }),`
  ].join("\n");
}

function buildFixtureVocabularyPatch(token) {
  return [
    `+const expected = { semverDecision: "minor", hardSignals: ["additive-api-contract"] };`,
    `+const branch = "security/${token}-runtime";`,
    `+const summary = "compatibility api runtime security";`
  ].join("\n");
}

function buildPythonPatch(functionName, returnValue) {
  return [
    `+def ${functionName}(payload):`,
    `+    return ${returnValue}`
  ].join("\n");
}

function deriveExpectedPipelineSemverDecision(scenario) {
  if (scenario.pipelineSemverDecision !== undefined) {
    return normalizeSemverDecision(scenario.pipelineSemverDecision);
  }
  if (!scenario.pipelineReleaseRelevant) {
    return "none";
  }
  const expected = getScenarioOptimalResult(scenario);
  return normalizeSemverDecision(expected.semverDecision);
}

function deriveExpectedPipelineMilestoneTitle(scenario) {
  if (scenario.pipelineMilestoneTitle !== undefined) {
    return scenario.pipelineMilestoneTitle;
  }
  if (!scenario.pipelineReleaseRelevant) {
    return null;
  }
  const expected = getScenarioOptimalResult(scenario);
  if (expected.semverDecision === "major") return "v1.0.0";
  if (expected.semverDecision === "minor") return "v0.1.0";
  if (expected.semverDecision === "patch") return "v0.0.2";
  return null;
}

function deriveExpectedSharedLabel(scenario) {
  const expected = getScenarioOptimalResult(scenario);
  if (Array.isArray(expected.requiredLabels) && expected.requiredLabels.length > 0) {
    return expected.requiredLabels[0];
  }
  if (Array.isArray(expected.requiredAnyLabels)) {
    const group = expected.requiredAnyLabels.find((entry) => Array.isArray(entry) && entry.length > 0);
    if (group) return group[0];
  }
  return "runtime";
}

function buildAutobotPipelinePullRequest(scenario) {
  const pullRequest = scenario.input.pullRequest;
  const head = {
    ref: String(pullRequest.headRef || `autobot/${pullRequest.number}`),
    sha: `sha-${pullRequest.number}`
  };
  if (pullRequest.headRepoFullName) {
    head.repo = { full_name: String(pullRequest.headRepoFullName) };
  }
  return {
    author_association: String(pullRequest.authorAssociation || "MEMBER"),
    base: { ref: "main" },
    body: String(pullRequest.body || ""),
    head,
    html_url: `https://example.invalid/pull/${pullRequest.number}`,
    id: pullRequest.number * 10,
    labels: [],
    milestone: null,
    number: pullRequest.number,
    state: "open",
    title: String(pullRequest.title || ""),
    updated_at: String(pullRequest.updatedAt || "2026-04-13T00:00:00Z")
  };
}

function buildAutobotPipelineContext(pullRequest) {
  return {
    eventName: "pull_request",
    issue: { number: pullRequest.number },
    payload: {
      action: String(pullRequest.eventAction || "opened"),
      pull_request: pullRequest
    }
  };
}

function parseVersionTag(rawVersion) {
  const match = String(rawVersion || "").trim().match(/^v?(\d+)\.(\d+)\.(\d+)$/);
  if (!match) {
    return { major: 0, minor: 0, patch: 1 };
  }
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3])
  };
}

function formatVersionTag(version) {
  return `v${version.major}.${version.minor}.${version.patch}`;
}

function bumpPatchVersion(rawVersion) {
  const version = parseVersionTag(rawVersion);
  version.patch += 1;
  return formatVersionTag(version);
}

function derivePrimaryPackageName(files) {
  const sourceFile = (files || [])
    .map((file) => String(file.filename || "").replace(/\\/g, "/"))
    .find((filename) => filename.startsWith("src/"));
  if (!sourceFile) return "repo";
  const parts = sourceFile.split("/").filter(Boolean);
  return parts[1] || "repo";
}

function buildMockIssueRecordFromEntity(entity, overrides = {}) {
  const issueNumber = Number(overrides.number || entity.number || 0);
  return {
    author_association: String(overrides.author_association || entity.authorAssociation || "MEMBER"),
    body: overrides.body !== undefined ? String(overrides.body) : String(entity.body || ""),
    html_url: String(overrides.html_url || entity.htmlUrl || `https://example.invalid/issues/${issueNumber}`),
    id: Number(overrides.id || entity.id || issueNumber * 100),
    labels: overrides.labels || entity.labels || [],
    milestone: overrides.milestone !== undefined ? overrides.milestone : entity.milestone,
    number: issueNumber,
    repository_full_name: String(overrides.repository_full_name || entity.repoFullName || "octo/repo"),
    state: String(overrides.state || entity.state || "open"),
    title: String(overrides.title || entity.title || `Issue ${issueNumber}`),
    updated_at: String(overrides.updated_at || entity.timestamps?.updatedAt || "2026-04-13T00:00:00Z"),
    ...(overrides.pull_request || entity.kind === "pull_request"
      ? { pull_request: overrides.pull_request || { url: String(overrides.html_url || entity.htmlUrl || `https://example.invalid/pull/${issueNumber}`) } }
      : {})
  };
}

function buildMockPullRequestRecordFromEntity(entity, overrides = {}) {
  const pullNumber = Number(overrides.number || entity.number || 0);
  const pullRequest = {
    author_association: String(overrides.author_association || entity.authorAssociation || "MEMBER"),
    base: { ref: String(overrides.baseRef || "main") },
    body: overrides.body !== undefined ? String(overrides.body) : String(entity.body || ""),
    head: {
      ref: String(overrides.headRef || entity.branch || `feature/${pullNumber}`),
      sha: String(overrides.headSha || entity.metadata?.headSha || `sha-${pullNumber}`)
    },
    html_url: String(overrides.html_url || entity.htmlUrl || `https://example.invalid/pull/${pullNumber}`),
    id: Number(overrides.id || entity.id || pullNumber * 10),
    labels: overrides.labels || entity.labels || [],
    milestone: overrides.milestone !== undefined ? overrides.milestone : entity.milestone,
    number: pullNumber,
    state: String(overrides.state || entity.state || "open"),
    title: String(overrides.title || entity.title || `Pull ${pullNumber}`),
    updated_at: String(overrides.updated_at || entity.timestamps?.updatedAt || "2026-04-13T00:00:00Z")
  };
  if (overrides.merged !== undefined) {
    pullRequest.merged = Boolean(overrides.merged);
  }
  if (overrides.headRepoFullName) {
    pullRequest.head.repo = { full_name: String(overrides.headRepoFullName) };
  }
  return pullRequest;
}

function createVersionMilestoneData(number, title, overrides = {}) {
  return {
    closed_issues: Number(overrides.closed_issues ?? 0),
    description: String(overrides.description || ""),
    number,
    open_issues: Number(overrides.open_issues ?? 0),
    state: String(overrides.state || "open"),
    title
  };
}

function buildStaleSmartLinkBlock(label) {
  return [
    "User content above.",
    "",
    "<!-- smart-autolinker:start -->",
    "### Smart Link Intelligence",
    "",
    `Stale ${label} block.`,
    "<!-- smart-autolinker:end -->",
    "",
    "User content below."
  ].join("\n");
}

function buildGraphCommentSeed(sourceKind, sourceNumber) {
  return {
    body: [
      `<!-- smart-link-graph:${sourceKind}:${sourceNumber} -->`,
      "### Relationship Graph",
      "",
      "```mermaid",
      "graph TD",
      "```"
    ].join("\n"),
    id: 7000 + sourceNumber,
    user: { login: "github-actions[bot]", type: "Bot" }
  };
}

function buildIssuePayloadFromRecord(issueRecord) {
  const payload = { ...issueRecord };
  delete payload.pull_request;
  return payload;
}

function createMaintenanceSnapshot(random, index, seed) {
  const sourcePackages = ["server", "balancer", "shared"];
  const dependencyFiles = ["package-lock.json", "pnpm-lock.yaml", "yarn.lock"];
  const workflowFiles = [".github/workflows/autobot.yml", ".github/workflows/release.yml", ".github/workflows/smart-link.yml"];
  const noise = buildNeutralWordSet(random, 3);
  return {
    expected: {
      forbiddenLabels: ["enhancement"],
      hardSignals: [],
      requiredAnyLabels: [["dependencies", "vulnerability", "compliance", "hardening", "pen-test"]],
      requiredLabels: ["dependencies"],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 200 + index,
        title: `Patch vulnerable dependencies ${noise[0]} ${noise[1]}`,
        body: "",
        headRef: `security/patch-${index}`
      },
      totals: {
        filesChanged: 5,
        additions: 170 + index,
        deletions: 90 + index,
        totalChanges: 260 + index * 2
      },
      files: [
        {
          filename: pick(random, workflowFiles),
          status: "modified",
          additions: 18,
          deletions: 14,
          patch: [
            `-    runs-on: ${pick(random, ["windows-latest", "ubuntu-22.04", "macos-latest"])}`,
            `-    runs-on: ${pick(random, ["windows-latest", "ubuntu-latest", "macos-13"])}`,
            "+    uses: actions/checkout@v6",
            "+    uses: actions/github-script@v9"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: pick(random, dependencyFiles),
          status: "modified",
          additions: 32,
          deletions: 28,
          patch: [
            `-      "os": ["${pick(random, ["linux", "win32", "darwin"])}"]`,
            `+      "integrity": "sha512-updated"`,
            `+      "version": "1.${index}.3"`
          ].join("\n"),
          rawPatchAvailable: true
        },
        ...sourcePackages.map((packageName, offset) => ({
          filename: `src/${packageName}/${noise[offset % noise.length]}_${index}.py`,
          status: "added",
          additions: 30 + randomInt(random, 0, 12),
          deletions: 0,
          patch: `+def ${packageName}_${noise[offset % noise.length]}_${index}():\n+    return "patched"`,
          rawPatchAvailable: true
        }))
      ]
    },
    kind: "maintenance",
    name: buildScenarioName("autobot", "maintenance", index, seed)
    ,pipelineReleaseRelevant: true
  };
}

function createRuntimeDropSnapshot(random, index, seed) {
  const runtimeRequirement = pick(random, [">=3.9", ">=3.10", ">=3.11"]);
  return {
    expected: {
      acceptableLabelSets: [["breaking-change", "runtime"]],
      hardSignals: ["runtime-support-dropped"],
      requiredLabels: ["runtime"],
      semverDecision: "major"
    },
    input: {
      pullRequest: {
        number: 400 + index,
        title: `Drop runtime support ${index}`,
        body: "",
        headRef: `breaking/runtime-${index}`
      },
      totals: {
        filesChanged: 1,
        additions: 0,
        deletions: 1,
        totalChanges: 1
      },
      files: [
        {
          filename: pick(random, ["pyproject.toml", "setup.cfg", "tox.ini"]),
          status: "modified",
          additions: 0,
          deletions: 1,
          patch: `-requires-python = "${runtimeRequirement}"`,
          rawPatchAvailable: true
        }
      ]
    },
    kind: "runtime-drop",
    name: buildScenarioName("autobot", "runtime-drop", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createAdditiveFeatureSnapshot(random, index, seed) {
  const featureWords = buildNeutralWordSet(random, 2);
  return {
    expected: {
      requiredAnyLabels: [["api", "enhancement"]],
      semverDecision: "minor"
    },
    input: {
      pullRequest: {
        number: 600 + index,
        title: `Feature: add ${featureWords[0]} ${featureWords[1]} endpoint`,
        body: "",
        headRef: `feature/${featureWords.join("-")}-${index}`
      },
      totals: {
        filesChanged: 2,
        additions: 130 + index,
        deletions: 6,
        totalChanges: 136 + index
      },
      files: [
        {
          filename: `src/repo/api/${featureWords[0]}_${index}.py`,
          status: "added",
          additions: 95,
          deletions: 0,
          patch: [
            `+def create_${featureWords[0]}_${index}(request):`,
            `+    return {"status": "ok"}`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: "src/repo/__init__.py",
          status: "modified",
          additions: 3,
          deletions: 0,
          patch: `+from .api.${featureWords[0]}_${index} import create_${featureWords[0]}_${index}`,
          rawPatchAvailable: true
        }
      ]
    },
    kind: "additive-feature",
    name: buildScenarioName("autobot", "additive-feature", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createSecurityPatchSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      forbiddenLabels: ["enhancement"],
      requiredAnyLabels: [["security", "vulnerability", "compliance", "hardening", "pen-test"], ["dependencies"]],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 800 + index,
        title: `Patch ${token} vulnerability`,
        body: "",
        headRef: `security/${token}-${index}`
      },
      totals: {
        filesChanged: 3,
        additions: 70 + index,
        deletions: 18,
        totalChanges: 88 + index
      },
      files: [
        {
          filename: pick(random, ["package-lock.json", "pnpm-lock.yaml", "yarn.lock"]),
          status: "modified",
          additions: 14,
          deletions: 12,
          patch: [
            `+      "integrity": "sha512-hardened"`,
            `+      "version": "2.${index}.1"`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/security/${token}_${index}.py`,
          status: "modified",
          additions: 20,
          deletions: 3,
          patch: [
            `+def sanitize_${token}(value):`,
            "+    if not value:",
            `+        raise ValueError("missing credential")`,
            "+    return value"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/test_${token}_${index}.py`,
          status: "modified",
          additions: 12,
          deletions: 3,
          patch: "+def test_security_patch():\n+    assert True",
          rawPatchAvailable: true
        }
      ]
    },
    kind: "security-patch",
    name: buildScenarioName("autobot", "security-patch", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createBreakingContractSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      requiredAnyLabels: [["api", "compatibility"]],
      requiredLabels: ["breaking-change"],
      semverDecision: "major"
    },
    input: {
      pullRequest: {
        number: 1000 + index,
        title: `Remove legacy ${token} public adapter`,
        body: "Breaking change for legacy consumers.",
        headRef: `breaking/remove-${token}-${index}`
      },
      totals: {
        filesChanged: 5,
        additions: 54,
        deletions: 172,
        totalChanges: 226
      },
      files: [
        {
          filename: `src/repo/compat/legacy_${token}.py`,
          status: "removed",
          additions: 0,
          deletions: 58,
          patch: [
            `-def legacy_${token}(payload):`,
            "-    return payload"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/api/legacy_${token}_contract.py`,
          status: "removed",
          additions: 0,
          deletions: 42,
          patch: [
            `-class Legacy${token[0].toUpperCase()}${token.slice(1)}Contract:`,
            "-    pass"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/legacy_${token}_adapter.py`,
          status: "renamed",
          additions: 8,
          deletions: 31,
          patch: [
            "-# backward compatibility shim removed",
            "+# breaking change: adapter renamed"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: "src/repo/__init__.py",
          status: "modified",
          additions: 2,
          deletions: 3,
          patch: [
            `-from .compat.legacy_${token} import legacy_${token}`,
            `+from .api.${token}_adapter import create_${token}_adapter`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/api/${token}_adapter.py`,
          status: "added",
          additions: 44,
          deletions: 0,
          patch: [
            `+def create_${token}_adapter(request):`,
            `+    return {"status": "migrated"}`
          ].join("\n"),
          rawPatchAvailable: true
        }
      ]
    },
    kind: "breaking-contract",
    name: buildScenarioName("autobot", "breaking-contract", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createRuntimeSupportAddSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  const pythonVersion = pick(random, ["3.12", "3.13"]);
  return {
    expected: {
      hardSignals: ["docker-runtime-expansion", "runtime-support-added"],
      requiredLabels: ["runtime", "docker"],
      semverDecision: "minor"
    },
    input: {
      pullRequest: {
        number: 1200 + index,
        title: `Add Python ${pythonVersion} runtime support for ${token}`,
        body: "",
        headRef: `feature/runtime-${token}-${index}`
      },
      totals: {
        filesChanged: 4,
        additions: 122,
        deletions: 8,
        totalChanges: 130
      },
      files: [
        {
          filename: "pyproject.toml",
          status: "modified",
          additions: 2,
          deletions: 0,
          patch: [
            `+requires-python = ">=${pythonVersion}"`,
            `+classifiers = ["Programming Language :: Python :: 3.12"]`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `docker/${token}/Dockerfile`,
          status: "modified",
          additions: 6,
          deletions: 1,
          patch: [
            `+FROM python:${pythonVersion}-slim`,
            "+RUN apt-get update && apt-get install -y ffmpeg"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/runtime/${token}_support.py`,
          status: "added",
          additions: 48,
          deletions: 0,
          patch: [
            `+def enable_${token}_runtime():`,
            `+    return "python-${pythonVersion}"`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/test_runtime_${token}_${index}.py`,
          status: "added",
          additions: 24,
          deletions: 0,
          patch: "+def test_runtime_support_added():\n+    assert True",
          rawPatchAvailable: true
        }
      ]
    },
    kind: "runtime-support-add",
    name: buildScenarioName("autobot", "runtime-support-add", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createWorkflowAutomationSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      forbiddenLabels: ["enhancement"],
      requiredLabels: ["test", "workflow"],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 1400 + index,
        title: `Refine autobot workflow orchestration for ${token}`,
        body: "",
        headRef: `automation/workflow-${token}-${index}`
      },
      totals: {
        filesChanged: 4,
        additions: 92,
        deletions: 44,
        totalChanges: 136
      },
      files: [
        {
          filename: pick(random, [".github/workflows/autobot.yml", ".github/workflows/release.yml"]),
          status: "modified",
          additions: 22,
          deletions: 12,
          patch: [
            "+  workflow_dispatch:",
            "+  schedule:",
            '+    - cron: "0 5 * * *"',
            "+    runs-on: ubuntu-latest"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/scripts/autobot_triage.cjs",
          status: "added",
          additions: 38,
          deletions: 0,
          patch: [
            `+function triage_${token}() {`,
            `+  return "automation";`,
            "+}"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/ISSUE_TEMPLATE/bug.yml",
          status: "modified",
          additions: 8,
          deletions: 4,
          patch: [
            "+labels:",
            "+  - bug"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/autobot_${token}_${index}.test.cjs`,
          status: "added",
          additions: 24,
          deletions: 0,
          patch: `+test("workflow sync", () => {\n+  assert.equal(true, true);\n+});`,
          rawPatchAvailable: true
        }
      ]
    },
    kind: "workflow-automation",
    name: buildScenarioName("autobot", "workflow-automation", index, seed),
    pipelineReleaseRelevant: false
  };
}

function createCompatibilityShimSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      requiredLabels: ["compatibility"],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 1600 + index,
        title: `Add backward compatibility shim for ${token}`,
        body: "",
        headRef: `compat/${token}-${index}`
      },
      totals: {
        filesChanged: 3,
        additions: 74,
        deletions: 6,
        totalChanges: 80
      },
      files: [
        {
          filename: `src/repo/compat/${token}_shim.py`,
          status: "added",
          additions: 36,
          deletions: 0,
          patch: [
            `+def ${token}_shim(payload):`,
            "+    # backward compatibility shim",
            "+    return payload"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/test_${token}_shim_${index}.py`,
          status: "added",
          additions: 16,
          deletions: 0,
          patch: "+def test_backward_compatibility_shim():\n+    assert True",
          rawPatchAvailable: true
        },
        {
          filename: `docs/reference/${token}_shim.md`,
          status: "added",
          additions: 12,
          deletions: 0,
          patch: "+This shim preserves backward compatibility for legacy callers.",
          rawPatchAvailable: true
        }
      ]
    },
    kind: "compatibility-shim",
    name: buildScenarioName("autobot", "compatibility-shim", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createDatabaseMigrationSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      acceptableLabelSets: [["breaking-change"]],
      requiredLabels: ["schema"],
      requiredAnyLabels: [["database", "migration"]],
      semverDecision: "major"
    },
    input: {
      pullRequest: {
        number: 1800 + index,
        title: `Reshape ${token} persistence schema`,
        body: "Migration required for deployed environments.",
        headRef: `database/${token}-${index}`
      },
      totals: {
        filesChanged: 4,
        additions: 112,
        deletions: 48,
        totalChanges: 160
      },
      files: [
        {
          filename: `migrations/${token}_${index}.sql`,
          status: "added",
          additions: 18,
          deletions: 0,
          patch: [
            "+ALTER TABLE repo_records DROP COLUMN legacy_value;",
            "+ALTER TABLE repo_records ADD COLUMN normalized_value TEXT;"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/database/${token}_repository.py`,
          status: "modified",
          additions: 28,
          deletions: 14,
          patch: [
            "+def migrate_records(connection):",
            "+    return connection.execute('alter table')"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `src/repo/schema/${token}_record.py`,
          status: "modified",
          additions: 12,
          deletions: 5,
          patch: [
            "+class RecordSchema:",
            "+    normalized_value: str"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/test_${token}_migration_${index}.py`,
          status: "added",
          additions: 18,
          deletions: 0,
          patch: "+def test_database_migration():\n+    assert True",
          rawPatchAvailable: true
        }
      ]
    },
    kind: "database-migration",
    name: buildScenarioName("autobot", "database-migration", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createInfrastructureEchoSnapshot(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  return {
    expected: {
      forbiddenLabels: ["api", "compatibility", "enhancement", "runtime", "security"],
      requiredLabels: ["dependencies", "test", "workflow"],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 2000 + index,
        title: `Refine autobot maintenance scaffolding for ${token}`,
        body: "",
        headRef: `autobot/${token}-${index}`
      },
      totals: {
        filesChanged: 5,
        additions: 238,
        deletions: 94,
        totalChanges: 332
      },
      files: [
        {
          filename: ".github/scripts/autobot/measurement/scorer.cjs",
          status: "modified",
          additions: 70,
          deletions: 18,
          patch: buildRuleVocabularyPatch(token),
          rawPatchAvailable: true
        },
        {
          filename: ".github/scripts/autobot/project_manager.cjs",
          status: "modified",
          additions: 34,
          deletions: 10,
          patch: [
            `+const releaseSummary = "${token} security runtime api compatibility";`,
            `+const previousSignal = "additive-api-contract";`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/workflows/autobot.yml",
          status: "modified",
          additions: 18,
          deletions: 6,
          patch: buildWorkflowPatch(random, token),
          rawPatchAvailable: true
        },
        {
          filename: "tests/deterministic_scenario_engine.cjs",
          status: "modified",
          additions: 48,
          deletions: 14,
          patch: buildFixtureVocabularyPatch(token),
          rawPatchAvailable: true
        },
        {
          filename: "package-lock.json",
          status: "modified",
          additions: 30,
          deletions: 18,
          patch: [
            `+      "version": "2.${index}.4"`,
            `+      "integrity": "sha512-${token}"`
          ].join("\n"),
          rawPatchAvailable: true
        }
      ]
    },
    kind: "infrastructure-echo",
    name: buildScenarioName("autobot", "infrastructure-echo", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createFixtureVocabularyTrapSnapshot(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  return {
    expected: {
      forbiddenLabels: ["api", "compatibility", "runtime", "security"],
      requiredLabels: ["test"],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 2200 + index,
        title: `Refresh fixture coverage for ${token}`,
        body: "",
        headRef: `tests/${token}-${index}`
      },
      totals: {
        filesChanged: 4,
        additions: 146,
        deletions: 39,
        totalChanges: 185
      },
      files: [
        {
          filename: "tests/autobot_scripts.test.cjs",
          status: "modified",
          additions: 42,
          deletions: 12,
          patch: buildFixtureVocabularyPatch(token),
          rawPatchAvailable: true
        },
        {
          filename: "tests/javascript_test_harness.cjs",
          status: "modified",
          additions: 24,
          deletions: 8,
          patch: [
            `+const title = "${token} api runtime compatibility security";`,
            `+const branch = "feature/${token}-runtime";`
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: `tests/test_${token}_fixture_${index}.py`,
          status: "added",
          additions: 28,
          deletions: 0,
          patch: buildPythonPatch(`test_${token}_fixture_${index}`, "True"),
          rawPatchAvailable: true
        },
        {
          filename: `docs/reference/${token}_fixture.md`,
          status: "added",
          additions: 16,
          deletions: 0,
          patch: `+Fixture vocabulary covers api runtime compatibility and security markers for ${token}.`,
          rawPatchAvailable: true
        }
      ]
    },
    kind: "fixture-vocabulary-trap",
    name: buildScenarioName("autobot", "fixture-vocabulary-trap", index, seed),
    pipelineReleaseRelevant: false
  };
}

function createPatchUnavailableFeatureSnapshot(random, index, seed) {
  const token = pickNeutralWord(random);
  return {
    expected: {
      forbiddenLabels: ["breaking-change", "security"],
      requiredAnyLabels: [["api", "enhancement"]],
      semverDecision: "minor"
    },
    input: {
      pullRequest: {
        number: 2400 + index,
        title: `Feature: add ${token} preview endpoint with oversized diff`,
        body: "GitHub did not provide inline patches for the largest files in this change.",
        headRef: `feature/${token}-oversized-${index}`
      },
      totals: {
        filesChanged: 2,
        additions: 148,
        deletions: 4,
        totalChanges: 152
      },
      files: [
        {
          filename: `src/repo/api/${token}_preview_${index}.py`,
          status: "added",
          additions: 124,
          deletions: 0,
          patch: "",
          rawPatchAvailable: false
        },
        {
          filename: "src/repo/__init__.py",
          status: "modified",
          additions: 24,
          deletions: 4,
          patch: "",
          rawPatchAvailable: false
        }
      ]
    },
    kind: "patch-unavailable-feature",
    name: buildScenarioName("autobot", "patch-unavailable-feature", index, seed),
    pipelineReleaseRelevant: true
  };
}

function createLabelRaceRecoverySnapshot(random, index, seed) {
  const scenario = createWorkflowAutomationSnapshot(random, 900 + index, seed);
  const transientLabels = ["test", "workflow"];
  return {
    ...scenario,
    expected: {
      ...scenario.expected,
      failureOperations: ["issues.addLabels"]
    },
    githubOptions: {
      failures: {
        "issues.addLabels": {
          data: {
            errors: transientLabels.map((label) => ({
              code: "invalid",
              field: "name",
              resource: "Label",
              value: label
            }))
          },
          message: "label provisioning race",
          status: 422,
          times: 1
        }
      }
    },
    kind: "label-race-recovery",
    name: buildScenarioName("autobot", "label-race-recovery", index, seed)
  };
}

function createAdversarialEmptyPayloadSnapshot(random, index, seed) {
  return {
    expected: {
      forbiddenLabels: ["api", "rest", "route", "enhancement", "breaking-change"],
      requiredAnyLabels: [["github", "workflow", "automation", "config", "tooling", "chore"]],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 7000 + index,
        title: "Empty patch edge case",
        body: "",
        headRef: `adversarial/empty-${index}`
      },
      totals: {
        filesChanged: 2,
        additions: 0,
        deletions: 0,
        totalChanges: 0
      },
      files: [
        {
          filename: ".github/workflows/check.yml",
          status: "modified",
          additions: 0,
          deletions: 0,
          patch: "",
          rawPatchAvailable: false
        },
        {
          filename: ".github/scripts/autobot/constants.cjs",
          status: "modified",
          additions: 0,
          deletions: 0,
          patch: "",
          rawPatchAvailable: false
        }
      ]
    },
    kind: "adversarial-empty-payload",
    name: buildScenarioName("autobot", "adversarial-empty-payload", index, seed)
  };
}

function createAdversarialCrossDomainFalsePositiveSnapshot(random, index, seed) {
  const workflowFiles = [".github/workflows/pr-size.yml", ".github/workflows/check.yml", ".github/workflows/autobot.yml"];
  return {
    expected: {
      forbiddenLabels: ["api", "rest", "route", "route param", "query param", "graphql", "webhook"],
      requiredAnyLabels: [["github", "workflow", "automation"]],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 7100 + index,
        title: "Update workflow configuration",
        body: "",
        headRef: `adversarial/cross-domain-${index}`
      },
      totals: {
        filesChanged: 3,
        additions: 40,
        deletions: 20,
        totalChanges: 60
      },
      files: workflowFiles.map((filename, offset) => ({
        filename,
        status: "modified",
        additions: 12 + offset,
        deletions: 6 + offset,
        patch: [
          "+    uses: actions/github-script@v9",
          "+        await github.rest.issues.createComment({",
          "+          owner: context.repo.owner,",
          "+          repo: context.repo.repo,",
          "+          issue_number: context.issue.number,",
          "+          body: `Route check: http endpoint verified`",
          "+        });",
          "+    runs-on: ubuntu-latest"
        ].join("\n"),
        rawPatchAvailable: true
      }))
    },
    kind: "adversarial-cross-domain-false-positive",
    name: buildScenarioName("autobot", "adversarial-cross-domain-fp", index, seed)
  };
}

function createAdversarialConflictingEvidenceSnapshot(random, index, seed) {
  const conflictingKeywords = ["route", "rest", "endpoint", "webhook", "graphql", "resolver"];
  const keyword = pick(random, conflictingKeywords);
  return {
    expected: {
      forbiddenLabels: ["api", "rest", "route", "graphql", "webhook"],
      requiredAnyLabels: [["github", "workflow", "automation", "test"]],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 7200 + index,
        title: `Autobot infrastructure update ${index}`,
        body: "",
        headRef: `adversarial/conflict-${index}`
      },
      totals: {
        filesChanged: 4,
        additions: 80,
        deletions: 40,
        totalChanges: 120
      },
      files: [
        {
          filename: ".github/scripts/autobot/measurement/scorer.cjs",
          status: "modified",
          additions: 30,
          deletions: 15,
          patch: [
            `+  "${keyword}": Object.freeze({ description: "api ${keyword} ${keyword}" }),`,
            `+  "rest-${keyword}": Object.freeze({ weight: 1.0, description: "http route endpoint" }),`,
            "+  const routeScore = evidenceScores.api || 0;",
            "+  const webhookScore = evidenceScores.webhook || 0;",
            "+  const graphqlEndpoint = evidenceScores.graphql || 0;"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/scripts/autobot/pr_analysis.cjs",
          status: "modified",
          additions: 20,
          deletions: 10,
          patch: [
            "+function hasApiEvidence(normalizedPath, patch) {",
            "+  return hasApiPathEvidence(normalizedPath);",
            "+}",
            "+function hasRouteParameterEvidence(text) {",
            "+  return /route param/.test(text);",
            "+}"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: "tests/autobot_scripts.test.cjs",
          status: "modified",
          additions: 20,
          deletions: 10,
          patch: [
            '+test("rest route detection excludes workflow files", () => {',
            "+  const result = analyzePullRequestSnapshotData(snapshot);",
            "+  assert.ok(!labels.includes('rest'));",
            "+  assert.ok(!labels.includes('route'));",
            "+  assert.ok(!labels.includes('api'));",
            "+});"
          ].join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/workflows/autobot.yml",
          status: "modified",
          additions: 10,
          deletions: 5,
          patch: [
            "+      - name: Run adversarial scenario coverage",
            "+        run: node --test tests/autobot_adversarial.test.cjs",
            "+    uses: actions/github-script@v9",
            "+          await github.rest.pulls.createReview({"
          ].join("\n"),
          rawPatchAvailable: true
        }
      ]
    },
    kind: "adversarial-conflicting-evidence",
    name: buildScenarioName("autobot", "adversarial-conflicting-evidence", index, seed)
  };
}

function createAdversarialMaxBoundarySnapshot(random, index, seed) {
  const maxLines = [];
  for (let lineIndex = 0; lineIndex < 200; lineIndex += 1) {
    const neutralWord = pickNeutralWord(random);
    maxLines.push(`+${neutralWord}_${lineIndex} = "value_${lineIndex}"`);
  }
  return {
    expected: {
      forbiddenLabels: ["api", "rest", "route"],
      requiredAnyLabels: [["github", "workflow", "automation", "tooling"]],
      semverDecision: "patch"
    },
    input: {
      pullRequest: {
        number: 7300 + index,
        title: "Large workflow refactor",
        body: "",
        headRef: `adversarial/boundary-${index}`
      },
      totals: {
        filesChanged: 2,
        additions: 200,
        deletions: 50,
        totalChanges: 250
      },
      files: [
        {
          filename: ".github/workflows/autobot.yml",
          status: "modified",
          additions: 200,
          deletions: 50,
          patch: maxLines.join("\n"),
          rawPatchAvailable: true
        },
        {
          filename: ".github/scripts/autobot/unified.cjs",
          status: "modified",
          additions: 30,
          deletions: 10,
          patch: [
            "+function resolveUnifiedAutomationFiles(input) {",
            "+  return { autobotSnapshotFile: input.stateFile };",
            "+}"
          ].join("\n"),
          rawPatchAvailable: true
        }
      ]
    },
    kind: "adversarial-max-boundary",
    name: buildScenarioName("autobot", "adversarial-max-boundary", index, seed)
  };
}

const AUTOBOT_SCENARIO_FACTORY_ENTRIES = [
  { factory: createMaintenanceSnapshot, name: "maintenance", weight: 7 },
  { factory: createRuntimeDropSnapshot, name: "runtime-drop", weight: 2 },
  { factory: createAdditiveFeatureSnapshot, name: "additive-feature", weight: 5 },
  { factory: createSecurityPatchSnapshot, name: "security-patch", weight: 4 },
  { factory: createBreakingContractSnapshot, name: "breaking-contract", weight: 2 },
  { factory: createRuntimeSupportAddSnapshot, name: "runtime-support-add", weight: 3 },
  { factory: createWorkflowAutomationSnapshot, name: "workflow-automation", weight: 4 },
  { factory: createCompatibilityShimSnapshot, name: "compatibility-shim", weight: 2 },
  { factory: createDatabaseMigrationSnapshot, name: "database-migration", weight: 2 },
  { factory: createPatchUnavailableFeatureSnapshot, name: "patch-unavailable-feature", weight: 2 },
  { factory: createLabelRaceRecoverySnapshot, name: "label-race-recovery", weight: 2 },
  { factory: createInfrastructureEchoSnapshot, name: "infrastructure-echo", weight: 1 },
  { factory: createFixtureVocabularyTrapSnapshot, name: "fixture-vocabulary-trap", weight: 1 },
  { factory: createAdversarialEmptyPayloadSnapshot, name: "adversarial-empty-payload", weight: 2 },
  { factory: createAdversarialCrossDomainFalsePositiveSnapshot, name: "adversarial-cross-domain-fp", weight: 3 },
  { factory: createAdversarialConflictingEvidenceSnapshot, name: "adversarial-conflicting-evidence", weight: 2 },
  { factory: createAdversarialMaxBoundarySnapshot, name: "adversarial-max-boundary", weight: 1 }
];

function buildAutobotScenarioCases(config = {}) {
  return buildScenarioCases("AUTOBOT", config, DEFAULT_AUTOBOT_SCENARIO_COUNT, AUTOBOT_SCENARIO_FACTORY_ENTRIES);
}

function buildAutobotPipelineScenarioCases(config = {}) {
  return buildScenarioCases("AUTOBOT_PIPELINE", config, DEFAULT_AUTOBOT_PIPELINE_SCENARIO_COUNT, AUTOBOT_SCENARIO_FACTORY_ENTRIES);
}

function validateAutobotScenarioResult(scenario, result) {
  return validateAutobotScenarioContract(scenario, result).deviations;
}

function buildAutobotActualResult(result) {
  const labels = parseJson(result.deterministic_labels_json);
  const semver = parseJson(result.deterministic_semver_json);
  return {
    hardSignals: Array.isArray(semver.hardSignals) ? semver.hardSignals : [],
    labels,
    semverDecision: semver.decision
  };
}

function buildAutobotPipelineRawResult(result) {
  return {
    deterministic_labels_json: JSON.stringify(result.issueLabels || []),
    deterministic_semver_json: JSON.stringify({
      decision: result.semverDecision || "none",
      hardSignals: result.hardSignals || []
    })
  };
}

function validateAutobotPipelineScenarioContract(scenario, result) {
  const expectedPipelineSemverDecision = deriveExpectedPipelineSemverDecision(scenario);
  const pipelineSemverScenario = {
    ...scenario,
    expected: {
      ...(scenario.expected || {}),
      semverDecision: expectedPipelineSemverDecision
    },
    optimalResult: {
      ...getScenarioOptimalResult(scenario),
      semverDecision: expectedPipelineSemverDecision
    }
  };
  const validation = validateAutobotScenarioContract(
    pipelineSemverScenario,
    buildAutobotPipelineRawResult(result)
  );
  const expectedMilestoneTitle = deriveExpectedPipelineMilestoneTitle(scenario);
  const expected = getScenarioOptimalResult(scenario);

  if (expectedMilestoneTitle !== result.milestoneTitle) {
    validation.deviations.push(
      "expected milestone " + (expectedMilestoneTitle || "(none)") + " but got " + (result.milestoneTitle || "(none)")
    );
  }
  if (!result.commentPresent) {
    validation.deviations.push("expected managed bot comment to be created");
  }
  validateTextFragments(validation, {
    actualText: result.commentBody,
    acceptableFragmentGroups: validation.contract.acceptable.acceptableAutobotCommentIncludesAny,
    description: "autobot comment",
    requiredFragments: ["## Autobot Summary", ...(expected.autobotCommentIncludes || [])]
  });
  if (Array.isArray(expected.failureOperations) && JSON.stringify(expected.failureOperations) !== JSON.stringify(result.failureOperations)) {
    validation.deviations.push(
      "expected failure operations " + (expected.failureOperations.join(", ") || "(none)")
      + " but got " + (result.failureOperations.join(", ") || "(none)")
    );
  }
  return validation;
}

function validateAutobotPipelineScenarioResult(scenario, result) {
  return validateAutobotPipelineScenarioContract(scenario, result).deviations;
}

function buildAutobotPipelineActualResult(result) {
  return {
    failureOperations: result.failureOperations,
    hardSignals: result.hardSignals,
    issueLabels: result.issueLabels,
    managedLabels: result.managedLabels,
    milestoneTitle: result.milestoneTitle,
    semverDecision: result.semverDecision
  };
}

async function executeAutobotPipelineScenario(scenario) {
  const owner = "octo";
  const repo = "repo";
  const pullRequest = buildAutobotPipelinePullRequest(scenario);
  const issueNumber = pullRequest.number;
  const github = createAutobotPipelineGithubMock({
    issueLabels: [],
    issueNumber,
    issueTitle: pullRequest.title,
    pullFiles: scenario.input.files,
    pullRequest,
    releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }],
    ...(scenario.githubOptions || {})
  });
  const context = buildAutobotPipelineContext(pullRequest);
  return withScenarioTempWorkspace(["autobot-pipeline", scenario.kind, issueNumber], async (workspace) => {
    const snapshotFile = workspace.resolveFile("snapshot.json");
    const stateFile = workspace.resolveFile("state.json");

    await collectPullRequestSnapshot({ github, owner, repo, pullRequest, snapshotFile });
    const analysisOutputs = analyzePullRequestSnapshot({ snapshotFile });
    await prepareProjectState({
      autobotLabelsRaw: analysisOutputs.deterministic_labels_json,
      context,
      deterministicSemverRaw: analysisOutputs.deterministic_semver_json,
      github,
      issueNumber,
      owner,
      repo,
      stateFile,
      summaryText: analysisOutputs.deterministic_summary
    });
    await syncPreparedProjectState({ github, issueNumber, owner, repo, stateFile });
    await syncProjectMilestone({ context, github, issueNumber, owner, repo });

    const issue = github.state.issuesByNumber.get(issueNumber);
    const comments = github.state.commentsByIssue.get(issueNumber) || [];
    const managedComment = getManagedBotComment(comments);
    const metadata = extractManagedBotCommentMetadata(managedComment?.body || "");

    return {
      analysisOutputs,
      commentBody: managedComment?.body || "",
      commentPresent: Boolean(managedComment),
      failureOperations: github.state.failureLog.map((entry) => entry.operationName),
      hardSignals: Array.isArray(metadata.deterministicSemver?.hardSignals) ? metadata.deterministicSemver.hardSignals : [],
      issueLabels: (issue?.labels || []).map((label) => label.name),
      managedLabels: Array.isArray(metadata.autobotLabels) ? metadata.autobotLabels : [],
      milestoneTitle: issue?.milestone?.title || null,
      semverDecision: String(metadata.semverDecision || metadata.deterministicSemver?.decision || "none")
    };
  });
}

async function evaluateAutobotPipelineScenarioSuite({ config } = {}) {
  const suite = buildAutobotPipelineScenarioCases(config || {});
  const evaluations = [];

  for (const scenario of suite.scenarios) {
    const rawResult = await executeAutobotPipelineScenario(scenario);
    const validation = validateAutobotPipelineScenarioContract(scenario, rawResult);
    evaluations.push({
      actualResult: buildAutobotPipelineActualResult(rawResult),
      acceptedOutcome: validation.acceptedOutcome,
      acceptedReasons: validation.acceptedReasons,
      contract: validation.contract,
      deviations: validation.deviations,
      kind: scenario.kind,
      name: scenario.name,
      optimalResult: {
        ...getScenarioOptimalResult(scenario),
        milestoneTitle: deriveExpectedPipelineMilestoneTitle(scenario),
        semverDecision: deriveExpectedPipelineSemverDecision(scenario)
      },
      passed: validation.deviations.length === 0,
      rawResult,
      scenario,
      warnings: validation.warnings
    });
  }

  const summary = summarizeScenarioEvaluations(evaluations, { suiteKey: suite.key });
  return {
    ...suite,
    evaluations,
    reportText: formatScenarioSuiteReport({
      evaluations,
      seed: suite.seed,
      suiteKey: suite.key,
      summary
    }),
    summary
  };
}

function createCloseIssueScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 4));
  const issueNumber = 100 + index;
  return {
    candidate: createSmartLinkEntity({
      body: `${token} request ${randomWords(random, 2).join(" ")}`,
      labels: [pick(random, ["enhancement", "runtime"])],
      number: issueNumber,
      title: `Add ${token}`
    }),
    expected: {
      minScore: 80,
      relationKind: "closes",
      suppressionExcludes: ["lexical-only", "below-threshold"],
      thresholdPassed: true
    },
    kind: "closes-issue",
    name: buildScenarioName("smart-link", "closes-issue", index, seed),
    source: createSmartLinkEntity({
      body: `Implements #${issueNumber}. ${randomWords(random, 3).join(" ")}`,
      changedFiles: [`src/repo/${token}_${index}.py`],
      kind: "pull_request",
      labels: [pick(random, ["enhancement", "runtime", "security"])],
      number: 800 + index,
      title: `Implement ${token}`
    }),
    threshold: 80
  };
}

function createConnectPullRequestScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 4));
  const pullRequestNumber = 400 + index;
  return {
    candidate: createSmartLinkEntity({
      body: `${token} shared refactor ${randomWords(random, 2).join(" ")}`,
      kind: "pull_request",
      number: pullRequestNumber,
      title: `Refactor ${token}`
    }),
    expected: {
      minScore: 80,
      relationKind: "connects",
      thresholdPassed: true
    },
    kind: "connects-pr",
    name: buildScenarioName("smart-link", "connects-pr", index, seed),
    source: createSmartLinkEntity({
      body: `Implements #${pullRequestNumber}. ${randomWords(random, 2).join(" ")}`,
      kind: "pull_request",
      number: 900 + index,
      title: `Coordinate ${token}`
    }),
    threshold: 80
  };
}

function createLexicalSuppressionScenario(random, index, seed) {
  const lexicalWords = randomWords(random, 5);
  return {
    candidate: createSmartLinkEntity({
      body: lexicalWords.slice().reverse().join(" "),
      kind: "issue",
      number: 1100 + index,
      title: lexicalWords.slice(1, 4).join(" ")
    }),
    expected: {
      relationKind: "related",
      suppressionIncludes: ["lexical-only"],
      thresholdPassed: false
    },
    kind: "lexical-suppressed",
    name: buildScenarioName("smart-link", "lexical-suppressed", index, seed),
    source: createSmartLinkEntity({
      body: lexicalWords.join(" "),
      kind: "issue",
      number: 1000 + index,
      title: lexicalWords.slice(0, 3).join(" ")
    }),
    threshold: 80
  };
}

function createAdvisoryFixScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const advisoryId = `CVE-2026-${1200 + index}`;
  const ecosystem = [pick(random, ["python", "node"])];
  return {
    candidate: createSmartLinkEntity({
      alertIdentifiers: [advisoryId],
      body: `Patch ${advisoryId} in ${token}.`,
      ecosystemSignals: ecosystem,
      kind: "pull_request",
      number: 1200 + index,
      packageSignals: [`package:${token}`],
      title: `Patch ${token}`
    }),
    expected: {
      minScore: 80,
      relationKind: "advisory_fix",
      suppressionExcludes: ["advisory-without-alert-signal"],
      thresholdPassed: true
    },
    kind: "advisory-fix",
    name: buildScenarioName("smart-link", "advisory-fix", index, seed),
    source: createSmartLinkEntity({
      alertIdentifiers: [advisoryId],
      body: `${advisoryId} affects package:${token}.`,
      ecosystemSignals: ecosystem,
      kind: "security_alert",
      packageSignals: [`package:${token}`],
      remediationReferences: [1200 + index],
      title: `Advisory for ${token}`
    }),
    threshold: 80
  };
}

function createClosedIssueSuppressionScenario(random, index, seed) {
  const issueNumber = 1400 + index;
  return {
    candidate: createSmartLinkEntity({
      body: `Already fixed ${randomWords(random, 2).join(" ")}`,
      number: issueNumber,
      state: "closed",
      title: `Resolved ${pick(random, randomWords(random, 2))}`
    }),
    expected: {
      relationKind: "closes",
      suppressionIncludes: ["close-target-not-open"],
      thresholdPassed: false
    },
    kind: "closed-close-suppressed",
    name: buildScenarioName("smart-link", "closed-close-suppressed", index, seed),
    source: createSmartLinkEntity({
      body: `Closes #${issueNumber}. ${randomWords(random, 2).join(" ")}`,
      kind: "pull_request",
      number: 1300 + index,
      title: `Close ${issueNumber}`
    }),
    threshold: 80
  };
}

function createDependsOnIssueScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const issueNumber = 1500 + index;
  const milestone = createMilestone(200 + index, `v0.${index}.0`);
  return {
    candidate: createSmartLinkEntity({
      body: `Dependency gate for ${token}.`,
      changedFiles: [`src/repo/${token}/dependency_${index}.py`],
      labels: ["runtime"],
      milestone,
      number: issueNumber,
      title: `Prepare ${token} dependency`
    }),
    expected: {
      minScore: 80,
      relationKind: "depends_on",
      thresholdPassed: true
    },
    kind: "depends-on-issue",
    name: buildScenarioName("smart-link", "depends-on-issue", index, seed),
    source: createSmartLinkEntity({
      body: `Depends on #${issueNumber}. ${randomWords(random, 2).join(" ")}`,
      changedFiles: [`src/repo/${token}/runtime_${index}.py`],
      kind: "pull_request",
      labels: ["runtime"],
      milestone,
      number: 1450 + index,
      title: `Ship ${token} runtime`
    }),
    threshold: 80
  };
}

function createReciprocalConnectScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const candidateNumber = 1600 + index;
  const sourceNumber = 1650 + index;
  const milestone = createMilestone(250 + index, `v1.${index}.0`);
  return {
    candidate: createSmartLinkEntity({
      body: `References #${sourceNumber}. Reciprocal ${token} work.`,
      changedFiles: [`src/repo/${token}/shared_${index}.py`],
      labels: ["compatibility"],
      milestone,
      number: candidateNumber,
      title: `Shared ${token} rollout`
    }),
    expected: {
      minScore: 80,
      relationKind: "connects",
      thresholdPassed: true
    },
    kind: "reciprocal-connect",
    name: buildScenarioName("smart-link", "reciprocal-connect", index, seed),
    source: createSmartLinkEntity({
      body: `References #${candidateNumber}. ${randomWords(random, 2).join(" ")}`,
      changedFiles: [`src/repo/${token}/shared_${index}.py`],
      kind: "issue",
      labels: ["compatibility"],
      milestone,
      number: sourceNumber,
      title: `Coordinate ${token} rollout`
    }),
    threshold: 80
  };
}

function createAdvisoryMismatchScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const advisoryId = `GHSA-${String(2000 + index).padStart(4, "0")}-abcd-efgh`;
  return {
    candidate: createSmartLinkEntity({
      alertIdentifiers: [advisoryId],
      body: `Patch ${advisoryId} for ${token}.`,
      ecosystemSignals: ["node"],
      kind: "pull_request",
      number: 1700 + index,
      packageSignals: [`package:${token}-ui`],
      title: `Patch ${token} ui advisory`
    }),
    expected: {
      minScore: 1,
      relationKind: "advisory_fix",
      suppressionIncludes: ["ecosystem-mismatch", "package-mismatch", "below-threshold"],
      thresholdPassed: false
    },
    kind: "advisory-mismatch",
    name: buildScenarioName("smart-link", "advisory-mismatch", index, seed),
    source: createSmartLinkEntity({
      alertIdentifiers: [advisoryId],
      body: `${advisoryId} affects package:${token}-core.`,
      ecosystemSignals: ["python"],
      kind: "security_alert",
      packageSignals: [`package:${token}-core`],
      title: `Core advisory for ${token}`
    }),
    threshold: 80
  };
}

function createBranchReferenceScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const issueNumber = 1800 + index;
  const sourceNumber = 1850 + index;
  const milestone = createMilestone(300 + index, `v2.${index}.0`);
  return {
    candidate: createSmartLinkEntity({
      body: `References #${sourceNumber}. ${token} dependency issue.`,
      changedFiles: [`src/repo/${token}/bridge_${index}.py`],
      labels: ["compatibility"],
      milestone,
      number: issueNumber,
      title: `Investigate ${token} compatibility`
    }),
    expected: {
      minScore: 80,
      relationKind: "connects",
      thresholdPassed: true
    },
    kind: "branch-reference",
    name: buildScenarioName("smart-link", "branch-reference", index, seed),
    source: createSmartLinkEntity({
      body: `${randomWords(random, 2).join(" ")}`,
      branch: `feature/${issueNumber}-${token}`,
      changedFiles: [`src/repo/${token}/bridge_${index}.py`],
      kind: "pull_request",
      labels: ["compatibility"],
      milestone,
      number: sourceNumber,
      title: `Bridge ${token} compatibility`
    }),
    threshold: 80
  };
}

function createBatchDirectiveScenario(random, index, seed) {
  const token = pick(random, randomWords(random, 3));
  const closeId = 1900 + index;
  const dependencyId = 1950 + index;
  const connectId = 2000 + index;
  const sourceNumber = 2050 + index;
  const milestone = createMilestone(350 + index, `v3.${index}.0`);
  return {
    candidates: [
      createSmartLinkEntity({
        body: `${token} feature request`,
        changedFiles: [`src/repo/${token}/feature_${index}.py`],
        kind: "issue",
        labels: ["runtime"],
        milestone,
        number: closeId,
        title: `Add ${token} feature`
      }),
      createSmartLinkEntity({
        body: `${token} dependency gate`,
        changedFiles: [`src/repo/${token}/dependency_${index}.py`],
        kind: "issue",
        labels: ["runtime"],
        milestone,
        number: dependencyId,
        title: `Prepare ${token} dependency`
      }),
      createSmartLinkEntity({
        body: `${token} refactor rollout`,
        changedFiles: [`src/repo/${token}/refactor_${index}.py`],
        kind: "pull_request",
        labels: ["runtime"],
        milestone,
        number: connectId,
        title: `Refactor ${token} pipeline`
      }),
      createSmartLinkEntity({
        body: randomWords(random, 5).join(" "),
        kind: "issue",
        number: 2100 + index,
        title: randomWords(random, 3).join(" ")
      })
    ],
    expected: {
      directiveSummary: {
        closeIds: [closeId],
        connectIds: [connectId],
        dependencyIds: [dependencyId]
      },
      emittedCandidateNumbers: [closeId, dependencyId, connectId],
      graphContains: ["graph TD", `ISSUE${closeId}`, `ISSUE${dependencyId}`, `PR${connectId}`],
      relationKinds: ["closes", "depends_on", "connects"]
    },
    kind: "batch-directives",
    mode: "batch",
    name: buildScenarioName("smart-link", "batch-directives", index, seed),
    source: createSmartLinkEntity({
      body: `Implements #${closeId}. Depends on #${dependencyId}. Connects to #${connectId}.`,
      changedFiles: [`src/repo/${token}/feature_${index}.py`, `src/repo/${token}/dependency_${index}.py`],
      kind: "pull_request",
      labels: ["runtime"],
      milestone,
      number: sourceNumber,
      title: `Ship ${token} stack`
    }),
    threshold: 80
  };
}

function buildSmartLinkScenarioCases(config = {}) {
  return buildScenarioCases("SMART_LINK", config, DEFAULT_SMART_LINK_SCENARIO_COUNT, [
    { factory: createCloseIssueScenario, name: "close-issue", weight: 5 },
    { factory: createConnectPullRequestScenario, name: "connect-pull-request", weight: 4 },
    { factory: createDependsOnIssueScenario, name: "depends-on-issue", weight: 4 },
    { factory: createReciprocalConnectScenario, name: "reciprocal-connect", weight: 2 },
    { factory: createLexicalSuppressionScenario, name: "lexical-suppression", weight: 2 },
    { factory: createAdvisoryFixScenario, name: "advisory-fix", weight: 2 },
    { factory: createAdvisoryMismatchScenario, name: "advisory-mismatch", weight: 1 },
    { factory: createClosedIssueSuppressionScenario, name: "closed-issue-suppression", weight: 2 },
    { factory: createBranchReferenceScenario, name: "branch-reference", weight: 2 },
    { factory: createBatchDirectiveScenario, name: "batch-directives", weight: 2 }
  ]);
}

function createUnifiedPullRequestDirectiveScenario(random, index, seed) {
  const autobotFactory = pickWeighted(random, [
    { factory: createAdditiveFeatureSnapshot, weight: 4 },
    { factory: createSecurityPatchSnapshot, weight: 3 },
    { factory: createRuntimeSupportAddSnapshot, weight: 2 },
    { factory: createCompatibilityShimSnapshot, weight: 2 }
  ]);
  const autobotScenario = withOptimalResult(autobotFactory.factory(random, 300 + index, seed));
  const sourceNumber = autobotScenario.input.pullRequest.number;
  const packageName = derivePrimaryPackageName(autobotScenario.input.files);
  const token = buildCodeIdentifier(random, 2);
  const closeIssueNumber = 3000 + index;
  const dependencyIssueNumber = 3100 + index;
  const connectPullNumber = 3200 + index;
  const noiseIssueNumber = 3300 + index;
  const sharedLabel = deriveExpectedSharedLabel(autobotScenario);
  const pathHint = `src/${packageName}/${token}_${index}.py`;
  const sourceEntity = createSmartLinkEntity({
    body: [
      `Implements #${closeIssueNumber}.`,
      `Depends on #${dependencyIssueNumber}.`,
      `Connects to #${connectPullNumber}.`,
      `Observed in ${pathHint}.`,
      String(autobotScenario.input.pullRequest.body || "")
    ].filter(Boolean).join("\n"),
    branch: autobotScenario.input.pullRequest.headRef,
    changedFiles: autobotScenario.input.files.map((file) => ({ filename: file.filename, status: file.status })),
    kind: "pull_request",
    number: sourceNumber,
    repoFullName: "octo/repo",
    title: autobotScenario.input.pullRequest.title,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const closeIssue = createSmartLinkEntity({
    body: `Bug trace for ${pathHint}.`,
    kind: "issue",
    labels: ["bug"],
    number: closeIssueNumber,
    repoFullName: "octo/repo",
    title: `Fix ${token} bug`,
    updatedAt: "2026-04-12T00:00:00Z"
  });
  const dependencyIssue = createSmartLinkEntity({
    body: `Documentation follow-up for ${pathHint}.`,
    kind: "issue",
    labels: ["documentation"],
    number: dependencyIssueNumber,
    repoFullName: "octo/repo",
    title: `Document ${token} dependency`,
    updatedAt: "2026-04-12T00:00:00Z"
  });
  const connectPull = createSmartLinkEntity({
    body: `References #${sourceNumber}. Shared rollout for ${pathHint}.`,
    kind: "pull_request",
    labels: [sharedLabel],
    number: connectPullNumber,
    repoFullName: "octo/repo",
    title: `Coordinate ${token} rollout`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const noiseIssue = createSmartLinkEntity({
    body: randomWords(random, 6).join(" "),
    kind: "issue",
    number: noiseIssueNumber,
    repoFullName: "octo/repo",
    title: randomWords(random, 3).join(" "),
    updatedAt: "2025-01-01T00:00:00Z"
  });
  const pullRequest = buildMockPullRequestRecordFromEntity(sourceEntity, {
    body: sourceEntity.body,
    headRef: autobotScenario.input.pullRequest.headRef,
    title: sourceEntity.title
  });

  return {
    autobotScenario,
    context: buildAutobotPipelineContext(pullRequest),
    expected: {
      autobotCommentIncludes: ["## Autobot Summary"],
      autobotCommentPresent: true,
      autobotReady: true,
      checkConclusion: "success",
      graphCommentIncludes: ["graph TD", `ISSUE${closeIssueNumber}`, `ISSUE${dependencyIssueNumber}`, `PR${connectPullNumber}`],
      graphCommentPresent: true,
      linkedLabels: [],
      milestoneTitle: deriveExpectedPipelineMilestoneTitle(autobotScenario),
      pullBodyIncludes: [
        "### Smart Link Intelligence",
        `Closes #${closeIssueNumber}`,
        `Depends on #${dependencyIssueNumber}`,
        `Connects to #${connectPullNumber}`
      ],
      smartLinkReady: true
    },
    githubOptions: {
      additionalIssues: [
        buildMockIssueRecordFromEntity(closeIssue),
        buildMockIssueRecordFromEntity(dependencyIssue),
        buildMockIssueRecordFromEntity(noiseIssue)
      ],
      additionalPullRequests: [
        buildMockPullRequestRecordFromEntity(connectPull, {
          body: connectPull.body,
          headRef: `feature/${connectPull.number}-${token}`
        })
      ],
      issueNumber: sourceNumber,
      issueTitle: sourceEntity.title,
      pullFiles: autobotScenario.input.files,
      pullRequest,
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: sourceNumber,
    kind: "unified-pr-directives",
    name: buildScenarioName("unified", "pr-directives", index, seed),
    thresholdInput: 80
  };
}

function createUnifiedIssueRelationshipScenario(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  const sourceNumber = 3400 + index;
  const dependencyIssueNumber = 3500 + index;
  const connectPullNumber = 3600 + index;
  const milestone = createMilestone(800 + index, `Tracking ${token}`);
  const pathHint = `src/${token}/runtime_${index}.py`;
  const sourceEntity = createSmartLinkEntity({
    body: [
      "Thank you for helping us squash this bug",
      "Detailed steps to reproduce",
      "Potential causes / workarounds / related issues (optional)",
      "Custom modifications / configuration",
      `Depends on #${dependencyIssueNumber}.`,
      `Connects to #${connectPullNumber}.`,
      `Observed in ${pathHint}.`
    ].join("\n"),
    kind: "issue",
    milestone,
    number: sourceNumber,
    repoFullName: "octo/repo",
    title: `Bug: ${token} runtime failure on Windows`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const dependencyIssue = createSmartLinkEntity({
    body: `Repro details for ${pathHint}.`,
    kind: "issue",
    labels: ["bug"],
    milestone,
    number: dependencyIssueNumber,
    repoFullName: "octo/repo",
    title: `Investigate ${token} runtime bug`,
    updatedAt: "2026-04-12T00:00:00Z"
  });
  const connectPull = createSmartLinkEntity({
    body: `References #${sourceNumber}. Mirrors ${pathHint}.`,
    kind: "pull_request",
    labels: ["bug"],
    milestone,
    number: connectPullNumber,
    repoFullName: "octo/repo",
    title: `Patch ${token} runtime path`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const issueRecord = buildMockIssueRecordFromEntity(sourceEntity, { milestone });

  return {
    context: {
      eventName: "issues",
      issue: { number: sourceNumber },
      payload: {
        action: "opened",
        issue: buildIssuePayloadFromRecord(issueRecord)
      }
    },
    expected: {
      autobotCommentPresent: false,
      autobotReady: true,
      checkConclusion: null,
      graphCommentPresent: false,
      issueLabelsInclude: ["windows"],
      linkedLabels: [],
      milestoneTitle: milestone.title,
      smartLinkCommentIncludes: ["### Smart Link Intelligence", `#${dependencyIssueNumber}`, `#${connectPullNumber}`, "depends on", "connects"],
      smartLinkCommentPresent: true,
      smartLinkReady: true
    },
    githubOptions: {
      additionalIssues: [issueRecord, buildMockIssueRecordFromEntity(dependencyIssue, { milestone })],
      additionalPullRequests: [
        buildMockPullRequestRecordFromEntity(connectPull, {
          body: connectPull.body,
          headRef: `fix/${connectPull.number}-${token}`,
          milestone
        })
      ],
      issueBody: issueRecord.body,
      issueNumber: sourceNumber,
      issueState: issueRecord.state,
      issueTitle: issueRecord.title,
      milestones: [createVersionMilestoneData(milestone.number, milestone.title, { open_issues: 2 })],
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: sourceNumber,
    kind: "unified-issue-relationships",
    name: buildScenarioName("unified", "issue-relationships", index, seed),
    thresholdInput: 80
  };
}

function createUnifiedStaleReopenedIssueScenario(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  const sourceNumber = 3650 + index;
  const dependencyIssueNumber = 3660 + index;
  const stalePullNumber = 3670 + index;
  const milestone = createMilestone(820 + index, `Tracking ${token} reopen`);
  const pathHint = `src/${token}/runtime_${index}.py`;
  const staleUpdatedAt = "2024-01-10T00:00:00Z";
  const recentUpdatedAt = "2026-04-12T00:00:00Z";
  const sourceEntity = createSmartLinkEntity({
    body: [
      "Thank you for helping us squash this bug",
      "Detailed steps to reproduce",
      "Potential causes / workarounds / related issues (optional)",
      "Custom modifications / configuration",
      `Depends on #${dependencyIssueNumber}.`,
      `Connects to #${stalePullNumber}.`,
      `Observed in ${pathHint}.`
    ].join("\n"),
    kind: "issue",
    milestone,
    number: sourceNumber,
    repoFullName: "octo/repo",
    title: `Bug: ${token} runtime issue reopened after partial fix`,
    updatedAt: recentUpdatedAt
  });
  const dependencyIssue = createSmartLinkEntity({
    body: `Fresh dependency follow-up for ${pathHint}.`,
    kind: "issue",
    labels: ["bug"],
    milestone,
    number: dependencyIssueNumber,
    repoFullName: "octo/repo",
    title: `Investigate ${token} dependency regression`,
    updatedAt: recentUpdatedAt
  });
  const stalePull = createSmartLinkEntity({
    body: `References #${sourceNumber}. Mirrors ${pathHint}.`,
    kind: "pull_request",
    labels: ["bug"],
    milestone,
    number: stalePullNumber,
    repoFullName: "octo/repo",
    title: `Historic ${token} runtime workaround`,
    updatedAt: staleUpdatedAt
  });
  const issueRecord = buildMockIssueRecordFromEntity(sourceEntity, { milestone, updated_at: recentUpdatedAt });

  return {
    context: {
      eventName: "issues",
      issue: { number: sourceNumber },
      payload: {
        action: "reopened",
        issue: buildIssuePayloadFromRecord(issueRecord)
      }
    },
    expected: {
      autobotCommentPresent: false,
      autobotReady: true,
      checkConclusion: null,
      graphCommentPresent: false,
      issueLabelsInclude: ["platform"],
      linkedLabels: [],
      milestoneTitle: milestone.title,
      smartLinkCommentExcludes: [`#${stalePullNumber}`],
      smartLinkCommentIncludes: ["### Smart Link Intelligence", `#${dependencyIssueNumber}`, "Depends On"],
      smartLinkCommentPresent: true,
      smartLinkReady: true
    },
    githubOptions: {
      additionalIssues: [issueRecord, buildMockIssueRecordFromEntity(dependencyIssue, { milestone, updated_at: recentUpdatedAt })],
      additionalPullRequests: [
        buildMockPullRequestRecordFromEntity(stalePull, {
          body: stalePull.body,
          headRef: `fix/${stalePull.number}-${token}`,
          milestone,
          updated_at: staleUpdatedAt
        })
      ],
      issueAuthorAssociation: "CONTRIBUTOR",
      issueBody: issueRecord.body,
      issueNumber: sourceNumber,
      issueState: issueRecord.state,
      issueTitle: issueRecord.title,
      issueUpdatedAt: recentUpdatedAt,
      milestones: [createVersionMilestoneData(milestone.number, milestone.title, { open_issues: 2 })],
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: sourceNumber,
    kind: "unified-issue-stale-reopened",
    name: buildScenarioName("unified", "issue-stale-reopened", index, seed),
    thresholdInput: 80
  };
}

function createUnifiedAlertScenario(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  const renderTargetIssueNumber = 3800 + index;
  const remediationPullNumber = 3900 + index;
  const advisoryId = index % 2 === 0
    ? `GHSA-${String(5000 + index).padStart(4, "0")}-abcd-efgh`
    : `CVE-2026-${5000 + index}`;
  const ecosystem = pick(random, ["python", "node"]);
  const pathHint = `src/${token}/security_${index}.py`;
  const alertPayload = {
    alertIdentifiers: [advisoryId],
    body: `${advisoryId} affects package:${token}. Impact observed in ${pathHint}.`,
    ecosystemSignals: [ecosystem],
    kind: "security_alert",
    packageSignals: [`package:${token}`],
    remediationReferences: [remediationPullNumber],
    renderTarget: { kind: "issue", number: renderTargetIssueNumber },
    title: `Security advisory for ${token}`
  };
  const targetIssue = createSmartLinkEntity({
    body: `Track ${token} advisory resolution.`,
    kind: "issue",
    number: renderTargetIssueNumber,
    repoFullName: "octo/repo",
    title: `Track ${token} advisory`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const remediationPull = createSmartLinkEntity({
    alertIdentifiers: [advisoryId],
    body: `Remediates ${advisoryId} for package:${token}. Touches ${pathHint}.`,
    ecosystemSignals: [ecosystem],
    kind: "pull_request",
    number: remediationPullNumber,
    packageSignals: [`package:${token}`],
    repoFullName: "octo/repo",
    title: `Patch ${token} advisory`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const dispatchMode = index % 2 === 0 ? "repository_dispatch" : "workflow_dispatch";
  const context = dispatchMode === "repository_dispatch"
    ? {
        eventName: "repository_dispatch",
        payload: {
          action: "triggered",
          client_payload: alertPayload
        }
      }
    : {
        eventName: "workflow_dispatch",
        payload: {
          inputs: {
            payload: JSON.stringify(alertPayload)
          }
        }
      };

  return {
    context,
    expected: {
      autobotCommentPresent: false,
      autobotReady: false,
      checkConclusion: null,
      graphCommentPresent: false,
      linkedLabels: [],
      milestoneTitle: null,
      smartLinkCommentIncludes: ["### Smart Link Intelligence", `#${remediationPullNumber}`, "advisory fix"],
      smartLinkCommentPresent: true,
      smartLinkReady: true
    },
    githubOptions: {
      additionalIssues: [buildMockIssueRecordFromEntity(targetIssue)],
      additionalPullRequests: [
        buildMockPullRequestRecordFromEntity(remediationPull, {
          body: remediationPull.body,
          headRef: `security/${remediationPull.number}-${token}`
        })
      ],
      issueNumber: renderTargetIssueNumber,
      issueTitle: targetIssue.title,
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: renderTargetIssueNumber,
    kind: "unified-alert-routing",
    name: buildScenarioName("unified", `alert-${dispatchMode}`, index, seed),
    renderTargetIssueNumber,
    thresholdInput: 80
  };
}

function createUnifiedSuppressedPullRequestScenario(random, index, seed) {
  const autobotScenario = withOptimalResult(createMaintenanceSnapshot(random, 500 + index, seed));
  const sourceNumber = autobotScenario.input.pullRequest.number;
  const packageName = derivePrimaryPackageName(autobotScenario.input.files);
  const token = buildCodeIdentifier(random, 2);
  const closedIssueNumber = 4000 + index;
  const pathHint = `src/${packageName}/${token}_${index}.py`;
  const initialBody = [
    buildStaleSmartLinkBlock(token),
    "",
    `Closes #${closedIssueNumber}.`,
    `Observed in ${pathHint}.`
  ].join("\n");
  const sourceEntity = createSmartLinkEntity({
    body: initialBody,
    branch: autobotScenario.input.pullRequest.headRef,
    changedFiles: autobotScenario.input.files.map((file) => ({ filename: file.filename, status: file.status })),
    kind: "pull_request",
    number: sourceNumber,
    repoFullName: "octo/repo",
    title: autobotScenario.input.pullRequest.title,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const closedIssue = createSmartLinkEntity({
    body: `Already fixed in ${pathHint}.`,
    kind: "issue",
    labels: ["bug"],
    number: closedIssueNumber,
    repoFullName: "octo/repo",
    state: "closed",
    title: `Closed ${token} bug`,
    updatedAt: "2026-04-12T00:00:00Z"
  });
  const pullRequest = buildMockPullRequestRecordFromEntity(sourceEntity, {
    body: initialBody,
    headRef: autobotScenario.input.pullRequest.headRef,
    title: sourceEntity.title
  });

  return {
    autobotScenario,
    context: buildAutobotPipelineContext(pullRequest),
    expected: {
      autobotCommentPresent: true,
      autobotReady: true,
      checkConclusion: "neutral",
      graphCommentPresent: false,
      linkedLabels: [],
      milestoneTitle: deriveExpectedPipelineMilestoneTitle(autobotScenario),
      pullBodyExcludes: ["<!-- smart-autolinker:start -->", "### Smart Link Intelligence"],
      pullBodyIncludes: ["User content above.", "User content below.", `Closes #${closedIssueNumber}.`],
      smartLinkReady: true
    },
    githubOptions: {
      additionalIssues: [buildMockIssueRecordFromEntity(closedIssue)],
      commentsByIssue: {
        [sourceNumber]: [buildGraphCommentSeed("pull_request", sourceNumber)]
      },
      issueNumber: sourceNumber,
      issueTitle: sourceEntity.title,
      pullFiles: autobotScenario.input.files,
      pullRequest,
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: sourceNumber,
    kind: "unified-pr-suppressed",
    name: buildScenarioName("unified", "pr-suppressed", index, seed),
    thresholdInput: 80
  };
}

function createUnifiedReleaseFinalizeScenario(random, index, seed) {
  const token = buildCodeIdentifier(random, 2);
  const sourceNumber = 4200 + index;
  const patchNumber = 2 + (index % 4);
  const milestoneTitle = `v0.${index % 3}.${patchNumber}`;
  const previousTag = `v0.${index % 3}.${patchNumber - 1}`;
  const milestone = createVersionMilestoneData(900 + index, milestoneTitle, {
    closed_issues: 1,
    open_issues: 0,
    state: "open"
  });
  const sourceEntity = createSmartLinkEntity({
    body: `Patch ${token} security flow.`,
    kind: "pull_request",
    labels: ["security"],
    number: sourceNumber,
    repoFullName: "octo/repo",
    state: "closed",
    title: `Patch ${token} security flow`,
    updatedAt: "2026-04-13T00:00:00Z"
  });
  const pullRequest = buildMockPullRequestRecordFromEntity(sourceEntity, {
    body: sourceEntity.body,
    headRef: `security/${token}-${index}`,
    merged: true,
    milestone: { number: milestone.number, title: milestone.title },
    state: "closed"
  });
  const issueRecord = buildMockIssueRecordFromEntity(sourceEntity, {
    labels: ["security"],
    milestone: { number: milestone.number, title: milestone.title },
    state: "closed"
  });

  return {
    context: {
      eventName: "pull_request",
      issue: { number: sourceNumber },
      payload: {
        action: "closed",
        pull_request: pullRequest
      }
    },
    expected: {
      autobotCommentPresent: false,
      autobotReady: true,
      checkConclusion: null,
      closedMilestoneTitles: [milestoneTitle],
      createdMilestoneTitles: [bumpPatchVersion(milestoneTitle)],
      createdReleaseTags: [milestoneTitle],
      graphCommentPresent: false,
      issueLabelsInclude: ["security"],
      linkedLabels: [],
      milestoneTitle,
      smartLinkCommentPresent: false,
      smartLinkReady: false
    },
    githubOptions: {
      additionalIssues: [issueRecord],
      issueBody: issueRecord.body,
      issueLabels: ["security"],
      issueMilestone: { number: milestone.number, title: milestone.title },
      issueNumber: sourceNumber,
      issueState: "closed",
      issueTitle: issueRecord.title,
      milestones: [milestone],
      pullRequest,
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: previousTag }]
    },
    issueNumber: sourceNumber,
    kind: "unified-release-finalize",
    name: buildScenarioName("unified", "release-finalize", index, seed),
    thresholdInput: 80
  };
}

function createUnifiedUntrustedForkScenario(random, index, seed) {
  const autobotScenario = withOptimalResult(createAdditiveFeatureSnapshot(random, 700 + index, seed));
  const sourceNumber = autobotScenario.input.pullRequest.number;
  const linkedIssueNumber = 4300 + index;
  const token = buildCodeIdentifier(random, 2);
  const body = [
    `Implements #${linkedIssueNumber}.`,
    String(autobotScenario.input.pullRequest.body || "")
  ].filter(Boolean).join("\n");
  const pullRequest = buildMockPullRequestRecordFromEntity(createSmartLinkEntity({
    body,
    kind: "pull_request",
    number: sourceNumber,
    repoFullName: "octo/repo",
    title: autobotScenario.input.pullRequest.title,
    updatedAt: "2026-04-13T00:00:00Z"
  }), {
    author_association: "FIRST_TIME_CONTRIBUTOR",
    body,
    headRef: autobotScenario.input.pullRequest.headRef,
    headRepoFullName: `forks/${token}`,
    title: autobotScenario.input.pullRequest.title
  });

  return {
    autobotScenario,
    context: buildAutobotPipelineContext(pullRequest),
    expected: {
      autobotCommentIncludes: ["## Autobot Summary"],
      autobotCommentPresent: true,
      autobotReady: true,
      checkConclusion: null,
      graphCommentPresent: false,
      linkedLabels: [],
      milestoneTitle: deriveExpectedPipelineMilestoneTitle(autobotScenario),
      smartLinkCommentPresent: false,
      smartLinkReady: false
    },
    githubOptions: {
      additionalIssues: [buildMockIssueRecordFromEntity(createSmartLinkEntity({
        body: `Issue for ${token} followup.`,
        kind: "issue",
        labels: ["enhancement"],
        number: linkedIssueNumber,
        repoFullName: "octo/repo",
        title: `Track ${token} followup`
      }))],
      issueNumber: sourceNumber,
      issueTitle: pullRequest.title,
      pullFiles: autobotScenario.input.files,
      pullRequest,
      releases: [{ draft: false, id: 1, prerelease: false, tag_name: "v0.0.1" }]
    },
    issueNumber: sourceNumber,
    kind: "unified-pr-untrusted-fork",
    name: buildScenarioName("unified", "pr-untrusted-fork", index, seed),
    thresholdInput: 80
  };
}

const UNIFIED_AUTOMATION_SCENARIO_FACTORY_ENTRIES = [
  { factory: createUnifiedPullRequestDirectiveScenario, name: "pr-directives", weight: 5 },
  { factory: createUnifiedIssueRelationshipScenario, name: "issue-relationships", weight: 4 },
  { factory: createUnifiedStaleReopenedIssueScenario, name: "issue-stale-reopened", weight: 2 },
  { factory: createUnifiedAlertScenario, name: "alert-routing", weight: 2 },
  { factory: createUnifiedSuppressedPullRequestScenario, name: "pr-suppressed", weight: 2 },
  { factory: createUnifiedReleaseFinalizeScenario, name: "release-finalize", weight: 2 },
  { factory: createUnifiedUntrustedForkScenario, name: "pr-untrusted-fork", weight: 2 }
];

function buildUnifiedAutomationScenarioCases(config = {}) {
  return buildScenarioCases("UNIFIED_AUTOMATION", config, DEFAULT_UNIFIED_AUTOMATION_SCENARIO_COUNT, UNIFIED_AUTOMATION_SCENARIO_FACTORY_ENTRIES);
}

function buildUnifiedAutomationAutobotRawResult(result) {
  return {
    deterministic_labels_json: JSON.stringify(result.issueLabels || []),
    deterministic_semver_json: JSON.stringify({
      decision: result.semverDecision || "none",
      hardSignals: result.hardSignals || []
    })
  };
}

function mergeValidationResults(target, source) {
  target.deviations.push(...(source.deviations || []));
  appendWarnings(target, source.warnings || []);
  if (source.acceptedOutcome) {
    acceptValidation(target, source.acceptedReasons.join("; "));
  }
}

function mergeContractMetrics(targetContract, sourceContract) {
  if (!targetContract?.metrics || !sourceContract?.metrics) {
    return;
  }
  targetContract.metrics.allowedCriticalLabels = uniqueStrings([
    ...(targetContract.metrics.allowedCriticalLabels || []),
    ...(sourceContract.metrics.allowedCriticalLabels || [])
  ]);
  targetContract.metrics.negativeCriticalLabels = uniqueStrings([
    ...(targetContract.metrics.negativeCriticalLabels || []),
    ...(sourceContract.metrics.negativeCriticalLabels || [])
  ]);
  targetContract.metrics.positiveCriticalLabels = uniqueStrings([
    ...(targetContract.metrics.positiveCriticalLabels || []),
    ...(sourceContract.metrics.positiveCriticalLabels || [])
  ]);
  targetContract.metrics.primarySemverDecision = targetContract.metrics.primarySemverDecision || sourceContract.metrics.primarySemverDecision || null;
}

function validateUnifiedAutomationScenarioContract(scenario, result) {
  const validation = createValidationResult(buildScenarioContract(scenario));
  const expected = validation.contract.expected;

  if (scenario.autobotScenario) {
    const autobotValidation = validateAutobotScenarioContract(scenario.autobotScenario, buildUnifiedAutomationAutobotRawResult(result));
    mergeValidationResults(validation, autobotValidation);
    mergeContractMetrics(validation.contract, autobotValidation.contract);
    validation.metricContract = autobotValidation.contract;
  }
  if (typeof expected.autobotReady === "boolean" && result.autobotReady !== expected.autobotReady) {
    validation.deviations.push(`expected autobotReady=${expected.autobotReady} but got ${result.autobotReady}`);
  }
  if (typeof expected.smartLinkReady === "boolean" && result.smartLinkReady !== expected.smartLinkReady) {
    validation.deviations.push(`expected smartLinkReady=${expected.smartLinkReady} but got ${result.smartLinkReady}`);
  }
  validateLabelCollection(validation, {
    actualLabels: result.linkedLabels,
    acceptableLabelSets: validation.contract.acceptable.acceptableLinkedLabelSets,
    description: "linked",
    requiredLabels: expected.linkedLabels
  });
  validateLabelCollection(validation, {
    actualLabels: result.issueLabels,
    acceptableLabelSets: validation.contract.acceptable.acceptableIssueLabelSets,
    description: "issue",
    forbiddenLabels: expected.issueLabelsExclude,
    requiredLabels: expected.issueLabelsInclude
  });
  if (expected.milestoneTitle !== undefined && expected.milestoneTitle !== result.milestoneTitle) {
    validation.deviations.push(`expected milestone ${expected.milestoneTitle || "(none)"} but got ${result.milestoneTitle || "(none)"}`);
  }
  if (typeof expected.autobotCommentPresent === "boolean" && result.autobotCommentPresent !== expected.autobotCommentPresent) {
    validation.deviations.push(`expected autobotCommentPresent=${expected.autobotCommentPresent} but got ${result.autobotCommentPresent}`);
  }
  if (typeof expected.smartLinkCommentPresent === "boolean" && result.smartLinkCommentPresent !== expected.smartLinkCommentPresent) {
    validation.deviations.push(`expected smartLinkCommentPresent=${expected.smartLinkCommentPresent} but got ${result.smartLinkCommentPresent}`);
  }
  if (typeof expected.graphCommentPresent === "boolean" && result.graphCommentPresent !== expected.graphCommentPresent) {
    validation.deviations.push(`expected graphCommentPresent=${expected.graphCommentPresent} but got ${result.graphCommentPresent}`);
  }
  validateTextFragments(validation, {
    actualText: result.autobotCommentBody,
    acceptableFragmentGroups: validation.contract.acceptable.acceptableAutobotCommentIncludesAny,
    description: "autobot comment",
    requiredFragments: expected.autobotCommentIncludes
  });
  validateTextFragments(validation, {
    actualText: result.smartLinkCommentBody,
    acceptableFragmentGroups: validation.contract.acceptable.acceptableSmartLinkCommentIncludesAny,
    description: "smart-link comment",
    requiredFragments: expected.smartLinkCommentIncludes
  });
  for (const fragment of expected.smartLinkCommentExcludes || []) {
    if (String(result.smartLinkCommentBody || "").includes(fragment)) {
      validation.deviations.push(`expected smart-link comment to exclude ${fragment}`);
    }
  }
  validateTextFragments(validation, {
    actualText: result.graphCommentBody,
    acceptableFragmentGroups: validation.contract.acceptable.acceptableGraphCommentIncludesAny,
    description: "graph comment",
    requiredFragments: expected.graphCommentIncludes
  });
  validateTextFragments(validation, {
    actualText: result.pullBody,
    acceptableFragmentGroups: validation.contract.acceptable.acceptablePullBodyIncludesAny,
    description: "pull body",
    requiredFragments: expected.pullBodyIncludes
  });
  for (const fragment of expected.pullBodyExcludes || []) {
    if (String(result.pullBody || "").includes(fragment)) {
      validation.deviations.push(`expected pull body to exclude ${fragment}`);
    }
  }
  if (expected.checkConclusion !== undefined && result.checkConclusion !== expected.checkConclusion) {
    validation.deviations.push(`expected check conclusion ${expected.checkConclusion || "(none)"} but got ${result.checkConclusion || "(none)"}`);
  }
  if (Array.isArray(expected.createdReleaseTags) && JSON.stringify(result.createdReleaseTags) !== JSON.stringify(expected.createdReleaseTags)) {
    validation.deviations.push(`expected created releases ${expected.createdReleaseTags.join(", ") || "(none)"} but got ${result.createdReleaseTags.join(", ") || "(none)"}`);
  }
  if (Array.isArray(expected.createdMilestoneTitles) && JSON.stringify(result.createdMilestoneTitles) !== JSON.stringify(expected.createdMilestoneTitles)) {
    validation.deviations.push(`expected created milestones ${expected.createdMilestoneTitles.join(", ") || "(none)"} but got ${result.createdMilestoneTitles.join(", ") || "(none)"}`);
  }
  if (Array.isArray(expected.closedMilestoneTitles) && JSON.stringify(result.closedMilestoneTitles) !== JSON.stringify(expected.closedMilestoneTitles)) {
    validation.deviations.push(`expected closed milestones ${expected.closedMilestoneTitles.join(", ") || "(none)"} but got ${result.closedMilestoneTitles.join(", ") || "(none)"}`);
  }
  if (Array.isArray(expected.failureOperations) && JSON.stringify(result.failureOperations) !== JSON.stringify(expected.failureOperations)) {
    validation.deviations.push(`expected failure operations ${expected.failureOperations.join(", ") || "(none)"} but got ${result.failureOperations.join(", ") || "(none)"}`);
  }
  if (result.failedMessages.length > 0) {
    validation.deviations.push(`expected no workflow core failures but got ${result.failedMessages.join(" | ")}`);
  }
  addUnexpectedCriticalLabelWarnings(validation, result.issueLabels);
  if (!validation.metricContract) {
    validation.metricContract = validation.contract;
  }
  return validation;
}

function validateUnifiedAutomationScenarioResult(scenario, result) {
  return validateUnifiedAutomationScenarioContract(scenario, result).deviations;
}

function buildUnifiedAutomationActualResult(result) {
  return {
    autobotReady: result.autobotReady,
    checkConclusion: result.checkConclusion,
    closedMilestoneTitles: result.closedMilestoneTitles,
    createdMilestoneTitles: result.createdMilestoneTitles,
    createdReleaseTags: result.createdReleaseTags,
    failureOperations: result.failureOperations,
    graphCommentPresent: result.graphCommentPresent,
    issueLabels: result.issueLabels,
    linkedLabels: result.linkedLabels,
    milestoneTitle: result.milestoneTitle,
    semverDecision: result.semverDecision,
    smartLinkCommentPresent: result.smartLinkCommentPresent,
    smartLinkReady: result.smartLinkReady
  };
}

async function executeUnifiedAutomationScenario(scenario) {
  const owner = "octo";
  const repo = "repo";
  const github = createAutobotPipelineGithubMock(scenario.githubOptions);
  const core = createWorkflowCoreMock();
  const workspaceKey = scenario.issueNumber || scenario.renderTargetIssueNumber || 0;
  return withScenarioTempWorkspace(["unified-automation", scenario.kind, workspaceKey], async (workspace) => {
    const stateFile = workspace.resolveFile("state.json");
    const outputs = await analyzeUnifiedAutomationState({
      context: scenario.context,
      core,
      github,
      owner,
      repo,
      stateFile,
      thresholdInput: scenario.thresholdInput
    });
    const persistedState = await applyUnifiedAutomationState({
      context: scenario.context,
      github,
      owner,
      repo,
      stateFile
    });
    const issueNumber = scenario.issueNumber
      || scenario.context.issue?.number
      || scenario.context.payload?.issue?.number
      || scenario.context.payload?.pull_request?.number
      || null;
    const renderTargetIssueNumber = scenario.renderTargetIssueNumber || issueNumber;
    const issue = issueNumber ? github.state.issuesByNumber.get(issueNumber) : null;
    const renderTargetIssue = renderTargetIssueNumber ? github.state.issuesByNumber.get(renderTargetIssueNumber) : null;
    const pull = issueNumber ? github.state.pullRequestsByNumber.get(issueNumber) : null;
    const sourceComments = issueNumber ? (github.state.commentsByIssue.get(issueNumber) || []) : [];
    const targetComments = renderTargetIssueNumber ? (github.state.commentsByIssue.get(renderTargetIssueNumber) || []) : [];
    const autobotComment = getManagedBotComment(sourceComments);
    const graphComment = findCommentContaining(sourceComments, "<!-- smart-link-graph:");
    const smartLinkComment = findCommentContaining(targetComments, "<!-- smart-link-comment:");
    const autobotMetadata = extractManagedBotCommentMetadata(autobotComment?.body || "");

    return {
      autobotCommentBody: autobotComment?.body || "",
      autobotCommentPresent: Boolean(autobotComment),
      autobotReady: outputs.autobot_ready === "true",
      checkConclusion: github.state.checks.at(-1)?.conclusion || null,
      closedMilestoneTitles: github.state.milestones
        .filter((milestone) => milestone.state === "closed")
        .map((milestone) => milestone.title)
        .sort((left, right) => left.localeCompare(right)),
      combinedLabels: parseJson(outputs.autobot_labels_json || "[]"),
      coreOutputs: Object.fromEntries(core.outputs.entries()),
      createdMilestoneTitles: github.state.createdMilestones.slice(),
      createdReleaseTags: github.state.createdReleases.map((release) => release.tag_name),
      failureOperations: github.state.failureLog.map((entry) => entry.operationName),
      failedMessages: core.failedMessages.slice(),
      graphCommentBody: graphComment?.body || "",
      graphCommentPresent: Boolean(graphComment),
      hardSignals: Array.isArray(autobotMetadata.deterministicSemver?.hardSignals) ? autobotMetadata.deterministicSemver.hardSignals : [],
      issue,
      issueLabels: (issue?.labels || []).map((label) => label.name),
      linkedLabels: parseJson(outputs.linked_labels_json || "[]"),
      milestoneTitle: issue?.milestone?.title || renderTargetIssue?.milestone?.title || null,
      outputs,
      persistedState,
      pull,
      pullBody: pull?.body || "",
      renderTargetIssue,
      semverDecision: String(autobotMetadata.semverDecision || autobotMetadata.deterministicSemver?.decision || "none"),
      smartLinkCommentBody: smartLinkComment?.body || "",
      smartLinkCommentPresent: Boolean(smartLinkComment),
      smartLinkReady: outputs.smart_link_ready === "true"
    };
  });
}

async function evaluateUnifiedAutomationScenarioSuite({ config } = {}) {
  const suite = buildUnifiedAutomationScenarioCases(config || {});
  const evaluations = [];

  for (const scenario of suite.scenarios) {
    const rawResult = await executeUnifiedAutomationScenario(scenario);
    const validation = validateUnifiedAutomationScenarioContract(scenario, rawResult);
    evaluations.push({
      actualResult: buildUnifiedAutomationActualResult(rawResult),
      acceptedOutcome: validation.acceptedOutcome,
      acceptedReasons: validation.acceptedReasons,
      contract: validation.contract,
      deviations: validation.deviations,
      kind: scenario.kind,
      metricContract: validation.metricContract,
      name: scenario.name,
      optimalResult: getScenarioOptimalResult(scenario),
      passed: validation.deviations.length === 0,
      rawResult,
      scenario,
      warnings: validation.warnings
    });
  }

  const summary = summarizeScenarioEvaluations(evaluations, { suiteKey: suite.key });
  return {
    ...suite,
    evaluations,
    reportText: formatScenarioSuiteReport({
      evaluations,
      seed: suite.seed,
      suiteKey: suite.key,
      summary
    }),
    summary
  };
}

function validateSmartLinkScenarioContract(scenario, result) {
  const validation = createValidationResult(buildScenarioContract(scenario));
  const expected = validation.contract.expected;
  const acceptableRelationKinds = validation.contract.acceptable.acceptableRelationKinds;
  if (Array.isArray(expected.emittedCandidateNumbers)) {
    const actualCandidateNumbers = Array.isArray(result.emittedCandidateNumbers) ? result.emittedCandidateNumbers : [];
    if (JSON.stringify(actualCandidateNumbers) !== JSON.stringify(expected.emittedCandidateNumbers)) {
      validation.deviations.push(`expected emitted candidates ${expected.emittedCandidateNumbers.join(", ")} but got ${actualCandidateNumbers.join(", ") || "(none)"}`);
    }
  }
  if (Array.isArray(expected.relationKinds)) {
    const actualRelationKinds = Array.isArray(result.relationKinds) ? result.relationKinds : [];
    if (JSON.stringify(actualRelationKinds) !== JSON.stringify(expected.relationKinds)) {
      if (acceptableRelationKinds.length > 0 && actualRelationKinds.every((relationKind) => acceptableRelationKinds.includes(String(relationKind).toLowerCase()))) {
        acceptValidation(validation, "relation kinds matched an acceptable outcome set");
      } else {
        validation.deviations.push(`expected relation kinds ${expected.relationKinds.join(", ")} but got ${actualRelationKinds.join(", ") || "(none)"}`);
      }
    }
  }
  if (expected.relationKind && result.relationKind !== expected.relationKind) {
    if (acceptableRelationKinds.includes(String(result.relationKind || "").toLowerCase())) {
      acceptValidation(validation, `relation kind ${result.relationKind} matched an acceptable outcome`);
    } else {
      validation.deviations.push(`expected relation ${expected.relationKind} but got ${result.relationKind}`);
    }
  }
  if (typeof expected.thresholdPassed === "boolean" && result.thresholdPassed !== expected.thresholdPassed) {
    validation.deviations.push(`expected thresholdPassed=${expected.thresholdPassed} but got ${result.thresholdPassed}`);
  }
  if (expected.minScore && result.emittedScore < expected.minScore) {
    validation.deviations.push(`expected emitted score >= ${expected.minScore} but got ${result.emittedScore}`);
  }
  const actualSuppressionReasons = uniqueStrings(result.suppressionReasons || []);
  const suppressionPresenceDeviations = [];
  for (const reason of expected.suppressionIncludes || []) {
    if (!actualSuppressionReasons.includes(reason)) {
      suppressionPresenceDeviations.push(`expected suppression reason ${reason} to be present`);
    }
  }
  if (suppressionPresenceDeviations.length > 0) {
    if (matchesAnyRequiredSet(actualSuppressionReasons, validation.contract.acceptable.acceptableSuppressionReasonSets)) {
      acceptValidation(validation, "suppression reasons matched an acceptable alternative set");
    } else {
      validation.deviations.push(...suppressionPresenceDeviations);
    }
  }
  for (const reason of expected.suppressionExcludes || []) {
    if (actualSuppressionReasons.includes(reason)) {
      validation.deviations.push(`expected suppression reason ${reason} to be absent`);
    }
  }
  if (expected.directiveSummary && result.directiveSummary) {
    for (const [key, value] of Object.entries(expected.directiveSummary)) {
      const actualValue = result.directiveSummary[key] || [];
      if (JSON.stringify(actualValue) !== JSON.stringify(value)) {
        validation.deviations.push(`expected directive ${key}=${value.join(", ") || "(none)"} but got ${actualValue.join(", ") || "(none)"}`);
      }
    }
  }
  for (const fragment of expected.graphContains || []) {
    if (!Array.isArray(result.graphLines) || !result.graphLines.some((line) => String(line).includes(fragment))) {
      validation.deviations.push(`expected graph fragment ${fragment} to be present`);
    }
  }
  return validation;
}

function validateSmartLinkScenarioResult(scenario, result) {
  return validateSmartLinkScenarioContract(scenario, result).deviations;
}

function buildSmartLinkActualResult(result) {
  if (Array.isArray(result.emittedResults)) {
    return {
      directiveSummary: result.directiveSummary,
      emittedCandidateNumbers: result.emittedResults.map((entry) => entry.candidate.number),
      graphLines: result.graphLines,
      relationKinds: result.emittedResults.map((entry) => entry.relationKind)
    };
  }
  return {
    emittedScore: result.emittedScore,
    relationKind: result.relationKind,
    suppressionReasons: Array.isArray(result.suppressionReasons) ? result.suppressionReasons : [],
    thresholdPassed: Boolean(result.thresholdPassed)
  };
}

function extractAutobotMetricContract(evaluation, suiteKey) {
  if (suiteKey === "UNIFIED_AUTOMATION") {
    return evaluation.metricContract || evaluation.contract || null;
  }
  if (suiteKey === "AUTOBOT" || suiteKey === "AUTOBOT_PIPELINE") {
    return evaluation.contract || null;
  }
  return null;
}

function extractEvaluationLabels(evaluation, suiteKey) {
  if (suiteKey === "AUTOBOT") {
    return uniqueStrings(evaluation.actualResult.labels || []);
  }
  if (suiteKey === "AUTOBOT_PIPELINE" || suiteKey === "UNIFIED_AUTOMATION") {
    return uniqueStrings(evaluation.actualResult.issueLabels || []);
  }
  return [];
}

function buildReleaseMetrics(evaluations, suiteKey) {
  const semverConfusion = createMatrix(SEMVER_DIMENSIONS);
  const criticalLabelMetrics = Object.fromEntries(CRITICAL_LABELS.map((label) => [label, {
    fn: 0,
    fp: 0,
    precision: 1,
    recall: 1,
    tn: 0,
    tp: 0
  }]));

  for (const evaluation of evaluations) {
    const contract = extractAutobotMetricContract(evaluation, suiteKey);
    if (!contract) {
      continue;
    }
    const expectedSemver = contract.metrics.primarySemverDecision;
    const actualSemver = normalizeSemverDecision(evaluation.actualResult.semverDecision);
    if (expectedSemver && semverConfusion[expectedSemver] && Object.prototype.hasOwnProperty.call(semverConfusion[expectedSemver], actualSemver)) {
      semverConfusion[expectedSemver][actualSemver] += 1;
    }
    const actualLabels = new Set(extractEvaluationLabels(evaluation, suiteKey));
    const positiveCriticalLabels = new Set(contract.metrics.positiveCriticalLabels || []);
    const negativeCriticalLabels = new Set(contract.metrics.negativeCriticalLabels || []);
    const actualLabelList = [...actualLabels];
    for (const label of CRITICAL_LABELS) {
      const metric = criticalLabelMetrics[label];
      const actualHas = actualLabels.has(label)
        || actualLabelList.some((actual) => AutobotLabelRegistry.matchesExpectedLabel(actual, label));
      if (positiveCriticalLabels.has(label)) {
        if (actualHas) {
          metric.tp += 1;
        } else {
          metric.fn += 1;
        }
      } else if (negativeCriticalLabels.has(label)) {
        if (actualHas) {
          metric.fp += 1;
        } else {
          metric.tn += 1;
        }
      }
    }
  }

  for (const metric of Object.values(criticalLabelMetrics)) {
    metric.precision = metric.tp + metric.fp === 0 ? 1 : metric.tp / (metric.tp + metric.fp);
    metric.recall = metric.tp + metric.fn === 0 ? 1 : metric.tp / (metric.tp + metric.fn);
  }

  return {
    criticalLabelMetrics,
    semverConfusion
  };
}

function buildSmartLinkMetrics(evaluations) {
  const relationConfusion = createMatrix(SMART_LINK_RELATION_DIMENSIONS);
  for (const evaluation of evaluations) {
    const expected = evaluation.contract.expected;
    const expectedRelation = expected.relationKind || (Array.isArray(expected.relationKinds) && expected.relationKinds.length === 1 ? expected.relationKinds[0] : null);
    const actualRelation = evaluation.actualResult.relationKind || (Array.isArray(evaluation.actualResult.relationKinds) && evaluation.actualResult.relationKinds.length === 1 ? evaluation.actualResult.relationKinds[0] : null);
    if (!expectedRelation || !actualRelation) {
      continue;
    }
    const normalizedExpected = String(expectedRelation).toLowerCase();
    const normalizedActual = String(actualRelation).toLowerCase();
    if (relationConfusion[normalizedExpected] && Object.prototype.hasOwnProperty.call(relationConfusion[normalizedExpected], normalizedActual)) {
      relationConfusion[normalizedExpected][normalizedActual] += 1;
    }
  }
  return { relationConfusion };
}

function buildScenarioMetrics(evaluations, suiteKey) {
  if (suiteKey === "SMART_LINK") {
    return buildSmartLinkMetrics(evaluations);
  }
  return buildReleaseMetrics(evaluations, suiteKey);
}

function summarizeScenarioEvaluations(evaluations, { suiteKey } = {}) {
  const summary = evaluations.reduce((result, evaluation) => {
    if (!evaluation.passed) {
      result.deviationScenarioCount += 1;
      result.deviationCount += evaluation.deviations.length;
    }
    if (evaluation.passed) {
      result.passedScenarioCount += 1;
      if (!evaluation.acceptedOutcome) {
        result.exactPassCount += 1;
      }
    }
    if (evaluation.acceptedOutcome) {
      result.acceptedOutcomeCount += 1;
    }
    if ((evaluation.warnings || []).length > 0) {
      result.warningScenarioCount += 1;
      result.warningCount += evaluation.warnings.length;
    }
    if (evaluation.passed && (evaluation.warnings || []).some((warning) => warning.code === "unexpected-critical-label")) {
      result.falseGreenCount += 1;
    }
    return result;
  }, {
    acceptedOutcomeCount: 0,
    deviationCount: 0,
    deviationScenarioCount: 0,
    exactPassCount: 0,
    falseGreenCount: 0,
    passedScenarioCount: 0,
    totalScenarios: evaluations.length,
    warningCount: 0,
    warningScenarioCount: 0
  });

  summary.exactPassRate = summary.totalScenarios === 0 ? 1 : summary.exactPassCount / summary.totalScenarios;
  summary.falseGreenRate = summary.passedScenarioCount === 0 ? 0 : summary.falseGreenCount / summary.passedScenarioCount;
  summary.metrics = buildScenarioMetrics(evaluations, suiteKey);
  return summary;
}

function formatRate(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatMatrixLines(matrix) {
  return Object.entries(matrix)
    .map(([expected, row]) => {
      const nonZeroCells = Object.entries(row)
        .filter(([, count]) => count > 0)
        .map(([actual, count]) => `${expected}->${actual}=${count}`);
      return nonZeroCells.length > 0 ? nonZeroCells.join(", ") : "";
    })
    .filter(Boolean);
}

function formatCriticalLabelMetricLines(metrics) {
  return Object.entries(metrics)
    .filter(([, metric]) => metric.tp > 0 || metric.fp > 0 || metric.fn > 0)
    .map(([label, metric]) => `${label}: precision=${metric.precision.toFixed(2)}, recall=${metric.recall.toFixed(2)}, tp=${metric.tp}, fp=${metric.fp}, fn=${metric.fn}`);
}

function formatScenarioEvaluationBlock(evaluation) {
  const lines = [
    `Scenario: ${evaluation.name}`,
    `Status: ${(evaluation.deviations || []).length > 0 ? "deviation" : (evaluation.warnings || []).length > 0 ? "warning" : evaluation.acceptedOutcome ? "accepted" : "pass"}`,
    `Kind: ${evaluation.kind}`,
    `Replay: ${evaluation.scenario?.replayKey || "(unavailable)"}`
  ];
  if (evaluation.scenario?.realismProfile) {
    lines.push(`Realism: ${stableStringify(evaluation.scenario.realismProfile)}`);
  }
  lines.push(`Scenario Detail: ${stableStringify(evaluation.scenario)}`);
  lines.push(`Optimal: ${stableStringify(evaluation.optimalResult)}`);
  lines.push(`Actual: ${stableStringify(evaluation.actualResult)}`);
  if (evaluation.rawResult !== undefined) {
    lines.push(`Raw Result: ${stableStringify(evaluation.rawResult)}`);
  }
  if ((evaluation.acceptedReasons || []).length > 0) {
    lines.push("Accepted Reasons:");
    for (const reason of evaluation.acceptedReasons) {
      lines.push(`- ${reason}`);
    }
  }
  if ((evaluation.deviations || []).length > 0) {
    lines.push("Deviations:");
    for (const deviation of evaluation.deviations) {
      lines.push(`- ${deviation}`);
    }
  }
  if ((evaluation.warnings || []).length > 0) {
    lines.push("Warnings:");
    for (const warning of evaluation.warnings) {
      lines.push(`- ${warning.message}`);
    }
  }
  return lines;
}

function formatScenarioSuiteReport(input) {
  const summary = input.summary;
  const lines = [
    `Scenario audit: ${input.suiteKey}`,
    `Seed: ${input.seed}`,
    `Scenarios: ${summary.totalScenarios}`,
    `Deviation scenarios: ${summary.deviationScenarioCount}`,
    `Deviation items: ${summary.deviationCount}`,
    `Warning scenarios: ${summary.warningScenarioCount}`,
    `Warning items: ${summary.warningCount}`,
    `Accepted outcomes: ${summary.acceptedOutcomeCount}`,
    `Exact pass rate: ${formatRate(summary.exactPassRate)}`,
    `False-green rate: ${formatRate(summary.falseGreenRate)}`,
    ""
  ];

  const metricLines = [];
  if (summary.metrics.semverConfusion) {
    metricLines.push(...formatMatrixLines(summary.metrics.semverConfusion).map((line) => `Semver: ${line}`));
  }
  if (summary.metrics.criticalLabelMetrics) {
    metricLines.push(...formatCriticalLabelMetricLines(summary.metrics.criticalLabelMetrics).map((line) => `Critical Labels: ${line}`));
  }
  if (summary.metrics.relationConfusion) {
    metricLines.push(...formatMatrixLines(summary.metrics.relationConfusion).map((line) => `Relations: ${line}`));
  }

  if (summary.deviationCount === 0) {
    lines.push("No deviations from scenario contracts.");
    const warningEvaluations = input.evaluations.filter((evaluation) => (evaluation.warnings || []).length > 0);
    if (warningEvaluations.length > 0) {
      lines.push("", "Detailed warnings:");
      for (const evaluation of warningEvaluations) {
        lines.push(...formatScenarioEvaluationBlock(evaluation));
        lines.push("");
      }
    }
    if (metricLines.length > 0) {
      lines.push("", ...metricLines);
    }
    return lines.join("\n");
  }

  for (const evaluation of input.evaluations.filter((entry) => !entry.passed || (entry.warnings || []).length > 0)) {
    lines.push(...formatScenarioEvaluationBlock(evaluation));
    lines.push("");
  }

  if (metricLines.length > 0) {
    lines.push("Metrics:");
    lines.push(...metricLines.map((line) => `- ${line}`));
  }

  return lines.join("\n");
}

function evaluateAutobotScenarioSuite({ analyzeScenario, config } = {}) {
  const suite = buildAutobotScenarioCases(config);
  const evaluations = suite.scenarios.map((scenario) => {
    const rawResult = analyzeScenario(scenario.input);
    const validation = validateAutobotScenarioContract(scenario, rawResult);
    return {
      actualResult: buildAutobotActualResult(rawResult),
      acceptedOutcome: validation.acceptedOutcome,
      acceptedReasons: validation.acceptedReasons,
      contract: validation.contract,
      deviations: validation.deviations,
      kind: scenario.kind,
      name: scenario.name,
      optimalResult: getScenarioOptimalResult(scenario),
      passed: validation.deviations.length === 0,
      rawResult,
      scenario,
      warnings: validation.warnings
    };
  });
  const summary = summarizeScenarioEvaluations(evaluations, { suiteKey: suite.key });
  return {
    ...suite,
    evaluations,
    reportText: formatScenarioSuiteReport({
      evaluations,
      seed: suite.seed,
      suiteKey: suite.key,
      summary
    }),
    summary
  };
}

function evaluateSmartLinkScenarioSuite({ analyzeScenario, config } = {}) {
  const suite = buildSmartLinkScenarioCases(config);
  const evaluations = suite.scenarios.map((scenario) => {
    const rawResult = scenario.mode === "batch"
      ? (() => {
          const candidateResults = scenario.candidates.map((candidate) => analyzeScenario({
            candidate,
            source: scenario.source,
            threshold: scenario.threshold
          }));
          const emittedResults = selectSmartLinkResults({ candidateResults, threshold: scenario.threshold });
          return {
            candidateResults,
            directiveSummary: buildDirectiveSummary(emittedResults),
            emittedCandidateNumbers: emittedResults.map((entry) => entry.candidate.number),
            emittedResults,
            graphLines: buildMermaidGraphLines(scenario.source, emittedResults),
            relationKinds: emittedResults.map((entry) => entry.relationKind)
          };
        })()
      : analyzeScenario({
          candidate: scenario.candidate,
          source: scenario.source,
          threshold: scenario.threshold
        });
    const validation = validateSmartLinkScenarioContract(scenario, rawResult);
    return {
      actualResult: buildSmartLinkActualResult(rawResult),
      acceptedOutcome: validation.acceptedOutcome,
      acceptedReasons: validation.acceptedReasons,
      contract: validation.contract,
      deviations: validation.deviations,
      kind: scenario.kind,
      name: scenario.name,
      optimalResult: getScenarioOptimalResult(scenario),
      passed: validation.deviations.length === 0,
      rawResult,
      scenario,
      warnings: validation.warnings
    };
  });
  const summary = summarizeScenarioEvaluations(evaluations, { suiteKey: suite.key });
  return {
    ...suite,
    evaluations,
    reportText: formatScenarioSuiteReport({
      evaluations,
      seed: suite.seed,
      suiteKey: suite.key,
      summary
    }),
    summary
  };
}

module.exports = {
  buildAutobotPipelineScenarioCases,
  buildAutobotScenarioCases,
  buildScenarioContract,
  buildSmartLinkScenarioCases,
  buildUnifiedAutomationScenarioCases,
  createAdversarialConflictingEvidenceSnapshot,
  createAdversarialCrossDomainFalsePositiveSnapshot,
  createAdversarialEmptyPayloadSnapshot,
  createAdversarialMaxBoundarySnapshot,
  evaluateAutobotPipelineScenarioSuite,
  evaluateAutobotScenarioSuite,
  evaluateSmartLinkScenarioSuite,
  evaluateUnifiedAutomationScenarioSuite,
  formatScenarioEvaluationBlock,
  formatScenarioSuiteReport,
  getScenarioOptimalResult,
  resolveScenarioConfig,
  summarizeScenarioEvaluations,
  validateAutobotPipelineScenarioContract,
  validateAutobotPipelineScenarioResult,
  validateAutobotScenarioContract,
  validateAutobotScenarioResult,
  validateSmartLinkScenarioContract,
  validateSmartLinkScenarioResult,
  validateUnifiedAutomationScenarioContract,
  validateUnifiedAutomationScenarioResult
};
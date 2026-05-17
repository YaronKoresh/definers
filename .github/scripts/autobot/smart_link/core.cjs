const crypto = require("crypto");

const { normalizeAccumulatedScore } = require("../measurement/scorer.cjs");

const { DEFAULT_THRESHOLD } = require("../constants.cjs");
const MAX_TOTAL_CANDIDATES = 80;
const MAX_EMITTED_RESULTS = 20;
const MANAGED_BODY_START = "<!-- smart-autolinker:start -->";
const MANAGED_BODY_END = "<!-- smart-autolinker:end -->";
const MANAGED_COMMENT_MARKER_PREFIX = "<!-- smart-link-comment:";
const GRAPH_COMMENT_MARKER_PREFIX = "<!-- smart-link-graph:";
const CHECK_NAME = "Smart Link Dependency Integrity";
const TRUSTED_ASSOCIATIONS = new Set(["MEMBER", "OWNER", "COLLABORATOR"]);
const LOW_SIGNAL_LABELS = new Set([
  "backlog",
  "bug",
  "discussion",
  "duplicate",
  "enhancement",
  "good first issue",
  "help wanted",
  "invalid",
  "question",
  "task",
  "triage",
  "wontfix"
]);
const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "been",
  "being",
  "but",
  "by",
  "can",
  "change",
  "changed",
  "changes",
  "changing",
  "ci",
  "close",
  "closed",
  "closes",
  "closing",
  "connect",
  "connected",
  "connects",
  "connecting",
  "depend",
  "depends",
  "doc",
  "docs",
  "documentation",
  "feature",
  "features",
  "feat",
  "fix",
  "fixed",
  "fixes",
  "fixing",
  "for",
  "from",
  "has",
  "have",
  "implements",
  "implement",
  "implemented",
  "implementing",
  "in",
  "into",
  "is",
  "issue",
  "issues",
  "it",
  "its",
  "merge",
  "merged",
  "new",
  "of",
  "on",
  "or",
  "our",
  "pull",
  "pr",
  "ref",
  "refs",
  "reference",
  "references",
  "related",
  "relates",
  "relating",
  "resolve",
  "resolved",
  "resolves",
  "resolving",
  "see",
  "that",
  "the",
  "their",
  "this",
  "those",
  "to",
  "update",
  "updated",
  "updates",
  "updating",
  "with"
]);
const DIRECTIVE_REGEX = /\b(?<directive>close[sd]?|fix(?:e[sd])?|resolve[sd]?|implement(?:s|ed|ing)?|connect(?:s|ed|ing)?|relate(?:s|d|ing)?|reference[sd]?|see|depend(?:s|ed|ing)?(?:\s+on)?|blocked(?:\s+by)?|duplicate(?:\s+of)?)\s+(?:to\s+|on\s+|by\s+)?(?:(?<repo>[a-z0-9_.-]+\/[a-z0-9_.-]+))?#(?<id>\d+)\b/gi;
const GENERIC_REF_REGEX = /(?:(?<repo>[a-z0-9_.-]+\/[a-z0-9_.-]+))?#(?<id>\d+)\b/gi;
const URL_REF_REGEX = /https:\/\/github\.com\/(?<repo>[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)\/(?:issues|pull)\/(?<id>\d+)\b/gi;
const PATH_SIGNAL_REGEX = /\b(?:\.github|docs|docker|scripts|src|tests)(?:\/[A-Za-z0-9_.-]+){1,5}\b/gi;
const GHSA_REGEX = /\bGHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}\b/gi;
const CVE_REGEX = /\bCVE-\d{4}-\d{4,}\b/gi;
const CWE_REGEX = /\bCWE-\d+\b/gi;
const SCORE_WEIGHTS = Object.freeze({
  branchReference: 0.75,
  dependencyReference: 1.2,
  explicitClose: 1.7,
  explicitImplement: 1.65,
  explicitReference: 0.9,
  explicitRelated: 0.38,
  explicitConnect: 1.05,
  reciprocalClose: 0.8,
  reciprocalReference: 0.78,
  sharedAlertIdentifier: 1.55,
  remediationReference: 1.3,
  sameMilestone: 0.22,
  recencyRecent: 0.08
});
const SCORE_CAPS = Object.freeze({
  lexicalBody: 0.18,
  lexicalTitle: 0.28,
  sharedEcosystem: 0.48,
  sharedFileSurface: 0.72,
  sharedLabels: 0.54,
  sharedPackage: 0.78
});
const PENALTIES = Object.freeze({
  closedTargetForClosure: 1.25,
  docsSecurityConflict: 0.7,
  ecosystemMismatch: 0.9,
  packageMismatch: 0.65,
  staleTarget: 0.2
});
const EMISSION_LIMITS = Object.freeze({
  advisory_fix: 8,
  closes: 8,
  connects: 12,
  depends_on: 12,
  related: 12
});
const RELATION_PRIORITY = Object.freeze({
  closes: 0,
  advisory_fix: 1,
  depends_on: 2,
  connects: 3,
  related: 4
});
const EVIDENCE_CLASS_PRIORITY = Object.freeze({
  explicit: 0,
  reciprocal: 1,
  structural: 2,
  lexical: 3,
  temporal: 4
});

function asString(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function normalizeRepoFullName(repoFullName) {
  return asString(repoFullName).trim().toLowerCase();
}

function normalizeWhitespace(text) {
  return asString(text).replace(/\r/g, "").trim();
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function uniqueStrings(values) {
  return [...new Set((Array.isArray(values) ? values : []).map((value) => asString(value).trim()).filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function uniqueNumbers(values) {
  return [...new Set((Array.isArray(values) ? values : []).map((value) => Number.parseInt(String(value), 10)).filter((value) => Number.isFinite(value) && value > 0))].sort((left, right) => left - right);
}

function normalizeLabels(labels) {
  return uniqueStrings((Array.isArray(labels) ? labels : []).map((label) => {
    if (typeof label === "string") return label.toLowerCase();
    return asString(label && label.name).toLowerCase();
  }));
}

function normalizeMilestone(milestone) {
  if (!milestone || typeof milestone !== "object") return null;
  const number = Number.parseInt(String(milestone.number || 0), 10);
  const title = normalizeWhitespace(milestone.title);
  if (!Number.isFinite(number) || number <= 0) {
    if (!title) return null;
    return { number: null, title };
  }
  return { number, title };
}

function normalizeChangedFiles(changedFiles) {
  return (Array.isArray(changedFiles) ? changedFiles : [])
    .map((file) => {
      if (typeof file === "string") {
        return { filename: file, status: "modified" };
      }
      return {
        filename: asString(file && file.filename),
        status: asString(file && file.status).toLowerCase() || "modified"
      };
    })
    .filter((file) => file.filename)
    .sort((left, right) => left.filename.localeCompare(right.filename));
}

function tokenizeText(text) {
  return asString(text)
    .toLowerCase()
    .replace(/[^a-z0-9_./\-\s]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token) && !/^\d+$/.test(token));
}

function extractAlertIdentifiers(text) {
  const identifiers = new Set();
  for (const regex of [GHSA_REGEX, CVE_REGEX, CWE_REGEX]) {
    for (const match of asString(text).matchAll(regex)) {
      identifiers.add(String(match[0]).toUpperCase());
    }
  }
  return [...identifiers].sort((left, right) => left.localeCompare(right));
}

function extractPathSignalsFromText(text) {
  const signals = new Set();
  for (const match of asString(text).matchAll(PATH_SIGNAL_REGEX)) {
    const normalized = match[0].toLowerCase();
    const parts = normalized.split("/").filter(Boolean);
    if (parts.length >= 2) {
      signals.add(parts.slice(0, Math.min(parts.length, 3)).join("/"));
      signals.add(parts.slice(0, 2).join("/"));
    }
  }
  return [...signals].sort((left, right) => left.localeCompare(right));
}

function buildPackageSignalsFromPaths(paths) {
  const packageSignals = new Set();
  for (const value of Array.isArray(paths) ? paths : []) {
    const pathValue = typeof value === "string" ? value : value && value.filename;
    const normalizedPath = asString(pathValue).toLowerCase().replace(/\\/g, "/");
    if (!normalizedPath) continue;
    const parts = normalizedPath.split("/").filter(Boolean);
    if (parts.length >= 2 && parts[0] === "src") {
      packageSignals.add(`package:${parts[1]}`);
      packageSignals.add(`path:${parts.slice(0, Math.min(parts.length, 3)).join("/")}`);
    }
    if (parts.length >= 1) {
      packageSignals.add(`root:${parts[0]}`);
    }
    if (parts.length >= 2) {
      packageSignals.add(`path:${parts.slice(0, 2).join("/")}`);
    }
    if (/^requirements.*\.txt$/.test(normalizedPath) || normalizedPath === "pyproject.toml" || normalizedPath === "setup.py" || normalizedPath === "setup.cfg" || normalizedPath === "poetry.lock") {
      packageSignals.add("manifest:python");
    }
    if (normalizedPath === "package.json" || normalizedPath.endsWith("/package.json") || normalizedPath === "pnpm-lock.yaml" || normalizedPath === "yarn.lock") {
      packageSignals.add("manifest:node");
    }
  }
  return [...packageSignals].sort((left, right) => left.localeCompare(right));
}

function buildFileSurfaceSignals(changedFiles, text) {
  const signals = new Set(extractPathSignalsFromText(text));
  for (const file of normalizeChangedFiles(changedFiles)) {
    const normalizedPath = file.filename.toLowerCase().replace(/\\/g, "/");
    const parts = normalizedPath.split("/").filter(Boolean);
    if (parts.length >= 2) signals.add(parts.slice(0, 2).join("/"));
    if (parts.length >= 3) signals.add(parts.slice(0, 3).join("/"));
  }
  return [...signals].sort((left, right) => left.localeCompare(right));
}

function inferEcosystemSignals({ body, changedFiles, labels, title }) {
  const text = `${asString(title)}\n${asString(body)}`.toLowerCase();
  const pathText = normalizeChangedFiles(changedFiles).map((file) => file.filename.toLowerCase()).join("\n");
  const labelText = normalizeLabels(labels).join("\n");
  const ecosystems = new Set();
  const combined = `${text}\n${pathText}\n${labelText}`;
  if (/pyproject\.toml|requirements[^\n]*\.txt|setup\.py|poetry\.lock|\bpython\b/.test(combined)) ecosystems.add("python");
  if (/package\.json|pnpm-lock\.yaml|yarn\.lock|\bnode\b|\bnpm\b/.test(combined)) ecosystems.add("node");
  if (/dockerfile|docker\/|compose\.ya?ml|\bdocker\b/.test(combined)) ecosystems.add("docker");
  if (/\.github\/workflows|actions\//.test(combined)) ecosystems.add("github-actions");
  return [...ecosystems].sort((left, right) => left.localeCompare(right));
}

function parseRepoBoundIds(text, repoFullName, regex) {
  const ids = [];
  const normalizedRepo = normalizeRepoFullName(repoFullName);
  for (const match of asString(text).matchAll(regex)) {
    const matchRepo = normalizeRepoFullName(match.groups && match.groups.repo);
    if (matchRepo && normalizedRepo && matchRepo !== normalizedRepo) continue;
    ids.push(match.groups && match.groups.id);
  }
  return uniqueNumbers(ids);
}

function normalizeDirectiveBucket(directive) {
  const value = asString(directive).toLowerCase();
  if (value.startsWith("close") || value.startsWith("fix") || value.startsWith("resolve")) return "closeIds";
  if (value.startsWith("implement")) return "implementIds";
  if (value.startsWith("depend") || value.startsWith("blocked")) return "dependencyIds";
  if (value.startsWith("connect")) return "connectIds";
  if (value.startsWith("duplicate")) return "relatedIds";
  return "relatedIds";
}

function extractExplicitReferenceSignals({ body, branch, repoFullName, title }) {
  const closeIds = [];
  const connectIds = [];
  const dependencyIds = [];
  const implementIds = [];
  const relatedIds = [];
  const directiveText = `${asString(title)}\n${asString(body)}`;
  const normalizedRepo = normalizeRepoFullName(repoFullName);
  for (const match of directiveText.matchAll(DIRECTIVE_REGEX)) {
    const matchRepo = normalizeRepoFullName(match.groups && match.groups.repo);
    if (matchRepo && normalizedRepo && matchRepo !== normalizedRepo) continue;
    const bucket = normalizeDirectiveBucket(match.groups && match.groups.directive);
    const target = Number.parseInt(String(match.groups && match.groups.id), 10);
    if (!Number.isFinite(target) || target <= 0) continue;
    if (bucket === "closeIds") closeIds.push(target);
    if (bucket === "connectIds") connectIds.push(target);
    if (bucket === "dependencyIds") dependencyIds.push(target);
    if (bucket === "implementIds") implementIds.push(target);
    if (bucket === "relatedIds") relatedIds.push(target);
  }
  const branchIds = [];
  for (const match of asString(branch).matchAll(/(?:^|\/|[_-])(?<id>\d{1,7})(?=$|[_/-])/g)) {
    const target = Number.parseInt(String(match.groups && match.groups.id), 10);
    if (Number.isFinite(target) && target > 0) branchIds.push(target);
  }
  const genericIds = parseRepoBoundIds(`${directiveText}\n${branch}`, repoFullName, GENERIC_REF_REGEX);
  const urlIds = parseRepoBoundIds(directiveText, repoFullName, URL_REF_REGEX);
  return {
    branchIds: uniqueNumbers(branchIds),
    closeIds: uniqueNumbers(closeIds),
    connectIds: uniqueNumbers(connectIds),
    dependencyIds: uniqueNumbers(dependencyIds),
    genericIds,
    implementIds: uniqueNumbers(implementIds),
    relatedIds: uniqueNumbers(relatedIds),
    urlIds
  };
}

function normalizeRenderTarget(renderTarget) {
  if (!renderTarget || typeof renderTarget !== "object") return null;
  const kind = asString(renderTarget.kind).toLowerCase();
  const number = Number.parseInt(String(renderTarget.number || 0), 10);
  if (!Number.isFinite(number) || number <= 0) return null;
  if (!["issue", "pull_request"].includes(kind)) return null;
  return { kind, number };
}

function uniqueSortedSignals(values) {
  return [...new Set((Array.isArray(values) ? values : []).map((value) => asString(value).trim().toLowerCase()).filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function normalizeRemediationReferences(remediationReferences) {
  return uniqueNumbers((Array.isArray(remediationReferences) ? remediationReferences : []).map((value) => {
    if (typeof value === "number") return value;
    if (typeof value === "string") return Number.parseInt(value, 10);
    if (value && typeof value === "object") {
      return Number.parseInt(String(value.number || value.issueNumber || value.pullRequestNumber || 0), 10);
    }
    return 0;
  }));
}

function normalizeSmartLinkEntity(input) {
  const numericId = Number.parseInt(String(input.id || input.number || 0), 10);
  const changedFiles = normalizeChangedFiles(input.changedFiles);
  const title = normalizeWhitespace(input.title);
  const body = normalizeWhitespace(input.body);
  const fileSurfaceSignals = buildFileSurfaceSignals(changedFiles, body);
  const explicitReferences = extractExplicitReferenceSignals({
    body,
    branch: input.branch,
    repoFullName: input.repoFullName,
    title
  });
  const pathPackageSignals = buildPackageSignalsFromPaths(changedFiles.map((file) => file.filename).concat(fileSurfaceSignals));
  const providedAlertIdentifiers = uniqueStrings(Array.isArray(input.alertIdentifiers) ? input.alertIdentifiers.map((value) => String(value).toUpperCase()) : []);
  const providedPackageSignals = uniqueSortedSignals(input.packageSignals);
  const providedEcosystemSignals = uniqueSortedSignals(input.ecosystemSignals);
  const titleTokens = tokenizeText(title);
  const bodyTokens = tokenizeText(body);
  const combinedTokens = uniqueStrings(titleTokens.concat(bodyTokens));
  const labels = normalizeLabels(input.labels);
  const alertIdentifiers = uniqueStrings(providedAlertIdentifiers.concat(extractAlertIdentifiers(`${title}\n${body}`)).map((value) => value.toUpperCase()));
  const packageSignals = uniqueSortedSignals(pathPackageSignals.concat(providedPackageSignals));
  const ecosystemSignals = uniqueSortedSignals(providedEcosystemSignals.concat(inferEcosystemSignals({ body, changedFiles, labels, title })));
  return {
    alertIdentifiers,
    authorAssociation: asString(input.authorAssociation).toUpperCase() || "NONE",
    body,
    branch: normalizeWhitespace(input.branch),
    changedFiles,
    explicitReferences,
    ecosystemSignals,
    fileSurfaceSignals,
    htmlUrl: normalizeWhitespace(input.htmlUrl),
    id: Number.isFinite(numericId) && numericId > 0 ? numericId : normalizeWhitespace(input.id || input.alertId || input.alert_id),
    kind: asString(input.kind).toLowerCase(),
    labels,
    metadata: input.metadata && typeof input.metadata === "object" ? input.metadata : {},
    milestone: normalizeMilestone(input.milestone),
    number: Number.parseInt(String(input.number || 0), 10) || null,
    packageSignals,
    remediationReferences: normalizeRemediationReferences(input.remediationReferences),
    renderTarget: normalizeRenderTarget(input.renderTarget),
    repoFullName: normalizeRepoFullName(input.repoFullName),
    state: asString(input.state).toLowerCase() || "open",
    timestamps: {
      createdAt: normalizeWhitespace(input.createdAt),
      updatedAt: normalizeWhitespace(input.updatedAt)
    },
    title,
    tokens: {
      body: bodyTokens,
      combined: combinedTokens,
      title: titleTokens
    },
    urls: {
      html: normalizeWhitespace(input.htmlUrl)
    }
  };
}

function getEntityKey(entity) {
  const suffix = Number.isFinite(entity.number) && entity.number > 0 ? entity.number : entity.id || "unknown";
  return `${entity.kind}:${suffix}`;
}

function computeIntersection(left, right) {
  const rightSet = new Set(Array.isArray(right) ? right : []);
  return [...new Set((Array.isArray(left) ? left : []).filter((value) => rightSet.has(value)))].sort((a, b) => String(a).localeCompare(String(b)));
}

function pushEvidence(target, evidenceClass, label, weight, relationHint, values) {
  if (!(weight > 0)) return;
  target.push({
    evidenceClass,
    label,
    relationHint,
    values: Array.isArray(values) ? values : [],
    weight
  });
}

function computeCappedWeight(count, multiplier, cap) {
  if (!(count > 0)) return 0;
  return Math.min(cap, count * multiplier);
}

function targetCanBeClosed(candidate) {
  return candidate.kind === "issue";
}

function relationReasonFromResult(result) {
  const primaryLabels = result.evidenceItems
    .slice()
    .sort((left, right) => right.weight - left.weight || left.label.localeCompare(right.label))
    .slice(0, 4)
    .map((item) => item.label.replace(/-/g, " "));
  return primaryLabels.length > 0 ? primaryLabels.join(", ") : "no corroborating evidence";
}

function selectRequestedRelation(source, candidate, evidenceState) {
  if (targetCanBeClosed(candidate) && (evidenceState.explicitClose || evidenceState.explicitImplement)) return "closes";
  if (!targetCanBeClosed(candidate) && (evidenceState.explicitClose || evidenceState.explicitImplement)) return "connects";
  if (source.kind === "security_alert" && (evidenceState.sharedAlertIdentifier || evidenceState.remediationReference)) return "advisory_fix";
  if (evidenceState.dependencyReference) return "depends_on";
  if (evidenceState.explicitConnect) return "connects";
  if (evidenceState.explicitReference || evidenceState.reciprocalReference || evidenceState.strongStructural) return "connects";
  return "related";
}

function summarizeSuppressionReasons(result) {
  return uniqueStrings(result.suppressionReasons);
}

function analyzeCandidatePair({ candidate, source, threshold = DEFAULT_THRESHOLD }) {
  const evidenceItems = [];
  const suppressionReasons = [];
  const closeIds = new Set(source.explicitReferences.closeIds);
  const implementIds = new Set(source.explicitReferences.implementIds);
  const connectIds = new Set(source.explicitReferences.connectIds);
  const relatedIds = new Set(source.explicitReferences.relatedIds);
  const genericIds = new Set(source.explicitReferences.genericIds.concat(source.explicitReferences.urlIds));
  const branchIds = new Set(source.explicitReferences.branchIds);
  const dependencyIds = new Set(source.explicitReferences.dependencyIds);
  const candidateId = candidate.number;
  const evidenceState = {
    dependencyReference: dependencyIds.has(candidateId),
    explicitClose: closeIds.has(candidateId),
    explicitConnect: connectIds.has(candidateId),
    explicitImplement: implementIds.has(candidateId),
    explicitReference: genericIds.has(candidateId),
    reciprocalReference: false,
    remediationReference: source.remediationReferences.includes(candidateId),
    sharedAlertIdentifier: false,
    strongStructural: false
  };

  if (evidenceState.explicitClose) pushEvidence(evidenceItems, "explicit", "explicit-close", SCORE_WEIGHTS.explicitClose, "closes", [candidateId]);
  if (evidenceState.explicitImplement) pushEvidence(evidenceItems, "explicit", "explicit-implement", SCORE_WEIGHTS.explicitImplement, "closes", [candidateId]);
  if (evidenceState.explicitConnect) pushEvidence(evidenceItems, "explicit", "explicit-connect", SCORE_WEIGHTS.explicitConnect, "connects", [candidateId]);
  if (relatedIds.has(candidateId)) pushEvidence(evidenceItems, "explicit", "explicit-related", SCORE_WEIGHTS.explicitRelated, "related", [candidateId]);
  if (evidenceState.explicitReference) pushEvidence(evidenceItems, "explicit", "explicit-reference", SCORE_WEIGHTS.explicitReference, "connects", [candidateId]);
  if (branchIds.has(candidateId)) pushEvidence(evidenceItems, "explicit", "branch-reference", SCORE_WEIGHTS.branchReference, "connects", [candidateId]);
  if (evidenceState.dependencyReference) pushEvidence(evidenceItems, "explicit", "dependency-reference", SCORE_WEIGHTS.dependencyReference, "depends_on", [candidateId]);
  if (evidenceState.remediationReference) pushEvidence(evidenceItems, "explicit", "remediation-reference", SCORE_WEIGHTS.remediationReference, source.kind === "security_alert" ? "advisory_fix" : "connects", [candidateId]);

  if (Number.isFinite(source.number) && source.number > 0) {
    const candidateReferencesSource = candidate.explicitReferences.closeIds.includes(source.number)
      || candidate.explicitReferences.implementIds.includes(source.number)
      || candidate.explicitReferences.connectIds.includes(source.number)
      || candidate.explicitReferences.relatedIds.includes(source.number)
      || candidate.explicitReferences.genericIds.includes(source.number)
      || candidate.explicitReferences.urlIds.includes(source.number);
    if (candidate.explicitReferences.closeIds.includes(source.number) || candidate.explicitReferences.implementIds.includes(source.number)) {
      evidenceState.reciprocalReference = true;
      pushEvidence(evidenceItems, "reciprocal", "reciprocal-close", SCORE_WEIGHTS.reciprocalClose, "connects", [source.number]);
    } else if (candidateReferencesSource) {
      evidenceState.reciprocalReference = true;
      pushEvidence(evidenceItems, "reciprocal", "reciprocal-reference", SCORE_WEIGHTS.reciprocalReference, "connects", [source.number]);
    }
  }

  const sharedAlertIdentifiers = computeIntersection(source.alertIdentifiers, candidate.alertIdentifiers);
  if (sharedAlertIdentifiers.length > 0) {
    evidenceState.sharedAlertIdentifier = true;
    pushEvidence(evidenceItems, "explicit", "shared-alert-identifier", SCORE_WEIGHTS.sharedAlertIdentifier, source.kind === "security_alert" ? "advisory_fix" : "connects", sharedAlertIdentifiers);
  }

  const sharedMilestone = source.milestone && candidate.milestone && (
    (source.milestone.number && candidate.milestone.number && source.milestone.number === candidate.milestone.number)
    || (source.milestone.title && candidate.milestone.title && source.milestone.title.toLowerCase() === candidate.milestone.title.toLowerCase())
  );
  if (sharedMilestone) {
    evidenceState.strongStructural = true;
    pushEvidence(evidenceItems, "structural", "same-milestone", SCORE_WEIGHTS.sameMilestone, "connects", [source.milestone.title || String(source.milestone.number)]);
  }

  const sharedLabels = computeIntersection(
    source.labels.filter((label) => !LOW_SIGNAL_LABELS.has(label)),
    candidate.labels.filter((label) => !LOW_SIGNAL_LABELS.has(label))
  );
  const sharedLabelWeight = computeCappedWeight(sharedLabels.length, 0.85, SCORE_CAPS.sharedLabels);
  if (sharedLabelWeight > 0) {
    evidenceState.strongStructural = true;
    pushEvidence(evidenceItems, "structural", "shared-labels", sharedLabelWeight, "connects", sharedLabels);
  }

  const sharedPackages = computeIntersection(source.packageSignals, candidate.packageSignals);
  const sharedPackageWeight = computeCappedWeight(sharedPackages.length, 1.05, SCORE_CAPS.sharedPackage);
  if (sharedPackageWeight > 0) {
    evidenceState.strongStructural = true;
    pushEvidence(evidenceItems, "structural", "shared-package-signals", sharedPackageWeight, "connects", sharedPackages);
  }

  const sharedFileSurfaces = computeIntersection(source.fileSurfaceSignals, candidate.fileSurfaceSignals);
  const sharedFileSurfaceWeight = computeCappedWeight(sharedFileSurfaces.length, 1.0, SCORE_CAPS.sharedFileSurface);
  if (sharedFileSurfaceWeight > 0) {
    evidenceState.strongStructural = true;
    pushEvidence(evidenceItems, "structural", "shared-file-surfaces", sharedFileSurfaceWeight, "connects", sharedFileSurfaces);
  }

  const sharedEcosystems = computeIntersection(source.ecosystemSignals, candidate.ecosystemSignals);
  const sharedEcosystemWeight = computeCappedWeight(sharedEcosystems.length, 1.2, SCORE_CAPS.sharedEcosystem);
  if (sharedEcosystemWeight > 0) {
    evidenceState.strongStructural = true;
    pushEvidence(evidenceItems, "structural", "shared-ecosystems", sharedEcosystemWeight, "connects", sharedEcosystems);
  }

  const titleOverlap = computeIntersection(source.tokens.title, candidate.tokens.title);
  const titleOverlapWeight = computeCappedWeight(titleOverlap.length, 0.07, SCORE_CAPS.lexicalTitle);
  if (titleOverlapWeight > 0) {
    pushEvidence(evidenceItems, "lexical", "title-token-overlap", titleOverlapWeight, "related", titleOverlap);
  }

  const bodyOverlap = computeIntersection(source.tokens.combined, candidate.tokens.combined).filter((token) => !titleOverlap.includes(token));
  const bodyOverlapWeight = computeCappedWeight(bodyOverlap.length, 0.03, SCORE_CAPS.lexicalBody);
  if (bodyOverlapWeight > 0) {
    pushEvidence(evidenceItems, "lexical", "body-token-overlap", bodyOverlapWeight, "related", bodyOverlap.slice(0, 12));
  }

  const updatedAt = candidate.timestamps.updatedAt;
  if (updatedAt) {
    const ageDays = Math.max(0, (Date.now() - Date.parse(updatedAt)) / 86400000);
    if (ageDays <= 14) {
      pushEvidence(evidenceItems, "temporal", "recently-updated", SCORE_WEIGHTS.recencyRecent, "related", [updatedAt]);
    } else if (ageDays >= 365) {
      suppressionReasons.push("stale-target");
    }
  }

  const relationKind = selectRequestedRelation(source, candidate, evidenceState);
  let rawPenalty = 0;
  const closeFamilyRequested = targetCanBeClosed(candidate) && (evidenceState.explicitClose || evidenceState.explicitImplement);
  if (relationKind === "closes" && candidate.state !== "open") {
    rawPenalty += PENALTIES.closedTargetForClosure;
    suppressionReasons.push("close-target-not-open");
  }
  if (source.kind === "security_alert" && source.ecosystemSignals.length > 0 && candidate.ecosystemSignals.length > 0 && sharedEcosystems.length === 0) {
    rawPenalty += PENALTIES.ecosystemMismatch;
    suppressionReasons.push("ecosystem-mismatch");
  }
  if (source.kind === "security_alert" && source.packageSignals.length > 0 && candidate.packageSignals.length > 0 && sharedPackages.length === 0) {
    rawPenalty += PENALTIES.packageMismatch;
    suppressionReasons.push("package-mismatch");
  }
  if (source.kind === "security_alert" && (candidate.labels.includes("documentation") || candidate.labels.includes("docs"))) {
    rawPenalty += PENALTIES.docsSecurityConflict;
    suppressionReasons.push("docs-security-conflict");
  }
  if (candidate.timestamps.updatedAt) {
    const ageDays = Math.max(0, (Date.now() - Date.parse(candidate.timestamps.updatedAt)) / 86400000);
    if (ageDays >= 365) {
      rawPenalty += PENALTIES.staleTarget;
    }
  }

  const rawPositive = evidenceItems.reduce((total, item) => total + item.weight, 0);
  const rawScore = Math.max(0, rawPositive - rawPenalty);
  const normalizedScore = clampNumber(normalizeAccumulatedScore(rawScore), 0, 1);
  const emittedScore = clampNumber(Math.round(normalizedScore * 100), 1, 100);
  const hasExplicitEvidence = evidenceItems.some((item) => item.evidenceClass === "explicit");
  const hasReciprocalEvidence = evidenceItems.some((item) => item.evidenceClass === "reciprocal");
  const hasStructuralEvidence = evidenceItems.some((item) => item.evidenceClass === "structural");
  const hasLexicalEvidence = evidenceItems.some((item) => item.evidenceClass === "lexical");
  const lexicalOnly = hasLexicalEvidence && !hasExplicitEvidence && !hasReciprocalEvidence && !hasStructuralEvidence;
  if (lexicalOnly) suppressionReasons.push("lexical-only");
  if (relationKind === "closes" && !(evidenceState.explicitClose || evidenceState.explicitImplement)) suppressionReasons.push("close-without-close-family-signal");
  if (relationKind === "advisory_fix" && !(evidenceState.sharedAlertIdentifier || evidenceState.remediationReference)) suppressionReasons.push("advisory-without-alert-signal");
  if (relationKind === "connects" && !hasExplicitEvidence && !(hasReciprocalEvidence && hasStructuralEvidence) && !(evidenceState.sharedAlertIdentifier && (hasStructuralEvidence || hasReciprocalEvidence))) {
    suppressionReasons.push("connect-without-strong-support");
  }
  if (relationKind === "related" && !hasStructuralEvidence && !hasReciprocalEvidence && !evidenceState.explicitReference) {
    suppressionReasons.push("related-without-structural-support");
  }
  if (emittedScore < threshold) suppressionReasons.push("below-threshold");

  const summarizedSuppressionReasons = summarizeSuppressionReasons({ suppressionReasons });
  const primaryEvidenceClass = evidenceItems
    .slice()
    .sort((left, right) => {
      const classDelta = EVIDENCE_CLASS_PRIORITY[left.evidenceClass] - EVIDENCE_CLASS_PRIORITY[right.evidenceClass];
      if (classDelta !== 0) return classDelta;
      return right.weight - left.weight;
    })
    .map((item) => item.evidenceClass)[0] || "lexical";

  return {
    candidate,
    emittedScore,
    evidenceItems: evidenceItems.slice().sort((left, right) => right.weight - left.weight || left.label.localeCompare(right.label)),
    evidenceSummary: relationReasonFromResult({ evidenceItems }),
    lexicalOnly,
    normalizedScore,
    primaryEvidenceClass,
    rawPenalty,
    rawPositive,
    rawScore,
    relationKind,
    requestedCloseFamily: closeFamilyRequested,
    suppressionReasons: summarizedSuppressionReasons,
    thresholdPassed: emittedScore >= threshold && summarizedSuppressionReasons.length === 0,
    threshold
  };
}

function compareCandidateResults(left, right) {
  if (right.emittedScore !== left.emittedScore) return right.emittedScore - left.emittedScore;
  if (RELATION_PRIORITY[left.relationKind] !== RELATION_PRIORITY[right.relationKind]) {
    return RELATION_PRIORITY[left.relationKind] - RELATION_PRIORITY[right.relationKind];
  }
  if (EVIDENCE_CLASS_PRIORITY[left.primaryEvidenceClass] !== EVIDENCE_CLASS_PRIORITY[right.primaryEvidenceClass]) {
    return EVIDENCE_CLASS_PRIORITY[left.primaryEvidenceClass] - EVIDENCE_CLASS_PRIORITY[right.primaryEvidenceClass];
  }
  const leftUpdated = asString(left.candidate.timestamps.updatedAt);
  const rightUpdated = asString(right.candidate.timestamps.updatedAt);
  if (leftUpdated !== rightUpdated) return rightUpdated.localeCompare(leftUpdated);
  return (left.candidate.number || 0) - (right.candidate.number || 0);
}

function selectSmartLinkResults({ candidateResults, threshold = DEFAULT_THRESHOLD }) {
  const emitted = [];
  const relationCounts = {
    advisory_fix: 0,
    closes: 0,
    connects: 0,
    depends_on: 0,
    related: 0
  };
  for (const result of candidateResults.slice().sort(compareCandidateResults)) {
    if (!(result.emittedScore >= threshold)) continue;
    if (result.suppressionReasons.length > 0) continue;
    if (relationCounts[result.relationKind] >= EMISSION_LIMITS[result.relationKind]) continue;
    emitted.push(result);
    relationCounts[result.relationKind] += 1;
    if (emitted.length >= MAX_EMITTED_RESULTS) break;
  }
  return emitted;
}

function buildResultFingerprint(source, emittedResults) {
  const payload = JSON.stringify({
    source: getEntityKey(source),
    targets: emittedResults.map((result) => ({
      id: result.candidate.number || result.candidate.id,
      relationKind: result.relationKind,
      score: result.emittedScore
    }))
  });
  return crypto.createHash("sha256").update(payload).digest("hex").slice(0, 20);
}

function escapeMermaidLabel(value) {
  return asString(value).replace(/"/g, "&quot;");
}

function buildSourceNodeDescriptor(source) {
  if (source.kind === "pull_request") {
    return {
      id: `PR${source.number}`,
      label: `PR #${source.number}<br/>${escapeMermaidLabel(source.title)}`,
      style: "fill:#9f7,stroke:#333,stroke-width:2px"
    };
  }
  if (source.kind === "issue") {
    return {
      id: `ISSUE${source.number}`,
      label: `Issue #${source.number}<br/>${escapeMermaidLabel(source.title)}`,
      style: "fill:#ffd966,stroke:#333,stroke-width:2px"
    };
  }
  const suffix = source.number || source.id || "ALERT";
  return {
    id: `ALERT${suffix}`,
    label: `Alert ${escapeMermaidLabel(source.title || String(suffix))}`,
    style: "fill:#ff9c9c,stroke:#333,stroke-width:2px"
  };
}

function buildTargetNodeDescriptor(candidate) {
  const prefix = candidate.kind === "pull_request" ? "PR" : "ISSUE";
  const id = `${prefix}${candidate.number}`;
  let style = candidate.kind === "pull_request"
    ? "fill:#9cf,stroke:#333,stroke-width:2px"
    : "fill:#f9f,stroke:#333,stroke-width:2px";
  if (candidate.kind === "pull_request" && candidate.state === "closed") style = "fill:#c9f,stroke:#333,stroke-width:2px";
  if (candidate.kind === "issue" && candidate.state !== "open") style = "fill:#ccc,stroke:#333,stroke-width:2px";
  return {
    id,
    label: `${candidate.kind === "pull_request" ? "PR" : "Issue"} #${candidate.number}<br/>${escapeMermaidLabel(candidate.title)}`,
    style
  };
}

function buildMermaidGraphLines(source, emittedResults) {
  if (!Array.isArray(emittedResults) || emittedResults.length === 0) return [];
  const lines = ["graph TD"];
  const sourceNode = buildSourceNodeDescriptor(source);
  lines.push(`    ${sourceNode.id}[\"${sourceNode.label}\"]`);
  lines.push(`    style ${sourceNode.id} ${sourceNode.style}`);
  const relationLabels = {
    advisory_fix: "Advisory Fix",
    closes: "Closes",
    connects: "Connects",
    depends_on: "Depends On",
    related: "Related"
  };
  const relationArrows = {
    advisory_fix: "-.->",
    closes: "-->",
    connects: "-.->",
    depends_on: "-.->",
    related: "-.->"
  };
  const seenNodes = new Set([sourceNode.id]);
  for (const result of emittedResults) {
    const targetNode = buildTargetNodeDescriptor(result.candidate);
    if (!seenNodes.has(targetNode.id)) {
      seenNodes.add(targetNode.id);
      lines.push(`    ${targetNode.id}[\"${targetNode.label}\"]`);
      lines.push(`    style ${targetNode.id} ${targetNode.style}`);
    }
    lines.push(`    ${sourceNode.id} ${relationArrows[result.relationKind]}|${relationLabels[result.relationKind]}| ${targetNode.id}`);
  }
  return lines;
}

function buildDirectiveSummary(emittedResults) {
  const advisoryFixIds = [];
  const closeIds = [];
  const connectIds = [];
  const dependencyIds = [];
  const relatedIds = [];
  for (const result of emittedResults) {
    const candidateNumber = result.candidate.number;
    if (!candidateNumber) continue;
    if (result.relationKind === "closes" && result.candidate.kind === "issue") closeIds.push(candidateNumber);
    if (result.relationKind === "advisory_fix") advisoryFixIds.push(candidateNumber);
    if (result.relationKind === "connects") connectIds.push(candidateNumber);
    if (result.relationKind === "depends_on") dependencyIds.push(candidateNumber);
    if (result.relationKind === "related") relatedIds.push(candidateNumber);
  }
  return {
    advisoryFixIds: uniqueNumbers(advisoryFixIds),
    closeIds: uniqueNumbers(closeIds),
    connectIds: uniqueNumbers(connectIds),
    dependencyIds: uniqueNumbers(dependencyIds),
    relatedIds: uniqueNumbers(relatedIds)
  };
}

function buildResultTableRows(emittedResults) {
  return emittedResults.map((result) => {
    const targetKind = result.candidate.kind === "pull_request" ? "PR" : "Issue";
    return `| #${result.candidate.number} | ${targetKind} | ${result.candidate.state} | ${result.emittedScore} | ${result.relationKind.replace(/_/g, " ")} | ${result.evidenceSummary} |`;
  });
}

module.exports = {
  CHECK_NAME,
  DEFAULT_THRESHOLD,
  DEFAULT_MAX_EMITTED_RESULTS: MAX_EMITTED_RESULTS,
  GRAPH_COMMENT_MARKER_PREFIX,
  MANAGED_BODY_END,
  MANAGED_BODY_START,
  MANAGED_COMMENT_MARKER_PREFIX,
  MAX_TOTAL_CANDIDATES,
  TRUSTED_ASSOCIATIONS,
  analyzeCandidatePair,
  buildDirectiveSummary,
  buildFileSurfaceSignals,
  buildMermaidGraphLines,
  buildPackageSignalsFromPaths,
  buildResultFingerprint,
  buildResultTableRows,
  computeIntersection,
  extractAlertIdentifiers,
  extractExplicitReferenceSignals,
  extractPathSignalsFromText,
  getEntityKey,
  normalizeRenderTarget,
  normalizeSmartLinkEntity,
  selectSmartLinkResults,
  tokenizeText
};
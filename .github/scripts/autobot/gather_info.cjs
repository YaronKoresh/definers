const {
  MAX_PATCH_CHARS_PER_FILE,
  SNAPSHOT_FILE
} = require("./constants.cjs");
const { readJson, writeJson } = require("./utils.cjs");
const {
  DEFAULT_EMISSION_THRESHOLD,
  MAX_TOTAL_CANDIDATES,
  TRUSTED_ASSOCIATIONS,
  normalizeSmartLinkEntity
} = require("./smart_link/core.cjs");

const SMART_LINK_ALLOWED_DISPATCH_SOURCE_KINDS = new Set(["security_alert"]);

function mergeLabels(primaryLabels, additionalLabels) {
  const left = Array.isArray(primaryLabels) ? primaryLabels : [];
  const right = Array.isArray(additionalLabels) ? additionalLabels : [];
  return left.concat(right);
}

function clampThreshold(value) {
  const parsed = Number.parseInt(String(value || ""), 10);
  if (!Number.isFinite(parsed)) return DEFAULT_EMISSION_THRESHOLD;
  return Math.max(1, Math.min(100, parsed));
}

function normalizeDispatchSourcePayload(payload) {
  if (!payload || typeof payload !== "object") return null;
  const source = payload.source && typeof payload.source === "object" ? payload.source : payload;
  return source;
}

function normalizeRenderTargetForAlert(payload) {
  if (!payload || typeof payload !== "object") return null;
  if (payload.renderTarget && typeof payload.renderTarget === "object") return payload.renderTarget;
  if (Number.isFinite(Number.parseInt(String(payload.issueNumber || 0), 10))) {
    return { kind: "issue", number: Number.parseInt(String(payload.issueNumber), 10) };
  }
  if (Number.isFinite(Number.parseInt(String(payload.pullRequestNumber || 0), 10))) {
    return { kind: "pull_request", number: Number.parseInt(String(payload.pullRequestNumber), 10) };
  }
  return null;
}

function resolveSnapshotFile(input) {
  const explicit = typeof input === "string"
    ? input
    : input && typeof input === "object"
      ? input.snapshotFile
      : "";
  return String(explicit || process.env.AUTOBOT_PR_SNAPSHOT_FILE || SNAPSHOT_FILE).trim() || SNAPSHOT_FILE;
}

function normalizePatch(patch) {
  const text = String(patch || "");
  if (!text) {
    return "[patch unavailable: binary, generated, or too large for the API]";
  }
  if (text.length <= MAX_PATCH_CHARS_PER_FILE) {
    return text;
  }
  return text.substring(0, MAX_PATCH_CHARS_PER_FILE) + "\n[...patch truncated for prompt budget...]";
}

async function collectPullRequestSnapshot({ github, owner, repo, pullRequest, snapshotFile }) {
  const files = await github.paginate(github.rest.pulls.listFiles, { owner, repo, pull_number: pullRequest.number, per_page: 100 });
  const totalAdditions = files.reduce((sum, file) => sum + file.additions, 0);
  const totalDeletions = files.reduce((sum, file) => sum + file.deletions, 0);
  const snapshot = {
    pullRequest: {
      number: pullRequest.number,
      title: String(pullRequest.title || ""),
      body: String(pullRequest.body || ""),
      headRef: String(pullRequest.head?.ref || "")
    },
    totals: {
      filesChanged: files.length,
      additions: totalAdditions,
      deletions: totalDeletions,
      totalChanges: totalAdditions + totalDeletions
    },
    files: files.map((file) => ({
      filename: file.filename,
      status: file.status,
      additions: file.additions,
      deletions: file.deletions,
      patch: normalizePatch(file.patch),
      rawPatchAvailable: Boolean(file.patch)
    }))
  };
  writeJson(resolveSnapshotFile({ snapshotFile }), snapshot);
  return snapshot;
}

async function collectPullRequestSource({ additionalLabels, github, owner, repo, pullRequest }) {
  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner,
    per_page: 100,
    pull_number: pullRequest.number,
    repo
  });
  return normalizeSmartLinkEntity({
    authorAssociation: pullRequest.author_association,
    body: pullRequest.body,
    branch: pullRequest.head && pullRequest.head.ref,
    changedFiles: files.map((file) => ({ filename: file.filename, status: file.status })),
    htmlUrl: pullRequest.html_url,
    id: pullRequest.id,
    kind: "pull_request",
    labels: mergeLabels(pullRequest.labels, additionalLabels),
    metadata: {
      headSha: pullRequest.head && pullRequest.head.sha
    },
    milestone: pullRequest.milestone,
    number: pullRequest.number,
    repoFullName: `${owner}/${repo}`,
    state: pullRequest.state,
    title: pullRequest.title,
    updatedAt: pullRequest.updated_at
  });
}

function collectIssueSource({ additionalLabels, issue, owner, repo }) {
  return normalizeSmartLinkEntity({
    authorAssociation: issue.author_association,
    body: issue.body,
    htmlUrl: issue.html_url,
    id: issue.id,
    kind: "issue",
    labels: mergeLabels(issue.labels, additionalLabels),
    milestone: issue.milestone,
    number: issue.number,
    repoFullName: `${owner}/${repo}`,
    state: issue.state,
    title: issue.title,
    updatedAt: issue.updated_at
  });
}

function collectAlertSource({ owner, payload, repo }) {
  const source = normalizeDispatchSourcePayload(payload);
  if (!source) return null;
  return normalizeSmartLinkEntity({
    alertIdentifiers: source.alertIdentifiers || source.alert_identifiers || source.identifiers,
    body: source.body || source.summary || source.description,
    ecosystemSignals: source.ecosystemSignals || source.ecosystem_signals,
    htmlUrl: source.htmlUrl || source.html_url,
    id: source.id || source.alertId || source.alert_id,
    kind: source.kind || source.sourceKind || "security_alert",
    labels: source.labels,
    milestone: source.milestone,
    number: source.number,
    packageSignals: source.packageSignals || source.package_signals || source.affectedPackages || source.affected_packages,
    remediationReferences: source.remediationReferences || source.remediation_references || source.relatedIssueNumbers || source.related_issue_numbers || source.relatedPullRequestNumbers || source.related_pull_request_numbers,
    renderTarget: normalizeRenderTargetForAlert(source),
    repoFullName: `${owner}/${repo}`,
    state: source.state || "open",
    title: source.title || source.alertTitle || source.alert_title || source.summary,
    updatedAt: source.updatedAt || source.updated_at
  });
}

function buildSearchTerms(source) {
  if (source.alertIdentifiers.length > 0) {
    return source.alertIdentifiers.slice(0, 4);
  }
  return source.tokens.title.slice(0, 6).filter((token) => token.length >= 4);
}

function scoreSeedCandidate(priority, updatedAt) {
  return { priority, updatedAt: String(updatedAt || "") };
}

function compareSeedCandidates(left, right) {
  if (left.priority !== right.priority) return left.priority - right.priority;
  if (left.updatedAt !== right.updatedAt) return right.updatedAt.localeCompare(left.updatedAt);
  return left.id - right.id;
}

function getSeedPriority(provenance) {
  const order = {
    explicit: 0,
    remediation: 1,
    alert_identifier: 2,
    reciprocal: 3,
    milestone: 4,
    search: 5,
    labels: 6,
    recent: 7
  };
  return order[provenance] ?? 8;
}

function addCandidateSeed(candidateMap, item, provenance, sourceNumber) {
  const id = Number.parseInt(String(item && item.number), 10);
  if (!Number.isFinite(id) || id <= 0 || id === sourceNumber) return;
  const entry = candidateMap.get(id) || {
    id,
    priority: 999,
    provenances: new Set(),
    updatedAt: ""
  };
  entry.priority = Math.min(entry.priority, getSeedPriority(provenance));
  entry.provenances.add(provenance);
  entry.updatedAt = String(item && item.updated_at || item && item.updatedAt || entry.updatedAt || "");
  candidateMap.set(id, entry);
}

async function collectSearchResults({ github, owner, query, repo }) {
  if (!query) return [];
  try {
    const { data } = await github.rest.search.issuesAndPullRequests({
      per_page: 20,
      q: `repo:${owner}/${repo} ${query}`,
      sort: "updated"
    });
    return Array.isArray(data.items) ? data.items : [];
  } catch (error) {
    return [];
  }
}

async function collectMilestoneResults({ github, owner, repo, source }) {
  if (!source.milestone || !source.milestone.number) return [];
  try {
    const { data } = await github.rest.issues.listForRepo({
      direction: "desc",
      milestone: String(source.milestone.number),
      owner,
      per_page: 50,
      repo,
      sort: "updated",
      state: "all"
    });
    return Array.isArray(data) ? data : [];
  } catch (error) {
    return [];
  }
}

async function collectRecentResults({ github, owner, repo }) {
  try {
    const { data } = await github.rest.issues.listForRepo({
      direction: "desc",
      owner,
      per_page: 60,
      repo,
      sort: "updated",
      state: "all"
    });
    return Array.isArray(data) ? data : [];
  } catch (error) {
    return [];
  }
}

async function hydrateCandidate({ github, id, owner, repo }) {
  try {
    const { data } = await github.rest.issues.get({ issue_number: id, owner, repo });
    const isPullRequest = Boolean(data.pull_request);
    return normalizeSmartLinkEntity({
      body: data.body,
      htmlUrl: data.html_url,
      id: data.id,
      kind: isPullRequest ? "pull_request" : "issue",
      labels: data.labels,
      milestone: data.milestone,
      number: data.number,
      repoFullName: `${owner}/${repo}`,
      state: data.state,
      title: data.title,
      updatedAt: data.updated_at
    });
  } catch (error) {
    return null;
  }
}

async function retrieveCandidateSnapshots({ github, owner, repo, source }) {
  const candidateMap = new Map();
  const explicitIds = new Set([
    ...source.explicitReferences.closeIds,
    ...source.explicitReferences.connectIds,
    ...source.explicitReferences.dependencyIds,
    ...source.explicitReferences.genericIds,
    ...source.explicitReferences.implementIds,
    ...source.explicitReferences.relatedIds,
    ...source.explicitReferences.urlIds,
    ...source.remediationReferences
  ]);
  for (const id of explicitIds) {
    addCandidateSeed(candidateMap, { number: id }, source.remediationReferences.includes(id) ? "remediation" : "explicit", source.number);
  }

  const searchTerms = buildSearchTerms(source);
  const searchQueries = [];
  if (searchTerms.length > 0) searchQueries.push(searchTerms.join(" "));
  for (const identifier of source.alertIdentifiers.slice(0, 3)) {
    searchQueries.push(`\"${identifier}\"`);
  }
  for (const label of source.labels.slice(0, 2)) {
    searchQueries.push(`label:\"${label}\"`);
  }
  const [recentResults, milestoneResults, ...searchResults] = await Promise.all([
    collectRecentResults({ github, owner, repo }),
    collectMilestoneResults({ github, owner, repo, source }),
    ...searchQueries.map((query) => collectSearchResults({ github, owner, query, repo }))
  ]);
  for (const item of recentResults) addCandidateSeed(candidateMap, item, "recent", source.number);
  for (const item of milestoneResults) addCandidateSeed(candidateMap, item, "milestone", source.number);
  for (let index = 0; index < searchResults.length; index += 1) {
    const query = searchQueries[index] || "";
    const provenance = source.alertIdentifiers.some((identifier) => query.includes(identifier)) ? "alert_identifier" : query.startsWith("label:") ? "labels" : "search";
    for (const item of searchResults[index]) addCandidateSeed(candidateMap, item, provenance, source.number);
  }

  const selectedCandidates = [...candidateMap.values()]
    .map((entry) => ({ ...entry, score: scoreSeedCandidate(entry.priority, entry.updatedAt) }))
    .sort((left, right) => compareSeedCandidates(left, right))
    .slice(0, MAX_TOTAL_CANDIDATES);

  const hydrated = await Promise.all(selectedCandidates.map((entry) => hydrateCandidate({ github, id: entry.id, owner, repo })));
  return hydrated.filter(Boolean);
}

async function collectSmartLinkSource({ additionalLabels, context, core, github, owner, repo, sourceFile, thresholdInput }) {
  const threshold = clampThreshold(thresholdInput);
  let source = null;
  const eventName = context.eventName;
  if (eventName === "pull_request") {
    const pullRequest = context.payload.pull_request;
    const isForkSource = pullRequest.head && pullRequest.head.repo && pullRequest.head.repo.full_name !== `${owner}/${repo}`;
    if (isForkSource && !TRUSTED_ASSOCIATIONS.has(String(pullRequest.author_association || "").toUpperCase())) {
      core.setOutput("ready", "false");
      core.setOutput("reason", "untrusted-fork");
      return { ready: "false", reason: "untrusted-fork", threshold: String(threshold) };
    }
    source = await collectPullRequestSource({ additionalLabels, github, owner, pullRequest, repo });
    source.renderTarget = { kind: "pull_request", number: source.number };
  }
  if (eventName === "issues") {
    source = collectIssueSource({ additionalLabels, issue: context.payload.issue, owner, repo });
    source.renderTarget = { kind: "issue", number: source.number };
  }
  if (eventName === "repository_dispatch") {
    source = collectAlertSource({ owner, payload: context.payload.client_payload, repo });
  }
  if (eventName === "workflow_dispatch") {
    const payloadText = String(context.payload.inputs && context.payload.inputs.payload || "").trim();
    if (!payloadText) {
      core.setOutput("ready", "false");
      core.setOutput("reason", "missing-payload");
      return { ready: "false", reason: "missing-payload", threshold: String(threshold) };
    }
    let payload = null;
    try {
      payload = JSON.parse(payloadText);
    } catch (error) {
      core.setFailed(`Invalid workflow_dispatch payload JSON: ${error.message}`);
      return { ready: "false", reason: "invalid-payload", threshold: String(threshold) };
    }
    source = collectAlertSource({ owner, payload, repo });
    if (!source) {
      core.setOutput("ready", "false");
      core.setOutput("reason", "invalid-payload");
      return { ready: "false", reason: "invalid-payload", threshold: String(threshold) };
    }
  }
  if (!source) {
    core.setOutput("ready", "false");
    core.setOutput("reason", "unsupported-event");
    return { ready: "false", reason: "unsupported-event", threshold: String(threshold) };
  }
  if (eventName === "repository_dispatch" || eventName === "workflow_dispatch") {
    const normalizedKind = String(source.kind || "").trim().toLowerCase();
    if (!SMART_LINK_ALLOWED_DISPATCH_SOURCE_KINDS.has(normalizedKind)) {
      core.setOutput("ready", "false");
      core.setOutput("reason", "unsupported-dispatch-source");
      return { ready: "false", reason: "unsupported-dispatch-source", threshold: String(threshold) };
    }
  }
  if (!source.renderTarget && source.kind === "security_alert") {
    const remediationReference = source.remediationReferences[0];
    if (Number.isFinite(remediationReference) && remediationReference > 0) {
      source.renderTarget = { kind: "issue", number: remediationReference };
    }
  }
  writeJson(sourceFile, { source, threshold });
  const outputs = {
    ready: "true",
    source_kind: source.kind,
    threshold: String(threshold)
  };
  for (const [name, value] of Object.entries(outputs)) {
    core.setOutput(name, value);
  }
  return outputs;
}

function readStoredSource(sourceFile) {
  return readJson(sourceFile);
}

module.exports = {
  clampThreshold,
  collectAlertSource,
  collectIssueSource,
  collectMilestoneResults,
  collectPullRequestSnapshot,
  collectPullRequestSource,
  collectRecentResults,
  collectSearchResults,
  collectSmartLinkSource,
  mergeLabels,
  normalizeDispatchSourcePayload,
  normalizePatch,
  normalizeRenderTargetForAlert,
  readStoredSource,
  resolveSnapshotFile,
  retrieveCandidateSnapshots
};

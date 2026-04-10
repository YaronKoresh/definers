const fs = require("fs");

const { analyzeIssueIntake } = require("./autobot_issue_intake");

const {
  FORCE_RELEASE_TYPES,
  LABEL_DEFINITIONS,
  RELEASE_RELEVANT_LABELS,
  SECONDARY_LABELS,
  VERSION_BUMP_BY_LABEL,
  VERSION_LABEL_ALIASES,
  VERSION_SENSITIVE_LABELS
} = require("./autobot_labels");

const MIN_RELEASE_SIZE = 3;
const MAX_AUTOBOT_LABELS = 12;
const BUMP_ORDER = { none: 0, patch: 1, minor: 2, major: 3 };
const BOT_COMMENT_SIGNATURE = "<!-- autobot-summary -->";
const LEGACY_BOT_COMMENT_SIGNATURE = "<!-- autobot-ai-summary -->";
const MILESTONE_COMMENT_SIGNATURE = "<!-- autobot-milestone-update -->";
const VALID_LABELS = new Set(Object.keys(LABEL_DEFINITIONS));

function readState(stateFile) {
  return JSON.parse(fs.readFileSync(stateFile, "utf8"));
}

function writeState(stateFile, value) {
  fs.writeFileSync(stateFile, JSON.stringify(value), "utf8");
}

function normalizeLabelName(label) {
  const normalized = String(label || "").trim().toLowerCase();
  return VERSION_LABEL_ALIASES[normalized] || normalized;
}

function hasLabelName(labels, expectedLabel) {
  return (labels || []).some((label) => {
    const labelName = typeof label === "string" ? label : label?.name;
    return normalizeLabelName(labelName) === expectedLabel;
  });
}

function hasReleaseRelevantLabel(labels) {
  return RELEASE_RELEVANT_LABELS.some((label) => hasLabelName(labels, label));
}

function uniqueValidLabels(labels) {
  return [...new Set((labels || []).map((label) => normalizeLabelName(label)).filter((label) => VALID_LABELS.has(label)))];
}

function trimLowSignalLabels(labels) {
  if (labels.length <= 3) return labels;
  const uniqueLabels = uniqueValidLabels(labels);
  const versionCritical = VERSION_SENSITIVE_LABELS.filter((label) => uniqueLabels.includes(label));
  const primary = uniqueLabels.filter((label) => !SECONDARY_LABELS.includes(label) && !versionCritical.includes(label));
  const secondary = uniqueLabels.filter((label) => SECONDARY_LABELS.includes(label));
  const cappedPrimary = [...versionCritical, ...primary].slice(0, MAX_AUTOBOT_LABELS);
  const remainingSlots = Math.max(MAX_AUTOBOT_LABELS - cappedPrimary.length, 0);
  const cappedSecondary = secondary.slice(0, remainingSlots);
  return [...cappedPrimary, ...cappedSecondary].slice(0, MAX_AUTOBOT_LABELS);
}

function parseVersionTag(rawVersion) {
  const match = String(rawVersion || "").trim().match(/^v?(\d+)\.(\d+)\.(\d+)$/);
  if (!match) return { major: 0, minor: 0, patch: 1 };
  return { major: Number(match[1]), minor: Number(match[2]), patch: Number(match[3]) };
}

function isVersionTag(rawVersion) {
  return /^v?\d+\.\d+\.\d+$/.test(String(rawVersion || "").trim());
}

function formatVersionTag(version) {
  return `v${version.major}.${version.minor}.${version.patch}`;
}

function maxBump(current, next) {
  return BUMP_ORDER[next] > BUMP_ORDER[current] ? next : current;
}

function normalizeBumpType(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(BUMP_ORDER, normalized) ? normalized : "none";
}

function parseDeterministicSemver(raw) {
  if (!raw) return null;
  const source = typeof raw === "string" ? String(raw).trim() : raw;
  if (!source) return null;
  let parsed = source;
  if (typeof source === "string") {
    if (["major", "minor", "patch", "none"].includes(source.toLowerCase())) {
      parsed = { decision: source.toLowerCase() };
    } else {
      try {
        parsed = JSON.parse(source);
      } catch (error) {
        return null;
      }
    }
  }
  if (!parsed || typeof parsed !== "object") return null;
  const decision = normalizeBumpType(parsed.decision);
  if (decision === "none") return null;
  return {
    decision,
    hardRule: Boolean(parsed.hardRule),
    hardSignals: Array.isArray(parsed.hardSignals) ? parsed.hardSignals : [],
    majorScore: Number(parsed.majorScore) || 0,
    minorScore: Number(parsed.minorScore) || 0,
    patchScore: Number(parsed.patchScore) || 0
  };
}

function bumpForLabels(labels) {
  return labels.reduce((bump, label) => {
    const normalized = normalizeLabelName(label);
    const next = VERSION_BUMP_BY_LABEL[normalized] || "none";
    return maxBump(bump, next);
  }, "none");
}

function computeTargetVersion(baseVersion, bumpType) {
  const parsed = parseVersionTag(baseVersion);
  if (bumpType === "major") return formatVersionTag({ major: parsed.major + 1, minor: 0, patch: 0 });
  if (bumpType === "minor") return formatVersionTag({ major: parsed.major, minor: parsed.minor + 1, patch: 0 });
  if (bumpType === "patch") return formatVersionTag({ major: parsed.major, minor: parsed.minor, patch: parsed.patch + 1 });
  return formatVersionTag(parsed);
}

function compareVersions(left, right) {
  const a = parseVersionTag(left);
  const b = parseVersionTag(right);
  if (a.major !== b.major) return a.major > b.major ? 1 : -1;
  if (a.minor !== b.minor) return a.minor > b.minor ? 1 : -1;
  if (a.patch !== b.patch) return a.patch > b.patch ? 1 : -1;
  return 0;
}

function parseMilestoneMetadata(description) {
  const text = String(description || "");
  const baseMatch = text.match(/autobot-base:(v\d+\.\d+\.\d+)/i);
  const managedMatch = text.match(/autobot-managed-title:(v\d+\.\d+\.\d+)/i);
  return {
    baseVersion: baseMatch ? baseMatch[1] : null,
    managedTitle: managedMatch ? managedMatch[1] : null
  };
}

function buildMilestoneMetadataDescription(description, baseVersion, managedTitle) {
  const rows = String(description || "")
    .split(/\r?\n/)
    .filter((line) => !/^autobot-base:v\d+\.\d+\.\d+$/i.test(line.trim()) && !/^autobot-managed-title:v\d+\.\d+\.\d+$/i.test(line.trim()));
  if (baseVersion) rows.push(`autobot-base:${baseVersion}`);
  if (managedTitle) rows.push(`autobot-managed-title:${managedTitle}`);
  return rows.join("\n").trim();
}

function parseAutobotLabels(raw) {
  if (!raw) return [];
  try {
    const cleaned = String(raw).replace(/```json\s*/gi, "").replace(/```\s*/gi, "").trim();
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) {
      return [...new Set(parsed.map((label) => normalizeLabelName(label)).filter((label) => VALID_LABELS.has(label)))];
    }
  } catch (error) {
    const lines = String(raw).replace(/[\[\]"'`]/g, "").split(/[,\n]/).map((label) => normalizeLabelName(label)).filter((label) => VALID_LABELS.has(label));
    return [...new Set(lines)];
  }
  return [];
}

function resolvePrLabelDelta(input) {
  const { action, previousBotLabels, currentPrLabels, autobotLabelsRaw } = input;
  const previousLabels = uniqueValidLabels(previousBotLabels);
  const currentLabels = uniqueValidLabels(currentPrLabels);
  const rawAutobotLabels = String(autobotLabelsRaw ?? input.aiLabelsRaw ?? "").trim();
  const parsedAutobotLabels = trimLowSignalLabels(parseAutobotLabels(rawAutobotLabels)).slice(0, MAX_AUTOBOT_LABELS);
  const hasFreshPrLabelResult = rawAutobotLabels !== "";

  let nextAutobotLabels = [];
  if (hasFreshPrLabelResult) {
    nextAutobotLabels = parsedAutobotLabels;
  } else if (action === "synchronize" && previousLabels.length > 0) {
    nextAutobotLabels = previousLabels;
  }

  const labelsToRemove = hasFreshPrLabelResult
    ? previousLabels.filter((label) => currentLabels.includes(label) && !nextAutobotLabels.includes(label))
    : [];
  const labelsToAdd = nextAutobotLabels.filter((label) => !currentLabels.includes(label));

  return {
    currentPrLabels: currentLabels,
    hasFreshPrLabelResult,
    labelsToAdd,
    labelsToRemove,
    nextAutobotLabels,
    parsedAutobotLabels,
    previousBotLabels: previousLabels
  };
}

function labelNamesFromIssue(issue) {
  return uniqueValidLabels((issue?.labels || []).map((label) => typeof label === "string" ? label : label.name));
}

function inferIssueLabels(issue) {
  return analyzeIssueIntake(issue)
    .labels
    .map((label) => normalizeLabelName(label))
    .filter((label) => VALID_LABELS.has(label));
}

function issueFallbackSupportLabels(issue) {
  return inferIssueLabels(issue).filter((label) => ["improvement", "proposal"].includes(label));
}

function extractBotMetadata(body) {
  const match = String(body || "").match(/<!-- autobot-metadata:(.+?) -->/s);
  if (!match) return {};
  try {
    return JSON.parse(match[1].trim());
  } catch (error) {
    return {};
  }
}

function getManagedLabelsFromMetadata(metadata) {
  const storedLabels = Array.isArray(metadata?.autobotLabels)
    ? metadata.autobotLabels
    : Array.isArray(metadata?.aiLabels)
      ? metadata.aiLabels
      : [];
  return storedLabels.map((label) => normalizeLabelName(label));
}

function isManagedBotCommentBody(body) {
  const text = String(body || "");
  return text.includes(BOT_COMMENT_SIGNATURE) || text.includes(LEGACY_BOT_COMMENT_SIGNATURE);
}

function semverDecisionFromMetadata(metadata) {
  if (!metadata || typeof metadata !== "object") return "none";
  const directDecision = normalizeBumpType(metadata.semverDecision);
  if (directDecision !== "none") return directDecision;
  return normalizeBumpType(metadata.deterministicSemver?.decision);
}

async function getComments({ github, owner, repo, issueNumber }) {
  return github.paginate(github.rest.issues.listComments, { owner, repo, issue_number: issueNumber });
}

async function getExistingBotComment({ github, owner, repo, issueNumber }) {
  const comments = await getComments({ github, owner, repo, issueNumber });
  return comments.find((comment) => comment.user.type === "Bot" && isManagedBotCommentBody(comment.body));
}

async function getExistingBotCommentForIssue({ github, owner, repo, issueNumber }) {
  const comments = await getComments({ github, owner, repo, issueNumber });
  return comments.find((comment) => comment.user.type === "Bot" && isManagedBotCommentBody(comment.body));
}

async function getBotMetadataForIssue({ github, owner, repo, issueNumber, metadataCache }) {
  if (metadataCache?.has(issueNumber)) {
    return metadataCache.get(issueNumber);
  }
  const existingBotComment = await getExistingBotCommentForIssue({ github, owner, repo, issueNumber });
  const metadata = existingBotComment ? extractBotMetadata(existingBotComment.body) : {};
  if (metadataCache) {
    metadataCache.set(issueNumber, metadata);
  }
  return metadata;
}

async function getExistingMilestoneComment({ github, owner, repo, issueNumber }) {
  const comments = await getComments({ github, owner, repo, issueNumber });
  return comments.find((comment) => comment.user.type === "Bot" && comment.body.includes(MILESTONE_COMMENT_SIGNATURE));
}

async function getExistingMajorAlertComment({ github, owner, repo, issueNumber }) {
  const comments = await getComments({ github, owner, repo, issueNumber });
  return comments.find((comment) => comment.user.type === "Bot" && comment.body.includes("MAJOR RELEASE ALERT"));
}

async function upsertBotComment({ github, owner, repo, issueNumber, body, metadata }) {
  const existing = await getExistingBotComment({ github, owner, repo, issueNumber });
  const MAX_COMMENT_CHARS = 60000;
  let safeBody = body;
  if (safeBody.length > MAX_COMMENT_CHARS) {
    safeBody = safeBody.slice(0, MAX_COMMENT_CHARS - 200) + "\n\n---\n\n_(Autobot note: comment truncated to fit GitHub size limits.)_";
  }
  const fullBody = BOT_COMMENT_SIGNATURE + "\n" + `<!-- autobot-metadata:${JSON.stringify(metadata || {})} -->` + "\n" + safeBody;
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body: fullBody });
    return;
  }
  await github.rest.issues.createComment({ owner, repo, issue_number: issueNumber, body: fullBody });
}

async function upsertMilestoneComment({ github, owner, repo, issueNumber, body, metadata }) {
  const existing = await getExistingMilestoneComment({ github, owner, repo, issueNumber });
  const fullBody = MILESTONE_COMMENT_SIGNATURE + "\n" + `<!-- autobot-metadata:${JSON.stringify(metadata || {})} -->` + "\n" + body;
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body: fullBody });
    return;
  }
  await github.rest.issues.createComment({ owner, repo, issue_number: issueNumber, body: fullBody });
}

async function getLatestPublishedVersion({ github, owner, repo }) {
  try {
    const releases = await github.paginate(github.rest.repos.listReleases, { owner, repo, per_page: 100 });
    const published = releases
      .filter((release) => !release.draft && !release.prerelease)
      .map((release) => String(release.tag_name || "").trim())
      .filter((tag) => /^v?\d+\.\d+\.\d+$/.test(tag));
    if (published.length === 0) return null;
    return published.reduce((maxTag, currentTag) => compareVersions(currentTag, maxTag) > 0 ? currentTag : maxTag, published[0]);
  } catch (error) {
    return null;
  }
}

async function ensureMilestoneBaseVersion({ github, owner, repo, milestone }) {
  const description = String(milestone.description || "");
  const metadata = parseMilestoneMetadata(description);
  const latestPublished = await getLatestPublishedVersion({ github, owner, repo });
  const baseVersion = metadata.baseVersion
    ? metadata.baseVersion
    : (latestPublished ? formatVersionTag(parseVersionTag(latestPublished)) : formatVersionTag(parseVersionTag(milestone.title)));
  const managedTitle = metadata.managedTitle || formatVersionTag(parseVersionTag(milestone.title));
  const manualTitleLocked = Boolean(metadata.managedTitle && compareVersions(formatVersionTag(parseVersionTag(milestone.title)), metadata.managedTitle) !== 0);
  const nextDescription = buildMilestoneMetadataDescription(description, baseVersion, managedTitle);
  if (nextDescription === description.trim()) {
    return { baseVersion, managedTitle, manualTitleLocked, hadManagedMarker: Boolean(metadata.managedTitle), milestone };
  }
  const updated = await github.rest.issues.updateMilestone({ owner, repo, milestone_number: milestone.number, description: nextDescription });
  return { baseVersion, managedTitle, manualTitleLocked, hadManagedMarker: Boolean(metadata.managedTitle), milestone: updated.data };
}

async function ensureLabelExists({ github, owner, repo, name }) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name });
  } catch (error) {
    if (error.status === 404) {
      const definition = LABEL_DEFINITIONS[name] || { color: "ededed", description: "" };
      try {
        await github.rest.issues.createLabel({ owner, repo, name, color: definition.color, description: definition.description });
      } catch (createError) {
        if (createError.status !== 422) throw createError;
        await github.rest.issues.getLabel({ owner, repo, name });
      }
      return;
    }
    throw error;
  }
}

function isInvalidLabelNameError(error, name) {
  if (error?.status !== 422) return false;
  const expectedName = normalizeLabelName(name);
  const errors = Array.isArray(error?.response?.data?.errors) ? error.response.data.errors : [];
  return errors.some((entry) => {
    const resource = String(entry?.resource || "").toLowerCase();
    const field = String(entry?.field || "").toLowerCase();
    const code = String(entry?.code || "").toLowerCase();
    const value = normalizeLabelName(entry?.value || "");
    return resource === "label" && field === "name" && code === "invalid" && value === expectedName;
  });
}

async function addLabelToIssue({ github, owner, repo, issueNumber, label }) {
  await ensureLabelExists({ github, owner, repo, name: label });
  try {
    await github.rest.issues.addLabels({ owner, repo, issue_number: issueNumber, labels: [label] });
  } catch (error) {
    if (!isInvalidLabelNameError(error, label)) throw error;
    await ensureLabelExists({ github, owner, repo, name: label });
    await github.rest.issues.addLabels({ owner, repo, issue_number: issueNumber, labels: [label] });
  }
}

async function addLabelsToIssue({ github, owner, repo, issueNumber, labels }) {
  for (const label of uniqueValidLabels(labels)) {
    await addLabelToIssue({ github, owner, repo, issueNumber, label });
  }
}

async function getOrCreateMilestone({ github, owner, repo }) {
  const milestones = await github.rest.issues.listMilestones({ owner, repo, state: "open", sort: "due_on", direction: "asc" });
  const versionedMilestones = milestones.data.filter((milestone) => isVersionTag(milestone.title));
  if (versionedMilestones.length > 0) {
    return versionedMilestones.reduce((selected, current) => {
      if (!selected) return current;
      return compareVersions(current.title, selected.title) > 0 ? current : selected;
    }, null);
  }
  let nextVersion = "v0.0.1";
  try {
    const releases = await github.rest.repos.listReleases({ owner, repo });
    const latest = releases.data[0];
    if (latest) {
      const parts = latest.tag_name.replace("v", "").split(".").map(Number);
      parts[2] += 1;
      nextVersion = `v${parts.join(".")}`;
    }
  } catch (error) {
  }
  const created = await github.rest.issues.createMilestone({ owner, repo, title: nextVersion });
  return created.data;
}

async function getOrCreateMilestoneByTitle({ github, owner, repo, title }) {
  const milestones = await github.rest.issues.listMilestones({ owner, repo, state: "open", sort: "due_on", direction: "asc" });
  const existing = milestones.data.find((milestone) => String(milestone.title || "").trim() === title);
  if (existing) return existing;
  const created = await github.rest.issues.createMilestone({ owner, repo, title });
  return created.data;
}

async function seedMilestoneMetadata({ github, owner, repo, milestone, baseVersion, managedTitle }) {
  const nextDescription = buildMilestoneMetadataDescription(milestone.description || "", baseVersion, managedTitle);
  if (nextDescription === String(milestone.description || "").trim()) return milestone;
  const updated = await github.rest.issues.updateMilestone({ owner, repo, milestone_number: milestone.number, description: nextDescription });
  return updated.data;
}

async function listManagedSemverMilestones({ github, owner, repo, targetMilestone }) {
  if (!isVersionTag(targetMilestone.title)) return [targetMilestone];
  const milestones = await github.rest.issues.listMilestones({ owner, repo, state: "open", sort: "due_on", direction: "asc" });
  return milestones.data.filter((milestone) => isVersionTag(milestone.title) && compareVersions(milestone.title, targetMilestone.title) <= 0);
}

async function listReleaseItemsForMilestones({ github, owner, repo, milestones }) {
  const releaseItems = [];
  for (const milestone of milestones) {
    const milestoneItems = await github.paginate(github.rest.issues.listForRepo, { owner, repo, milestone: milestone.number, state: "all" });
    releaseItems.push(...milestoneItems.filter((item) => item.pull_request && hasReleaseRelevantLabel(item.labels)));
  }
  return releaseItems;
}

function getItemLabelNames(item) {
  return (item.labels || []).map((label) => normalizeLabelName(label.name));
}

async function resolveReleaseItemBump({ github, owner, repo, item, currentIssueNumber, currentSemverDecision, metadataCache }) {
  if (item?.number === currentIssueNumber) {
    const normalizedCurrentDecision = normalizeBumpType(currentSemverDecision);
    if (normalizedCurrentDecision !== "none") {
      return normalizedCurrentDecision;
    }
  }
  const metadata = await getBotMetadataForIssue({ github, owner, repo, issueNumber: item.number, metadataCache });
  const metadataDecision = semverDecisionFromMetadata(metadata);
  if (metadataDecision !== "none") {
    return metadataDecision;
  }
  return bumpForLabels(getItemLabelNames(item));
}

async function resolveRequiredBump({ github, owner, repo, releaseItems, currentLabelNames, currentIssueNumber, currentSemverDecision, metadataCache }) {
  let releaseBump = normalizeBumpType(currentSemverDecision);
  if (releaseBump === "none") {
    releaseBump = bumpForLabels(currentLabelNames);
  }
  for (const item of releaseItems) {
    const itemBump = await resolveReleaseItemBump({
      github,
      owner,
      repo,
      item,
      currentIssueNumber,
      currentSemverDecision,
      metadataCache
    });
    releaseBump = maxBump(releaseBump, itemBump);
  }
  return releaseBump === "none" ? "patch" : releaseBump;
}

function buildPrCommentBody({ summaryForComment, pullRequest, nextAutobotLabels }) {
  const appliedLabels = nextAutobotLabels.map((label) => `\`${label}\``).join(" ") || "_none_";
  return [
    "# Autobot — Changes Analysis",
    "",
    `> **PR #${pullRequest.number}** · ${pullRequest.head.ref} → ${pullRequest.base.ref} · ${new Date().toISOString().split("T")[0]}`,
    "",
    "---",
    "",
    summaryForComment,
    "",
    "---",
    "",
    "<details>",
    "<summary><strong>🏷️ Label Classification</strong></summary>",
    "",
    `**Applied labels:** ${appliedLabels}`,
    "",
    "Labels were determined from the code diff evidence collected by Autobot, not from the PR title or description alone.",
    "",
    "</details>"
  ].join("\n");
}

async function selectivelyConsolidateSupersededMilestones({ github, owner, repo, targetMilestone, targetBump, currentIssueNumber, metadataCache }) {
  if (!isVersionTag(targetMilestone.title)) {
    return { milestone: targetMilestone, consolidatedMilestones: [], migratedPullRequests: [], retainedMilestones: [] };
  }
  const milestones = await github.rest.issues.listMilestones({ owner, repo, state: "open", sort: "due_on", direction: "asc" });
  const supersededMilestones = milestones.data.filter((milestone) => {
    if (milestone.number === targetMilestone.number) return false;
    if (!isVersionTag(milestone.title)) return false;
    return compareVersions(milestone.title, targetMilestone.title) <= 0;
  });
  const consolidatedMilestones = [];
  const migratedPullRequests = [];
  const retainedMilestones = [];

  for (const milestone of supersededMilestones) {
    const milestoneItems = await github.paginate(github.rest.issues.listForRepo, { owner, repo, milestone: milestone.number, state: "all" });
    let movableOpenPrCount = 0;
    for (const item of milestoneItems) {
      if (!item.pull_request) continue;
      if (item.state !== "open") continue;
      if (item.number === currentIssueNumber) continue;
      const itemLabels = getItemLabelNames(item);
      if (!hasReleaseRelevantLabel(itemLabels)) continue;
      const itemBump = await resolveReleaseItemBump({
        github,
        owner,
        repo,
        item,
        currentIssueNumber,
        currentSemverDecision: "none",
        metadataCache
      });
      if (BUMP_ORDER[itemBump] > BUMP_ORDER[targetBump]) continue;
      const existingBotCommentForItem = await getExistingBotCommentForIssue({ github, owner, repo, issueNumber: item.number });
      if (!existingBotCommentForItem) continue;
      await github.rest.issues.update({ owner, repo, issue_number: item.number, milestone: targetMilestone.number });
      migratedPullRequests.push({ number: item.number, title: item.title, fromMilestone: milestone.title });
      movableOpenPrCount += 1;
    }

    const remainingItems = await github.paginate(github.rest.issues.listForRepo, { owner, repo, milestone: milestone.number, state: "all" });
    if (remainingItems.length === 0) {
      await github.rest.issues.updateMilestone({ owner, repo, milestone_number: milestone.number, state: "closed" });
      consolidatedMilestones.push(milestone.title);
    } else {
      retainedMilestones.push({ title: milestone.title, remainingCount: remainingItems.length, migratedPullRequestCount: movableOpenPrCount });
    }
  }

  const refreshed = await github.rest.issues.getMilestone({ owner, repo, milestone_number: targetMilestone.number });
  return { milestone: refreshed.data, consolidatedMilestones, migratedPullRequests, retainedMilestones };
}

function buildMilestoneComment({ milestoneChanged, previousMilestoneTitle, finalMilestoneTitle, consolidation }) {
  const noteLines = [
    "# Autobot — Milestone Update",
    "",
    milestoneChanged
      ? `This PR was moved ${previousMilestoneTitle ? `from **${previousMilestoneTitle}** ` : ""}to **${finalMilestoneTitle}** based on the current semantic-version impact of the PR and the canonical open release milestone.`
      : `Autobot kept this PR on **${finalMilestoneTitle}** and selectively consolidated compatible lower semantic-version PRs into it.`,
    ""
  ];
  if (consolidation.migratedPullRequests.length > 0) {
    noteLines.push(`Moved compatible PRs: ${consolidation.migratedPullRequests.map((item) => `#${item.number}`).join(", ")}`);
    noteLines.push("");
  }
  if (consolidation.consolidatedMilestones.length > 0) {
    noteLines.push(`Closed empty older milestones: ${consolidation.consolidatedMilestones.map((title) => `**${title}**`).join(", ")}`);
    noteLines.push("");
  }
  if (consolidation.retainedMilestones.length > 0) {
    noteLines.push(`Retained older milestones for manual review: ${consolidation.retainedMilestones.map((item) => `**${item.title}** (${item.remainingCount} remaining)`).join(", ")}`);
    noteLines.push("");
  }
  noteLines.push("This only moves compatible open Autobot-managed PRs. Issues, closed items, manual milestones, and incompatible PRs stay where they are.");
  return noteLines.join("\n");
}

async function prepareProjectState(input) {
  const {
    github,
    owner,
    repo,
    context,
    issueNumber,
    summaryText,
    autobotLabelsRaw,
    deterministicSemverRaw,
    stateFile
  } = input;
  const payload = context.payload;
  const isPR = context.eventName === "pull_request";
  const livePrIssue = isPR ? await github.rest.issues.get({ owner, repo, issue_number: issueNumber }) : null;
  const summaryForComment = String(summaryText ?? input.aiSummary ?? "").replace(/\nEND_OF_REPORT\s*$/m, "").trim();
  const payloadIssue = payload.issue || {};
  const currentPrLabels = isPR ? labelNamesFromIssue(livePrIssue?.data) : [];
  const existingIssueLabels = labelNamesFromIssue(payloadIssue);
  const rawAutobotLabels = String(autobotLabelsRaw ?? input.aiLabelsRaw ?? "").trim();
  const parsedAutobotLabels = trimLowSignalLabels(parseAutobotLabels(rawAutobotLabels)).slice(0, MAX_AUTOBOT_LABELS);
  const deterministicSemver = isPR ? parseDeterministicSemver(deterministicSemverRaw) : null;

  let issueLabelsToAdd = [];
  let previousBotLabels = [];
  let nextAutobotLabels = [];
  let labelsToRemove = [];
  let labelsToAdd = [];
  let commentBody = "";

  if (!isPR) {
    const fallbackSupportLabels = issueFallbackSupportLabels(payloadIssue);
    const inferredIssueLabels = parsedAutobotLabels.length > 0
      ? uniqueValidLabels([...existingIssueLabels, ...parsedAutobotLabels, ...fallbackSupportLabels])
      : uniqueValidLabels([...existingIssueLabels, ...inferIssueLabels(payloadIssue)]);
    issueLabelsToAdd = inferredIssueLabels.filter((label) => !existingIssueLabels.includes(label));
  }

  const existingBotComment = isPR ? await getExistingBotComment({ github, owner, repo, issueNumber }) : null;
  previousBotLabels = existingBotComment ? getManagedLabelsFromMetadata(extractBotMetadata(existingBotComment.body)) : [];
  if (isPR) {
    const prLabelDelta = resolvePrLabelDelta({
      action: payload.action,
      currentPrLabels,
      previousBotLabels,
      autobotLabelsRaw: rawAutobotLabels
    });
    previousBotLabels = prLabelDelta.previousBotLabels;
    nextAutobotLabels = prLabelDelta.nextAutobotLabels;
    labelsToRemove = prLabelDelta.labelsToRemove;
    labelsToAdd = prLabelDelta.labelsToAdd;
  }

  if (isPR && summaryForComment) {
    commentBody = buildPrCommentBody({
      summaryForComment,
      pullRequest: payload.pull_request,
      nextAutobotLabels
    });
  }

  const state = {
    action: payload.action,
    eventName: context.eventName,
    isPR,
    issueNumber,
    issueLabelsToAdd,
    labelsToRemove,
    labelsToAdd,
    nextAutobotLabels,
    deterministicSemver,
    semverDecision: deterministicSemver?.decision || bumpForLabels(nextAutobotLabels),
    commentBody
  };
  writeState(stateFile, state);
  return state;
}

async function syncPreparedProjectState({ github, owner, repo, issueNumber, stateFile }) {
  const state = readState(stateFile);
  if (!["opened", "synchronize", "reopened"].includes(state.action)) {
    return state;
  }

  if (!state.isPR && state.issueLabelsToAdd.length > 0) {
    await addLabelsToIssue({ github, owner, repo, issueNumber, labels: state.issueLabelsToAdd });
  }

  if (state.isPR) {
    for (const label of state.labelsToRemove) {
      try {
        await github.rest.issues.removeLabel({ owner, repo, issue_number: issueNumber, name: label });
      } catch (error) {
        if (error.status !== 404) throw error;
      }
    }
    if (state.labelsToAdd.length > 0) {
      await addLabelsToIssue({ github, owner, repo, issueNumber, labels: state.labelsToAdd });
    }
    if (state.commentBody) {
      await upsertBotComment({
        github,
        owner,
        repo,
        issueNumber,
        body: state.commentBody,
        metadata: {
          autobotLabels: state.nextAutobotLabels,
          maxAutobotLabels: MAX_AUTOBOT_LABELS,
          semverDecision: normalizeBumpType(state.semverDecision),
          deterministicSemver: state.deterministicSemver || null
        }
      });
    }
  }

  return state;
}

async function syncProjectMilestone({ github, owner, repo, context, issueNumber }) {
  const payload = context.payload;
  if (!["opened", "synchronize", "reopened"].includes(payload.action)) {
    return;
  }

  const isPR = context.eventName === "pull_request";
  const freshIssue = await github.rest.issues.get({ owner, repo, issue_number: issueNumber });
  const currentLabelNames = freshIssue.data.labels.map((label) => normalizeLabelName(label.name));
  const currentBotComment = isPR ? await getExistingBotComment({ github, owner, repo, issueNumber }) : null;
  const currentBotMetadata = currentBotComment ? extractBotMetadata(currentBotComment.body) : {};
  const currentSemverDecision = semverDecisionFromMetadata(currentBotMetadata);
  const metadataCache = new Map();
  if (isPR && currentBotComment) {
    metadataCache.set(issueNumber, currentBotMetadata);
  }
  const releaseRelevant = hasReleaseRelevantLabel(currentLabelNames);
  if (!isPR) {
    if (freshIssue.data.milestone && !releaseRelevant) {
      await github.rest.issues.update({ owner, repo, issue_number: issueNumber, milestone: null });
    }
    return;
  }
  if (!releaseRelevant) {
    if (freshIssue.data.milestone) {
      await github.rest.issues.update({ owner, repo, issue_number: issueNumber, milestone: null });
    }
    return;
  }
  const issueHadMilestoneBeforeAutobot = Boolean(freshIssue.data.milestone);
  const existingMilestone = freshIssue.data.milestone
    ? (await github.rest.issues.getMilestone({ owner, repo, milestone_number: freshIssue.data.milestone.number })).data
    : null;
  if (existingMilestone && !isVersionTag(existingMilestone.title)) {
    return;
  }
  const highestOpenVersionMilestone = await getOrCreateMilestone({ github, owner, repo });
  let milestone = existingMilestone || highestOpenVersionMilestone;
  if (isVersionTag(highestOpenVersionMilestone.title) && (!existingMilestone || isVersionTag(milestone.title) && compareVersions(highestOpenVersionMilestone.title, milestone.title) > 0)) {
    milestone = highestOpenVersionMilestone;
  }
  const previewMilestoneWithBase = await ensureMilestoneBaseVersion({ github, owner, repo, milestone });
  const previewManagedMilestones = await listManagedSemverMilestones({ github, owner, repo, targetMilestone: previewMilestoneWithBase.milestone });
  const previewReleaseItems = await listReleaseItemsForMilestones({ github, owner, repo, milestones: previewManagedMilestones });
  const previewRequiredBump = await resolveRequiredBump({
    github,
    owner,
    repo,
    releaseItems: previewReleaseItems,
    currentLabelNames,
    currentIssueNumber: issueNumber,
    currentSemverDecision,
    metadataCache
  });
  const targetVersion = computeTargetVersion(previewMilestoneWithBase.baseVersion, previewRequiredBump);
  if (isVersionTag(targetVersion) && compareVersions(targetVersion, milestone.title) > 0) {
    milestone = await getOrCreateMilestoneByTitle({ github, owner, repo, title: targetVersion });
    milestone = await seedMilestoneMetadata({ github, owner, repo, milestone, baseVersion: previewMilestoneWithBase.baseVersion, managedTitle: targetVersion });
  }
  const consolidation = await selectivelyConsolidateSupersededMilestones({ github, owner, repo, targetMilestone: milestone, targetBump: previewRequiredBump, currentIssueNumber: issueNumber, metadataCache });
  milestone = consolidation.milestone;
  const milestoneChanged = !freshIssue.data.milestone || freshIssue.data.milestone.number !== milestone.number;
  if (milestoneChanged) {
    await github.rest.issues.update({ owner, repo, issue_number: issueNumber, milestone: milestone.number });
  }
  const items = await github.paginate(github.rest.issues.listForRepo, { owner, repo, milestone: milestone.number, state: "all" });
  const releaseItems = items.filter((item) => item.pull_request && hasReleaseRelevantLabel(item.labels));
  const currentPrRequiresMajorRelease = currentSemverDecision === "major" || currentLabelNames.includes("breaking-change");
  const existingMajorAlertComment = await getExistingMajorAlertComment({ github, owner, repo, issueNumber });
  if (currentPrRequiresMajorRelease && isPR) {
    if (!existingMajorAlertComment) {
      await github.rest.issues.createComment({ owner, repo, issue_number: issueNumber, body: `🚨 **MAJOR RELEASE ALERT** 🚨\n\n@${owner} This PR triggers a **major** version bump based on deterministic release-impact analysis.` });
    }
  } else if (existingMajorAlertComment) {
    await github.rest.issues.deleteComment({ owner, repo, comment_id: existingMajorAlertComment.id });
  }
  const milestoneWithBase = await ensureMilestoneBaseVersion({ github, owner, repo, milestone });
  const shouldRespectManualMilestone = milestoneWithBase.manualTitleLocked || (issueHadMilestoneBeforeAutobot && !milestoneWithBase.hadManagedMarker && !isVersionTag(milestoneWithBase.milestone.title));
  if (shouldRespectManualMilestone) return;
  const requiredBump = await resolveRequiredBump({
    github,
    owner,
    repo,
    releaseItems,
    currentLabelNames,
    currentIssueNumber: issueNumber,
    currentSemverDecision,
    metadataCache
  });
  const computedTitle = computeTargetVersion(milestoneWithBase.baseVersion, requiredBump);
  const newTitle = compareVersions(computedTitle, milestoneWithBase.milestone.title) > 0 ? computedTitle : milestoneWithBase.milestone.title;
  if (newTitle !== milestoneWithBase.milestone.title) {
    const metadataDescription = buildMilestoneMetadataDescription(milestoneWithBase.milestone.description || milestone.description || "", milestoneWithBase.baseVersion, newTitle);
    await github.rest.issues.updateMilestone({ owner, repo, milestone_number: milestone.number, title: newTitle, description: metadataDescription });
  }
  if (isPR && (milestoneChanged || consolidation.consolidatedMilestones.length > 0 || consolidation.migratedPullRequests.length > 0 || consolidation.retainedMilestones.length > 0 || newTitle !== milestoneWithBase.milestone.title)) {
    const previousMilestoneTitle = existingMilestone ? existingMilestone.title : null;
    const finalMilestoneTitle = newTitle;
    const noteBody = buildMilestoneComment({ milestoneChanged: milestoneChanged || newTitle !== milestoneWithBase.milestone.title, previousMilestoneTitle, finalMilestoneTitle, consolidation });
    await upsertMilestoneComment({
      github,
      owner,
      repo,
      issueNumber,
      body: noteBody,
      metadata: {
        milestone: finalMilestoneTitle,
        previousMilestone: previousMilestoneTitle,
        consolidatedMilestones: consolidation.consolidatedMilestones,
        migratedPullRequests: consolidation.migratedPullRequests.map((item) => item.number),
        retainedMilestones: consolidation.retainedMilestones,
        targetVersion: finalMilestoneTitle,
        requiredBump
      }
    });
  }
}

async function finalizeClosedPullRequestRelease({ github, owner, repo, context }) {
  const payload = context.payload;
  if (payload.action !== "closed") return;
  if (!payload.pull_request || !payload.pull_request.merged) return;
  const milestoneData = payload.pull_request.milestone;
  if (!milestoneData) return;
  const freshMilestone = await github.rest.issues.getMilestone({ owner, repo, milestone_number: milestoneData.number });
  if (!(freshMilestone.data.open_issues === 0 && freshMilestone.data.state === "open" && freshMilestone.data.closed_issues > 0)) {
    return;
  }
  const closedItems = await github.paginate(github.rest.issues.listForRepo, { owner, repo, milestone: freshMilestone.data.number, state: "closed" });
  const closedReleaseItems = closedItems.filter((item) => item.pull_request && hasReleaseRelevantLabel(item.labels));
  const metadataCache = new Map();
  const closedReleaseBumps = [];
  for (const item of closedReleaseItems) {
    closedReleaseBumps.push(await resolveReleaseItemBump({
      github,
      owner,
      repo,
      item,
      currentIssueNumber: payload.pull_request.number,
      currentSemverDecision: "none",
      metadataCache
    }));
  }
  const hasBreaking = closedReleaseBumps.some((bump) => bump === "major");
  const hasForcedType = closedReleaseItems.some((item) => item.labels.some((label) => FORCE_RELEASE_TYPES.includes(normalizeLabelName(label.name))));
  if (closedReleaseItems.length < MIN_RELEASE_SIZE && !hasForcedType && !hasBreaking) return;
  let targetVersion = freshMilestone.data.title;
  const releases = await github.rest.repos.listReleases({ owner, repo });
  const latestRelease = releases.data[0];
  if (latestRelease && latestRelease.draft) {
    const currentParts = targetVersion.replace("v", "").split(".").map(Number);
    const latestParts = latestRelease.tag_name.replace("v", "").split(".").map(Number);
    const isLatestLarger = latestParts[0] > currentParts[0] || (latestParts[0] === currentParts[0] && latestParts[1] > currentParts[1]) || (latestParts[0] === currentParts[0] && latestParts[1] === currentParts[1] && latestParts[2] > currentParts[2]);
    if (isLatestLarger) targetVersion = latestRelease.tag_name;
    await github.rest.repos.deleteRelease({ owner, repo, release_id: latestRelease.id });
    try {
      await github.rest.git.deleteRef({ owner, repo, ref: `tags/${latestRelease.tag_name}` });
    } catch (error) {
    }
  }
  await github.rest.issues.updateMilestone({ owner, repo, milestone_number: freshMilestone.data.number, state: "closed" });
  await github.rest.repos.createRelease({ owner, repo, tag_name: targetVersion, name: targetVersion, generate_release_notes: true, draft: true });
  const parts = targetVersion.replace("v", "").split(".").map(Number);
  parts[2] += 1;
  const nextVersion = `v${parts.join(".")}`;
  await github.rest.issues.createMilestone({ owner, repo, title: nextVersion });
}

module.exports = {
  finalizeClosedPullRequestRelease,
  hasReleaseRelevantLabel,
  inferIssueLabels,
  normalizeLabelName,
  parseDeterministicSemver,
  prepareProjectState,
  buildPrCommentBody,
  resolveRequiredBump,
  resolvePrLabelDelta,
  resolveReleaseItemBump,
  semverDecisionFromMetadata,
  syncPreparedProjectState,
  syncProjectMilestone
};
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const {
  analyzePullRequestSnapshotData,
  normalizePatch
} = require("../.github/scripts/autobot/pr_analysis.cjs");
const {
  buildPrCommentBody,
  hasReleaseRelevantLabel,
  parseDeterministicSemver
} = require("../.github/scripts/autobot/project_manager.cjs");
const {
  formatRegressionFixtureEntry,
  normalizeScenarioRegressionFixture
} = require("../tests/deterministic_scenario_regression_fixtures.cjs");

function readCaptureInputText(input = {}) {
  const env = input.env || process.env;
  const argv = input.argv || process.argv;
  const readFileSync = input.readFileSync || fs.readFileSync;
  const captureFile = String(env.SCENARIO_CAPTURE_FILE || argv[2] || "").trim();
  if (captureFile) {
    return readFileSync(captureFile, "utf8");
  }
  return readFileSync(0, "utf8");
}

function selectCaptureCandidate(payload, input = {}) {
  const env = input.env || process.env;
  const argv = input.argv || process.argv;
  const candidates = Array.isArray(payload)
    ? payload
    : Array.isArray(payload.captureCandidates)
      ? payload.captureCandidates
      : Array.isArray(payload.candidates)
        ? payload.candidates
        : [];
  const requestedId = String(env.SCENARIO_CAPTURE_ID || argv[3] || "").trim();
  if (!requestedId) {
    return candidates[0] || null;
  }
  return candidates.find((entry) => String(entry.fixtureId || "") === requestedId) || null;
}

function formatCaptureFixtureFromPayload(payload, input = {}) {
  const candidate = selectCaptureCandidate(payload, input);
  if (!candidate) {
    throw new Error("No capture candidate was found in the provided payload.");
  }
  const fixture = normalizeScenarioRegressionFixture({
    fixtureId: candidate.fixtureId,
    kind: candidate.kind,
    materialization: candidate.materialization,
    reason: candidate.reason,
    replayKey: candidate.replayKey,
    severity: candidate.severity,
    suiteKey: candidate.suiteKey
  });
  return formatRegressionFixtureEntry(fixture);
}

function hasRedirectedStdin(fsModule = fs) {
  try {
    return !fsModule.fstatSync(0).isCharacterDevice();
  } catch {
    return false;
  }
}

function shouldFormatCaptureFixture(input = {}) {
  const env = input.env || process.env;
  const argv = input.argv || process.argv;
  const fsModule = input.fsModule || fs;
  return Boolean(String(env.SCENARIO_CAPTURE_FILE || argv[2] || "").trim()) || hasRedirectedStdin(fsModule);
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

function parseIntegerFlag(value, fallback, minimum, maximum) {
  const parsed = Number.parseInt(String(value ?? "").trim(), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(minimum, Math.min(maximum, parsed));
}

function resolveIncludeUntracked(input = {}, previewMode = "working-tree", env = process.env) {
  if (input.includeUntracked !== undefined) {
    return Boolean(input.includeUntracked);
  }
  return parseBooleanFlag(env.AUTOBOT_LOCAL_INCLUDE_UNTRACKED, false);
}

function resolveMaxUntrackedFiles(input = {}, env = process.env) {
  if (input.maxUntrackedFiles !== undefined) {
    return parseIntegerFlag(input.maxUntrackedFiles, 200, 0, 20000);
  }
  return parseIntegerFlag(env.AUTOBOT_LOCAL_MAX_UNTRACKED_FILES, 200, 0, 20000);
}

function splitTrackedDiffSections(diffText) {
  const normalized = String(diffText || "").replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const sections = [];
  let current = [];
  for (const line of lines) {
    if (line.startsWith("diff --git ")) {
      if (current.length > 0) {
        sections.push(current.join("\n").trim());
      }
      current = [line];
      continue;
    }
    if (current.length > 0) {
      current.push(line);
    }
  }
  if (current.length > 0) {
    sections.push(current.join("\n").trim());
  }
  return sections.filter(Boolean);
}

function stripDiffPrefix(value) {
  return String(value || "").replace(/^a\//, "").replace(/^b\//, "");
}

function unescapeGitPath(value) {
  return String(value || "")
    .replace(/^"(.*)"$/, "$1")
    .replace(/\\([\\"])/g, "$1");
}

function parseDiffHeaderPaths(headerLine) {
  const text = String(headerLine || "");
  let match = text.match(/^diff --git "([^"]+)" "([^"]+)"$/);
  if (match) {
    return {
      beforePath: stripDiffPrefix(unescapeGitPath(match[1])),
      afterPath: stripDiffPrefix(unescapeGitPath(match[2]))
    };
  }
  match = text.match(/^diff --git a\/(.+) b\/(.+)$/);
  if (match) {
    return {
      beforePath: match[1],
      afterPath: match[2]
    };
  }
  return {
    beforePath: "",
    afterPath: ""
  };
}

function findDiffMetadataValue(lines, prefix) {
  const match = lines.find((line) => line.startsWith(prefix));
  return match ? match.slice(prefix.length).trim() : "";
}

function countTrackedPatchChanges(lines) {
  return lines.reduce((summary, line) => {
    if (line.startsWith("+") && !line.startsWith("+++ ")) {
      summary.additions += 1;
    } else if (line.startsWith("-") && !line.startsWith("--- ")) {
      summary.deletions += 1;
    }
    return summary;
  }, {
    additions: 0,
    deletions: 0
  });
}

function normalizeRepoRelativePath(value) {
  return String(value || "").replace(/\\/g, "/");
}

function buildTrackedFileEntry(sectionText) {
  const lines = String(sectionText || "").split("\n");
  const headerPaths = parseDiffHeaderPaths(lines[0]);
  const renameFrom = findDiffMetadataValue(lines, "rename from ");
  const renameTo = findDiffMetadataValue(lines, "rename to ");
  const copyFrom = findDiffMetadataValue(lines, "copy from ");
  const copyTo = findDiffMetadataValue(lines, "copy to ");
  let status = "modified";
  if (lines.some((line) => line.startsWith("new file mode "))) {
    status = "added";
  } else if (lines.some((line) => line.startsWith("deleted file mode "))) {
    status = "removed";
  } else if (renameFrom || renameTo) {
    status = "renamed";
  } else if (copyFrom || copyTo) {
    status = "copied";
  }
  const filename = normalizeRepoRelativePath(
    renameTo
      || copyTo
      || (status === "removed" ? headerPaths.beforePath : headerPaths.afterPath)
      || headerPaths.beforePath
  );
  const changes = countTrackedPatchChanges(lines);
  return {
    additions: changes.additions,
    deletions: changes.deletions,
    filename,
    patch: normalizePatch(sectionText),
    rawPatchAvailable: true,
    status
  };
}

function splitContentLines(content) {
  const normalized = String(content || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!normalized) {
    return [];
  }
  const withoutTerminalNewline = normalized.endsWith("\n") ? normalized.slice(0, -1) : normalized;
  return withoutTerminalNewline ? withoutTerminalNewline.split("\n") : [];
}

function buildSyntheticAddedPatch(filename, content) {
  const lines = splitContentLines(content);
  const header = [
    `diff --git a/${filename} b/${filename}`,
    "new file mode 100644",
    "--- /dev/null",
    `+++ b/${filename}`
  ];
  if (lines.length === 0) {
    return header.join("\n");
  }
  return [...header, `@@ -0,0 +1,${lines.length} @@`, ...lines.map((line) => `+${line}`)].join("\n");
}

function buildUntrackedFileEntry(filename, content) {
  const normalizedFilename = normalizeRepoRelativePath(filename);
  const text = String(content || "");
  const additions = text.includes("\u0000") ? 0 : splitContentLines(text).length;
  return {
    additions,
    deletions: 0,
    filename: normalizedFilename,
    patch: text.includes("\u0000") ? normalizePatch("") : normalizePatch(buildSyntheticAddedPatch(normalizedFilename, text)),
    rawPatchAvailable: !text.includes("\u0000"),
    status: "added"
  };
}

function buildLocalGitDiffSnapshot(input = {}) {
  const trackedFiles = splitTrackedDiffSections(input.diffText).map((section) => buildTrackedFileEntry(section));
  const untrackedFiles = (input.untrackedFiles || []).map((filename) => buildUntrackedFileEntry(filename, input.readFileText(filename)));
  const files = [...trackedFiles, ...untrackedFiles];
  const totals = files.reduce((summary, file) => {
    summary.additions += Number(file.additions || 0);
    summary.deletions += Number(file.deletions || 0);
    return summary;
  }, {
    additions: 0,
    deletions: 0
  });
  const parsedNumber = Number.parseInt(String(input.number || 0), 10);
  return {
    files,
    pullRequest: {
      body: String(input.body || ""),
      headRef: String(input.headRef || ""),
      number: Number.isFinite(parsedNumber) ? parsedNumber : 0,
      title: String(input.title || "")
    },
    totals: {
      additions: totals.additions,
      deletions: totals.deletions,
      filesChanged: files.length,
      totalChanges: totals.additions + totals.deletions
    }
  };
}

function runGitCommand(args, input = {}) {
  const execFile = input.execFileSync || execFileSync;
  const cwd = input.cwd || process.cwd();
  try {
    return String(execFile("git", args, {
      cwd,
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024
    }) || "").trimEnd();
  } catch (error) {
    const stderr = String(error && error.stderr ? error.stderr : "").trim();
    const suffix = stderr ? `: ${stderr}` : "";
    throw new Error(`git ${args.join(" ")} failed${suffix}`);
  }
}

function parseNullSeparatedList(text) {
  return String(text || "")
    .split("\u0000")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function safeParseJson(value, fallback) {
  try {
    return JSON.parse(String(value || ""));
  } catch {
    return fallback;
  }
}

function runGitCommandOptional(args, input = {}) {
  try {
    return runGitCommand(args, input);
  } catch {
    return "";
  }
}

function runCommandOptional(command, args, input = {}) {
  const execFile = input.execFileSync || execFileSync;
  const cwd = input.cwd || process.cwd();
  try {
    return String(execFile(command, args, {
      cwd,
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024
    }) || "").trimEnd();
  } catch {
    return "";
  }
}

function resolveVerifiedRef(refName, input = {}) {
  const normalized = String(refName || "").trim();
  if (!normalized) return "";
  const verification = runGitCommandOptional(
    ["rev-parse", "--verify", "--quiet", normalized + "^{commit}"],
    input
  ).trim();
  return verification ? normalized : "";
}

function readPullRequestFromEnvironment(input = {}) {
  const env = input.env || process.env;
  const parsedNumber = Number.parseInt(String(env.AUTOBOT_PR_NUMBER || "").trim(), 10);
  const baseRefName = String(env.AUTOBOT_PR_BASE_REF || "").trim();
  const headRef = String(env.AUTOBOT_PR_HEAD_REF || "").trim();
  const title = String(env.AUTOBOT_PR_TITLE || "").trim();
  const body = String(env.AUTOBOT_PR_BODY || "");
  if (!Number.isFinite(parsedNumber) && !baseRefName && !headRef && !title && !body) {
    return null;
  }
  return {
    baseRefName,
    body,
    headRef,
    number: Number.isFinite(parsedNumber) ? parsedNumber : 0,
    source: "env",
    title
  };
}

function readPullRequestFromEventPayload(input = {}) {
  const env = input.env || process.env;
  const readFileSync = input.readFileSync || fs.readFileSync;
  const eventName = String(env.GITHUB_EVENT_NAME || "").toLowerCase();
  const eventPath = String(env.GITHUB_EVENT_PATH || "").trim();
  if (!eventPath || (eventName !== "pull_request" && eventName !== "pull_request_target")) {
    return null;
  }
  try {
    const payload = JSON.parse(readFileSync(eventPath, "utf8"));
    const pullRequest = payload && payload.pull_request ? payload.pull_request : null;
    if (!pullRequest) return null;
    const parsedNumber = Number.parseInt(String(pullRequest.number || 0), 10);
    return {
      baseRefName: String((pullRequest.base && pullRequest.base.ref) || ""),
      body: String(pullRequest.body || ""),
      headRef: String((pullRequest.head && pullRequest.head.ref) || ""),
      number: Number.isFinite(parsedNumber) ? parsedNumber : 0,
      source: "event",
      title: String(pullRequest.title || "")
    };
  } catch {
    return null;
  }
}

function readPullRequestFromGh(input = {}) {
  const raw = runCommandOptional(
    "gh",
    ["pr", "view", "--json", "number,title,body,baseRefName,headRefName"],
    input
  ).trim();
  if (!raw) return null;
  try {
    const pullRequest = JSON.parse(raw);
    if (!pullRequest || !pullRequest.baseRefName) return null;
    const parsedNumber = Number.parseInt(String(pullRequest.number || 0), 10);
    return {
      baseRefName: String(pullRequest.baseRefName || ""),
      body: String(pullRequest.body || ""),
      headRef: String(pullRequest.headRefName || ""),
      number: Number.isFinite(parsedNumber) ? parsedNumber : 0,
      source: "gh-view",
      title: String(pullRequest.title || "")
    };
  } catch {
    return null;
  }
}

function readPullRequestFromGhByHead(input = {}) {
  const headRef = runGitCommandOptional(["rev-parse", "--abbrev-ref", "HEAD"], input).trim();
  if (!headRef || headRef === "HEAD") {
    return null;
  }

  const headSelectors = [headRef];
  const repoOwner = runCommandOptional("gh", ["repo", "view", "--json", "owner", "--jq", ".owner.login"], input).trim();
  if (repoOwner) {
    headSelectors.push(`${repoOwner}:${headRef}`);
  }

  for (const selector of headSelectors) {
    const raw = runCommandOptional(
      "gh",
      ["pr", "list", "--state", "open", "--head", selector, "--json", "number,title,body,baseRefName,headRefName", "--limit", "1"],
      input
    ).trim();
    if (!raw) {
      continue;
    }
    try {
      const items = JSON.parse(raw);
      const pullRequest = Array.isArray(items) ? items[0] : null;
      if (!pullRequest || !pullRequest.baseRefName) {
        continue;
      }
      const parsedNumber = Number.parseInt(String(pullRequest.number || 0), 10);
      return {
        baseRefName: String(pullRequest.baseRefName || ""),
        body: String(pullRequest.body || ""),
        headRef: String(pullRequest.headRefName || headRef),
        number: Number.isFinite(parsedNumber) ? parsedNumber : 0,
        source: "gh-list",
        title: String(pullRequest.title || "")
      };
    } catch { }
  }

  return null;
}

function readPullRequestFromRemoteRefs(input = {}) {
  const env = input.env || process.env;
  const remote = String(input.remote || env.AUTOBOT_PR_REMOTE || "origin").trim() || "origin";
  const headRef = runGitCommandOptional(["rev-parse", "--abbrev-ref", "HEAD"], input).trim();
  const localHeadSha = runGitCommandOptional(["rev-parse", "HEAD"], input).trim();

  let remoteHeadSha = "";
  if (headRef && headRef !== "HEAD") {
    const remoteHeadLine = runGitCommandOptional(["ls-remote", "--heads", remote, `refs/heads/${headRef}`], input).trim();
    if (remoteHeadLine) {
      remoteHeadSha = String(remoteHeadLine.split(/\s+/)[0] || "").trim();
    }
  }

  const candidateShas = new Set([localHeadSha, remoteHeadSha].filter(Boolean));
  if (candidateShas.size === 0) {
    return null;
  }

  const pullRefsText = runGitCommandOptional(["ls-remote", "--refs", remote, "refs/pull/*/head"], input).trim();
  if (!pullRefsText) {
    return null;
  }

  const matchingPrNumbers = [];
  for (const line of pullRefsText.split(/\r?\n/)) {
    const trimmed = String(line || "").trim();
    if (!trimmed) {
      continue;
    }
    const parts = trimmed.split(/\s+/);
    if (parts.length < 2) {
      continue;
    }
    const sha = String(parts[0] || "").trim();
    const ref = String(parts[1] || "").trim();
    const match = ref.match(/^refs\/pull\/(\d+)\/head$/);
    if (!match || !candidateShas.has(sha)) {
      continue;
    }
    const parsedNumber = Number.parseInt(String(match[1]), 10);
    if (Number.isFinite(parsedNumber)) {
      matchingPrNumbers.push(parsedNumber);
    }
  }

  if (matchingPrNumbers.length === 0) {
    return null;
  }

  matchingPrNumbers.sort((left, right) => left - right);

  return {
    baseRefName: "",
    body: "",
    headRef: headRef && headRef !== "HEAD" ? headRef : "",
    number: matchingPrNumbers[0],
    source: "git-refs",
    title: ""
  };
}

function resolvePreviewBase(input = {}) {
  const env = input.env || process.env;
  const explicitBaseRef = String(input.baseRef || env.AUTOBOT_LOCAL_BASE_REF || "").trim();
  if (explicitBaseRef) {
    return {
      baseRef: explicitBaseRef,
      previewMode: "working-tree"
    };
  }

  const prBaseRef = String(input.pullRequestBaseRef || "").trim();
  if (prBaseRef) {
    const preferredRef = resolveVerifiedRef("origin/" + prBaseRef, input)
      || resolveVerifiedRef(prBaseRef, input)
      || prBaseRef;
    return {
      baseRef: preferredRef,
      previewMode: "pull-request"
    };
  }

  const ciBaseRef = String(env.GITHUB_BASE_REF || env.AUTOBOT_PR_BASE_REF || "").trim();
  if (ciBaseRef) {
    const preferredRef = resolveVerifiedRef("origin/" + ciBaseRef, input)
      || resolveVerifiedRef(ciBaseRef, input)
      || ciBaseRef;
    return {
      baseRef: preferredRef,
      previewMode: "pull-request"
    };
  }

  const currentBranch = runGitCommandOptional(["rev-parse", "--abbrev-ref", "HEAD"], input).trim();
  const originHead = runGitCommandOptional(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], input).trim();
  const defaultBranch = originHead.replace(/^origin\//, "");
  if (currentBranch && originHead && defaultBranch && currentBranch !== "HEAD" && currentBranch !== defaultBranch) {
    return {
      baseRef: originHead,
      previewMode: "pull-request"
    };
  }

  return {
    baseRef: "HEAD",
    previewMode: "working-tree"
  };
}

function collectLocalGitDiffSnapshot(input = {}) {
  const env = input.env || process.env;
  const cwd = input.cwd || process.cwd();
  const readFileSync = input.readFileSync || fs.readFileSync;

  const repoRoot = String(input.repoRoot || runGitCommand(["rev-parse", "--show-toplevel"], {
    cwd,
    execFileSync: input.execFileSync
  })).trim();

  const repoExecInput = {
    cwd: repoRoot,
    execFileSync: input.execFileSync
  };

  const envPullRequest = readPullRequestFromEnvironment({ env });
  const eventPullRequest = envPullRequest ? null : readPullRequestFromEventPayload({
    env,
    readFileSync
  });
  const ghPullRequest = envPullRequest || eventPullRequest
    ? null
    : readPullRequestFromGh(repoExecInput);
  const ghHeadPullRequest = envPullRequest || eventPullRequest || ghPullRequest
    ? null
    : readPullRequestFromGhByHead(repoExecInput);
  const gitRefsPullRequest = envPullRequest || eventPullRequest || ghPullRequest || ghHeadPullRequest
    ? null
    : readPullRequestFromRemoteRefs({
        ...repoExecInput,
        env
      });

  const pullRequestContext = envPullRequest || eventPullRequest || ghPullRequest || ghHeadPullRequest || gitRefsPullRequest;
  const pullRequestContextSource = pullRequestContext ? pullRequestContext.source : "none";

  const baseResolution = resolvePreviewBase({
    baseRef: input.baseRef,
    cwd: repoRoot,
    env,
    execFileSync: input.execFileSync,
    pullRequestBaseRef: pullRequestContext ? pullRequestContext.baseRefName : ""
  });

  let baseRef = String(baseResolution.baseRef || "HEAD").trim() || "HEAD";
  let previewMode = String(baseResolution.previewMode || "working-tree");
  let diffSpec = baseRef;

  if (previewMode === "pull-request") {
    const mergeBase = String(
      input.mergeBase || runGitCommandOptional(["merge-base", baseRef, "HEAD"], repoExecInput)
    ).trim();
    if (mergeBase) {
      diffSpec = mergeBase + "..HEAD";
    }
  }

  let diffText;
  if (input.diffText !== undefined) {
    diffText = String(input.diffText);
  } else {
    try {
      diffText = runGitCommand([
        "diff",
        "--find-renames",
        "--no-ext-diff",
        "--no-color",
        "--unified=3",
        diffSpec
      ], repoExecInput);
    } catch (error) {
      if (previewMode === "pull-request") {
        previewMode = "working-tree";
        baseRef = "HEAD";
        diffSpec = "HEAD";
        diffText = runGitCommand([
          "diff",
          "--find-renames",
          "--no-ext-diff",
          "--no-color",
          "--unified=3",
          "HEAD"
        ], repoExecInput);
      } else {
        throw error;
      }
    }
  }

  const includeUntracked = resolveIncludeUntracked(input, previewMode, env);
  const maxUntrackedFiles = resolveMaxUntrackedFiles(input, env);

  const untrackedFiles = includeUntracked
    ? (input.untrackedFiles || parseNullSeparatedList(runGitCommand([
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z"
      ], repoExecInput))).slice(0, maxUntrackedFiles)
    : [];

const resolvedTitle = input.title !== undefined
  ? input.title
  : (env.AUTOBOT_LOCAL_TITLE !== undefined
    ? env.AUTOBOT_LOCAL_TITLE
    : (pullRequestContext ? pullRequestContext.title : ""));

const resolvedBody = input.body !== undefined
  ? input.body
  : (env.AUTOBOT_LOCAL_BODY !== undefined
    ? env.AUTOBOT_LOCAL_BODY
    : (pullRequestContext ? pullRequestContext.body : ""));

const resolvedNumber = input.number !== undefined
  ? Number.parseInt(String(input.number), 10)
  : (pullRequestContext ? pullRequestContext.number : 0);

const resolvedHeadRef = input.headRef
  || (pullRequestContext ? pullRequestContext.headRef : "")
  || runGitCommand(["rev-parse", "--abbrev-ref", "HEAD"], repoExecInput);

  const snapshot = buildLocalGitDiffSnapshot({
    baseRef,
    body: resolvedBody,
    diffText,
    headRef: resolvedHeadRef,
    number: Number.isFinite(resolvedNumber) ? resolvedNumber : 0,
    readFileText: function (filename) {
      return readFileSync(path.join(repoRoot, filename), "utf8");
    },
    title: resolvedTitle,
    untrackedFiles
  });

  return {
    baseRef,
    diffSpec,
    previewMode,
    repoRoot,
    snapshot,
    pullRequestContextSource,
  };
}

function analyzeCurrentGitDiff(input = {}) {
  const preview = collectLocalGitDiffSnapshot(input);
  return {
    ...preview,
    outputs: analyzePullRequestSnapshotData(preview.snapshot)
  };
}

function formatLocalDiffAnalysis(input) {
  const deterministicLabels = safeParseJson(input.outputs.deterministic_labels_json, []);
  const linkedLabels = safeParseJson(input.outputs.linked_labels_json, []);
  const nextAutobotLabels = [...new Set([
    ...(Array.isArray(deterministicLabels) ? deterministicLabels : []),
    ...(Array.isArray(linkedLabels) ? linkedLabels : [])
  ])];
  const deterministicSemver = parseDeterministicSemver(input.outputs.deterministic_semver_json);
  const releaseRelevant = hasReleaseRelevantLabel(nextAutobotLabels);
  const rawPrNumber = Number.parseInt(String(input.snapshot.pullRequest.number || 0), 10);
  const displayPrNumber = Number.isFinite(rawPrNumber) && rawPrNumber > 0
    ? rawPrNumber
    : "local-preview";

  const onlineBody = buildPrCommentBody({
    summaryForComment: input.outputs.deterministic_summary,
    pullRequest: {
      base: { ref: String(input.baseRef || "HEAD") },
      head: { ref: String(input.snapshot.pullRequest.headRef || "(detached)") },
      number: displayPrNumber
    },
    nextAutobotLabels,
    semverDecision: releaseRelevant
      ? String((deterministicSemver && deterministicSemver.decision) || "none")
      : "none"
  });

  const metadataLines = [
    "<!-- autobot-local-preview -->",
    "<!-- mode: " + String(input.previewMode || "working-tree") + " -->",
    "<!-- base: " + String(input.baseRef || "HEAD") + " -->",
    "<!-- diff: " + String(input.diffSpec || input.baseRef || "HEAD") + " -->",
    "<!-- pr-context: " + String(input.pullRequestContextSource || "none") + " -->",
    ""
  ];

  return metadataLines.join("\n") + onlineBody;
}

function main() {
  if (shouldFormatCaptureFixture()) {
    const payload = JSON.parse(readCaptureInputText());
    console.log(formatCaptureFixtureFromPayload(payload));
    return;
  }
  console.log(formatLocalDiffAnalysis(analyzeCurrentGitDiff()));
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }
}

module.exports = {
  analyzeCurrentGitDiff,
  buildLocalGitDiffSnapshot,
  formatCaptureFixtureFromPayload,
  formatLocalDiffAnalysis,
  readCaptureInputText,
  selectCaptureCandidate,
  shouldFormatCaptureFixture,
  splitTrackedDiffSections
};
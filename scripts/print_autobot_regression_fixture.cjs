const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const {
  analyzePullRequestSnapshotData,
  normalizePatch
} = require("../.github/scripts/autobot/pr_analysis.cjs");
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
  return {
    files,
    pullRequest: {
      body: String(input.body || ""),
      headRef: String(input.headRef || ""),
      number: 0,
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

function collectLocalGitDiffSnapshot(input = {}) {
  const cwd = input.cwd || process.cwd();
  const baseRef = String(input.baseRef || process.env.AUTOBOT_LOCAL_BASE_REF || "HEAD").trim() || "HEAD";
  const repoRoot = String(input.repoRoot || runGitCommand(["rev-parse", "--show-toplevel"], {
    cwd,
    execFileSync: input.execFileSync
  })).trim();
  const diffText = String(input.diffText !== undefined ? input.diffText : runGitCommand([
    "diff",
    "--find-renames",
    "--no-ext-diff",
    "--no-color",
    "--unified=3",
    baseRef
  ], {
    cwd: repoRoot,
    execFileSync: input.execFileSync
  }));
  const untrackedFiles = input.untrackedFiles || parseNullSeparatedList(runGitCommand([
    "ls-files",
    "--others",
    "--exclude-standard",
    "-z"
  ], {
    cwd: repoRoot,
    execFileSync: input.execFileSync
  }));
  const readFileSync = input.readFileSync || fs.readFileSync;
  const snapshot = buildLocalGitDiffSnapshot({
    baseRef,
    body: input.body !== undefined ? input.body : process.env.AUTOBOT_LOCAL_BODY,
    diffText,
    headRef: input.headRef || runGitCommand(["rev-parse", "--abbrev-ref", "HEAD"], {
      cwd: repoRoot,
      execFileSync: input.execFileSync
    }),
    readFileText: (filename) => readFileSync(path.join(repoRoot, filename), "utf8"),
    title: input.title !== undefined ? input.title : process.env.AUTOBOT_LOCAL_TITLE,
    untrackedFiles
  });
  return {
    baseRef,
    repoRoot,
    snapshot
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
  const lines = [
    "## Local Autobot Preview",
    "",
    `Repository: ${input.repoRoot}`,
    `Base ref: ${input.baseRef}`,
    `Branch: ${input.snapshot.pullRequest.headRef || "(detached)"}`,
    `Files changed: ${input.snapshot.totals.filesChanged} (+${input.snapshot.totals.additions} -${input.snapshot.totals.deletions})`,
    ""
  ];
  if (input.snapshot.files.length === 0) {
    lines.push("No local git diff changes were detected for this preview.");
    lines.push("");
  }
  lines.push(input.outputs.deterministic_summary);
  return lines.join("\n");
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
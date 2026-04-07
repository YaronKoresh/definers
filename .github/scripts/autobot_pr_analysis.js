const fs = require("fs");

const SNAPSHOT_FILE = "/tmp/autobot_pr_snapshot.json";
const MAX_FILES_PER_WINDOW = 20;
const MAX_PATCH_CHARS_PER_WINDOW = 7000;
const MAX_PATCH_CHARS_PER_FILE = 700;
const MAX_HIGHLIGHTED_FILES_PER_WINDOW = 6;
const MAX_PATCH_SNIPPETS_PER_WINDOW = 4;
const MAX_BATCH_SUMMARY_REQUESTS = 2;
const MAX_SUMMARY_BATCH_CONTENT_CHARS = 6400;
const MAX_SINGLE_AI_WINDOWS = 2;
const MAX_TOP_DIRECTORIES = 8;
const MAX_TOP_FILES = 10;
const HIGH_VOLUME_FILES = 120;
const HIGH_VOLUME_CHANGES = 12000;
const HIGH_VOLUME_WINDOWS = 5;
const WINDOW_SEPARATOR = "\n\n═══════════════════════════════════════════\n\n";
const MAINTENANCE_ONLY_CATEGORIES = new Set(["documentation", "test", "workflow", "github", "config", "dependencies"]);
const VERSION_CRITICAL_LABELS = ["breaking-change", "security", "api", "database", "schema", "compatibility", "migration", "feature-flag", "runtime", "performance"];
const LABEL_ORDER = [
  "breaking-change",
  "security",
  "api",
  "database",
  "schema",
  "compatibility",
  "migration",
  "feature-flag",
  "runtime",
  "performance",
  "bug",
  "enhancement",
  "ui",
  "documentation",
  "test",
  "workflow",
  "automation",
  "github",
  "ci",
  "config",
  "dependencies",
  "docker",
  "tooling",
  "dx",
  "cleanup",
  "chore"
];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeText(filePath, content) {
  fs.writeFileSync(filePath, content, "utf8");
}

function topDirectoryForFile(filename) {
  const parts = String(filename || "").split("/");
  parts.pop();
  return parts.length > 0 ? parts.slice(0, 2).join("/") : "(root)";
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

function scoreFile(file) {
  const changeVolume = Number(file.additions || 0) + Number(file.deletions || 0);
  const patchWeight = file.patch ? Math.min(file.patch.length, 4000) : 0;
  const structuralWeight = file.status === "renamed" || file.status === "removed" ? 800 : 0;
  return changeVolume * 5 + patchWeight + structuralWeight;
}

function sortLabels(labels) {
  return [...new Set(labels)]
    .filter(Boolean)
    .sort((left, right) => {
      const leftRank = LABEL_ORDER.indexOf(left);
      const rightRank = LABEL_ORDER.indexOf(right);
      const normalizedLeftRank = leftRank === -1 ? LABEL_ORDER.length : leftRank;
      const normalizedRightRank = rightRank === -1 ? LABEL_ORDER.length : rightRank;
      return normalizedLeftRank - normalizedRightRank || left.localeCompare(right);
    });
}

function formatBulletLines(items, fallback) {
  return items.length > 0 ? items.map((item) => `- ${item}`).join("\n") : `- ${fallback}`;
}

function matchesWorkflowFilePath(normalizedPath) {
  return /^\.github\/workflows\/.+\.ya?ml$/.test(normalizedPath)
    || /^\.circleci\/config\.ya?ml$/.test(normalizedPath)
    || /^\.gitlab-ci\.ya?ml$/.test(normalizedPath)
    || /^\.buildkite\/.+\.ya?ml$/.test(normalizedPath)
    || /(^|\/)azure-pipelines\.ya?ml$/.test(normalizedPath)
    || /(^|\/)jenkinsfile$/.test(normalizedPath)
    || /^\.github\/actions\//.test(normalizedPath);
}

function matchesWorkflowScriptPath(normalizedPath) {
  return /^scripts\/.+\.(js|ts|py|sh|ps1|bat)$/.test(normalizedPath)
    && /\b(workflow|pipeline|orchestr|release|triage|autobot|ci)\b/.test(normalizedPath.replace(/[/._-]+/g, " "));
}

function hasWorkflowTextEvidence(text) {
  return /workflow_dispatch|schedule:|cron:|jobs:|steps:|runs-on:|uses:\s*actions\/|pull_request:|push:|pipeline|orchestrat/.test(text);
}

function matchesUiPath(normalizedPath) {
  return /(^|\/)(components?|ui|views?|templates|static|styles?|themes?|frontend)(\/|$)/.test(normalizedPath)
    || /(^|\/)presentation\/apps\//.test(normalizedPath)
    || /(^|\/)(gui_[^/]+|[^/]+_gui)\.py$/.test(normalizedPath)
    || /\.(css|scss|sass|less|tsx|jsx|vue|svelte|html)$/.test(normalizedPath);
}

function hasUiTextEvidence(text) {
  return /\bgr\.(blocks|button|textbox|dropdown|accordion|slider|checkbox|row|column|tabs?|html|markdown)\b/.test(text)
    || /<(button|form|input|select|dialog|label)\b/.test(text)
    || /\bclassname=/.test(text);
}

function hasApiPathEvidence(normalizedPath) {
  return /(^|\/)(api|apis|routes?|controllers?|webhooks?)(\/|$)/.test(normalizedPath)
    || /(^|\/)(openapi|swagger)(\.[^/]+)?$/.test(normalizedPath);
}

function hasApiPatchEvidence(patch) {
  return /https:\/\/models\.github\.ai\/inference\//.test(patch)
    || /\b(rest api|graphql api|github api|webhook)\b/.test(patch)
    || /\b(router|app)\.(get|post|put|patch|delete)\b/.test(patch);
}

function hasRuntimePathEvidence(normalizedPath) {
  return /^docker\//.test(normalizedPath)
    || /(^|\/)(dockerfile|compose\.ya?ml)$/.test(normalizedPath)
    || /(^|\/)(platform|runtime|system|cuda)(\/|$)/.test(normalizedPath)
    || /(^|\/)(pyproject\.toml|tox\.ini|setup\.(cfg|py))$/.test(normalizedPath)
    || matchesWorkflowFilePath(normalizedPath);
}

function hasRuntimePatchEvidence(patch) {
  return /\brequires-python\b/.test(patch)
    || /\bpython 3\.\d+\b/.test(patch)
    || /\bcuda\b/.test(patch)
    || /\bffmpeg\b/.test(patch)
    || /\bubuntu\b|\bwindows\b|\blinux\b|\bmacos\b/.test(patch)
    || /\bplatform_system\b/.test(patch)
    || /\bapt-get\b|\bdnf\b|\bpacman\b/.test(patch)
    || /\bnvidia\b/.test(patch);
}

function hasSchemaEvidence(normalizedPath, patch) {
  return /(^|\/)(schema|schemas|openapi|swagger|graphql|protos?)(\/|$)/.test(normalizedPath)
    || /(^|\/)(openapi|swagger|graphql|schema)(\.[^/]+)?$/.test(normalizedPath)
    || /\.(proto|graphql|gql)$/.test(normalizedPath)
    || /\b(json schema|graphql schema|openapi|swagger)\b/.test(patch)
    || /syntax\s*=\s*"proto"/.test(patch);
}

function hasMigrationEvidence(normalizedPath, patch) {
  return /(^|\/)(migrations?|alembic)(\/|$)/.test(normalizedPath)
    || /\.sql$/.test(normalizedPath)
    || /\bmigrations?\b|\balembic\b|upgrade\s*\(|downgrade\s*\(/.test(patch);
}

function hasDatabaseEvidence(normalizedPath, patch) {
  const pathEvidence = /(^|\/)(database|databases|db|sql)(\/|$)/.test(normalizedPath)
    || /\.sql$/.test(normalizedPath);
  const patchEvidence = /\b(database|sql|sqlite|postgres(?:ql)?|mysql|orm)\b/.test(patch)
    || /\b(create|alter|drop)\s+table\b/.test(patch)
    || /\b(insert\s+into|delete\s+from)\b/.test(patch);
  return /\.sql$/.test(normalizedPath) || pathEvidence && patchEvidence || /\b(create|alter|drop)\s+table\b/.test(patch);
}

function hasApiEvidence(normalizedPath, patch) {
  return hasApiPathEvidence(normalizedPath) || hasApiPatchEvidence(patch);
}

function hasSecurityEvidence(normalizedPath, patch) {
  return /(^|\/)(security|auth|policy)(\/|$)/.test(normalizedPath)
    || /(^|\/)(codeql|dependabot|security)(\.[^/]+)?$/.test(normalizedPath)
    || /authorization:\s*bearer/.test(patch)
    || /permissions:\s*(\{|\n)/.test(patch)
    || /\b(access token|bearer token|secret|secrets|credential|sanitize|sanitiz|csrf|xss|auth(?:entication|orization)?|least privilege|permissions?)\b/.test(patch);
}

function hasBreakingChangeEvidence(patch) {
  return /\bbreaking[\s-]?change\b|\bbackward[\s-]?incompatible\b|\bincompatible api\b|\bconsumer adaptation\b|\brequires migration\b|\bmust update (callers|consumers)\b/.test(patch);
}

function deriveTitleSignals(prSignalText) {
  const bug = /\b(bug|fix|fixes|fixed|regression|hotfix|error|broken|failure|repair)\b/.test(prSignalText);
  const enhancement = !bug && /(^|\s)(feat|feature|enhancement)(:|\b)/.test(prSignalText);
  const documentation = /\b(doc|docs|documentation|readme)\b/.test(prSignalText);
  const ui = /\b(ui|ux|gradio|frontend)\b/.test(prSignalText);
  return {
    bug,
    enhancement,
    documentation,
    ui
  };
}

function classifyFile(filename) {
  const normalized = String(filename || "").toLowerCase();
  const categories = new Set();
  if (matchesWorkflowFilePath(normalized) || matchesWorkflowScriptPath(normalized)) {
    categories.add("workflow");
  }
  if (/^\.github\//.test(normalized)) {
    categories.add("github");
  }
  if (
    /^docs\//.test(normalized) ||
    /(^|\/)(readme|contributing|changelog|license)(\.[^/]+)?$/.test(normalized) ||
    /\.(md|mdx|rst|txt)$/.test(normalized)
  ) {
    categories.add("documentation");
  }
  if (
    /^tests\//.test(normalized) ||
    /(^|\/)test_[^/]+\.py$/.test(normalized) ||
    /\.(spec|test)\.(js|jsx|ts|tsx|py)$/.test(normalized)
  ) {
    categories.add("test");
  }
  if (matchesUiPath(normalized)) {
    categories.add("ui");
  }
  if (
    /^docker\//.test(normalized) ||
    /(^|\/)dockerfile$/.test(normalized) ||
    /compose\.ya?ml$/.test(normalized)
  ) {
    categories.add("docker");
  }
  if (/^scripts\//.test(normalized) || /^\.vscode\//.test(normalized)) {
    categories.add("tooling");
  }
  if (
    /^requirements.*\.txt$/.test(normalized) ||
    /(^|\/)(poetry\.lock|package(-lock)?\.json|pnpm-lock\.yaml|yarn\.lock|uv\.lock|pdm\.lock)$/.test(normalized) ||
    normalized === "pyproject.toml"
  ) {
    categories.add("dependencies");
  }
  if (
    /^pyproject\.toml$/.test(normalized) ||
    /^manifest\.in$/.test(normalized) ||
    /^tox\.ini$/.test(normalized) ||
    /^setup\.(cfg|py)$/.test(normalized) ||
    /^ruff\.toml$/.test(normalized) ||
    /^\.editorconfig$/.test(normalized) ||
    /^\.pre-commit-config\.(yaml|yml)$/.test(normalized) ||
    /\.(toml|ya?ml|json|ini|cfg)$/.test(normalized)
  ) {
    categories.add("config");
  }
  if (categories.size === 0) {
    categories.add("source");
  }
  return [...categories];
}

function deriveSignals(file) {
  const normalizedPath = String(file.filename || "").toLowerCase();
  const patch = String(file.patch || "").toLowerCase();
  const text = `${normalizedPath}\n${patch}`;
  const signals = new Set();
  const workflowSignal = matchesWorkflowFilePath(normalizedPath)
    || matchesWorkflowScriptPath(normalizedPath) && hasWorkflowTextEvidence(text);

  if (workflowSignal) {
    signals.add("workflow");
    if (/schedule:|workflow_dispatch|pull_request:|issues:|uses:\s*actions\/|jobs:|steps:|runs-on:|push:/.test(text)) {
      signals.add("ci");
    }
    if (/autobot|automation|label|triage|milestone|release/.test(text)) {
      signals.add("automation");
    }
  }
  if (/^\.github\//.test(normalizedPath)) {
    signals.add("github");
  }
  if (matchesUiPath(normalizedPath) || hasUiTextEvidence(text)) {
    signals.add("ui");
  }
  if (/^docker\//.test(normalizedPath) || /(^|\/)dockerfile$/.test(normalizedPath) || /compose\.ya?ml/.test(normalizedPath)) {
    signals.add("docker");
    signals.add("runtime");
  }
  if (hasSchemaEvidence(normalizedPath, patch)) {
    signals.add("schema");
  }
  if (hasMigrationEvidence(normalizedPath, patch)) {
    signals.add("migration");
  }
  if (hasDatabaseEvidence(normalizedPath, patch)) {
    signals.add("database");
  }
  if (hasApiEvidence(normalizedPath, patch)) {
    signals.add("api");
  }
  if (hasSecurityEvidence(normalizedPath, patch)) {
    signals.add("security");
  }
  if (hasBreakingChangeEvidence(patch)) {
    signals.add("breaking-change");
  }
  if (/(^|\/)(compat|compatibility|interop|polyfill|shim)(\/|$)/.test(normalizedPath)
    || /\b(backward compatibility|compatibility|interop|polyfill|shim)\b/.test(patch)) {
    signals.add("compatibility");
  }
  if (/feature[\s-]?flag|kill switch|rollout/.test(text)) {
    signals.add("feature-flag");
  }
  if (hasRuntimePathEvidence(normalizedPath) && hasRuntimePatchEvidence(patch)) {
    signals.add("runtime");
  }
  if (/\b(performance|latency|throughput|cache|memory|optimiz)\b/.test(patch)) {
    signals.add("performance");
  }
  return [...signals];
}

function formatWindow(windowFiles, windowCount, index, options = {}) {
  const highlightedFileLimit = options.highlightedFileLimit ?? MAX_HIGHLIGHTED_FILES_PER_WINDOW;
  const patchSnippetLimit = options.patchSnippetLimit ?? MAX_PATCH_SNIPPETS_PER_WINDOW;
  const coverageLevel = options.coverageLevel ?? (patchSnippetLimit > 0 ? "full" : "compact");
  const highlightedFiles = [...windowFiles]
    .sort((left, right) => right.score - left.score || left.filename.localeCompare(right.filename))
    .slice(0, highlightedFileLimit);
  const detailedPatchFiles = patchSnippetLimit > 0
    ? highlightedFiles.filter((file) => file.rawPatchAvailable).slice(0, patchSnippetLimit)
    : [];
  const omittedFileCount = Math.max(windowFiles.length - highlightedFiles.length, 0);

  return [
    `WINDOW ${index + 1}/${windowCount}`,
    `Coverage level: ${coverageLevel}`,
    `Files in window: ${windowFiles.length}`,
    "Highlighted files:",
    highlightedFiles.map((file) => `- ${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})${file.rawPatchAvailable ? "" : " [patch unavailable]"}`).join("\n"),
    omittedFileCount > 0 ? `Additional lower-signal files not expanded in this window: ${omittedFileCount}` : "All files in this window are highlighted.",
    detailedPatchFiles.length > 0 ? "Detailed patch snippets:" : "Detailed patch snippets: none available in this window.",
    detailedPatchFiles.map((file) => [
      `FILE: ${file.filename}`,
      `STATUS: ${file.status.toUpperCase()} (+${file.additions} -${file.deletions})`,
      file.patch
    ].join("\n")).join("\n\n")
  ].filter(Boolean).join("\n\n");
}

function formatMinimalWindow(windowFiles, windowCount, index) {
  const topFiles = [...windowFiles]
    .sort((left, right) => right.score - left.score || left.filename.localeCompare(right.filename))
    .slice(0, 3);
  const windowDirs = [...new Set(windowFiles.map((file) => topDirectoryForFile(file.filename)))].slice(0, 4);

  return [
    `WINDOW ${index + 1}/${windowCount}`,
    "Coverage level: minimal",
    `Files in window: ${windowFiles.length}`,
    `Top directories: ${windowDirs.join(", ") || "(none)"}`,
    `Top files: ${topFiles.map((file) => `${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})`).join("; ") || "(none)"}`
  ].join("\n");
}

function buildSummaryBatches(entries) {
  const batches = [];
  let currentBatch = [];
  let currentBatchChars = 0;

  for (const entry of [...entries].sort((left, right) => left.index - right.index)) {
    const separatorLength = currentBatch.length > 0 ? WINDOW_SEPARATOR.length : 0;
    const wouldOverflow = currentBatch.length > 0 && currentBatchChars + separatorLength + entry.selectedContent.length > MAX_SUMMARY_BATCH_CONTENT_CHARS;

    if (wouldOverflow) {
      batches.push(currentBatch);
      currentBatch = [];
      currentBatchChars = 0;
    }

    currentBatch.push(entry);
    currentBatchChars += (currentBatch.length > 1 ? WINDOW_SEPARATOR.length : 0) + entry.selectedContent.length;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}

function downgradeWindowCoverage(entries) {
  const candidates = entries
    .map((entry, index) => {
      if (entry.selectedVariant === "full") {
        return { index, nextVariant: "compact", nextContent: entry.compact, savings: entry.full.length - entry.compact.length, score: entry.score };
      }
      if (entry.selectedVariant === "compact") {
        return { index, nextVariant: "minimal", nextContent: entry.minimal, savings: entry.compact.length - entry.minimal.length, score: entry.score };
      }
      return null;
    })
    .filter(Boolean)
    .sort((left, right) => right.savings - left.savings || left.score - right.score || left.index - right.index);

  const candidate = candidates[0];
  if (!candidate || candidate.savings <= 0) {
    return false;
  }

  entries[candidate.index].selectedVariant = candidate.nextVariant;
  entries[candidate.index].selectedContent = candidate.nextContent;
  return true;
}

function buildOverflowWindow(entries, windowCount) {
  const overflowFiles = entries.flatMap((entry) => entry.files);
  const topOverflowFiles = overflowFiles
    .slice()
    .sort((left, right) => scoreFile(right) - scoreFile(left) || left.filename.localeCompare(right.filename))
    .slice(0, 8);
  const overflowDirs = [...new Set(overflowFiles.map((file) => topDirectoryForFile(file.filename)))].slice(0, 8);

  return {
    index: entries[0]?.index ?? windowCount,
    files: overflowFiles,
    score: 0,
    selectedVariant: "minimal",
    selectedContent: [
      "WINDOW OVERFLOW",
      "Coverage level: minimal",
      `Overflow windows: ${entries.length}`,
      `Top directories: ${overflowDirs.join(", ") || "(none)"}`,
      `Top files: ${topOverflowFiles.map((file) => `${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})`).join("; ") || "(none)"}`
    ].join("\n")
  };
}

function summarizeCoverage(entries) {
  return entries.reduce((counts, entry) => {
    counts[entry.selectedVariant] = (counts[entry.selectedVariant] || 0) + 1;
    return counts;
  }, { full: 0, compact: 0, minimal: 0 });
}

function buildSummaryBatchPrompt({ batchEntries, batchIndex, batchCount, totalFiles, totalAdditions, totalDeletions, topDirectories, orderedSignals, candidateLabels }) {
  const coverageCounts = summarizeCoverage(batchEntries);
  const globalContext = [
    `Total files: ${totalFiles}`,
    `Total additions: ${totalAdditions}`,
    `Total deletions: ${totalDeletions}`,
    `Top directories: ${topDirectories.join(", ") || "(none)"}`,
    `Deterministic signals: ${orderedSignals.join(", ") || "(none)"}`,
    `Candidate labels: ${candidateLabels.join(", ") || "(none)"}`
  ].join("\n");

  return [
    "You are a principal software engineer analyzing one batch of rolling windows from a large pull request.",
    "You are working under a strict request budget, so use the global context and the supplied windows instead of inferring missing detail.",
    "Each batch contains windows in full, compact, or minimal form to stay within prompt budget.",
    "Use only the evidence shown in this batch. Compact and minimal windows are only partially analyzed.",
    "Do not invent hidden behavior, missing files, or unsupported motivations.",
    "",
    "OUTPUT REQUIREMENTS:",
    "- Output MUST be valid Markdown.",
    "- Do NOT wrap the report in triple backticks.",
    "- Keep the result between 700 and 1600 characters.",
    "- Preserve explicit signals related to breaking changes, compatibility, migration, api, runtime, database, schema, security, performance, ui, workflow, tooling, tests, and documentation when supported.",
    "- End your response with the exact final line: END_OF_REPORT",
    "",
    "Use EXACTLY this structure:",
    "",
    "## Window Batch Analysis",
    "",
    "### Batch Scope",
    "2-4 sentences.",
    "",
    "### Release Signals",
    "2-5 bullets.",
    "",
    "### Risks And Checks",
    "2-5 bullets.",
    "",
    "### Classification Signals",
    "2-6 bullets listing direct technical signals useful for downstream label selection.",
    "",
    "BATCH METRICS",
    `Batch number: ${batchIndex + 1}`,
    `Total batches: ${batchCount}`,
    `Windows in batch: ${batchEntries.length}`,
    `Coverage mix: full=${coverageCounts.full || 0}, compact=${coverageCounts.compact || 0}, minimal=${coverageCounts.minimal || 0}`,
    "",
    "GLOBAL CONTEXT",
    globalContext,
    "",
    "WINDOWS",
    batchEntries.map((entry) => entry.selectedContent).join(WINDOW_SEPARATOR)
  ].join("\n");
}

async function collectPullRequestSnapshot({ github, owner, repo, pullRequest }) {
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
  writeText(SNAPSHOT_FILE, JSON.stringify(snapshot));
  return snapshot;
}

function analyzePullRequestSnapshotData(snapshot, options = {}) {
  const writeArtifacts = options.writeArtifacts !== false;
  const filesWithContext = snapshot.files.map((file) => {
    const fileWithContext = {
      filename: file.filename,
      status: file.status,
      additions: file.additions,
      deletions: file.deletions,
      patch: file.patch,
      rawPatchAvailable: Boolean(file.rawPatchAvailable)
    };
    fileWithContext.score = scoreFile(fileWithContext);
    fileWithContext.categories = classifyFile(file.filename);
    fileWithContext.signals = deriveSignals(fileWithContext);
    return fileWithContext;
  });
  const totalAdditions = Number(snapshot.totals.additions || 0);
  const totalDeletions = Number(snapshot.totals.deletions || 0);
  const totalChanges = Number(snapshot.totals.totalChanges || 0);
  const prSignalText = `${snapshot.pullRequest.title}`.toLowerCase();
  const titleSignals = deriveTitleSignals(prSignalText);
  const categoryCounts = new Map();
  const directoryScores = new Map();
  const signalSet = new Set();

  for (const fileWithContext of filesWithContext) {
    for (const category of fileWithContext.categories) {
      categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1);
    }
    const directory = topDirectoryForFile(fileWithContext.filename);
    directoryScores.set(directory, (directoryScores.get(directory) || 0) + fileWithContext.score);
    for (const signal of fileWithContext.signals) {
      signalSet.add(signal);
    }
  }

  const orderedSignals = sortLabels([...signalSet]);
  const topDirectories = [...directoryScores.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, MAX_TOP_DIRECTORIES)
    .map(([directory]) => directory);
  const topFiles = [...filesWithContext]
    .sort((left, right) => right.score - left.score || left.filename.localeCompare(right.filename))
    .slice(0, MAX_TOP_FILES);
  const categorySummary = [...categoryCounts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([category, count]) => `${category}=${count}`)
    .join(", ");
  const maintenanceCategoriesList = [...categoryCounts.entries()]
    .filter(([category]) => MAINTENANCE_ONLY_CATEGORIES.has(category))
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([category]) => category);
  const hasBehavioralSurfaceChange = Boolean(
    categoryCounts.get("source") || categoryCounts.get("ui")
  );
  const zeroAiEligible = filesWithContext.length > 0 && [...categoryCounts.keys()].every((category) => MAINTENANCE_ONLY_CATEGORIES.has(category)) && !VERSION_CRITICAL_LABELS.some((label) => signalSet.has(label));

  const deterministicLabelSet = new Set();
  for (const signal of orderedSignals) {
    deterministicLabelSet.add(signal);
  }
  if (categoryCounts.get("documentation")) deterministicLabelSet.add("documentation");
  if (categoryCounts.get("test")) deterministicLabelSet.add("test");
  if (categoryCounts.get("ui") || signalSet.has("ui")) deterministicLabelSet.add("ui");
  if (categoryCounts.get("workflow") || signalSet.has("workflow")) deterministicLabelSet.add("workflow");
  if (categoryCounts.get("github") || signalSet.has("github")) deterministicLabelSet.add("github");
  if (signalSet.has("automation")) deterministicLabelSet.add("automation");
  if (signalSet.has("ci")) deterministicLabelSet.add("ci");
  if (categoryCounts.get("config")) deterministicLabelSet.add("config");
  if (categoryCounts.get("dependencies")) deterministicLabelSet.add("dependencies");
  if (categoryCounts.get("docker")) deterministicLabelSet.add("docker");
  if (categoryCounts.get("tooling")) {
    deterministicLabelSet.add("tooling");
    deterministicLabelSet.add("dx");
  }
  if (filesWithContext.some((file) => file.status === "removed")) deterministicLabelSet.add("cleanup");
  if (zeroAiEligible) {
    if (categoryCounts.get("documentation")) deterministicLabelSet.add("documentation");
    if (categoryCounts.get("test")) deterministicLabelSet.add("test");
    if (categoryCounts.get("workflow") || signalSet.has("workflow")) deterministicLabelSet.add("workflow");
    if (categoryCounts.get("github") || signalSet.has("github")) deterministicLabelSet.add("github");
    if (categoryCounts.get("config")) deterministicLabelSet.add("config");
    if (categoryCounts.get("dependencies")) deterministicLabelSet.add("dependencies");
  }
  if (deterministicLabelSet.size === 0) deterministicLabelSet.add("chore");

  const candidateLabelSet = new Set(deterministicLabelSet);
  for (const signal of orderedSignals) {
    candidateLabelSet.add(signal);
  }
  if (!zeroAiEligible) {
    if (titleSignals.bug && hasBehavioralSurfaceChange) {
      candidateLabelSet.add("bug");
    }
    if (titleSignals.enhancement && hasBehavioralSurfaceChange) {
      candidateLabelSet.add("enhancement");
    }
    if (categoryCounts.get("documentation") || titleSignals.documentation) candidateLabelSet.add("documentation");
    if (categoryCounts.get("test")) candidateLabelSet.add("test");
    if (categoryCounts.get("ui") || signalSet.has("ui")) candidateLabelSet.add("ui");
    if (categoryCounts.get("workflow") || signalSet.has("workflow")) candidateLabelSet.add("workflow");
    if (categoryCounts.get("github") || signalSet.has("github")) candidateLabelSet.add("github");
    if (signalSet.has("automation")) candidateLabelSet.add("automation");
    if (signalSet.has("ci")) candidateLabelSet.add("ci");
    if (categoryCounts.get("config")) candidateLabelSet.add("config");
    if (categoryCounts.get("dependencies")) candidateLabelSet.add("dependencies");
    if (categoryCounts.get("docker")) candidateLabelSet.add("docker");
    if (categoryCounts.get("tooling")) {
      candidateLabelSet.add("tooling");
      candidateLabelSet.add("dx");
    }
    if (filesWithContext.some((file) => file.status === "removed")) {
      candidateLabelSet.add("cleanup");
    }
  }

  const deterministicLabels = sortLabels([...deterministicLabelSet]).slice(0, 6);
  const candidateLabels = sortLabels([...candidateLabelSet]).slice(0, 18);

  const windows = [];
  let currentWindow = [];
  let currentWindowPatchChars = 0;

  for (const file of filesWithContext) {
    const patchSize = file.patch.length;
    const shouldStartNewWindow = currentWindow.length > 0 && (
      currentWindow.length >= MAX_FILES_PER_WINDOW ||
      currentWindowPatchChars + patchSize > MAX_PATCH_CHARS_PER_WINDOW
    );

    if (shouldStartNewWindow) {
      windows.push(currentWindow);
      currentWindow = [];
      currentWindowPatchChars = 0;
    }

    currentWindow.push(file);
    currentWindowPatchChars += patchSize;
  }

  if (currentWindow.length > 0) {
    windows.push(currentWindow);
  }

  const windowCandidates = windows.map((windowFiles, index) => ({
    index,
    files: windowFiles,
    score: windowFiles.reduce((sum, file) => sum + file.score, 0),
    full: formatWindow(windowFiles, windows.length, index, { coverageLevel: "full" }),
    compact: formatWindow(windowFiles, windows.length, index, { highlightedFileLimit: 4, patchSnippetLimit: 0, coverageLevel: "compact" }),
    minimal: formatMinimalWindow(windowFiles, windows.length, index)
  }));
  const windowEntries = windowCandidates.map((candidate) => ({
    ...candidate,
    selectedVariant: "full",
    selectedContent: candidate.full
  }));

  let summaryBatches = buildSummaryBatches(windowEntries);
  while (summaryBatches.length > MAX_BATCH_SUMMARY_REQUESTS && downgradeWindowCoverage(windowEntries)) {
    summaryBatches = buildSummaryBatches(windowEntries);
  }

  if (summaryBatches.length > MAX_BATCH_SUMMARY_REQUESTS) {
    const overflowEntries = summaryBatches.slice(MAX_BATCH_SUMMARY_REQUESTS - 1).flat();
    summaryBatches = [
      ...summaryBatches.slice(0, MAX_BATCH_SUMMARY_REQUESTS - 1),
      [buildOverflowWindow(overflowEntries, windows.length)]
    ];
  }

  const highVolume = totalChanges >= HIGH_VOLUME_CHANGES || filesWithContext.length >= HIGH_VOLUME_FILES || windows.length > HIGH_VOLUME_WINDOWS;
  const summaryTier = zeroAiEligible ? "zero_ai" : highVolume ? "capped_batch_ai" : "single_ai";
  const directWindowEntries = [...windowEntries]
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, MAX_SINGLE_AI_WINDOWS)
    .sort((left, right) => left.index - right.index);
  const summaryBatchCount = summaryTier === "capped_batch_ai" ? summaryBatches.length : 0;
  const estimatedAiRequests = summaryTier === "zero_ai" ? 0 : summaryTier === "single_ai" ? 1 : summaryBatchCount + 1;

  if (writeArtifacts) {
    for (const [batchIndex, batchEntries] of summaryBatches.entries()) {
      const prompt = buildSummaryBatchPrompt({
        batchEntries,
        batchIndex,
        batchCount: summaryBatches.length,
        totalFiles: filesWithContext.length,
        totalAdditions,
        totalDeletions,
        topDirectories,
        orderedSignals,
        candidateLabels
      });
      writeText(`/tmp/summary_batch_${batchIndex + 1}.txt`, prompt);
    }
  }

  const directWindowContent = directWindowEntries.length > 0
    ? directWindowEntries.map((entry) => entry.compact).join(WINDOW_SEPARATOR)
    : "No diff windows selected.";
  if (writeArtifacts) {
    writeText("/tmp/pr_single_ai_windows.txt", directWindowContent);
  }

  function buildDeterministicSummary() {
    const whatChangedSentences = zeroAiEligible
      ? [
          `This pull request changes ${filesWithContext.length} files (+${totalAdditions} -${totalDeletions}) and stays within narrow maintenance surfaces: ${maintenanceCategoriesList.join(", ") || "none"}.`,
          `The highest-signal areas are ${topDirectories.join(", ") || "(root)"}, and the deterministic scan did not find API, schema, database, runtime, security, or breaking-change signals.`
        ]
      : [
          `This pull request changes ${filesWithContext.length} files (+${totalAdditions} -${totalDeletions}) across ${topDirectories.join(", ") || "(root)"} with category mix ${categorySummary || "(none)"}.`,
          orderedSignals.length > 0
            ? `Deterministic high-signal areas include ${orderedSignals.join(", ")}.`
            : "The deterministic scan did not find strong release-critical signals, so any deeper classification depends on the selected diff windows or batch analyses."
        ];
    const releaseBullets = zeroAiEligible
      ? ["Likely not release-relevant because the diff is limited to narrow maintenance areas."]
      : orderedSignals.length > 0
        ? [
            `Release-relevant signals detected: ${orderedSignals.join(", ")}.`,
            `Highest-signal files: ${topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "(none)"}.`
          ]
        : ["No direct version-critical path signals were detected by the deterministic scan."];
    const riskBullets = zeroAiEligible
      ? [
          `Review ${topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "the touched files"} for wording, workflow, or expectation drift.`,
          categoryCounts.get("workflow")
            ? "Validate repository automation behavior because workflow files changed."
            : "Validate that the change stays scoped to the intended maintenance surface."
        ]
      : [
          orderedSignals.length > 0
            ? `Focus review on ${orderedSignals.join(", ")} interactions in ${topDirectories.join(", ") || "(root)"}.`
            : `Focus review on ${topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "the highest-signal files"} because the deterministic scan is only partial.`,
          categoryCounts.get("test")
            ? "Run the touched regression surfaces and verify changed tests still reflect product behavior."
            : "Add or verify targeted checks for the changed runtime paths."
        ];
    const classificationSignals = [];
    if (categoryCounts.get("documentation")) classificationSignals.push("Documentation files are part of the changed scope.");
    if (categoryCounts.get("test")) classificationSignals.push("Tests are part of the changed scope.");
    if (categoryCounts.get("ui") || signalSet.has("ui")) classificationSignals.push("UI surfaces changed.");
    if (categoryCounts.get("workflow") || signalSet.has("workflow")) classificationSignals.push("Workflow or orchestration logic changed.");
    if (categoryCounts.get("config")) classificationSignals.push("Configuration surfaces changed.");
    if (categoryCounts.get("dependencies")) classificationSignals.push("Dependency or manifest files changed.");
    if (orderedSignals.length > 0) classificationSignals.push(`Deterministic release signals: ${orderedSignals.join(", ")}.`);
    if (classificationSignals.length === 0) classificationSignals.push("Deterministic path analysis did not isolate a narrow label family.");
    classificationSignals.push(`Top directories: ${topDirectories.join(", ") || "(root)"}.`);
    return [
      "## Autobot Summary",
      "",
      "### What Changed",
      whatChangedSentences.join(" "),
      "",
      "### Release Relevance",
      formatBulletLines(releaseBullets, "No release signals detected."),
      "",
      "### Risks And Testing",
      formatBulletLines(riskBullets, "Review the highest-signal files."),
      "",
      "### Classification Signals",
      formatBulletLines(classificationSignals.slice(0, 6), "No classification signals detected.")
    ].join("\n");
  }

  const deterministicSummary = buildDeterministicSummary();
  const evidenceScaffold = [
    "PR EVIDENCE",
    `Summary tier: ${summaryTier}`,
    `Estimated AI requests: ${estimatedAiRequests}`,
    `Files changed: ${filesWithContext.length}`,
    `Additions: ${totalAdditions}`,
    `Deletions: ${totalDeletions}`,
    `Windows discovered: ${windows.length}`,
    `Batch requests planned: ${summaryBatchCount}`,
    `Category mix: ${categorySummary || "(none)"}`,
    `Top directories: ${topDirectories.join(", ") || "(root)"}`,
    `Candidate labels: ${candidateLabels.join(", ") || "(none)"}`,
    orderedSignals.length > 0 ? `Deterministic signals: ${orderedSignals.join(", ")}` : "Deterministic signals: (none)",
    "",
    "Top files:",
    formatBulletLines(topFiles.map((file) => `${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})`), "(none)")
  ].join("\n");
  if (writeArtifacts) {
    writeText("/tmp/pr_evidence.txt", evidenceScaffold);
  }

  const totalSelectedWindowChars = directWindowEntries.reduce((sum, entry) => sum + entry.compact.length, 0);
  return {
    has_changes: filesWithContext.length > 0 ? "true" : "false",
    files_changed: String(filesWithContext.length),
    additions: String(totalAdditions),
    deletions: String(totalDeletions),
    windows_count: String(windows.length),
    windows_included: String(windowEntries.length),
    windows_omitted: String(Math.max(windows.length - directWindowEntries.length, 0)),
    summary_batch_count: String(summaryBatchCount),
    summary_prompt_chars: String(totalSelectedWindowChars),
    summary_tier: summaryTier,
    estimated_ai_requests: String(estimatedAiRequests),
    candidate_labels_json: JSON.stringify(candidateLabels),
    deterministic_labels_json: JSON.stringify(deterministicLabels),
    deterministic_summary: deterministicSummary,
    evidence_chars: String(evidenceScaffold.length)
  };
}

function analyzePullRequestSnapshot() {
  return analyzePullRequestSnapshotData(readJson(SNAPSHOT_FILE));
}

module.exports = {
  MAX_BATCH_SUMMARY_REQUESTS,
  SNAPSHOT_FILE,
  WINDOW_SEPARATOR,
  analyzePullRequestSnapshot,
  analyzePullRequestSnapshotData,
  collectPullRequestSnapshot
};
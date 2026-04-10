const fs = require("fs");

const { scoreDeterministicEvidence } = require("./autobot_deterministic_scorer");

const SNAPSHOT_FILE = "/tmp/autobot_pr_snapshot.json";
const MAX_AUTOBOT_LABELS = 12;
const MAX_PATCH_CHARS_PER_FILE = 700;
const MAX_TOP_DIRECTORIES = 8;
const MAX_TOP_FILES = 10;
const MIN_BEHAVIORAL_ADDITION_LINES = 80;
const MIN_PUBLIC_CONTRACT_MOVES = 3;
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
const ACCESSIBILITY_TEXT_PATTERN = /\baria-|accessib|a11y|screen reader|keyboard nav/;
const AUTOMATION_TEXT_PATTERN = /\b(autobot|automation|label|triage|milestone|release)\b/;
const FEATURE_FLAG_TEXT_PATTERN = /feature[\s-]?flag|kill switch|rollout/;
const LOCALIZATION_TEXT_PATTERN = /\bi18n\b|\bl10n\b|\blocale\b|\btranslations?\b|\bgettext\b/;
const RUNTIME_SUPPORT_TEXT_PATTERN = /\brequires-python\b|\bpython 3\.\d+\b|\bcuda\b|\bffmpeg\b|\bubuntu\b|\bwindows\b|\blinux\b|\bmacos\b|\bnvidia\b|\bplatform_system\b/;
const SECURITY_SECRET_TEXT_PATTERN = /authorization:\s*bearer|\b(access token|bearer token|secret|secrets|credential|permissions?)\b/;

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

function isAutobotClassificationInfrastructure(normalizedPath) {
  return /^\.github\/scripts\/autobot[^/]*\.js$/.test(normalizedPath)
    || normalizedPath === ".github/workflows/autobot.yml"
    || normalizedPath === "tests/test_autobot_workflow.py";
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

function isBehavioralSurfaceFile(file) {
  return file.categories.includes("source") || file.categories.includes("ui");
}

function isPublicPackageContractPath(normalizedPath) {
  return /^src\/[^/]+\/(?:__init__\.py|[^/]+\.py|[^/]+\/__init__\.py|[^/]+\/[^/]+\.py)$/.test(normalizedPath);
}

function countBehavioralSurfaceAdditions(filesWithContext) {
  return filesWithContext.filter((file) => file.status === "added" && isBehavioralSurfaceFile(file)).length;
}

function countPublicContractMoves(filesWithContext) {
  return filesWithContext.filter((file) => {
    if (!["removed", "renamed"].includes(file.status)) {
      return false;
    }
    return isPublicPackageContractPath(String(file.filename || "").toLowerCase());
  }).length;
}

function hasCapabilityExpansionSignal({ behavioralSurfaceAdditions, totalAdditions, totalChanges }) {
  return behavioralSurfaceAdditions >= 2
    || behavioralSurfaceAdditions >= 1 && (totalAdditions >= MIN_BEHAVIORAL_ADDITION_LINES || totalChanges >= MIN_BEHAVIORAL_ADDITION_LINES * 2);
}

function hasStructuralPublicBreakingSignal({ behavioralSurfaceAdditions, publicContractMoves, filesWithContext }) {
  if (publicContractMoves < MIN_PUBLIC_CONTRACT_MOVES) {
    return false;
  }
  const touchesPublicPackageInitializer = filesWithContext.some((file) => /^src\/[^/]+\/(?:__init__\.py|[^/]+\/__init__\.py)$/.test(String(file.filename || "").toLowerCase()));
  return touchesPublicPackageInitializer || behavioralSurfaceAdditions > 0;
}

function deriveSignals(file) {
  const normalizedPath = String(file.filename || "").toLowerCase();
  const patch = String(file.patch || "").toLowerCase();
  const text = `${normalizedPath}\n${patch}`;
  const signals = new Set();
  const autobotClassificationInfrastructure = isAutobotClassificationInfrastructure(normalizedPath);
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
  if (autobotClassificationInfrastructure) {
    return [...signals];
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

function isDocsSitePath(normalizedPath) {
  return /^docs\//.test(normalizedPath);
}

function isExamplePath(normalizedPath) {
  return /(^|\/)(examples?|samples?|demo)(\/|$)/.test(normalizedPath);
}

function isDevcontainerPath(normalizedPath) {
  return /^\.devcontainer\//.test(normalizedPath)
    || /(^|\/)devcontainer\.json$/.test(normalizedPath);
}

function isInfrastructurePath(normalizedPath) {
  return /(^|\/)(infra|infrastructure|deploy|k8s|kubernetes|terraform|helm)(\/|$)/.test(normalizedPath)
    || /\.(tf|tfvars)$/.test(normalizedPath)
    || /chart\.ya?ml$/.test(normalizedPath);
}

function patchLinesWithPrefix(patch, prefix) {
  return String(patch || "")
    .split("\n")
    .filter((line) => line.startsWith(prefix) && !line.startsWith(`${prefix}${prefix}${prefix}`));
}

function hasPatchLineMatch(file, prefix, pattern) {
  return patchLinesWithPrefix(file.patch, prefix).some((line) => pattern.test(line.toLowerCase()));
}

function isDestructiveFileChange(file) {
  const normalizedPath = String(file.filename || "").toLowerCase();
  const patch = String(file.patch || "").toLowerCase();
  return ["removed", "renamed"].includes(file.status)
    || hasBreakingChangeEvidence(patch)
    || hasPatchLineMatch(file, "-", /\b(remove|drop|delete|deprecat|disable|rename|migrate)\b/)
    || isPublicPackageContractPath(normalizedPath) && Number(file.deletions || 0) > Number(file.additions || 0);
}

function inferEvidenceScope(file) {
  const normalizedPath = String(file.filename || "").toLowerCase();
  if (
    isPublicPackageContractPath(normalizedPath)
    || hasApiPathEvidence(normalizedPath)
    || /\.(proto|graphql|gql)$/.test(normalizedPath)
    || matchesUiPath(normalizedPath)
  ) {
    return "public";
  }
  if (file.categories.some((category) => ["documentation", "test", "workflow", "github", "config", "dependencies", "docker", "tooling"].includes(category))) {
    return "repo";
  }
  return "subsystem";
}

function addEvidenceItem(evidenceMap, ruleId, options = {}) {
  const rawOccurrenceCount = options.occurrenceCount === undefined
    ? 1
    : Number(options.occurrenceCount);
  const occurrenceCount = Number.isFinite(rawOccurrenceCount)
    ? Math.max(rawOccurrenceCount, 0)
    : 1;
  if (occurrenceCount < 1) {
    return;
  }
  const scope = options.scope || "";
  const confidence = options.confidence || "";
  const polarity = options.polarity || "";
  const key = [ruleId, scope, confidence, polarity].join("|");
  let entry = evidenceMap.get(key);

  if (!entry) {
    entry = {
      ruleId,
      occurrenceCount: 0,
      metadata: {
        sampleFiles: []
      }
    };
    if (scope) entry.scope = scope;
    if (confidence) entry.confidence = confidence;
    if (polarity) entry.polarity = polarity;
    evidenceMap.set(key, entry);
  }

  entry.occurrenceCount += occurrenceCount;

  for (const sampleFile of options.sampleFiles || []) {
    if (!sampleFile || entry.metadata.sampleFiles.includes(sampleFile)) {
      continue;
    }
    if (entry.metadata.sampleFiles.length >= 5) {
      break;
    }
    entry.metadata.sampleFiles.push(sampleFile);
  }
}

function finalizeEvidenceItems(evidenceMap) {
  return [...evidenceMap.values()]
    .map((entry) => {
      const normalizedEntry = {
        ruleId: entry.ruleId,
        occurrenceCount: entry.occurrenceCount
      };
      if (entry.scope) normalizedEntry.scope = entry.scope;
      if (entry.confidence) normalizedEntry.confidence = entry.confidence;
      if (entry.polarity) normalizedEntry.polarity = entry.polarity;
      if (entry.metadata.sampleFiles.length > 0) {
        normalizedEntry.metadata = entry.metadata;
      }
      return normalizedEntry;
    })
    .sort((left, right) => right.occurrenceCount - left.occurrenceCount || left.ruleId.localeCompare(right.ruleId));
}

function compareRankedLabelEntries(left, right) {
  if (right.score !== left.score) {
    return right.score - left.score;
  }
  const leftRank = LABEL_ORDER.indexOf(left.label);
  const rightRank = LABEL_ORDER.indexOf(right.label);
  const normalizedLeftRank = leftRank === -1 ? LABEL_ORDER.length : leftRank;
  const normalizedRightRank = rightRank === -1 ? LABEL_ORDER.length : rightRank;
  return normalizedLeftRank - normalizedRightRank || left.label.localeCompare(right.label);
}

function rankScoredLabels(labelScores, propertyName) {
  return Object.values(labelScores)
    .filter((entry) => Boolean(entry[propertyName]))
    .sort(compareRankedLabelEntries)
    .map((entry) => entry.label);
}

function mergeRankedLabels(primaryLabels, secondaryLabels, limit) {
  const mergedLabels = [];
  for (const label of [...primaryLabels, ...secondaryLabels]) {
    if (!label || mergedLabels.includes(label)) {
      continue;
    }
    mergedLabels.push(label);
    if (mergedLabels.length >= limit) {
      break;
    }
  }
  return mergedLabels;
}

function buildDeterministicEvidence(filesWithContext, context) {
  const evidenceMap = new Map();
  const documentationFiles = [];
  const docsSiteFiles = [];
  const exampleFiles = [];
  const addedTestFiles = [];
  const removedTestFiles = [];
  const workflowFiles = [];
  const ciFiles = [];
  const githubFiles = [];
  const automationFiles = [];
  const configFiles = [];
  const dependencyExpansionFiles = [];
  const dependencyTighteningFiles = [];
  const dockerExpansionFiles = [];
  const dockerDroppedFiles = [];
  const devcontainerFiles = [];
  const toolingFiles = [];
  const infrastructureFiles = [];
  const uiFiles = [];
  const accessibilityFiles = [];
  const localizationFiles = [];
  const securityAuthFiles = [];
  const securitySecretFiles = [];
  const destructiveApiFiles = [];
  const additiveApiFiles = [];
  const destructiveSchemaFiles = [];
  const additiveSchemaFiles = [];
  const destructiveDatabaseFiles = [];
  const additiveDatabaseFiles = [];
  const migrationFiles = [];
  const compatibilityDropFiles = [];
  const compatibilityShimFiles = [];
  const featureFlagFiles = [];
  const featureFlagContractFiles = [];
  const runtimeSupportAddedFiles = [];
  const runtimeSupportDroppedFiles = [];
  const runtimePolicyFiles = [];
  const performanceFiles = [];
  const cleanupFiles = [];
  const publicRemovedFiles = [];
  const publicRenamedFiles = [];
  const publicAddedFiles = [];
  const behavioralAdditionFiles = [];

  for (const file of filesWithContext) {
    const normalizedPath = String(file.filename || "").toLowerCase();
    const patch = String(file.patch || "").toLowerCase();
    const categories = new Set(file.categories);
    const signals = new Set(file.signals);
    const scope = inferEvidenceScope(file);
    const destructive = isDestructiveFileChange(file);
    const runtimeAdded = hasPatchLineMatch(file, "+", RUNTIME_SUPPORT_TEXT_PATTERN) && !hasPatchLineMatch(file, "-", RUNTIME_SUPPORT_TEXT_PATTERN);
    const runtimeDropped = hasPatchLineMatch(file, "-", RUNTIME_SUPPORT_TEXT_PATTERN) && !hasPatchLineMatch(file, "+", RUNTIME_SUPPORT_TEXT_PATTERN);

    if (categories.has("documentation")) {
      if (isDocsSitePath(normalizedPath)) {
        docsSiteFiles.push(file.filename);
      } else if (isExamplePath(normalizedPath)) {
        exampleFiles.push(file.filename);
      } else {
        documentationFiles.push(file.filename);
      }
    }

    if (categories.has("test")) {
      if (file.status === "removed") {
        removedTestFiles.push(file.filename);
      } else {
        addedTestFiles.push(file.filename);
      }
    }

    if (categories.has("workflow")) {
      workflowFiles.push(file.filename);
      if (signals.has("ci")) {
        ciFiles.push(file.filename);
      }
    }

    if (categories.has("github")) {
      githubFiles.push(file.filename);
    }

    if (signals.has("automation") || AUTOMATION_TEXT_PATTERN.test(`${normalizedPath}\n${patch}`)) {
      automationFiles.push(file.filename);
    }

    if (categories.has("config")) {
      configFiles.push(file.filename);
    }

    if (categories.has("dependencies")) {
      if (runtimeDropped || destructive) {
        dependencyTighteningFiles.push(file.filename);
      } else {
        dependencyExpansionFiles.push(file.filename);
      }
    }

    if (categories.has("docker")) {
      if (runtimeDropped || destructive && !runtimeAdded) {
        dockerDroppedFiles.push(file.filename);
      } else {
        dockerExpansionFiles.push(file.filename);
      }
    }

    if (categories.has("tooling")) {
      if (isDevcontainerPath(normalizedPath)) {
        devcontainerFiles.push(file.filename);
      } else {
        toolingFiles.push(file.filename);
      }
    }

    if (isInfrastructurePath(normalizedPath)) {
      infrastructureFiles.push(file.filename);
    }

    if (categories.has("ui")) {
      uiFiles.push(file.filename);
      if (ACCESSIBILITY_TEXT_PATTERN.test(patch)) {
        accessibilityFiles.push(file.filename);
      }
      if (LOCALIZATION_TEXT_PATTERN.test(patch)) {
        localizationFiles.push(file.filename);
      }
    }

    if (signals.has("security")) {
      if (SECURITY_SECRET_TEXT_PATTERN.test(patch)) {
        securitySecretFiles.push(file.filename);
      } else {
        securityAuthFiles.push(file.filename);
      }
    }

    if (signals.has("api") && (hasApiPathEvidence(normalizedPath) || isPublicPackageContractPath(normalizedPath))) {
      if (destructive) {
        destructiveApiFiles.push(file.filename);
      } else {
        additiveApiFiles.push(file.filename);
      }
    }

    if (signals.has("schema") && (hasSchemaEvidence(normalizedPath, "") || isPublicPackageContractPath(normalizedPath))) {
      if (destructive) {
        destructiveSchemaFiles.push(file.filename);
      } else {
        additiveSchemaFiles.push(file.filename);
      }
    }

    if (signals.has("database") && hasDatabaseEvidence(normalizedPath, patch)) {
      if (destructive || /\b(drop|alter)\s+table\b/.test(patch)) {
        destructiveDatabaseFiles.push(file.filename);
      } else {
        additiveDatabaseFiles.push(file.filename);
      }
    }

    if (signals.has("migration")) {
      migrationFiles.push(file.filename);
    }

    if (signals.has("compatibility")) {
      if (destructive || runtimeDropped) {
        compatibilityDropFiles.push(file.filename);
      } else {
        compatibilityShimFiles.push(file.filename);
      }
    }

    if (signals.has("feature-flag") || FEATURE_FLAG_TEXT_PATTERN.test(patch)) {
      if (destructive) {
        featureFlagContractFiles.push(file.filename);
      } else {
        featureFlagFiles.push(file.filename);
      }
    }

    if (signals.has("runtime")) {
      if (runtimeDropped) {
        runtimeSupportDroppedFiles.push(file.filename);
      } else if (runtimeAdded || categories.has("docker") || categories.has("dependencies")) {
        runtimeSupportAddedFiles.push(file.filename);
      } else {
        runtimePolicyFiles.push(file.filename);
      }
    }

    if (signals.has("performance")) {
      performanceFiles.push(file.filename);
    }

    if (file.status === "removed" && !isPublicPackageContractPath(normalizedPath) && !categories.has("test")) {
      cleanupFiles.push(file.filename);
    }

    if (isPublicPackageContractPath(normalizedPath)) {
      if (file.status === "removed") {
        publicRemovedFiles.push(file.filename);
      }
      if (file.status === "renamed") {
        publicRenamedFiles.push(file.filename);
      }
      if (file.status === "added" && isBehavioralSurfaceFile(file)) {
        publicAddedFiles.push(file.filename);
      }
    }

    if (file.status === "added" && isBehavioralSurfaceFile(file)) {
      behavioralAdditionFiles.push(file.filename);
    }
  }

  const publicCapabilityFiles = publicAddedFiles.length > 0
    ? publicAddedFiles
    : publicRemovedFiles.length + publicRenamedFiles.length > 0
      ? behavioralAdditionFiles.filter((filename) => /^src\/[^/]+\//.test(String(filename).toLowerCase()))
      : [];

  addEvidenceItem(evidenceMap, "documentation-surface", { occurrenceCount: documentationFiles.length, scope: "repo", confidence: "structural", sampleFiles: documentationFiles });
  addEvidenceItem(evidenceMap, "docs-site-surface", { occurrenceCount: docsSiteFiles.length, scope: "repo", confidence: "structural", sampleFiles: docsSiteFiles });
  addEvidenceItem(evidenceMap, "examples-surface", { occurrenceCount: exampleFiles.length, scope: "repo", confidence: "structural", sampleFiles: exampleFiles });
  addEvidenceItem(evidenceMap, "tests-added", { occurrenceCount: addedTestFiles.length, scope: "repo", confidence: "structural", sampleFiles: addedTestFiles });
  addEvidenceItem(evidenceMap, "tests-removed", { occurrenceCount: removedTestFiles.length, scope: "repo", confidence: "structural", sampleFiles: removedTestFiles });
  addEvidenceItem(evidenceMap, "workflow-orchestration-change", { occurrenceCount: workflowFiles.length, scope: "repo", confidence: "structural", sampleFiles: workflowFiles });
  addEvidenceItem(evidenceMap, "ci-execution-change", { occurrenceCount: ciFiles.length, scope: "repo", confidence: "structural", sampleFiles: ciFiles });
  addEvidenceItem(evidenceMap, "github-management-change", { occurrenceCount: githubFiles.length, scope: "repo", confidence: "structural", sampleFiles: githubFiles });
  addEvidenceItem(evidenceMap, "automation-bot-change", { occurrenceCount: automationFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: automationFiles });
  addEvidenceItem(evidenceMap, "config-surface-change", { occurrenceCount: configFiles.length, scope: "repo", confidence: "structural", sampleFiles: configFiles });
  addEvidenceItem(evidenceMap, "dependency-capability-expansion", { occurrenceCount: dependencyExpansionFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: dependencyExpansionFiles });
  addEvidenceItem(evidenceMap, "dependency-compatibility-tightening", { occurrenceCount: dependencyTighteningFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: dependencyTighteningFiles });
  addEvidenceItem(evidenceMap, "docker-runtime-expansion", { occurrenceCount: dockerExpansionFiles.length, scope: "repo", confidence: "structural", sampleFiles: dockerExpansionFiles });
  addEvidenceItem(evidenceMap, "docker-runtime-drop", { occurrenceCount: dockerDroppedFiles.length, scope: "repo", confidence: "structural", sampleFiles: dockerDroppedFiles });
  addEvidenceItem(evidenceMap, "devcontainer-surface-change", { occurrenceCount: devcontainerFiles.length, scope: "repo", confidence: "structural", sampleFiles: devcontainerFiles });
  addEvidenceItem(evidenceMap, "tooling-surface-change", { occurrenceCount: toolingFiles.length, scope: "repo", confidence: "structural", sampleFiles: toolingFiles });
  addEvidenceItem(evidenceMap, "infrastructure-surface-change", { occurrenceCount: infrastructureFiles.length, scope: "repo", confidence: "structural", sampleFiles: infrastructureFiles });
  addEvidenceItem(evidenceMap, "ui-surface-change", { occurrenceCount: uiFiles.length, scope: "public", confidence: "corroborated", sampleFiles: uiFiles });
  addEvidenceItem(evidenceMap, "accessibility-improvement", { occurrenceCount: accessibilityFiles.length, scope: "public", confidence: "corroborated", sampleFiles: accessibilityFiles });
  addEvidenceItem(evidenceMap, "localization-change", { occurrenceCount: localizationFiles.length, scope: "public", confidence: "corroborated", sampleFiles: localizationFiles });
  addEvidenceItem(evidenceMap, "security-auth-change", { occurrenceCount: securityAuthFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: securityAuthFiles });
  addEvidenceItem(evidenceMap, "security-secret-handling", { occurrenceCount: securitySecretFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: securitySecretFiles });
  addEvidenceItem(evidenceMap, "destructive-api-contract", { occurrenceCount: destructiveApiFiles.length, scope: "public", confidence: "structural", sampleFiles: destructiveApiFiles });
  addEvidenceItem(evidenceMap, "additive-api-contract", { occurrenceCount: additiveApiFiles.length, scope: "public", confidence: "structural", sampleFiles: additiveApiFiles });
  addEvidenceItem(evidenceMap, "destructive-schema-contract", { occurrenceCount: destructiveSchemaFiles.length, scope: "public", confidence: "structural", sampleFiles: destructiveSchemaFiles });
  addEvidenceItem(evidenceMap, "additive-schema-contract", { occurrenceCount: additiveSchemaFiles.length, scope: "public", confidence: "structural", sampleFiles: additiveSchemaFiles });
  addEvidenceItem(evidenceMap, "destructive-database-change", { occurrenceCount: destructiveDatabaseFiles.length, scope: "public", confidence: "corroborated", sampleFiles: destructiveDatabaseFiles });
  addEvidenceItem(evidenceMap, "additive-database-change", { occurrenceCount: additiveDatabaseFiles.length, scope: "subsystem", confidence: "corroborated", sampleFiles: additiveDatabaseFiles });
  addEvidenceItem(evidenceMap, "explicit-migration-marker", { occurrenceCount: migrationFiles.length, scope: "public", confidence: "corroborated", sampleFiles: migrationFiles });
  addEvidenceItem(evidenceMap, "compatibility-drop", { occurrenceCount: compatibilityDropFiles.length, scope: "public", confidence: "corroborated", sampleFiles: compatibilityDropFiles });
  addEvidenceItem(evidenceMap, "compatibility-shim", { occurrenceCount: compatibilityShimFiles.length, scope: "public", confidence: "corroborated", sampleFiles: compatibilityShimFiles });
  addEvidenceItem(evidenceMap, "feature-flag-added", { occurrenceCount: featureFlagFiles.length, scope: "public", confidence: "corroborated", sampleFiles: featureFlagFiles });
  addEvidenceItem(evidenceMap, "feature-flag-contract-change", { occurrenceCount: featureFlagContractFiles.length, scope: "public", confidence: "corroborated", sampleFiles: featureFlagContractFiles });
  addEvidenceItem(evidenceMap, "runtime-support-added", { occurrenceCount: runtimeSupportAddedFiles.length, scope: "public", confidence: "corroborated", sampleFiles: runtimeSupportAddedFiles });
  addEvidenceItem(evidenceMap, "runtime-support-dropped", { occurrenceCount: runtimeSupportDroppedFiles.length, scope: "public", confidence: "corroborated", sampleFiles: runtimeSupportDroppedFiles });
  addEvidenceItem(evidenceMap, "runtime-policy-change", { occurrenceCount: runtimePolicyFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: runtimePolicyFiles });
  addEvidenceItem(evidenceMap, "performance-optimization", { occurrenceCount: performanceFiles.length, scope: "subsystem", confidence: "corroborated", sampleFiles: performanceFiles });
  addEvidenceItem(evidenceMap, "cleanup-removal", { occurrenceCount: cleanupFiles.length, scope: "subsystem", confidence: "structural", sampleFiles: cleanupFiles });
  addEvidenceItem(evidenceMap, "removed-public-export", { occurrenceCount: publicRemovedFiles.length, scope: "public", confidence: "structural", sampleFiles: publicRemovedFiles });
  addEvidenceItem(evidenceMap, "renamed-public-module", { occurrenceCount: publicRenamedFiles.length, scope: "public", confidence: "structural", sampleFiles: publicRenamedFiles });
  addEvidenceItem(evidenceMap, "added-public-capability", { occurrenceCount: publicCapabilityFiles.length, scope: "public", confidence: "structural", sampleFiles: publicCapabilityFiles });

  if (context.capabilityExpansionSignal) {
    addEvidenceItem(evidenceMap, "feature-capability-added", {
      occurrenceCount: Math.max(context.behavioralSurfaceAdditions, behavioralAdditionFiles.length, 1),
      scope: behavioralAdditionFiles.some((filename) => /(^|\/)(ui|components?|views?|templates|frontend)(\/|$)/.test(String(filename).toLowerCase())) ? "public" : "subsystem",
      confidence: "corroborated",
      sampleFiles: behavioralAdditionFiles
    });
  }

  return finalizeEvidenceItems(evidenceMap);
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

function analyzePullRequestSnapshotData(snapshot) {
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
    fileWithContext.signals.forEach(signal => signalSet.add(signal));
  }

  const behavioralSurfaceAdditions = countBehavioralSurfaceAdditions(filesWithContext);
  const publicContractMoves = countPublicContractMoves(filesWithContext);
  const capabilityExpansionSignal = hasCapabilityExpansionSignal({
    behavioralSurfaceAdditions,
    totalAdditions,
    totalChanges
  });
  const structuralPublicBreakingSignal = hasStructuralPublicBreakingSignal({
    behavioralSurfaceAdditions,
    publicContractMoves,
    filesWithContext
  });
  if (capabilityExpansionSignal) {
    signalSet.add("enhancement");
  }
  if (structuralPublicBreakingSignal) {
    signalSet.add("breaking-change");
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
  const maintenanceOnlyEligible = filesWithContext.length > 0 && [...categoryCounts.keys()].every((category) => MAINTENANCE_ONLY_CATEGORIES.has(category)) && !VERSION_CRITICAL_LABELS.some((label) => signalSet.has(label));
  const evidenceItems = buildDeterministicEvidence(filesWithContext, {
    behavioralSurfaceAdditions,
    capabilityExpansionSignal,
    structuralPublicBreakingSignal
  });
  const scoring = scoreDeterministicEvidence({
    evidenceItems,
    options: { maxLabels: MAX_AUTOBOT_LABELS }
  });
  const scorerEmittedLabels = scoring.emittedLabels.map((entry) => entry.label);
  const scorerRetainedLabels = rankScoredLabels(scoring.labelScores, "retained");
  const primaryScorerLabels = scoring.primaryLabels.map((entry) => entry.label);

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
  if (maintenanceOnlyEligible) {
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
  if (!maintenanceOnlyEligible) {
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

  const deterministicLabels = mergeRankedLabels(
    scorerEmittedLabels,
    sortLabels([...deterministicLabelSet]),
    MAX_AUTOBOT_LABELS
  );
  const candidateLabels = mergeRankedLabels(
    scorerRetainedLabels,
    sortLabels([...candidateLabelSet]),
    18
  );

  function buildDeterministicSummary() {
    const whatChangedSentences = maintenanceOnlyEligible
      ? [
          `This pull request changes ${filesWithContext.length} files (+${totalAdditions} -${totalDeletions}) and stays within narrow maintenance surfaces: ${maintenanceCategoriesList.join(", ") || "none"}.`,
          `The highest-signal areas are ${topDirectories.join(", ") || "(root)"}, and the deterministic scan did not find API, schema, database, runtime, security, or breaking-change signals.`
        ]
      : [
          `This pull request changes ${filesWithContext.length} files (+${totalAdditions} -${totalDeletions}) across ${topDirectories.join(", ") || "(root)"} with category mix ${categorySummary || "(none)"}.`,
          capabilityExpansionSignal
            ? `Added behavioral source or UI surfaces were detected across ${behavioralSurfaceAdditions} path${behavioralSurfaceAdditions === 1 ? "" : "s"}.`
            : null,
          structuralPublicBreakingSignal
            ? `Public package import surfaces were renamed or removed across ${publicContractMoves} path${publicContractMoves === 1 ? "" : "s"}, so consumer updates may be required.`
            : null,
          orderedSignals.length > 0
            ? `Deterministic high-signal areas include ${orderedSignals.join(", ")}.`
            : "The deterministic scan did not find strong release-critical signals, so any deeper classification depends on the selected diff windows or batch analyses."
        ].filter(Boolean);
    const releaseBullets = maintenanceOnlyEligible
      ? ["Likely not release-relevant because the diff is limited to narrow maintenance areas."]
      : orderedSignals.length > 0
        ? [
            `Release-relevant signals detected: ${orderedSignals.join(", ")}.`,
            structuralPublicBreakingSignal
              ? "Structural public package moves indicate a likely breaking import or integration surface."
              : null,
            `Highest-signal files: ${topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "(none)"}.`
          ].filter(Boolean)
        : ["No direct version-critical path signals were detected by the deterministic scan."];
    const riskBullets = maintenanceOnlyEligible
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
    if (behavioralSurfaceAdditions > 0) classificationSignals.push(`Added behavioral source or UI files: ${behavioralSurfaceAdditions}.`);
    if (publicContractMoves > 0) classificationSignals.push(`Renamed or removed public package paths: ${publicContractMoves}.`);
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
    `Scored semver decision: ${scoring.semver.decision}`,
    `Scored emitted labels: ${scorerEmittedLabels.join(", ") || "(none)"}`,
    `Scored primary labels: ${primaryScorerLabels.join(", ") || "(none)"}`,
    `Evidence items: ${evidenceItems.length}`,
    `Files changed: ${filesWithContext.length}`,
    `Additions: ${totalAdditions}`,
    `Deletions: ${totalDeletions}`,
    `Category mix: ${categorySummary || "(none)"}`,
    `Top directories: ${topDirectories.join(", ") || "(root)"}`,
    `Candidate labels: ${candidateLabels.join(", ") || "(none)"}`,
    orderedSignals.length > 0 ? `Deterministic signals: ${orderedSignals.join(", ")}` : "Deterministic signals: (none)",
    "",
    "Top files:",
    formatBulletLines(topFiles.map((file) => `${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})`), "(none)")
  ].join("\n");

  return {
    has_changes: filesWithContext.length > 0 ? "true" : "false",
    files_changed: String(filesWithContext.length),
    additions: String(totalAdditions),
    deletions: String(totalDeletions),
    candidate_labels_json: JSON.stringify(candidateLabels),
    deterministic_labels_json: JSON.stringify(deterministicLabels),
    deterministic_primary_labels_json: JSON.stringify(primaryScorerLabels),
    deterministic_evidence_json: JSON.stringify(evidenceItems),
    deterministic_evidence_count: String(evidenceItems.length),
    deterministic_category_scores_json: JSON.stringify(scoring.categoryScores),
    deterministic_impact_scores_json: JSON.stringify(scoring.impactScores),
    deterministic_label_scores_json: JSON.stringify(scoring.labelScores),
    deterministic_semver_json: JSON.stringify(scoring.semver),
    deterministic_semver_decision: String(scoring.semver.decision),
    deterministic_summary: deterministicSummary,
    evidence_chars: String(evidenceScaffold.length)
  };
}

function analyzePullRequestSnapshot() {
  return analyzePullRequestSnapshotData(readJson(SNAPSHOT_FILE));
}

module.exports = {
  SNAPSHOT_FILE,
  analyzePullRequestSnapshot,
  analyzePullRequestSnapshotData,
  collectPullRequestSnapshot
};
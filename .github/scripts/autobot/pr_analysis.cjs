const fs = require("fs");

const { scoreDeterministicEvidence } = require("./measurement/scorer.cjs");
const {
  applyLabelWordBudget: applyLabelWordBudgetFromMeasurement,
  applyPrLabelPolicy: applyPrLabelPolicyFromMeasurement,
  deriveGenericMaintenanceLabelLimit: deriveGenericMaintenanceLabelLimitFromMeasurement,
  derivePrLabelBudget: derivePrLabelBudgetFromMeasurement,
  isSmallPullRequest: isSmallPullRequestFromMeasurement,
  mergeRankedLabels: mergeRankedLabelsFromMeasurement,
  rankScoredLabels: rankScoredLabelsFromMeasurement,
  selectDeterministicLabels: selectDeterministicLabelsFromMeasurement
} = require("./measurement/label_selection.cjs");
const {
  buildLabelRationaleLines,
  buildPrDeterministicSummary,
  collectDirectEvidenceForLabel,
  collectSupportFilesForLabel,
  describeEvidenceItem,
  formatBulletLines
} = require("./phrasing/pr_summary.cjs");
const {
  collectPullRequestSnapshot: collectPullRequestSnapshotFromGatherInfo,
  normalizePatch: normalizePatchFromGatherInfo,
  resolveSnapshotFile: resolveSnapshotFileFromGatherInfo
} = require("./gather_info.cjs");
const { AutobotLabelRegistry, MAX_AUTOBOT_LABELS, sortLabels } = require("./labels.cjs");
const { LABEL_SUPPORT_PATTERNS } = require("./constants.cjs");

const SNAPSHOT_FILE = "/tmp/autobot_pr_snapshot.json";
const MAX_TOP_DIRECTORIES = 8;
const MAX_TOP_FILES = 10;
const MIN_BEHAVIORAL_ADDITION_LINES = 80;
const MIN_PUBLIC_CONTRACT_MOVES = 3;
const MAINTENANCE_ONLY_CATEGORIES = new Set(["documentation", "test", "workflow", "github", "config", "dependencies"]);
const PR_ABSTRACT_TYPE_LABELS = new Set(["bug", "enhancement", "improvement", "proposal"]);
const PR_GENERIC_MAINTENANCE_LABELS = Object.freeze(["cleanup", "config", "dependencies", "documentation", "test", "tooling"]);
const PR_INFRASTRUCTURE_LABELS = Object.freeze(["automation", "github", "workflow"]);
const PR_SUPPRESSED_BY_TECHNICAL_LABELS = new Set([
  "automation",
  "build",
  "ci",
  "config",
  "dependencies",
  "devcontainer",
  "docker",
  "docs-api",
  "docs-site",
  "documentation",
  "dx",
  "error-handling",
  "examples",
  "formatting",
  "github",
  "license",
  "lint",
  "packaging",
  "policy",
  "quality",
  "refactor",
  "release",
  "release-notes",
  "stability",
  "style",
  "test",
  "tooling",
  "validation",
  "versioning",
  "workflow"
]);
const PR_PIPELINE_CLUTTER_LABELS = new Set([
  "action pin",
  "artifact upload",
  "branch filter",
  "build job",
  "cache restore",
  "github actions",
  "os matrix",
  "python matrix"
]);
const ACCESSIBILITY_TEXT_PATTERN = /\baria-|accessib|a11y|screen reader|keyboard nav/;
const AUTOMATION_TEXT_PATTERN = /\b(autobot|automation|label|triage|milestone|release)\b/;
const FEATURE_FLAG_TEXT_PATTERN = /feature[\s-]?flag|kill switch|rollout/;
const LOCALIZATION_TEXT_PATTERN = /\bi18n\b|\bl10n\b|\blocale\b|\btranslations?\b|\bgettext\b/;
const PACKAGE_JSON_DEPENDENCY_CONTEXT_PATTERNS = Object.freeze([/^"(dependencies|devDependencies|peerDependencies|optionalDependencies|bundledDependencies|bundleDependencies)"\s*:\s*\{,?$/i]);
const PACKAGE_JSON_DEPENDENCY_METADATA_KEYS = new Set(["browser", "description", "directories", "engines", "exports", "keywords", "main", "module", "name", "overrides", "packageManager", "repository", "resolutions", "scripts", "type", "version", "volta"]);
const PACKAGE_JSON_DEPENDENCY_VALUE_PATTERN = /^(workspace:|file:|link:|npm:|github:|git\+|https?:|~|\^|>=?|<=?|=|\*|\d)/i;
const PYPROJECT_DEPENDENCY_CONTEXT_PATTERNS = Object.freeze([
  /^\[(project\.optional-dependencies|dependency-groups|tool\.poetry\.dependencies|tool\.poetry\.group\.[^\]]+\.dependencies|tool\.pdm(?:\.[^\]]+)?\.(?:dependencies|dev-dependencies)|build-system)\]$/i,
  /^(dependencies|optional-dependencies|requires)\s*=\s*[\[{]/i
]);
const PYPROJECT_DIRECT_REQUIREMENT_PATTERN = /^"[a-z0-9][a-z0-9._-]*(?:\[[^"\]]+\])?(?:\s*(?:[<>=!~]{1,2}|===|@).+)"[,]?$/i;
const PYPROJECT_SECTION_ENTRY_PATTERNS = Object.freeze([
  /^"[a-z0-9][a-z0-9._-]*(?:\[[^"\]]+\])?(?:\s*(?:[<>=!~]{1,2}|===|@).*)?"[,]?$/i,
  /^[a-z0-9._-]+\s*=\s*"(?:\^|~|>=?|<=?|==|!=|===|\*|file:|path:|git\+|https?:|ssh:).+"[,]?$/i,
  /^[a-z0-9._-]+\s*=\s*\[/i
]);
const RUNTIME_SUPPORT_TEXT_PATTERN = /\brequires-python\b|\bpython 3\.\d+\b|\bcuda\b|\bffmpeg\b|\bubuntu\b|\bwindows\b|\blinux\b|\bmacos\b|\bnvidia\b|\bplatform_system\b/;
const SECURITY_SECRET_TEXT_PATTERN = /authorization:\s*bearer|\b(access token|bearer token|secret|secrets|credential|permissions?)\b/;
const GENERAL_LABEL_REPLACEMENT_COUNTS = Object.freeze({
  moderatelySpecific: 5,
  verySpecific: 15
});
const GENERAL_LABEL_MIN_DESCENDANTS = GENERAL_LABEL_REPLACEMENT_COUNTS.moderatelySpecific + GENERAL_LABEL_REPLACEMENT_COUNTS.verySpecific;
const LOW_CONFIDENCE_LABEL = "low";
const MAX_LABEL_WORDS = 2;
const LOW_SPECIFICITY_TECHNICAL_LABELS = new Set([
  "filesystem",
  "linux",
  "macos",
  "os",
  "platform",
  "process",
  "python",
  "runtime",
  "windows"
]);
const TECHNICAL_BROAD_LABEL_DESCENDANT_THRESHOLD = 6;
const PATH_NORMALIZATION_TEXT_PATTERN = /\b(path normalization|normalize(?:d|r|s)? path|normpath|realpath|resolve\(|abspath|joinpath|ntpath|posixpath|path separator|filepath|pathlib)\b/;
const TEMP_DIRECTORY_TEXT_PATTERN = /\b(temp(?:orary)?(?:\s+directory|\s+dir)?|tempfile|mkdtemp|temp dir|tmp dir|temp path)\b/;
const ATOMIC_WRITE_TEXT_PATTERN = /\b(atomic(?:\s+write)?|os\.replace|fsync|write\s+tmp|rename\()\b/;
const SYMLINK_SAFETY_TEXT_PATTERN = /\b(symlink|readlink|lstat|realpath)\b/;
const SUBPROCESS_IO_TEXT_PATTERN = /\b(subprocess|child_process|popen|communicate|stdin|stdout|stderr|pipe)\b/;
const WORKER_POOL_TEXT_PATTERN = /\b(worker pool|thread ?pool|process ?pool|executor)\b/;
const RETRY_BUDGET_TEXT_PATTERN = /\b(retry budget|retry(?:ing)?|backoff|max[_\s-]?attempts?)\b/;
const DAEMON_MODE_TEXT_PATTERN = /\b(daemon(?:ize|ized)?|detached process|daemon mode)\b/;

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function resolveSnapshotFile(input) {
  return resolveSnapshotFileFromGatherInfo(input);
}

function topDirectoryForFile(filename) {
  const parts = String(filename || "").split("/");
  parts.pop();
  return parts.length > 0 ? parts.slice(0, 2).join("/") : "(root)";
}

function normalizePatch(patch) {
  return normalizePatchFromGatherInfo(patch);
}

function scoreFile(file) {
  const changeVolume = Number(file.additions || 0) + Number(file.deletions || 0);
  const patchWeight = file.patch ? Math.min(file.patch.length, 4000) : 0;
  const structuralWeight = file.status === "renamed" || file.status === "removed" ? 800 : 0;
  return changeVolume * 5 + patchWeight + structuralWeight;
}

function splitPatchLines(patch) {
  return String(patch || "").split("\n");
}

function stripDiffLinePrefix(line) {
  if (/^[ +-]/.test(line)) {
    return line.slice(1);
  }
  return line;
}

function isChangedPatchLine(line) {
  return /^[+-]/.test(line) && !/^(\+\+\+|---) /.test(line);
}

function matchesAnyPattern(value, patterns) {
  return patterns.some((pattern) => pattern.test(value));
}

function isDependencyLockfilePath(normalizedPath) {
  return /^requirements.*\.txt$/.test(normalizedPath)
    || /(^|\/)(poetry\.lock|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|uv\.lock|pdm\.lock)$/.test(normalizedPath);
}

function hasPyprojectDependencyChange(patch) {
  const normalizedLines = splitPatchLines(patch)
    .map((line) => stripDiffLinePrefix(line).trim())
    .filter(Boolean);
  const changedLines = splitPatchLines(patch)
    .filter((line) => isChangedPatchLine(line))
    .map((line) => stripDiffLinePrefix(line).trim())
    .filter(Boolean);
  const hasDependencyContext = normalizedLines.some((line) => matchesAnyPattern(line, PYPROJECT_DEPENDENCY_CONTEXT_PATTERNS));
  return changedLines.some((line) => matchesAnyPattern(line, PYPROJECT_DEPENDENCY_CONTEXT_PATTERNS))
    || changedLines.some((line) => PYPROJECT_DIRECT_REQUIREMENT_PATTERN.test(line))
    || hasDependencyContext && changedLines.some((line) => matchesAnyPattern(line, PYPROJECT_SECTION_ENTRY_PATTERNS));
}

function hasPackageJsonDependencyChange(patch) {
  const normalizedLines = splitPatchLines(patch)
    .map((line) => stripDiffLinePrefix(line).trim())
    .filter(Boolean);
  const changedLines = splitPatchLines(patch)
    .filter((line) => isChangedPatchLine(line))
    .map((line) => stripDiffLinePrefix(line).trim())
    .filter(Boolean);
  const hasDependencyContext = normalizedLines.some((line) => matchesAnyPattern(line, PACKAGE_JSON_DEPENDENCY_CONTEXT_PATTERNS));
  return changedLines.some((line) => matchesAnyPattern(line, PACKAGE_JSON_DEPENDENCY_CONTEXT_PATTERNS))
    || hasDependencyContext && changedLines.some((line) => {
      const entryMatch = line.match(/^"([^"]+)"\s*:\s*"([^"]+)"[,]?$/);
      if (!entryMatch) {
        return false;
      }
      const [, key, value] = entryMatch;
      return !PACKAGE_JSON_DEPENDENCY_METADATA_KEYS.has(key) && PACKAGE_JSON_DEPENDENCY_VALUE_PATTERN.test(value);
    });
}

function hasDependencySurfaceChange(normalizedPath, patch) {
  if (isDependencyLockfilePath(normalizedPath)) {
    return true;
  }
  if (normalizedPath === "pyproject.toml") {
    return hasPyprojectDependencyChange(patch);
  }
  if (normalizedPath === "package.json") {
    return hasPackageJsonDependencyChange(patch);
  }
  return false;
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
  return /^\.github\/scripts\/(?:(?:autobot|smart_link)[^/]*|(?:autobot|smart_link)\/.+)\.(cjs|mjs|js|ts)$/.test(normalizedPath)
    || /^\.github\/workflows\/(autobot|smart-link)\.ya?ml$/.test(normalizedPath)
    || /^(test|tests)\/(?:(test_)?autobot[^/]*|smart_link[^/]*|test_smart_link[^/]*|deterministic_scenario_engine|javascript_test_harness)(\.test)?\.(cjs|mjs|js|ts|py)$/.test(normalizedPath);
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
  return /(^|\/)(api|apis|webhooks?)(\/|$)/.test(normalizedPath)
    || /(^|\/)(openapi|swagger)(\.[^/]+)?$/.test(normalizedPath);
}

function hasRouteParameterEvidence(text) {
  return /\b(route param|path param)\b/.test(text)
    || /\/:[a-z_][a-z0-9_]*/.test(text)
    || /\/\{[a-z_][a-z0-9_]*\}/.test(text)
    || /\/<[a-z0-9_:.-]+>/.test(text);
}

function hasQueryParameterEvidence(text) {
  return /\bquery param\b|\b(searchparams?|urlsearchparams)\b/.test(text)
    || /\?[a-z_][a-z0-9_]*=/.test(text);
}

function hasRuntimePathEvidence(normalizedPath) {
  return /^docker\//.test(normalizedPath)
    || /(^|\/)(dockerfile|compose\.ya?ml)$/.test(normalizedPath)
    || /(^|\/)(platform|runtime|system|cuda)(\/|$)/.test(normalizedPath)
    || /(^|\/)(pyproject\.toml|tox\.ini|setup\.(cfg|py))$/.test(normalizedPath);
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

function hasShellCommandEvidence(patch) {
  return /\bcmd(?:\.exe)?\s*\/[ck]\b/.test(patch)
    || /\bpowershell(?:\.exe)?\b/.test(patch)
    || /\/(?:bin\/)?(?:sh|bash)\b/.test(patch)
    || /\b(?:bash|sh)\s+-[cl]\b/.test(patch)
    || /^#!.*\b(?:bash|sh)\b/m.test(patch)
    || /\bshell command\b|\bshell script\b/.test(patch)
    || /\bshell\s*=\s*(?:true|false)\b/.test(patch);
}

function hasFilesystemPatchEvidence(patch) {
  return /\b(filesystem|file system|path separator|filepath|ntpath|posixpath)\b/.test(patch)
    || /\\\\/.test(patch);
}

function hasPerformancePatchEvidence(patch) {
  return /\b(performance|latency|throughput|optimiz(?:e|ation|ing)?|cold start|init time|cache(?: hit| key| invalidat|\b)|heap|peak memory|memory usage|memory footprint|memory pressure|buffer copy|buffer copies|leak(?: risk)?|retain(?:ed|ing)?|retention)\b/.test(patch)
    || hasImportTimeEvidence(patch);
}

function hasHeapUsageEvidence(patch) {
  return /\b(heap|peak memory|memory usage|memory footprint|memory pressure|buffer copy|buffer copies)\b/.test(patch);
}

function hasImportTimeEvidence(patch) {
  return /\bimport[-\s]+time\b/.test(patch)
    && /\b(duration|latency|measure(?:d|ment)?|benchmark|startup|cold start|ms)\b/.test(patch);
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
  return hasApiPathEvidence(normalizedPath);
}

function hasVulnerabilityEvidence(text) {
  return /\b(vulnerab\w*|advisory|cve-\d{4}-\d+|ghsa-[a-z0-9-]+|exploit(?:able)?|security advisory)\b/.test(text);
}

function hasComplianceEvidence(text) {
  return /\b(compliance|soc ?2|pci(?:-dss)?|gdpr|hipaa|iso ?27001|nist|fedramp)\b/.test(text)
    || /\bsarbanes[-\s]+oxley\b/.test(text)
    || /\bsox\s+(?:compliance|controls?|control matrix|attestation|audit)\b/.test(text);
}

function hasHardeningEvidence(text) {
  return /\b(hardening|harden|least privilege|defense in depth|deny by default|sandbox|mitigation|sanitize|sanitiz)\b/.test(text);
}

function hasPenTestEvidence(text) {
  return /\b(pen[\s-]?test|penetration test|red team)\b/.test(text);
}

function hasSecurityEvidence(normalizedPath, patch) {
  const text = `${normalizedPath}\n${patch}`;
  return /(^|\/)(security|auth|policy)(\/|$)/.test(normalizedPath)
    || /(^|\/)(codeql|dependabot|security)(\.[^/]+)?$/.test(normalizedPath)
    || /authorization:\s*bearer/.test(patch)
    || /permissions:\s*(\{|\n)/.test(patch)
    || /\b(access token|bearer token|secret|secrets|credential|sanitize|sanitiz|csrf|xss|auth(?:entication|orization)?|least privilege|permissions?)\b/.test(patch)
    || hasVulnerabilityEvidence(text)
    || hasComplianceEvidence(text)
    || hasHardeningEvidence(text)
    || hasPenTestEvidence(text);
}

function hasBreakingChangeEvidence(patch) {
  return /\bbreaking[\s-]?change\b|\bbackward[\s-]?incompatible\b|\bincompatible api\b|\bconsumer adaptation\b|\brequires migration\b|\bmust update (callers|consumers)\b/.test(patch);
}

function deriveTitleSignals(prSignalText) {
  const bug = /\b(bug|fix|fixes|fixed|regression|hotfix|error|broken|failure|repair)\b/.test(prSignalText);
  const enhancement = !bug && /(^|\s)(feat|feature|enhancement)(:|\b)/.test(prSignalText);
  const documentation = /\b(doc|docs|documentation|readme)\b/.test(prSignalText);
  const security = /\b(security|vulnerab\w*|advisory|cve-\d{4}-\d+|ghsa-[a-z0-9-]+)\b/.test(prSignalText);
  const ui = /\b(ui|ux|gradio|frontend)\b/.test(prSignalText);
  return {
    bug,
    enhancement,
    documentation,
    security,
    ui
  };
}

function classifyFile(file) {
  const normalized = typeof file === "string"
    ? String(file || "").toLowerCase()
    : String(file?.filename || "").toLowerCase();
  const patch = typeof file === "string" ? "" : String(file?.patch || "");
  const categories = new Set();
  if (matchesWorkflowFilePath(normalized) || matchesWorkflowScriptPath(normalized)) {
    categories.add("workflow");
  }
  if (/^\.github\//.test(normalized)) {
    categories.add("github");
  }
  if (
    /^docs\//.test(normalized)
    || /(^|\/)(readme|contributing|changelog|license)(\.[^/]+)?$/.test(normalized)
    || /\.(md|mdx|rst|txt)$/.test(normalized)
  ) {
    categories.add("documentation");
  }
  if (
    /^(test|tests)\//.test(normalized)
    || /(^|\/)test_[^/]+\.py$/.test(normalized)
    || /\.(spec|test)\.(js|jsx|ts|tsx|py|cjs|mjs|cts|mts)$/.test(normalized)
  ) {
    categories.add("test");
  }
  if (matchesUiPath(normalized)) {
    categories.add("ui");
  }
  if (
    /^docker\//.test(normalized)
    || /(^|\/)dockerfile$/.test(normalized)
    || /compose\.ya?ml$/.test(normalized)
  ) {
    categories.add("docker");
  }
  if (/^scripts\//.test(normalized) || /^\.vscode\//.test(normalized)) {
    categories.add("tooling");
  }
  if (hasDependencySurfaceChange(normalized, patch)) {
    categories.add("dependencies");
  }
  if (
    /^pyproject\.toml$/.test(normalized)
    || /^manifest\.in$/.test(normalized)
    || /^tox\.ini$/.test(normalized)
    || /^setup\.(cfg|py)$/.test(normalized)
    || /^ruff\.toml$/.test(normalized)
    || /^\.editorconfig$/.test(normalized)
    || /^\.pre-commit-config\.(yaml|yml)$/.test(normalized)
    || /\.(toml|ya?ml|json|ini|cfg)$/.test(normalized)
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
  return /^src\/definers\/(?:__init__\.py|[^/]+\.py|[^/]+\/__init__\.py|[^/]+\/[^/]+\.py)$/.test(normalizedPath);
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

function isRuntimeSupportContractPath(normalizedPath) {
  return isPublicPackageContractPath(normalizedPath)
    || /^pyproject\.toml$/.test(normalizedPath)
    || /^setup\.(cfg|py)$/.test(normalizedPath)
    || /^tox\.ini$/.test(normalizedPath)
    || /^docker\//.test(normalizedPath)
    || /(^|\/)(dockerfile|compose\.ya?ml)$/.test(normalizedPath)
    || /(^|\/)(platform|runtime|system|cuda)(\/|$)/.test(normalizedPath);
}

function isExplicitPublicCapabilityFile(file) {
  if (file.status !== "added" || !isBehavioralSurfaceFile(file)) {
    return false;
  }
  const normalizedPath = String(file.filename || "").toLowerCase();
  const patch = String(file.patch || "").toLowerCase();
  return /(^|\/)__init__\.py$/.test(normalizedPath)
    || matchesUiPath(normalizedPath)
    || hasUiTextEvidence(patch)
    || hasApiEvidence(normalizedPath, patch)
    || hasSchemaEvidence(normalizedPath, patch)
    || FEATURE_FLAG_TEXT_PATTERN.test(patch);
}

function hasPublicFacingCapabilityAddition(filesWithContext) {
  return filesWithContext.some((file) => isExplicitPublicCapabilityFile(file));
}

function hasCapabilityExpansionSignal({ behavioralSurfaceAdditions, filesWithContext, totalAdditions, totalChanges }) {
  if (!hasPublicFacingCapabilityAddition(filesWithContext)) {
    return false;
  }
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
  const categories = new Set(file.categories || classifyFile(file));
  const allowsVersionCriticalTextSignals = !categories.has("documentation") && !categories.has("test");
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
  if (allowsVersionCriticalTextSignals && hasSchemaEvidence(normalizedPath, patch)) {
    signals.add("schema");
  }
  if (allowsVersionCriticalTextSignals && hasMigrationEvidence(normalizedPath, patch)) {
    signals.add("migration");
  }
  if (allowsVersionCriticalTextSignals && hasDatabaseEvidence(normalizedPath, patch)) {
    signals.add("database");
  }
  if (allowsVersionCriticalTextSignals && hasApiEvidence(normalizedPath, patch)) {
    signals.add("api");
  }
  if (allowsVersionCriticalTextSignals && hasSecurityEvidence(normalizedPath, patch)) {
    signals.add("security");
  }
  if (hasBreakingChangeEvidence(patch)) {
    signals.add("breaking-change");
  }
  if (allowsVersionCriticalTextSignals && (
    /(^|\/)(compat|compatibility|interop|polyfill|shim)(\/|$)/.test(normalizedPath)
    || /\b(backward compatibility|compatibility|interop|polyfill|shim)\b/.test(patch)
  )) {
    signals.add("compatibility");
  }
  if (allowsVersionCriticalTextSignals && /feature[\s-]?flag|kill switch|rollout/.test(text)) {
    signals.add("feature-flag");
  }
  if (allowsVersionCriticalTextSignals && hasRuntimePathEvidence(normalizedPath) && hasRuntimePatchEvidence(patch)) {
    signals.add("runtime");
  }
  if (hasPerformancePatchEvidence(patch)) {
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

function isTestPath(normalizedPath) {
  return /^(test|tests)\//.test(normalizedPath)
    || /(^|\/)test_[^/]+\.py$/.test(normalizedPath)
    || /\.(spec|test)\.(js|jsx|ts|tsx|py|cjs|mjs|cts|mts)$/.test(normalizedPath);
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

function addTechnicalLabel(labels, label) {
  if (label) {
    labels.add(label);
  }
}

function countLabelWords(label) {
  const normalizedLabel = String(label || "").trim().replace(/[_-]+/g, " ");
  if (!normalizedLabel) {
    return 0;
  }
  return normalizedLabel.split(/\s+/).filter(Boolean).length;
}

function addSpecificFilesystemLabels(technicalLabels, text) {
  if (PATH_NORMALIZATION_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "path normalization");
  }
  if (TEMP_DIRECTORY_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "temp directory");
  }
  if (ATOMIC_WRITE_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "atomic write");
  }
  if (SYMLINK_SAFETY_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "symlink safety");
  }
}

function addSpecificProcessLabels(technicalLabels, text) {
  if (SUBPROCESS_IO_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "subprocess io");
  }
  if (WORKER_POOL_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "worker pool");
  }
  if (RETRY_BUDGET_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "retry budget");
  }
  if (DAEMON_MODE_TEXT_PATTERN.test(text)) {
    addTechnicalLabel(technicalLabels, "daemon mode");
  }
}

function getLabelPrecisionMetrics(label, context = {}) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  const scoreEntry = context.technicalLabelScores?.[normalizedLabel] || {};
  const directEvidence = collectDirectEvidenceForLabel(normalizedLabel, context.evidenceItems || []);
  const supportFiles = collectSupportFilesForLabel(normalizedLabel, context.filesWithContext || []);
  const metadata = AutobotLabelRegistry.getLabelMetadata(normalizedLabel);
  const depth = AutobotLabelRegistry.getLabelDepth(normalizedLabel);
  const descendantCount = collectTechnicalDescendantsWithDistance(normalizedLabel).length;
  const broadLabel = descendantCount >= TECHNICAL_BROAD_LABEL_DESCENDANT_THRESHOLD || LOW_SPECIFICITY_TECHNICAL_LABELS.has(normalizedLabel);
  const wordCount = countLabelWords(normalizedLabel);
  const directEvidenceStrength = Math.min(
    directEvidence.reduce((total, evidence) => total + Math.max(Number(evidence?.occurrenceCount) || 1, 1), 0),
    4
  ) / 4;
  const supportStrength = Math.min(supportFiles.length, 3) / 3;
  const scoreStrength = Math.max(Math.min(Number(scoreEntry?.score || 0), 1), 0);
  const depthBoost = Math.min(depth * 0.06, 0.2);
  const releaseBoost = metadata?.releaseRelevant ? 0.04 : 0;
  const broadPenalty = broadLabel ? 0.14 : 0;
  const shallowPenalty = depth <= 1 ? 0.08 : 0;
  const wordPenalty = wordCount > MAX_LABEL_WORDS ? 0.08 : 0;
  const precisionScore = Math.max(
    Math.min(
      scoreStrength * 0.45
      + directEvidenceStrength * 0.3
      + supportStrength * 0.2
      + depthBoost
      + releaseBoost
      - broadPenalty
      - shallowPenalty
      - wordPenalty,
      1
    ),
    0
  );
  const minimumPrecision = broadLabel ? 0.5 : 0.4;
  return {
    broadLabel,
    descendantCount,
    minimumPrecision,
    normalizedLabel,
    precisionScore,
    scoreEntry
  };
}

function deriveSpecificReplacementLabels(labelMetrics, context = {}, options = {}) {
  const maxWords = Math.max(Number(options.maxWords) || MAX_LABEL_WORDS, 1);
  const descendants = collectTechnicalDescendantsWithDistance(labelMetrics.normalizedLabel)
    .filter((candidate) => countLabelWords(candidate.label) <= maxWords)
    .map((candidate) => ({
      candidate,
      metrics: getLabelPrecisionMetrics(candidate.label, context)
    }))
    .filter((entry) => entry.metrics.precisionScore >= 0.45)
    .sort((left, right) => {
      if (right.metrics.precisionScore !== left.metrics.precisionScore) {
        return right.metrics.precisionScore - left.metrics.precisionScore;
      }
      return right.candidate.distance - left.candidate.distance || left.candidate.label.localeCompare(right.candidate.label);
    });
  return descendants.slice(0, 2).map((entry) => entry.metrics.normalizedLabel);
}

function enforcePrecisionEmissionParameters(labels, context = {}, options = {}) {
  const limit = Math.max(Number(options.limit) || MAX_AUTOBOT_LABELS, 1);
  const maxWords = Math.max(Number(options.maxWords) || MAX_LABEL_WORDS, 1);
  const normalizedLabels = applyLabelWordBudgetFromMeasurement(labels, { maxWords });
  const selectedLabels = [];

  for (const label of normalizedLabels) {
    const labelMetrics = getLabelPrecisionMetrics(label, context);
    const requiresRefinement = LOW_SPECIFICITY_TECHNICAL_LABELS.has(labelMetrics.normalizedLabel)
      || countLabelWords(labelMetrics.normalizedLabel) > maxWords;
    if (!requiresRefinement) {
      selectedLabels.push(labelMetrics.normalizedLabel);
      continue;
    }
    if (labelMetrics.precisionScore >= labelMetrics.minimumPrecision) {
      selectedLabels.push(labelMetrics.normalizedLabel);
      continue;
    }
    const replacements = deriveSpecificReplacementLabels(labelMetrics, context, { maxWords });
    if (replacements.length > 0) {
      selectedLabels.push(...replacements);
      continue;
    }
    if (labelMetrics.precisionScore >= 0.35) {
      selectedLabels.push(labelMetrics.normalizedLabel);
    }
  }

  const finalLabels = applyPrLabelPolicy(selectedLabels, { ...options, limit, maxWords });
  if (finalLabels.length > 0) {
    return finalLabels;
  }
  return applyPrLabelPolicy(normalizedLabels, { ...options, limit, maxWords });
}

function deriveTechnicalLabels(filesWithContext) {
  const technicalLabels = new Set();

  for (const file of filesWithContext) {
    const normalizedPath = String(file.filename || "").toLowerCase();
    const patch = String(file.patch || "").toLowerCase();
    const text = `${normalizedPath}\n${patch}`;
    const categories = new Set(file.categories || classifyFile(file));
    const signals = new Set(file.signals || deriveSignals(file));
    const autobotClassificationInfrastructure = isAutobotClassificationInfrastructure(normalizedPath);
    const destructive = isDestructiveFileChange(file);
    const documentationSurface = /^docs\//.test(normalizedPath)
      || /(^|\/)(readme|contributing|changelog)(\.[^/]+)?$/.test(normalizedPath)
      || /\.(md|mdx|rst|txt)$/.test(normalizedPath);
    const testSurface = isTestPath(normalizedPath);
    const runtimeContractSurface = isRuntimeSupportContractPath(normalizedPath);
    const runtimeTextSurface = hasRuntimePatchEvidence(patch) && !categories.has("dependencies") && !categories.has("workflow");
    const cemExcludedFromApiText = categories.has("workflow") || categories.has("github") || autobotClassificationInfrastructure || matchesWorkflowFilePath(normalizedPath);
    const apiSurface = hasApiPathEvidence(normalizedPath)
      || (!cemExcludedFromApiText && /\b(fastapi|flask|router|route|endpoint|webhook|graphql)\b/.test(text));

    if (matchesWorkflowFilePath(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "github actions");
      if (/strategy:\s*|matrix:/.test(patch)) addTechnicalLabel(technicalLabels, "matrix job");
      if (/python-version|python_version/.test(patch)) addTechnicalLabel(technicalLabels, "python matrix");
      if (/ubuntu|windows|macos/.test(patch)) addTechnicalLabel(technicalLabels, "os matrix");
      if (/branches:|branches-ignore:|tags:|tags-ignore:|paths:|paths-ignore:/.test(patch)) addTechnicalLabel(technicalLabels, "branch filter");
      if (/upload-artifact|download-artifact/.test(patch)) addTechnicalLabel(technicalLabels, "artifact upload");
      if (/actions\/cache|cache-hit|restore-keys|cache:/.test(patch)) addTechnicalLabel(technicalLabels, "cache restore");
      if (/uses:\s*actions\//.test(patch)) addTechnicalLabel(technicalLabels, "action pin");
    }

    if (matchesUiPath(normalizedPath) || hasUiTextEvidence(text)) {
      addTechnicalLabel(technicalLabels, "view");
      if (/\.(html|tsx|jsx|vue|svelte)$/.test(normalizedPath) || /<(button|form|input|select|dialog|label)\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "template");
      }
      if (/\b(button|form|input|select|dialog|label|textbox|dropdown|accordion|slider|checkbox|tabs?)\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "form control");
      }
      if (/class(name)?=|styles?|themes?|tokens?|\.css|\.scss|\.sass|\.less|color:/.test(text) || /\.(css|scss|sass|less)$/.test(normalizedPath)) {
        addTechnicalLabel(technicalLabels, "theme token");
      }
    }

    if (ACCESSIBILITY_TEXT_PATTERN.test(text)) {
      if (/keyboard|keybind|shortcut|tabindex/.test(text)) addTechnicalLabel(technicalLabels, "keyboard nav");
      if (/screen reader|aria-/.test(text)) addTechnicalLabel(technicalLabels, "screen reader");
      if (/focus/.test(text)) addTechnicalLabel(technicalLabels, "focus order");
    }

    if (LOCALIZATION_TEXT_PATTERN.test(text)) {
      addTechnicalLabel(technicalLabels, "localization");
      if (/translation|translations|gettext|catalog/.test(text)) addTechnicalLabel(technicalLabels, "translation catalog");
      if (/locale|message|string/.test(text)) addTechnicalLabel(technicalLabels, "locale string");
      if (/\brtl\b|right-to-left/.test(text)) addTechnicalLabel(technicalLabels, "rtl layout");
    }

    if (isPublicPackageContractPath(normalizedPath)) {
      const packageInitializer = /__init__\.py$/.test(normalizedPath);
      const topLevelModule = /^src\/definers\/[^/]+\.py$/.test(normalizedPath);
      const destructivePublicMove = ["removed", "renamed"].includes(file.status) || destructive;
      if (packageInitializer || topLevelModule || destructivePublicMove) addTechnicalLabel(technicalLabels, "public export");
      if (packageInitializer) addTechnicalLabel(technicalLabels, "facade module");
      if (destructivePublicMove) addTechnicalLabel(technicalLabels, "import path");
    }

    if (testSurface) {
      if (/integration|scenario|lane|replay|regression|harness/.test(text)) {
        addTechnicalLabel(technicalLabels, "integration test");
      } else {
        addTechnicalLabel(technicalLabels, "unit test");
      }
      if (/fixture/.test(text)) addTechnicalLabel(technicalLabels, "test fixture");
      if (/\bmock\b/.test(text)) addTechnicalLabel(technicalLabels, "mock setup");
      if (/scenario|lane/.test(text)) addTechnicalLabel(technicalLabels, "scenario lane");
      if (/replay|seed/.test(text)) addTechnicalLabel(technicalLabels, "replay seed");
      if (/smoke/.test(text)) addTechnicalLabel(technicalLabels, "smoke test");
    }

    if (documentationSurface) {
      if (/^readme(\.[^/]+)?$/.test(normalizedPath) || /\/readme(\.[^/]+)?$/.test(normalizedPath)) {
        addTechnicalLabel(technicalLabels, "readme");
      }
      if (/changelog/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "changelog");
      if (/docs\/reference\//.test(normalizedPath) || /openapi|swagger|api reference/.test(text)) addTechnicalLabel(technicalLabels, "api doc");
      if (/docs\/(reference|capabilities|runtime)\//.test(normalizedPath) || /architecture|module map|data flow/.test(text)) {
        addTechnicalLabel(technicalLabels, "architecture doc");
      }
      if (/getting-started|install/.test(normalizedPath)) {
        addTechnicalLabel(technicalLabels, "install guide");
        if (/getting-started|quickstart/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "quickstart");
        if (/environment|env /.test(text)) addTechnicalLabel(technicalLabels, "env setup");
      }
      if (/troubleshoot|troubleshooting|faq/.test(normalizedPath) || /\bfaq\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "troubleshoot guide");
        if (/\berror\b/.test(text)) addTechnicalLabel(technicalLabels, "error guide");
        if (/\bfaq\b/.test(text)) addTechnicalLabel(technicalLabels, "faq");
      }
    }

    if (
      signals.has("compatibility")
      && !documentationSurface
      && !/^(test|tests)\//.test(normalizedPath)
      && !/\.(spec|test)\.(js|jsx|ts|tsx|py|cjs|mjs|cts|mts)$/.test(normalizedPath)
      && !destructive
    ) {
      addTechnicalLabel(technicalLabels, "shim module");
    }

    if (/codeowners$/i.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "codeowners");
      if (/review/.test(text)) addTechnicalLabel(technicalLabels, "review rule");
      if (/\*/.test(patch) || /src\//.test(patch)) addTechnicalLabel(technicalLabels, "path ownership");
    }
    if (/\.github\/(issue_template|issue-?templates?)\//.test(normalizedPath) || /issue forms?/.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "issue form");
      if (/bug/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "bug form");
      if (/feature|enhancement/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "feature form");
    }
    if (/pull[_-]?request[_-]?template|pull template/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "pull template");
    if (/dependabot/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "dependabot");

    if (normalizedPath === "pyproject.toml") {
      addTechnicalLabel(technicalLabels, "pyproject");
      if (hasPyprojectDependencyChange(patch)) addTechnicalLabel(technicalLabels, "dependency group");
      if (/scripts|task|tool\.poetry\.scripts|project\.scripts/.test(text)) addTechnicalLabel(technicalLabels, "task runner");
      if (/requires-python|python 3\./.test(text)) {
        addTechnicalLabel(technicalLabels, "python version");
        addTechnicalLabel(technicalLabels, "support matrix");
        if (/==|~=|\^/.test(patch)) addTechnicalLabel(technicalLabels, "version pin");
      }
    }
    if (normalizedPath === "package.json") {
      addTechnicalLabel(technicalLabels, "package manifest");
      if (hasPackageJsonDependencyChange(patch)) addTechnicalLabel(technicalLabels, "dependency group");
      if (/"scripts"\s*:/.test(text)) addTechnicalLabel(technicalLabels, "task runner");
    }
    if (/poetry\.lock$/.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "lockfile");
      addTechnicalLabel(technicalLabels, "poetry lock");
    }
    if (/package-lock\.json$/.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "lockfile");
      addTechnicalLabel(technicalLabels, "package lock");
    }
    if (/pnpm-lock\.yaml$|yarn\.lock$|uv\.lock$|pdm\.lock$|requirements.*\.txt$/.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "lockfile");
    }

    if (/dockerfile$/.test(normalizedPath)) {
      addTechnicalLabel(technicalLabels, "dockerfile");
      addTechnicalLabel(technicalLabels, "image");
      if (/^\+.*from .*:/m.test(patch) || /image:/.test(patch)) addTechnicalLabel(technicalLabels, "image tag");
      if (/cache-from|cache-to|buildx/.test(patch)) addTechnicalLabel(technicalLabels, "layer cache");
    }
    if (/compose\.ya?ml$/.test(normalizedPath)) addTechnicalLabel(technicalLabels, "compose");
    if (/^docker\//.test(normalizedPath)) addTechnicalLabel(technicalLabels, "image");
    if (matchesWorkflowFilePath(normalizedPath) && /build|publish|release|package/.test(text)) {
      addTechnicalLabel(technicalLabels, "build job");
      if (/wheel|pypi|python -m build/.test(text)) addTechnicalLabel(technicalLabels, "wheel");
      if (/docker|image/.test(text)) addTechnicalLabel(technicalLabels, "image");
    }
    if (/deploy|staging|prod|production/.test(normalizedPath) || /\bstaging\b|\bproduction\b|\bprod\b|rollback|rollout|smoke/.test(patch)) {
      addTechnicalLabel(technicalLabels, "deploy job");
      if (/staging/.test(text)) addTechnicalLabel(technicalLabels, "staging deploy");
      if (/production|prod/.test(text)) addTechnicalLabel(technicalLabels, "prod deploy");
      if (/rollout|gate/.test(text)) addTechnicalLabel(technicalLabels, "rollout gate");
      if (/smoke/.test(text)) addTechnicalLabel(technicalLabels, "smoke check");
      if (/rollback/.test(text)) addTechnicalLabel(technicalLabels, "rollback hook");
    }

    if (apiSurface) {
      if (/graphql|\.gql$|\.graphql$/.test(text)) {
        addTechnicalLabel(technicalLabels, "graphql");
        if (/resolver/.test(text)) addTechnicalLabel(technicalLabels, "resolver");
        if (/type\s+\w+|extend type|input\s+\w+/.test(text)) addTechnicalLabel(technicalLabels, "schema field");
      } else if (/webhook/.test(text)) {
        addTechnicalLabel(technicalLabels, "webhook");
        if (/payload|event/.test(text)) addTechnicalLabel(technicalLabels, "webhook payload");
        if (/retry/.test(text)) addTechnicalLabel(technicalLabels, "webhook retry");
      } else {
        addTechnicalLabel(technicalLabels, "route");
        if (hasRouteParameterEvidence(patch)) addTechnicalLabel(technicalLabels, "route param");
        if (hasQueryParameterEvidence(patch)) addTechnicalLabel(technicalLabels, "query param");
        if (/\brest\b|\bhttp\b/.test(text)) addTechnicalLabel(technicalLabels, "rest");
      }
    }
    if (apiSurface && /openapi|swagger/.test(text)) addTechnicalLabel(technicalLabels, "openapi spec");
    if (apiSurface && /request body|request_model|request schema|payload/.test(text)) addTechnicalLabel(technicalLabels, "request body");
    if (apiSurface && /response body|response_model|serializer|json response/.test(text)) addTechnicalLabel(technicalLabels, "response body");
    if (apiSurface && /validate|validator|pydantic/.test(text)) addTechnicalLabel(technicalLabels, "body validation");
    if (apiSurface && /input schema/.test(text)) addTechnicalLabel(technicalLabels, "input schema");
    if (apiSurface && /output schema/.test(text)) addTechnicalLabel(technicalLabels, "output schema");
    if (apiSurface && /error payload|error response/.test(text)) addTechnicalLabel(technicalLabels, "error payload");

    if (hasSecurityEvidence(normalizedPath, patch) && !matchesWorkflowFilePath(normalizedPath) && !autobotClassificationInfrastructure) {
      addTechnicalLabel(technicalLabels, "auth");
      if (/(?:^|[^a-z0-9])token(?:[^a-z0-9]|$)|authorization:\s*bearer/.test(text)) addTechnicalLabel(technicalLabels, "token");
      if (/\bpermissions?\b|\bauth(?:entication|orization)?\b/.test(text)) addTechnicalLabel(technicalLabels, "permission");
      if (/\bsession\b|\bcookie\b/.test(text)) addTechnicalLabel(technicalLabels, "session");
      if (/\bapi key\b/.test(text)) addTechnicalLabel(technicalLabels, "api key");
      if (/\btls\b|certificate/.test(text)) addTechnicalLabel(technicalLabels, "tls");
      if (/\bjwt\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "jwt");
        if (/expiry|expires|exp\b/.test(text)) addTechnicalLabel(technicalLabels, "jwt expiry");
        if (/claims?/.test(text)) addTechnicalLabel(technicalLabels, "jwt claim");
      }
      if (/refresh token/.test(text)) {
        addTechnicalLabel(technicalLabels, "refresh token");
        if (/rotation/.test(text)) addTechnicalLabel(technicalLabels, "token rotation");
        if (/revoke|revocation/.test(text)) addTechnicalLabel(technicalLabels, "token revocation");
      }
      if (/\brbac\b|\brole\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "rbac");
        if (/role map/.test(text)) addTechnicalLabel(technicalLabels, "role map");
        if (/role check|permission check|authorize/.test(text)) addTechnicalLabel(technicalLabels, "role check");
      }
      if (/\boauth\b|\bscope\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "oauth scope");
        if (/scope map/.test(text)) addTechnicalLabel(technicalLabels, "scope map");
        if (/scope check/.test(text)) addTechnicalLabel(technicalLabels, "scope check");
      }
      if (/\bcsrf\b/.test(text)) addTechnicalLabel(technicalLabels, "csrf");
      if (/\bauthz\b|authorization|authorize/.test(text)) addTechnicalLabel(technicalLabels, "authz");
      if (hasVulnerabilityEvidence(text)) addTechnicalLabel(technicalLabels, "vulnerability");
      if (hasComplianceEvidence(text)) addTechnicalLabel(technicalLabels, "compliance");
      if (hasHardeningEvidence(text)) addTechnicalLabel(technicalLabels, "hardening");
      if (hasPenTestEvidence(text)) addTechnicalLabel(technicalLabels, "pen-test");
    }

    if (hasSchemaEvidence(normalizedPath, patch)) {
      if (/\.proto$/.test(normalizedPath) || /\bproto\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "proto schema");
        if (/=\s*\d+\s*;/.test(patch)) addTechnicalLabel(technicalLabels, "field number");
        if (/\bwire\b|\breserved\b/.test(text)) addTechnicalLabel(technicalLabels, "wire format");
      }
      if (/json schema|schema\.json|\.schema\./.test(text) || (/\.json$/.test(normalizedPath) && /schema/.test(normalizedPath))) {
        addTechnicalLabel(technicalLabels, "json schema");
        if (/\benum\b/.test(text)) addTechnicalLabel(technicalLabels, "enum value");
        if (/\bminimum\b|\bmaximum\b|\bpattern\b|\brequired\b/.test(text)) addTechnicalLabel(technicalLabels, "field rule");
      }
      if (/typing|type alias|typeddict|dataclass/.test(text)) {
        addTechnicalLabel(technicalLabels, "public type");
        if (/alias/.test(text)) addTechnicalLabel(technicalLabels, "type alias");
        if (/export/.test(text)) addTechnicalLabel(technicalLabels, "type export");
      }
      if (/generated|codegen|sdk/.test(text)) {
        addTechnicalLabel(technicalLabels, "generated type");
        if (/generated|codegen/.test(text)) addTechnicalLabel(technicalLabels, "codegen output");
        if (/\bsdk\b/.test(text)) addTechnicalLabel(technicalLabels, "sdk type");
      }
    }

    if (!documentationSurface && !testSurface && !autobotClassificationInfrastructure && hasMigrationEvidence(normalizedPath, patch)) {
      addTechnicalLabel(technicalLabels, "migration file");
      if (/\bseed\b/.test(text)) addTechnicalLabel(technicalLabels, "seed data");
      if (/rename table|alter table.*rename/.test(text)) addTechnicalLabel(technicalLabels, "table rename");
      if (destructive) {
        addTechnicalLabel(technicalLabels, "destructive migration");
        if (/drop column|column .*drop/.test(text)) addTechnicalLabel(technicalLabels, "column drop");
        if (/drop table/.test(text)) addTechnicalLabel(technicalLabels, "table drop");
      } else {
        addTechnicalLabel(technicalLabels, "additive migration");
        if (/add column|column .*add/.test(text)) addTechnicalLabel(technicalLabels, "column add");
        if (/create table|add table/.test(text)) addTechnicalLabel(technicalLabels, "table add");
      }
    }

    if (hasDatabaseEvidence(normalizedPath, patch)) {
      if (/\bselect\b|\bread\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "read query");
        if (/\bindex\b/.test(text)) addTechnicalLabel(technicalLabels, "index usage");
        if (/result|shape|columns?/.test(text)) addTechnicalLabel(technicalLabels, "result shape");
      }
      if (/\binsert\b|\bupdate\b|\bdelete\b|\bupsert\b|\bwrite\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "write query");
        if (/\bupsert\b/.test(text)) addTechnicalLabel(technicalLabels, "upsert");
        if (/\bbulk\b/.test(text)) addTechnicalLabel(technicalLabels, "bulk write");
      }
      if (/create index|drop index|alter index/.test(text)) addTechnicalLabel(technicalLabels, "index ddl");
    }

    if (!matchesWorkflowFilePath(normalizedPath) && (runtimeContractSurface || runtimeTextSurface)) {
      const runtimeSpecificEvidence = hasRuntimePathEvidence(normalizedPath)
        || /requires-python|python 3\.|\bwindows\b|\blinux\b|\bubuntu\b|\bmacos\b|\bcuda\b|\bnvidia\b|\bffmpeg\b|platform_system|env[_ -]?var|os\.environ|getenv|dotenv|subprocess|spawn|fork|exec\(|timezone|\btz\b|locale|path separator|filesystem|file system/.test(text);
      if (runtimeSpecificEvidence) addTechnicalLabel(technicalLabels, "platform");
      if (/docker|compose/.test(text) || /^docker\//.test(normalizedPath)) addTechnicalLabel(technicalLabels, "container");
      if (!testSurface && hasShellCommandEvidence(patch)) addTechnicalLabel(technicalLabels, "shell");
      if (hasFilesystemPatchEvidence(patch)) {
        addTechnicalLabel(technicalLabels, "filesystem");
        addSpecificFilesystemLabels(technicalLabels, text);
      }
      if (/subprocess|spawn|fork|exec\(|child_process/.test(text)) {
        addTechnicalLabel(technicalLabels, "process");
        addSpecificProcessLabels(technicalLabels, text);
      }
      if (/locale/.test(text)) addTechnicalLabel(technicalLabels, "locale");
      if (/timezone|\btz\b/.test(text)) addTechnicalLabel(technicalLabels, "timezone");
      if (/\bx86_64\b|\barm64\b|\baarch64\b|\barchitecture\b/.test(text)) addTechnicalLabel(technicalLabels, "architecture");
      if (/requires-python|python 3\./.test(text)) {
        addTechnicalLabel(technicalLabels, "python");
        addTechnicalLabel(technicalLabels, "python version");
        addTechnicalLabel(technicalLabels, "support matrix");
      }
      if (/\bwindows\b/.test(text) || /windows/.test(normalizedPath)) {
        addTechnicalLabel(technicalLabels, "os");
        addTechnicalLabel(technicalLabels, "windows");
        if (/path separator|\\\\/.test(patch) || PATH_NORMALIZATION_TEXT_PATTERN.test(text)) addTechnicalLabel(technicalLabels, "path separator");
        if (hasShellCommandEvidence(patch)) addTechnicalLabel(technicalLabels, "shell command");
      }
      if (/\blinux\b|\bubuntu\b/.test(text) || /linux/.test(normalizedPath)) {
        addTechnicalLabel(technicalLabels, "os");
        addTechnicalLabel(technicalLabels, "linux");
        if (/apt-get|dnf|pacman|yum/.test(text)) addTechnicalLabel(technicalLabels, "package install");
        if (/shared lib|ld_library_path|\.so\b/.test(text)) addTechnicalLabel(technicalLabels, "shared lib");
      }
      if (/\bmacos\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "os");
        addTechnicalLabel(technicalLabels, "macos");
      }
      if (/\bcuda\b|\bnvidia\b/.test(text)) {
        addTechnicalLabel(technicalLabels, "cuda");
        if (/gpu memory|vram/.test(text)) addTechnicalLabel(technicalLabels, "gpu memory");
        if (/kernel launch|cuda launch/.test(text)) addTechnicalLabel(technicalLabels, "kernel launch");
      }
      if (/env[_ -]?var|os\.environ|getenv|dotenv/.test(text)) {
        addTechnicalLabel(technicalLabels, "env var");
        if (/dotenv|load env/.test(text)) addTechnicalLabel(technicalLabels, "env loading");
        if (/\bdefault\b/.test(text)) addTechnicalLabel(technicalLabels, "env default");
      }
    }

    if (FEATURE_FLAG_TEXT_PATTERN.test(text)) {
      addTechnicalLabel(technicalLabels, "feature toggle");
      if (/kill switch/.test(text)) addTechnicalLabel(technicalLabels, "kill switch");
      if (/default/.test(text)) addTechnicalLabel(technicalLabels, "default state");
      if (/cohort|segment|bucket|target/.test(text)) addTechnicalLabel(technicalLabels, "cohort rule");
    }

    const performanceEligibleSurface = !documentationSurface && !testSurface && !matchesWorkflowFilePath(normalizedPath) && !autobotClassificationInfrastructure;
    if (performanceEligibleSurface && hasPerformancePatchEvidence(patch)) {
      if (hasHeapUsageEvidence(patch)) {
        addTechnicalLabel(technicalLabels, "heap usage");
        if (/peak memory/.test(patch)) addTechnicalLabel(technicalLabels, "peak memory");
        if (/buffer copy|buffer copies/.test(patch)) addTechnicalLabel(technicalLabels, "buffer copy");
      }
      if (/\bleak\b|\bretain(?:ed|ing)?\b|\bretention\b/.test(patch)) {
        addTechnicalLabel(technicalLabels, "leak risk");
        if (/\bretain(?:ed|ing)?\b|\bretention\b/.test(patch)) addTechnicalLabel(technicalLabels, "object retention");
        if (/cache leak/.test(patch)) addTechnicalLabel(technicalLabels, "cache leak");
      }
      if (/latency|cold start|init time/.test(patch) || hasImportTimeEvidence(patch)) {
        addTechnicalLabel(technicalLabels, "latency");
        if (/cold start/.test(patch)) addTechnicalLabel(technicalLabels, "cold start");
        if (/init time/.test(patch)) addTechnicalLabel(technicalLabels, "init time");
        if (hasImportTimeEvidence(patch)) addTechnicalLabel(technicalLabels, "import time");
      }
      if (/cache/.test(patch)) {
        addTechnicalLabel(technicalLabels, "cache hit");
        if (/cache key/.test(patch)) addTechnicalLabel(technicalLabels, "cache keys");
        if (/invalidat/.test(patch)) addTechnicalLabel(technicalLabels, "cache invalidation");
      }
    }
  }

  return sortLabels([...technicalLabels]);
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

function rankScoredLabels(labelScores, propertyName) {
  return rankScoredLabelsFromMeasurement(labelScores, propertyName);
}

function mergeRankedLabels(primaryLabels, secondaryLabels, limit) {
  return mergeRankedLabelsFromMeasurement(primaryLabels, secondaryLabels, limit);
}

function isSmallPullRequest(filesChanged, totalChanges) {
  return isSmallPullRequestFromMeasurement(filesChanged, totalChanges);
}

function derivePrLabelBudget(smallPullRequest) {
  return derivePrLabelBudgetFromMeasurement(smallPullRequest);
}

function deriveGenericMaintenanceLabelLimit(smallPullRequest) {
  return deriveGenericMaintenanceLabelLimitFromMeasurement(smallPullRequest);
}

function applyPrLabelPolicy(labels, options = {}) {
  return applyPrLabelPolicyFromMeasurement(labels, options);
}

function ensureRequiredLabelFamily(labels, requiredLabel, preferredLabels = [], options = {}) {
  const limit = Math.max(Number(options.limit) || MAX_AUTOBOT_LABELS, 1);
  const rankedLabels = applyPrLabelPolicy(labels, {
    ...options,
    limit: Math.max(limit * 3, MAX_AUTOBOT_LABELS)
  });
  if (rankedLabels.some((label) => AutobotLabelRegistry.matchesExpectedLabel(label, requiredLabel))) {
    return rankedLabels.slice(0, limit);
  }
  const fallbackCandidates = [...preferredLabels, requiredLabel]
    .map((label) => AutobotLabelRegistry.normalizeLabelName(label))
    .filter((label) => label && AutobotLabelRegistry.isTechnicalLabel(label));
  const fallbackLabel = fallbackCandidates.find((label) => rankedLabels.includes(label)) || fallbackCandidates[0] || "";
  if (!fallbackLabel) {
    return rankedLabels.slice(0, limit);
  }
  const selected = rankedLabels.slice(0, limit);
  if (selected.includes(fallbackLabel)) {
    return selected;
  }
  if (selected.length < limit) {
    selected.push(fallbackLabel);
  } else if (selected.length > 0) {
    const reverseIndex = [...selected]
      .reverse()
      .findIndex((label) => !AutobotLabelRegistry.matchesExpectedLabel(label, requiredLabel));
    const targetIndex = reverseIndex === -1 ? selected.length - 1 : selected.length - 1 - reverseIndex;
    selected[targetIndex] = fallbackLabel;
  } else {
    selected.push(fallbackLabel);
  }

  const deduped = [];
  for (const label of selected) {
    if (!label || deduped.includes(label)) {
      continue;
    }
    deduped.push(label);
  }
  if (deduped.some((label) => AutobotLabelRegistry.matchesExpectedLabel(label, requiredLabel))) {
    return deduped.slice(0, limit);
  }
  return [fallbackLabel, ...deduped.filter((label) => label !== fallbackLabel)].slice(0, limit);
}

function selectDeterministicLabels(orderedSignals, deterministicLabelSet, options = {}) {
  return selectDeterministicLabelsFromMeasurement(orderedSignals, deterministicLabelSet, options);
}

function buildLabelConfidenceMapFromRationaleLines(lines) {
  const confidenceByLabel = {};
  for (const line of lines || []) {
    const match = String(line || "").match(/^([^:]+):[\s\S]*?Confidence\s+(high|medium|low)\./i);
    if (!match) {
      continue;
    }
    const label = String(match[1] || "").trim();
    if (!label) {
      continue;
    }
    confidenceByLabel[label] = String(match[2] || "").toLowerCase();
  }
  return confidenceByLabel;
}

function buildTechnicalChildrenIndex() {
  const index = new Map();
  for (const label of Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS || {})) {
    const metadata = AutobotLabelRegistry.getLabelMetadata(label);
    if (!metadata?.parent) {
      continue;
    }
    const parent = AutobotLabelRegistry.normalizeLabelName(metadata.parent);
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (!parent || !normalizedLabel || parent === normalizedLabel) {
      continue;
    }
    if (!index.has(parent)) {
      index.set(parent, new Set());
    }
    index.get(parent).add(normalizedLabel);
  }
  return new Map(
    [...index.entries()].map(([parent, children]) => [parent, [...children].sort((left, right) => left.localeCompare(right))])
  );
}

const TECHNICAL_CHILDREN_INDEX = buildTechnicalChildrenIndex();
const TECHNICAL_DESCENDANT_CACHE = new Map();

function collectTechnicalDescendantsWithDistance(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  if (TECHNICAL_DESCENDANT_CACHE.has(normalizedLabel)) {
    return TECHNICAL_DESCENDANT_CACHE.get(normalizedLabel);
  }
  const descendants = [];
  const visited = new Set();
  const queue = (TECHNICAL_CHILDREN_INDEX.get(normalizedLabel) || []).map((child) => ({
    distance: 1,
    label: child
  }));
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || visited.has(current.label)) {
      continue;
    }
    visited.add(current.label);
    const children = TECHNICAL_CHILDREN_INDEX.get(current.label) || [];
    descendants.push({
      distance: current.distance,
      label: current.label,
      leaf: children.length === 0,
      metadata: AutobotLabelRegistry.getLabelMetadata(current.label) || null
    });
    for (const child of children) {
      queue.push({
        distance: current.distance + 1,
        label: child
      });
    }
  }
  descendants.sort((left, right) => left.distance - right.distance || left.label.localeCompare(right.label));
  TECHNICAL_DESCENDANT_CACHE.set(normalizedLabel, descendants);
  return descendants;
}

function isGeneralTechnicalLabel(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  if (!AutobotLabelRegistry.getLabelMetadata(normalizedLabel)) {
    return false;
  }
  return collectTechnicalDescendantsWithDistance(normalizedLabel).length >= GENERAL_LABEL_MIN_DESCENDANTS;
}

function rankReplacementCandidate(candidate, context = {}) {
  const label = candidate?.label;
  const supportCheck = context.labelSupportCheck;
  const supported = typeof supportCheck === "function" ? Boolean(supportCheck(label)) : false;
  const score = Number(context.technicalLabelScores?.[label]?.score || 0);
  const releaseRelevant = Number(Boolean(candidate?.metadata?.releaseRelevant));
  const specificity = Number(candidate?.distance || 0);
  const leaf = Number(Boolean(candidate?.leaf));
  return (supported ? 100 : 0) + (score * 12) + (releaseRelevant * 1.5) + (specificity * 0.2) + (leaf * 0.4);
}

function selectReplacementLabels(candidates, targetCount, context = {}, excluded = new Set()) {
  const selected = [];
  const ranked = [...candidates].sort((left, right) => {
    const leftScore = rankReplacementCandidate(left, context);
    const rightScore = rankReplacementCandidate(right, context);
    if (rightScore !== leftScore) {
      return rightScore - leftScore;
    }
    return right.distance - left.distance || left.label.localeCompare(right.label);
  });
  for (const candidate of ranked) {
    if (!candidate?.label || excluded.has(candidate.label)) {
      continue;
    }
    selected.push(candidate.label);
    excluded.add(candidate.label);
    if (selected.length >= targetCount) {
      break;
    }
  }
  return selected;
}

function buildGeneralLabelReplacementPlan(label, context = {}) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  const metadata = AutobotLabelRegistry.getLabelMetadata(normalizedLabel);
  if (!metadata) {
    return null;
  }
  const labelDepth = AutobotLabelRegistry.getLabelDepth(normalizedLabel);
  const descendants = collectTechnicalDescendantsWithDistance(normalizedLabel).map((candidate) => ({
    ...candidate,
    absoluteDepth: AutobotLabelRegistry.getLabelDepth(candidate.label)
  }));
  const familyCandidates = Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS || {})
    .map((candidateLabel) => AutobotLabelRegistry.normalizeLabelName(candidateLabel))
    .filter((candidateLabel) => candidateLabel && candidateLabel !== normalizedLabel)
    .map((candidateLabel) => {
      const candidateMetadata = AutobotLabelRegistry.getLabelMetadata(candidateLabel);
      if (!candidateMetadata || candidateMetadata.family !== metadata.family) {
        return null;
      }
      const absoluteDepth = AutobotLabelRegistry.getLabelDepth(candidateLabel);
      return {
        absoluteDepth,
        distance: Math.max(absoluteDepth - labelDepth, 1),
        label: candidateLabel,
        leaf: (TECHNICAL_CHILDREN_INDEX.get(candidateLabel) || []).length === 0,
        metadata: candidateMetadata
      };
    })
    .filter(Boolean);
  const candidateMap = new Map();
  for (const candidate of [...descendants, ...familyCandidates]) {
    if (!candidate?.label || candidateMap.has(candidate.label)) {
      continue;
    }
    candidateMap.set(candidate.label, candidate);
  }
  let candidates = [...candidateMap.values()];
  if (candidates.length < GENERAL_LABEL_MIN_DESCENDANTS) {
    const globalTechnicalFallback = Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS || {})
      .map((candidateLabel) => AutobotLabelRegistry.normalizeLabelName(candidateLabel))
      .filter((candidateLabel) => candidateLabel && candidateLabel !== normalizedLabel)
      .map((candidateLabel) => {
        if (candidateMap.has(candidateLabel)) {
          return null;
        }
        const candidateMetadata = AutobotLabelRegistry.getLabelMetadata(candidateLabel);
        if (!candidateMetadata) {
          return null;
        }
        const absoluteDepth = AutobotLabelRegistry.getLabelDepth(candidateLabel);
        return {
          absoluteDepth,
          distance: Math.max(absoluteDepth - labelDepth, 1),
          label: candidateLabel,
          leaf: (TECHNICAL_CHILDREN_INDEX.get(candidateLabel) || []).length === 0,
          metadata: candidateMetadata
        };
      })
      .filter(Boolean);
    candidates = [...candidates, ...globalTechnicalFallback];
  }
  const moderatePool = candidates.filter((candidate) => (candidate.absoluteDepth || 0) <= labelDepth + 2 || candidate.distance <= 2);
  const verySpecificPool = candidates.filter((candidate) => (candidate.absoluteDepth || 0) >= labelDepth + 3 || candidate.leaf);
  const excluded = new Set([normalizedLabel]);
  const moderatelySpecific = selectReplacementLabels(
    moderatePool,
    GENERAL_LABEL_REPLACEMENT_COUNTS.moderatelySpecific,
    context,
    excluded
  );
  if (moderatelySpecific.length < GENERAL_LABEL_REPLACEMENT_COUNTS.moderatelySpecific) {
    moderatelySpecific.push(
      ...selectReplacementLabels(
        descendants,
        GENERAL_LABEL_REPLACEMENT_COUNTS.moderatelySpecific - moderatelySpecific.length,
        context,
        excluded
      )
    );
  }
  const verySpecific = selectReplacementLabels(
    verySpecificPool,
    GENERAL_LABEL_REPLACEMENT_COUNTS.verySpecific,
    context,
    excluded
  );
  if (verySpecific.length < GENERAL_LABEL_REPLACEMENT_COUNTS.verySpecific) {
    const deepestDescendants = [...descendants]
      .sort((left, right) => right.distance - left.distance || left.label.localeCompare(right.label));
    verySpecific.push(
      ...selectReplacementLabels(
        deepestDescendants,
        GENERAL_LABEL_REPLACEMENT_COUNTS.verySpecific - verySpecific.length,
        context,
        excluded
      )
    );
  }
  if (
    moderatelySpecific.length < GENERAL_LABEL_REPLACEMENT_COUNTS.moderatelySpecific
    || verySpecific.length < GENERAL_LABEL_REPLACEMENT_COUNTS.verySpecific
  ) {
    return null;
  }
  return {
    generalLabel: normalizedLabel,
    moderatelySpecific,
    verySpecific
  };
}

const CEM_API_FAMILY_LABELS = new Set([
  "api", "rest", "route", "route param", "query param", "graphql",
  "resolver", "schema field", "webhook", "webhook payload", "webhook retry",
  "openapi spec", "request body", "response body", "body validation",
  "input schema", "output schema", "error payload"
]);

const CEM_EXCLUDED_DOMAINS_FOR_API = new Set(["workflow", "github", "config"]);

function isFileCemExcludedForLabel(file, label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  if (!CEM_API_FAMILY_LABELS.has(normalizedLabel)) {
    return false;
  }
  const normalizedPath = String(file.filename || "").toLowerCase();
  const categories = new Set(file.categories || classifyFile(file));
  if (hasApiPathEvidence(normalizedPath)) {
    return false;
  }
  return [...categories].some((category) => CEM_EXCLUDED_DOMAINS_FOR_API.has(category))
    || matchesWorkflowFilePath(normalizedPath)
    || isAutobotClassificationInfrastructure(normalizedPath);
}

function buildContentEvidenceMap(filesWithContext, technicalSignals, evidenceItems) {
  const cemMap = new Map();
  for (const label of technicalSignals) {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (!CEM_API_FAMILY_LABELS.has(normalizedLabel)) {
      cemMap.set(normalizedLabel, { confirmed: true, reason: "non-api-family" });
      continue;
    }
    const hasStructuralEvidence = filesWithContext.some((file) => {
      if (isFileCemExcludedForLabel(file, normalizedLabel)) {
        return false;
      }
      const normalizedPath = String(file.filename || "").toLowerCase();
      if (hasApiPathEvidence(normalizedPath)) {
        return true;
      }
      const signals = new Set(file.signals || []);
      return signals.has("api");
    });
    const hasEvidenceItemSupport = evidenceItems.some((item) => {
      const ruleId = String(item.ruleId || "");
      return (ruleId.includes("api") || ruleId.includes("schema"))
        && Number(item.occurrenceCount || 0) > 0;
    });
    cemMap.set(normalizedLabel, {
      confirmed: hasStructuralEvidence || hasEvidenceItemSupport,
      reason: hasStructuralEvidence ? "structural-path" : hasEvidenceItemSupport ? "evidence-item" : "rejected-no-structural-proof"
    });
  }
  return cemMap;
}

function applyCemGate(labels, cemMap) {
  return labels.filter((label) => {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    const entry = cemMap.get(normalizedLabel);
    if (!entry) {
      return true;
    }
    return entry.confirmed;
  });
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
  const vulnerabilityFiles = [];
  const complianceFiles = [];
  const hardeningFiles = [];
  const penTestFiles = [];
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
  const securityContextText = String(context.securityContextText || "").toLowerCase();

  for (const file of filesWithContext) {
    const normalizedPath = String(file.filename || "").toLowerCase();
    const patch = String(file.patch || "").toLowerCase();
    const text = `${normalizedPath}\n${patch}`;
    const categories = new Set(file.categories);
    const signals = new Set(file.signals);
    const autobotClassificationInfrastructure = isAutobotClassificationInfrastructure(normalizedPath);
    const destructive = isDestructiveFileChange(file);
    const runtimeSupportContractPath = isRuntimeSupportContractPath(normalizedPath);
    const runtimeAdded = runtimeSupportContractPath && hasPatchLineMatch(file, "+", RUNTIME_SUPPORT_TEXT_PATTERN) && !hasPatchLineMatch(file, "-", RUNTIME_SUPPORT_TEXT_PATTERN);
    const runtimeDropped = runtimeSupportContractPath && hasPatchLineMatch(file, "-", RUNTIME_SUPPORT_TEXT_PATTERN) && !hasPatchLineMatch(file, "+", RUNTIME_SUPPORT_TEXT_PATTERN);

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
      if (hasVulnerabilityEvidence(text)) {
        vulnerabilityFiles.push(file.filename);
      }
      if (hasComplianceEvidence(text)) {
        complianceFiles.push(file.filename);
      }
      if (hasHardeningEvidence(text)) {
        hardeningFiles.push(file.filename);
      }
      if (hasPenTestEvidence(text)) {
        penTestFiles.push(file.filename);
      }
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

    if (!autobotClassificationInfrastructure && (signals.has("feature-flag") || FEATURE_FLAG_TEXT_PATTERN.test(patch))) {
      if (destructive) {
        featureFlagContractFiles.push(file.filename);
      } else {
        featureFlagFiles.push(file.filename);
      }
    }

    if (signals.has("runtime")) {
      if (runtimeDropped && runtimeSupportContractPath) {
        runtimeSupportDroppedFiles.push(file.filename);
      } else if (runtimeAdded && runtimeSupportContractPath) {
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
      if (isExplicitPublicCapabilityFile(file)) {
        publicAddedFiles.push(file.filename);
      }
    }

    if (file.status === "added" && isBehavioralSurfaceFile(file)) {
      behavioralAdditionFiles.push(file.filename);
    }
  }

  const publicCapabilityFiles = publicAddedFiles;

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
  addEvidenceItem(evidenceMap, "security-vulnerability", { occurrenceCount: vulnerabilityFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: vulnerabilityFiles });
  addEvidenceItem(evidenceMap, "security-compliance", { occurrenceCount: complianceFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: complianceFiles });
  addEvidenceItem(evidenceMap, "security-hardening", { occurrenceCount: hardeningFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: hardeningFiles });
  addEvidenceItem(evidenceMap, "security-pen-test", { occurrenceCount: penTestFiles.length, scope: "repo", confidence: "corroborated", sampleFiles: penTestFiles });
  const securityContextFiles = [...new Set([
    ...vulnerabilityFiles,
    ...complianceFiles,
    ...hardeningFiles,
    ...penTestFiles,
    ...securityAuthFiles,
    ...securitySecretFiles
  ])];
  if (securityContextFiles.length > 0) {
    addEvidenceItem(evidenceMap, "security-vulnerability", {
      occurrenceCount: hasVulnerabilityEvidence(securityContextText) ? 1 : 0,
      scope: "repo",
      confidence: "title",
      sampleFiles: securityContextFiles
    });
    addEvidenceItem(evidenceMap, "security-compliance", {
      occurrenceCount: hasComplianceEvidence(securityContextText) ? 1 : 0,
      scope: "repo",
      confidence: "title",
      sampleFiles: securityContextFiles
    });
    addEvidenceItem(evidenceMap, "security-hardening", {
      occurrenceCount: hasHardeningEvidence(securityContextText) ? 1 : 0,
      scope: "repo",
      confidence: "title",
      sampleFiles: securityContextFiles
    });
    addEvidenceItem(evidenceMap, "security-pen-test", {
      occurrenceCount: hasPenTestEvidence(securityContextText) ? 1 : 0,
      scope: "repo",
      confidence: "title",
      sampleFiles: securityContextFiles
    });
  }
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

async function collectPullRequestSnapshot({ github, owner, repo, pullRequest, snapshotFile }) {
  return collectPullRequestSnapshotFromGatherInfo({ github, owner, repo, pullRequest, snapshotFile });
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
    fileWithContext.categories = classifyFile(fileWithContext);
    fileWithContext.signals = deriveSignals(fileWithContext);
    return fileWithContext;
  });
  const totalAdditions = Number(snapshot.totals.additions || 0);
  const totalDeletions = Number(snapshot.totals.deletions || 0);
  const totalChanges = Number(snapshot.totals.totalChanges || 0);
  const prSignalText = `${snapshot.pullRequest.title}`.toLowerCase();
  const securityContextText = `${snapshot.pullRequest.title}\n${snapshot.pullRequest.body || ""}`.toLowerCase();
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
    filesWithContext,
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
  const hasDirectSecuritySurface = filesWithContext.some((file) => {
    const normalizedPath = String(file.filename || "").toLowerCase();
    return file.signals.includes("security")
      && !isAutobotClassificationInfrastructure(normalizedPath)
      && !isDependencyLockfilePath(normalizedPath);
  });
  if (titleSignals.security && hasDirectSecuritySurface) {
    signalSet.add("security");
  }

  const technicalSignalSet = new Set(deriveTechnicalLabels(filesWithContext));
  if (hasDirectSecuritySurface && hasVulnerabilityEvidence(securityContextText)) {
    technicalSignalSet.add("vulnerability");
  }
  if (hasDirectSecuritySurface && hasComplianceEvidence(securityContextText)) {
    technicalSignalSet.add("compliance");
  }
  if (hasDirectSecuritySurface && hasHardeningEvidence(securityContextText)) {
    technicalSignalSet.add("hardening");
  }
  if (hasDirectSecuritySurface && hasPenTestEvidence(securityContextText)) {
    technicalSignalSet.add("pen-test");
  }
  const technicalSignals = sortLabels([...technicalSignalSet]);
  const orderedSignals = sortLabels([...signalSet, ...technicalSignals]);
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
  const maintenanceOnlyEligible = filesWithContext.length > 0
    && [...categoryCounts.keys()].every((category) => MAINTENANCE_ONLY_CATEGORIES.has(category))
    && !AutobotLabelRegistry.RELEASE_CRITICAL_LABELS.some((label) => signalSet.has(label));
  const smallPullRequest = isSmallPullRequest(filesWithContext.length, totalChanges);
  const prLabelLimit = derivePrLabelBudget(smallPullRequest);
  const genericMaintenanceLabelLimit = deriveGenericMaintenanceLabelLimit(smallPullRequest);
  const prLabelPolicy = {
    genericMaintenanceLimit: genericMaintenanceLabelLimit,
    limit: prLabelLimit,
    maxWords: MAX_LABEL_WORDS
  };
  const prCollectionLimit = Math.max(prLabelLimit * 2, MAX_AUTOBOT_LABELS);
  const evidenceItems = buildDeterministicEvidence(filesWithContext, {
    behavioralSurfaceAdditions,
    capabilityExpansionSignal,
    securityContextText,
    structuralPublicBreakingSignal
  });
  const contentEvidenceMap = buildContentEvidenceMap(filesWithContext, technicalSignals, evidenceItems);
  const cemFilteredTechnicalSignals = applyCemGate(technicalSignals, contentEvidenceMap);
  const cemFilteredOrderedSignals = applyCemGate(orderedSignals, contentEvidenceMap);
  const scoring = scoreDeterministicEvidence({
    evidenceItems,
    options: { maxLabels: MAX_AUTOBOT_LABELS }
  });
  const scorerEmittedLabels = applyPrLabelPolicy(scoring.emittedLabels.map((entry) => entry.label), {
    ...prLabelPolicy,
    limit: prCollectionLimit
  });
  const scorerRetainedLabels = applyPrLabelPolicy(rankScoredLabels(scoring.labelScores, "retained"), {
    ...prLabelPolicy,
    limit: prCollectionLimit
  });
  const primaryScorerLabels = applyPrLabelPolicy(scoring.primaryLabels.map((entry) => entry.label), prLabelPolicy);

  const deterministicLabelSet = new Set();
  for (const signal of cemFilteredOrderedSignals) {
    deterministicLabelSet.add(signal);
  }
  for (const label of scorerRetainedLabels) {
    deterministicLabelSet.add(label);
  }
  for (const label of scorerEmittedLabels) {
    deterministicLabelSet.add(label);
  }
  if (categoryCounts.get("documentation")) deterministicLabelSet.add("documentation");
  if (categoryCounts.get("test")) deterministicLabelSet.add("test");
  if (categoryCounts.get("test") || categoryCounts.get("workflow") || signalSet.has("workflow")) deterministicLabelSet.add("test job");
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
    if (categoryCounts.get("test") || categoryCounts.get("workflow") || signalSet.has("workflow")) candidateLabelSet.add("test job");
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

  const technicalLabelScores = Object.fromEntries(
    Object.entries(scoring.labelScores).filter(([label]) => AutobotLabelRegistry.isTechnicalLabel(label))
  );
  const cemFilteredTechnicalSignalSet = new Set(cemFilteredTechnicalSignals);
  const rawDeterministicLabels = selectDeterministicLabels(cemFilteredOrderedSignals, deterministicLabelSet, prLabelPolicy);
  const maintenanceLabelHasDirectSurfaceSupport = (label) => {
    switch (label) {
      case "dependencies":
        return Boolean(categoryCounts.get("dependencies"));
      case "documentation":
        return Boolean(categoryCounts.get("documentation"));
      case "test":
        return Boolean(categoryCounts.get("test"));
      case "workflow":
        return Boolean(categoryCounts.get("workflow")) || signalSet.has("workflow");
      case "github":
        return Boolean(categoryCounts.get("github")) || signalSet.has("github");
      case "config":
        return Boolean(categoryCounts.get("config"));
      case "tooling":
        return Boolean(categoryCounts.get("tooling"));
      case "docker":
        return Boolean(categoryCounts.get("docker"));
      case "cleanup":
        return filesWithContext.some((file) => file.status === "removed");
      default:
        return false;
    }
  };
  const labelSupportCache = new Map();
  const labelHasStrongOutputSupport = (label) => {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (labelSupportCache.has(normalizedLabel)) {
      return labelSupportCache.get(normalizedLabel);
    }
    if (
      AutobotLabelRegistry.matchesExpectedLabel(normalizedLabel, "test")
      && (Boolean(categoryCounts.get("test")) || Boolean(categoryCounts.get("workflow")) || signalSet.has("workflow"))
    ) {
      labelSupportCache.set(normalizedLabel, true);
      return true;
    }
    const scoreEntry = technicalLabelScores[normalizedLabel];
    const directEvidence = collectDirectEvidenceForLabel(normalizedLabel, evidenceItems);
    const supportFiles = collectSupportFilesForLabel(normalizedLabel, filesWithContext);
    const supported = Boolean(
      cemFilteredTechnicalSignalSet.has(normalizedLabel)
      || scoreEntry?.primary
      || scoreEntry?.emitted
      || (scoreEntry?.retained && (directEvidence.length > 0 || supportFiles.length > 0 || maintenanceLabelHasDirectSurfaceSupport(normalizedLabel)))
      || maintenanceLabelHasDirectSurfaceSupport(normalizedLabel)
      || directEvidence.some((item) => Number(item?.occurrenceCount || 0) >= 2)
      || supportFiles.length > 0
    );
    labelSupportCache.set(normalizedLabel, supported);
    return supported;
  };
  let deterministicLabels = rawDeterministicLabels.filter((label) => labelHasStrongOutputSupport(label));
  if (deterministicLabels.length === 0) {
    deterministicLabels = applyPrLabelPolicy(
      mergeRankedLabels(
        scorerEmittedLabels,
        scorerRetainedLabels,
        prCollectionLimit
      ),
      {
        ...prLabelPolicy,
        limit: prCollectionLimit
      }
    );
  }
  if (deterministicLabels.length === 0) {
    deterministicLabels = rawDeterministicLabels.filter((label) => !isGeneralTechnicalLabel(label)).slice(0, prLabelLimit);
  }
  const generalLabelReplacementPlans = {};
  const expandedReplacementLabels = [];
  if (!maintenanceOnlyEligible) {
    for (const label of deterministicLabels) {
      const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
      if (!isGeneralTechnicalLabel(normalizedLabel)) {
        expandedReplacementLabels.push(normalizedLabel);
        continue;
      }
      const replacementPlan = buildGeneralLabelReplacementPlan(normalizedLabel, {
        labelSupportCheck: labelHasStrongOutputSupport,
        technicalLabelScores
      });
      if (!replacementPlan) {
        expandedReplacementLabels.push(normalizedLabel);
        continue;
      }
      generalLabelReplacementPlans[normalizedLabel] = replacementPlan;
      expandedReplacementLabels.push(normalizedLabel, ...replacementPlan.moderatelySpecific, ...replacementPlan.verySpecific);
    }
    if (expandedReplacementLabels.length > 0) {
      deterministicLabels = applyPrLabelPolicy(expandedReplacementLabels, {
        ...prLabelPolicy,
        limit: prCollectionLimit
      });
    }
  }
  const rationaleContext = {
    evidenceItems,
    filesWithContext,
    hardSignals: scoring.semver.hardSignals,
    labelScores: technicalLabelScores
  };
  let labelRationaleLines = buildLabelRationaleLines(deterministicLabels, rationaleContext);
  let confidenceByLabel = buildLabelConfidenceMapFromRationaleLines(labelRationaleLines);
  const computeEvidenceStrength = (label) => {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    const scoreEntry = technicalLabelScores[normalizedLabel];
    const directEvidence = collectDirectEvidenceForLabel(normalizedLabel, evidenceItems);
    const supportFiles = collectSupportFilesForLabel(normalizedLabel, filesWithContext);
    const scorerStrength = Math.max(Number(scoreEntry?.score || 0), 0);
    const evidenceOccurrences = directEvidence.reduce((total, item) => total + Math.max(Number(item?.occurrenceCount || 0), 0), 0);
    const evidenceStrength = Math.min(evidenceOccurrences, 6) / 6;
    const supportPatterns = LABEL_SUPPORT_PATTERNS[normalizedLabel] || [];
    const filenameOnlySupportCount = supportPatterns.length > 0
      ? filesWithContext.filter((file) => {
          const fn = String(file.filename || "").toLowerCase();
          return supportPatterns.some((pattern) => pattern.test(fn));
        }).length
      : 0;
    const effectiveSupportCount = Math.max(supportFiles.length, filenameOnlySupportCount);
    const supportStrength = Math.min(effectiveSupportCount, 3) / 3;
    const maintenanceStrength = maintenanceLabelHasDirectSurfaceSupport(normalizedLabel) ? 0.4 : 0;
    const hardSignalCorrelation = directEvidence.some((item) => scoring.semver.hardSignals.includes(item.ruleId)) ? 0.3 : 0;
    return scorerStrength * 0.4 + evidenceStrength * 0.3 + supportStrength * 0.2 + maintenanceStrength + hardSignalCorrelation;
  };
  const canRetainLowConfidenceLabel = (label) => {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (
      AutobotLabelRegistry.matchesExpectedLabel(normalizedLabel, "test")
      && (Boolean(categoryCounts.get("test")) || Boolean(categoryCounts.get("workflow")) || signalSet.has("workflow"))
    ) {
      return true;
    }
    if (maintenanceLabelHasDirectSurfaceSupport(normalizedLabel)) {
      return true;
    }
    if (cemFilteredTechnicalSignalSet.has(normalizedLabel)) {
      return true;
    }
    const scoreEntry = technicalLabelScores[normalizedLabel];
    if (scoreEntry?.primary || scoreEntry?.emitted) {
      return true;
    }
    if (collectDirectEvidenceForLabel(normalizedLabel, evidenceItems).length > 0) {
      return true;
    }
    if (collectSupportFilesForLabel(normalizedLabel, filesWithContext).length > 0) {
      return true;
    }
    return scoring.semver.hardSignals.length > 0 && AutobotLabelRegistry.isReleaseRelevantLabel(normalizedLabel);
  };
  deterministicLabels = deterministicLabels.filter((label) => {
    const confidence = confidenceByLabel[label];
    if (!confidence) {
      return false;
    }
    if (confidence !== LOW_CONFIDENCE_LABEL) {
      return true;
    }
    return canRetainLowConfidenceLabel(label);
  });
  if (deterministicLabels.length === 0) {
    const nonLowFallbackCandidates = mergeRankedLabels(
      scorerEmittedLabels,
      mergeRankedLabels(scorerRetainedLabels, rawDeterministicLabels, prCollectionLimit),
      prCollectionLimit
    ).filter((label) => {
      const confidenceMap = buildLabelConfidenceMapFromRationaleLines(
        buildLabelRationaleLines([label], rationaleContext)
      );
      const confidence = confidenceMap[label];
      if (!confidence) {
        return false;
      }
      if (confidence !== LOW_CONFIDENCE_LABEL) {
        return true;
      }
      return canRetainLowConfidenceLabel(label);
    });
    deterministicLabels = applyPrLabelPolicy(nonLowFallbackCandidates, prLabelPolicy);
    labelRationaleLines = buildLabelRationaleLines(deterministicLabels, rationaleContext);
    confidenceByLabel = buildLabelConfidenceMapFromRationaleLines(labelRationaleLines);
  }

  const prePrecisionDeterministicLabels = [...deterministicLabels];
  deterministicLabels = enforcePrecisionEmissionParameters(
    deterministicLabels,
    {
      evidenceItems,
      filesWithContext,
      technicalLabelScores
    },
    prLabelPolicy
  );
  if (deterministicLabels.length === 0) {
    deterministicLabels = enforcePrecisionEmissionParameters(
      mergeRankedLabels(
        scorerEmittedLabels,
        mergeRankedLabels(scorerRetainedLabels, rawDeterministicLabels, prCollectionLimit),
        prCollectionLimit
      ),
      {
        evidenceItems,
        filesWithContext,
        technicalLabelScores
      },
      prLabelPolicy
    );
  }
  if (deterministicLabels.length === 0) {
    deterministicLabels = applyPrLabelPolicy(
      prePrecisionDeterministicLabels.length > 0
        ? prePrecisionDeterministicLabels
        : mergeRankedLabels(
            scorerEmittedLabels,
            mergeRankedLabels(scorerRetainedLabels, rawDeterministicLabels, prCollectionLimit),
            prCollectionLimit
          ),
      prLabelPolicy
    );
  }
  if (maintenanceOnlyEligible && categoryCounts.get("test")) {
    deterministicLabels = ensureRequiredLabelFamily(
      deterministicLabels,
      "test",
      ["unit test", "test job"],
      prLabelPolicy
    );
  }
  if (maintenanceOnlyEligible && (categoryCounts.get("workflow") || signalSet.has("workflow"))) {
    deterministicLabels = ensureRequiredLabelFamily(
      deterministicLabels,
      "workflow",
      ["github actions", "workflow file"],
      prLabelPolicy
    );
  }
  if (signalSet.has("migration")
    || scoring.semver.hardSignals.includes("destructive-database-change")
    || scoring.semver.hardSignals.includes("additive-database-change")) {
    deterministicLabels = ensureRequiredLabelFamily(
      deterministicLabels,
      "migration file",
      ["destructive migration", "additive migration", "column drop", "column add"],
      prLabelPolicy
    );
  }

  for (const label of deterministicLabels) {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (!isGeneralTechnicalLabel(normalizedLabel) || generalLabelReplacementPlans[normalizedLabel]) {
      continue;
    }
    const replacementPlan = buildGeneralLabelReplacementPlan(normalizedLabel, {
      labelSupportCheck: labelHasStrongOutputSupport,
      technicalLabelScores
    });
    if (replacementPlan) {
      generalLabelReplacementPlans[normalizedLabel] = replacementPlan;
    }
  }
  const allGeneralLabelReplacementPlans = {};
  for (const label of Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS || {})) {
    const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
    if (!isGeneralTechnicalLabel(normalizedLabel) || allGeneralLabelReplacementPlans[normalizedLabel]) {
      continue;
    }
    const replacementPlan = buildGeneralLabelReplacementPlan(normalizedLabel, {
      labelSupportCheck: labelHasStrongOutputSupport,
      technicalLabelScores
    });
    if (replacementPlan) {
      allGeneralLabelReplacementPlans[normalizedLabel] = replacementPlan;
    }
  }
  const replacementPlanOutput = {
    ...allGeneralLabelReplacementPlans,
    ...generalLabelReplacementPlans
  };
  const releaseRelevant = AutobotLabelRegistry.hasReleaseRelevantLabel(deterministicLabels);
  const candidateLabels = applyPrLabelPolicy(
    mergeRankedLabels(
      scorerRetainedLabels,
      sortLabels([...candidateLabelSet]),
      prCollectionLimit
    ),
    prLabelPolicy
  );
  const consideredButNotEmittedLabels = [...new Set([...rawDeterministicLabels, ...primaryScorerLabels, ...candidateLabels])]
    .filter((label) => !deterministicLabels.includes(label));
  const topEvidenceItems = evidenceItems.slice(0, 4);

  if (deterministicLabels.length > 1) {
    const evidenceStrengths = deterministicLabels.map((label) => ({
      label,
      strength: computeEvidenceStrength(label)
    }));
    const hasStrongLabels = evidenceStrengths.some((entry) => entry.strength >= 0.15);
    if (hasStrongLabels) {
      const filteredLabels = evidenceStrengths
        .filter((entry) => entry.strength >= 0.05)
        .sort((left, right) => right.strength - left.strength)
        .map((entry) => entry.label);
      if (filteredLabels.length > 0) {
        deterministicLabels = applyPrLabelPolicy(filteredLabels, prLabelPolicy);
      }
    }
  }

  deterministicLabels = applyCemGate(deterministicLabels, contentEvidenceMap);

  labelRationaleLines = buildLabelRationaleLines(deterministicLabels, rationaleContext);

  const deterministicSummary = buildPrDeterministicSummary({
    behavioralSurfaceAdditions,
    capabilityExpansionSignal,
    categoryCounts,
    categorySummary,
    consideredButNotEmittedLabels,
    deterministicLabels,
    filesWithContext,
    labelRationaleLines,
    maintenanceCategoriesList,
    maintenanceOnlyEligible,
    publicContractMoves,
    releaseRelevant,
    scoring,
    structuralPublicBreakingSignal,
    topDirectories,
    topEvidenceItems,
    topFiles,
    totalAdditions,
    totalDeletions
  });
  const evidenceScaffold = [
    "PR EVIDENCE",
    `Scored semver decision: ${scoring.semver.decision}`,
    `Final emitted technical labels: ${deterministicLabels.join(", ") || "(none)"}`,
    `Held back context labels: ${consideredButNotEmittedLabels.join(", ") || "(none)"}`,
    `General label replacements: ${Object.keys(replacementPlanOutput).length}`,
    `Evidence items: ${evidenceItems.length}`,
    `Files changed: ${filesWithContext.length}`,
    `Additions: ${totalAdditions}`,
    `Deletions: ${totalDeletions}`,
    `Category mix: ${categorySummary || "(none)"}`,
    `Top directories: ${topDirectories.join(", ") || "(root)"}`,
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
    deterministic_general_label_replacements_json: JSON.stringify(replacementPlanOutput),
    release_relevant: releaseRelevant ? "true" : "false",
    deterministic_evidence_json: JSON.stringify(evidenceItems),
    deterministic_evidence_count: String(evidenceItems.length),
    deterministic_category_scores_json: JSON.stringify(scoring.categoryScores),
    deterministic_impact_scores_json: JSON.stringify(scoring.impactScores),
    deterministic_label_scores_json: JSON.stringify(technicalLabelScores),
    deterministic_semver_json: JSON.stringify(scoring.semver),
    deterministic_semver_decision: String(scoring.semver.decision),
    deterministic_summary: deterministicSummary,
    evidence_chars: String(evidenceScaffold.length)
  };
}

function analyzePullRequestSnapshot(input = {}) {
  return analyzePullRequestSnapshotData(readJson(resolveSnapshotFile(input)));
}

class AutobotPullRequestAnalyzer {
  static SNAPSHOT_FILE = SNAPSHOT_FILE;

  static collectPullRequestSnapshot(input) {
    return collectPullRequestSnapshot(input);
  }

  static analyzePullRequestSnapshotData(snapshot) {
    return analyzePullRequestSnapshotData(snapshot);
  }

  static analyzePullRequestSnapshot() {
    return analyzePullRequestSnapshot();
  }
}

module.exports = {
  AutobotPullRequestAnalyzer,
  CEM_API_FAMILY_LABELS,
  SNAPSHOT_FILE,
  analyzePullRequestSnapshot,
  analyzePullRequestSnapshotData,
  applyCemGate,
  buildContentEvidenceMap,
  collectPullRequestSnapshot,
  isFileCemExcludedForLabel,
  normalizePatch
};
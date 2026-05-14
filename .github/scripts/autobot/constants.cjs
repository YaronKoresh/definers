const SNAPSHOT_FILE = "/tmp/autobot_pr_snapshot.json";
const MAX_PATCH_CHARS_PER_FILE = 700;
const MAX_TOP_DIRECTORIES = 8;
const MAX_TOP_FILES = 10;
const MIN_BEHAVIORAL_ADDITION_LINES = 80;
const MIN_PUBLIC_CONTRACT_MOVES = 3;
const MAINTENANCE_ONLY_CATEGORIES = new Set(["documentation", "test", "workflow", "github", "config", "dependencies"]);
const PR_CANONICAL_LABEL_REPLACEMENTS = Object.freeze({
  accessibility: "navigation",
  ci: "workflow",
  database: "storage",
  "docs-api": "documentation",
  "docs-site": "documentation",
  dx: "tooling",
  examples: "documentation",
  "feature-flag": "feature toggle",
  migration: "migration file",
  monitoring: "request tracing",
  observability: "request tracing",
  "release-notes": "documentation",
  serialization: "format",
  telemetry: "request tracing",
  types: "type",
  ui: "view"
});
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
const SMALL_PR_FILE_LIMIT = 4;
const SMALL_PR_GENERIC_LABEL_LIMIT = 2;
const SMALL_PR_LABEL_LIMIT = 3;
const SMALL_PR_TOTAL_CHANGE_LIMIT = 120;
const LARGE_PR_GENERIC_LABEL_LIMIT = 3;
const LARGE_PR_LABEL_LIMIT = 4;
const ACCESSIBILITY_TEXT_PATTERN = /\baria-|accessib|a11y|screen reader|keyboard nav/;
const AUTOMATION_TEXT_PATTERN = /\b(autobot|release[-_ ]script|automation[-_ ]bot|repo automation)\b/;
const FEATURE_FLAG_TEXT_PATTERN = /\b(feature[\s_-]?(?:flag|toggle)|kill[\s_-]?switch|rollout(?:\s+gate|\s+policy)?|cohort(?:\s+rule)?|segment(?:ation)?|bucket(?:ing)?)\b/;
const LOCALIZATION_TEXT_PATTERN = /\bi18n\b|\bl10n\b|\blocale\b|\btranslations?\b|\bgettext\b/;
const PACKAGE_JSON_DEPENDENCY_CONTEXT_PATTERNS = Object.freeze([/^(?:\+|-)?\s*\"(dependencies|devDependencies|peerDependencies|optionalDependencies|bundledDependencies|bundleDependencies)\"\s*:\s*\{,?$/i]);
const PACKAGE_JSON_DEPENDENCY_METADATA_KEYS = new Set(["browser", "description", "directories", "engines", "exports", "keywords", "main", "module", "name", "overrides", "packageManager", "repository", "resolutions", "scripts", "type", "version", "volta"]);
const PACKAGE_JSON_DEPENDENCY_VALUE_PATTERN = /^(workspace:|file:|link:|npm:|github:|git\+|https?:|~|\^|>=?|<=?|=|\*|\d)/i;
const PYPROJECT_DEPENDENCY_CONTEXT_PATTERNS = Object.freeze([
  /^\[(project\.optional-dependencies|dependency-groups|tool\.poetry\.dependencies|tool\.poetry\.group\.[^\]]+\.dependencies|tool\.pdm(?:\.[^\]]+)?\.(?:dependencies|dev-dependencies)|build-system)\]$/i,
  /^(dependencies|optional-dependencies|requires)\s*=\s*[\[{]/i
]);
const PYPROJECT_DIRECT_REQUIREMENT_PATTERN = /^\"[a-z0-9][a-z0-9._-]*(?:\[[^\"\]]+\])?(?:\s*(?:[<>=!~]{1,2}|===|@).+)\"[,]?$/i;
const PYPROJECT_SECTION_ENTRY_PATTERNS = Object.freeze([
  /^\"[a-z0-9][a-z0-9._-]*(?:\[[^\"\]]+\])?(?:\s*(?:[<>=!~]{1,2}|===|@).*)?\"[,]?$/i,
  /^[a-z0-9._-]+\s*=\s*\"(?:\^|~|>=?|<=?|==|!=|===|\*|file:|path:|git\+|https?:|ssh:).+\"[,]?$/i,
  /^[a-z0-9._-]+\s*=\s*\[/i
]);
const RUNTIME_SUPPORT_TEXT_PATTERN = /\brequires-python\b|\bpython 3\.\d+\b|\bcuda\b|\bffmpeg\b|\bubuntu\b|\bwindows\b|\blinux\b|\bmacos\b|\bnvidia\b|\bplatform_system\b/;
const SECURITY_SECRET_TEXT_PATTERN = /authorization:\s*bearer|\b(access token|bearer token|secret|secrets|credential|permissions?)\b/;
const LABEL_SUPPORT_PATTERNS = Object.freeze({
  auth: Object.freeze([/\b(auth|authentication|authorization|permissions?|sessions?|cookies?|jwt|oauth|rbac|csrf|authz|credentials?)\b|(?:^|[^a-z0-9])token(?:[^a-z0-9]|$)/]),
  compliance: Object.freeze([/\b(compliance|pci|soc ?2|gdpr|hipaa|iso ?27001|nist|fedramp)\b/, /\bsarbanes[-\s]+oxley\b/, /\bsox\s+(?:compliance|controls?|control matrix|attestation|audit)\b/]),
  container: Object.freeze([/^docker\//, /dockerfile|compose\.ya?ml|\bcontainer\b|\bimage\b/]),
  cuda: Object.freeze([/\b(cuda|nvidia|gpu)\b/]),
  "destructive migration": Object.freeze([/\b(migration|migrate|drop column|drop table|table drop|column drop|rename table)\b/, /migrations?\//]),
  "facade module": Object.freeze([/__init__\.py$/, /^src\/[^/]+\/[^/]+\/__init__\.py$/, /^src\/[^/]+\/[^/]+\.py$/]),
  filesystem: Object.freeze([/\b(filesystem|file system|path separator|filepath|ntpath|posixpath|normpath|realpath|abspath)\b/, /\\\\/]),
  "heap usage": Object.freeze([/\b(heap|peak memory|memory usage|memory footprint|memory pressure|buffer copy|buffer copies)\b/]),
  "import time": Object.freeze([/\bimport[-\s]+time\b.*\b(duration|latency|measure(?:d|ment)?|benchmark|startup|cold start|ms)\b/, /\b(duration|latency|measure(?:d|ment)?|benchmark|startup|cold start|ms)\b.*\bimport[-\s]+time\b/]),
  "image tag": Object.freeze([/from\s+[^\n]+:[^\s]+/, /image:\s*[^\s]+:[^\s]+/]),
  lockfile: Object.freeze([/package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock|uv\.lock|pdm\.lock|requirements.*\.txt/]),
  process: Object.freeze([/(subprocess\.(?:popen|run|call|check_call|check_output)\s*\(|child_process\.(?:spawn|fork|exec|execfile|execsync|spawnsync)\s*\(|\bpopen\s*\(|\bcommunicate\s*\(|\bstdin\b|\bstdout\b|\bstderr\b|\bpipe(?:line)?\b)/]),
  "query param": Object.freeze([/\bquery param\b/, /\b(searchparams?|urlsearchparams)\b/, /\?[a-z_][a-z0-9_]*=/]),
  route: Object.freeze([/\broute\b/, /\brouter\b/, /@\w*router\.(get|post|put|delete|patch|options|head)\b/, /\.route\(/]),
  "route param": Object.freeze([/\b(route param|path param)\b/, /\/:[a-z_][a-z0-9_]*/, /\/\{[a-z_][a-z0-9_]*\}/, /\/<[a-z0-9_:.-]+>/]),
  shell: Object.freeze([/\bcmd(?:\.exe)?\b[^\n]{0,24}\/[ck]\b/, /\bpowershell(?:\.exe)?\b/, /\/(?:bin\/)?(?:sh|bash)\b/, /\b(?:bash|sh)\s+-[cl]\b/, /^#!.*\b(?:bash|sh)\b/m, /\bshell command\b/, /\bshell script\b/, /\bshell\s*=\s*(?:true|false)\b/]),
  "shell command": Object.freeze([/\bcmd(?:\.exe)?\b[^\n]{0,24}\/[ck]\b/, /\bpowershell(?:\.exe)?\b/, /\/(?:bin\/)?(?:sh|bash)\b/, /\b(?:bash|sh)\s+-[cl]\b/, /^#!.*\b(?:bash|sh)\b/m, /\bshell command\b/, /\bshell script\b/, /\bshell\s*=\s*(?:true|false)\b/]),
  "support matrix": Object.freeze([/requires-python|python 3\.|classifiers?.*python|support matrix|runs-on:|windows-latest|ubuntu-latest|macos-latest/]),
  windows: Object.freeze([/\b(windows|win32|winreg|shell32)\b/, /windows/]),
  token: Object.freeze([/(?:^|[^a-z0-9])token(?:[^a-z0-9]|$)|authorization:\s*bearer|\bjwt\b/]),
    ci: Object.freeze([
    /\b(ci|continuous integration|jobs:|steps:|runs-on:|uses:\s*actions\/)\b/
  ]),
  workflow: Object.freeze([
    /\.github\/workflows\/|\.gitlab-ci\.ya?ml|\.circleci\/config\.ya?ml|\.buildkite\/|jenkinsfile|\bworkflow\b|\bpipeline\b|\bjobs:\b|\bsteps:\b/
  ]),
  dependencies: Object.freeze([
    /\b(dependencies|devDependencies|peerDependencies|optionalDependencies|bundledDependencies|bundleDependencies)\b/,
    /package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock|uv\.lock|pdm\.lock|requirements.*\.txt/
  ]),
  security: Object.freeze([
    /(^|\/)(security|auth|policy|oauth|jwt|rbac|token)(\/|$)/,
    /(^|\/)(codeql|dependabot|security)(\.[^/]+)?$/
  ]),
  vulnerability: Object.freeze([
    /\b(vulnerab\w*|advisory|cve-\d{4}-\d+|ghsa-[a-z0-9-]+|exploit(?:able)?|security advisory)\b/,
    /(^|\/)(security|auth|policy|oauth|jwt|rbac|token)(\/|$)/
  ]),
  observability: Object.freeze([/\b(observability|telemetry|prometheus|opentelemetry|otel|metrics?|histogram|counter|gauge|tracing?|trace id|span(?: id)?|monitoring|datadog|newrelic)\b/, /(^|\/)(observability|telemetry|monitoring|metrics?|prometheus|tracing)(\/|$)/]),
  monitoring: Object.freeze([/\b(monitoring|prometheus|metrics?|histogram|counter|gauge|alert|sli|slo)\b/]),
  telemetry: Object.freeze([/\b(telemetry|opentelemetry|otel|trace(?:s|d|ing)?|span(?: id)?)\b/]),
  style: Object.freeze([/\b(class(name)?=|style=|styles?\.|theme|color|font|spacing|margin|padding|border(?:-radius)?|box-shadow|background|tailwind|var\(--|display\s*:\s*(flex|grid))\b/]),
  formatting: Object.freeze([/\b(prettier|format\(|formatted|whitespace|indent(?:ation)?)\b/, /\.(css|scss|sass|less)$/]),
});

module.exports = {
  ACCESSIBILITY_TEXT_PATTERN,
  AUTOMATION_TEXT_PATTERN,
  FEATURE_FLAG_TEXT_PATTERN,
  LABEL_SUPPORT_PATTERNS,
  LARGE_PR_GENERIC_LABEL_LIMIT,
  LARGE_PR_LABEL_LIMIT,
  LOCALIZATION_TEXT_PATTERN,
  MAINTENANCE_ONLY_CATEGORIES,
  MAX_PATCH_CHARS_PER_FILE,
  MAX_TOP_DIRECTORIES,
  MAX_TOP_FILES,
  MIN_BEHAVIORAL_ADDITION_LINES,
  MIN_PUBLIC_CONTRACT_MOVES,
  PACKAGE_JSON_DEPENDENCY_CONTEXT_PATTERNS,
  PACKAGE_JSON_DEPENDENCY_METADATA_KEYS,
  PACKAGE_JSON_DEPENDENCY_VALUE_PATTERN,
  PR_ABSTRACT_TYPE_LABELS,
  PR_CANONICAL_LABEL_REPLACEMENTS,
  PR_GENERIC_MAINTENANCE_LABELS,
  PR_INFRASTRUCTURE_LABELS,
  PR_PIPELINE_CLUTTER_LABELS,
  PR_SUPPRESSED_BY_TECHNICAL_LABELS,
  PYPROJECT_DEPENDENCY_CONTEXT_PATTERNS,
  PYPROJECT_DIRECT_REQUIREMENT_PATTERN,
  PYPROJECT_SECTION_ENTRY_PATTERNS,
  RUNTIME_SUPPORT_TEXT_PATTERN,
  SECURITY_SECRET_TEXT_PATTERN,
  SMALL_PR_FILE_LIMIT,
  SMALL_PR_GENERIC_LABEL_LIMIT,
  SMALL_PR_LABEL_LIMIT,
  SMALL_PR_TOTAL_CHANGE_LIMIT,
  SNAPSHOT_FILE
};
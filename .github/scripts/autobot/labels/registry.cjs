const {
  TECHNICAL_DEFAULT_ISSUE_LABELS,
  TECHNICAL_LABEL_DEFINITIONS,
  TECHNICAL_LABEL_GUIDANCE,
  TECHNICAL_LABEL_METADATA,
  TECHNICAL_LABELS,
  TECHNICAL_MAJOR_VERSION_LABELS,
  TECHNICAL_MINOR_VERSION_LABELS,
  TECHNICAL_RELEASE_RELEVANT_LABELS,
  TECHNICAL_SECONDARY_LABELS,
  collapseTechnicalLabels,
  getTechnicalLabelAncestors,
  getTechnicalLabelMetadata,
  isTechnicalDescendant,
  matchesTechnicalExpectation
} = require("./index.cjs");

class AutobotLabelRegistry {
  static LABEL_GUIDANCE = Object.freeze({
    "breaking-change": "Use when consumers must adapt because of an incompatible API, contract, config, or behavior change.",
    "security": "Use when auth, permissions, secrets, sanitization, or exploit mitigation clearly changed.",
    "api": "Use when endpoints, handlers, or request or response contracts changed.",
    "database": "Use when migrations, queries, schema storage, or database configuration materially changed.",
    "schema": "Use when shared schema artifacts such as GraphQL, JSON Schema, or Proto changed.",
    "compatibility": "Use when interoperability, backward compatibility, or adapter behavior changed.",
    "migration": "Use when upgrade steps, migration scripts, or transition logic changed.",
    "feature-flag": "Use when rollout toggles or gated behavior changed.",
    "runtime": "Use when runtime support, platform execution behavior, or environment compatibility changed.",
    "performance": "Use only when the summary explicitly supports speed, caching, latency, or memory effects.",
    "bug": "Use when the summary clearly describes incorrect behavior, failure, regression, or a broken path.",
    "enhancement": "Use when the change adds or expands meaningful capability.",
    "improvement": "Use mainly for issues that request a meaningful improvement or gap closure without a confirmed defect.",
    "proposal": "Use mainly for issues that describe a proposed direction or future work.",
    "documentation": "Use when documentation is a substantial part of the actual work.",
    "test": "Use when tests or test harnesses are added, changed, or repaired.",
    "workflow": "Use when GitHub workflow orchestration or job logic changed.",
    "automation": "Use when bot or autonomous repository automation behavior changed.",
    "github": "Use when GitHub-specific repository metadata, templates, or repo automation surfaces changed.",
    "ci": "Use when pipeline execution or validation job behavior changed.",
    "config": "Use when settings, manifests, or environment configuration changed.",
    "dependencies": "Use when dependency declarations or lockfiles changed.",
    "docker": "Use when Docker image or compose configuration changed.",
    "tooling": "Use when developer tools or scripts materially changed.",
    "dx": "Use when the main outcome is smoother local development workflow.",
    "cleanup": "Use when obsolete code or files were intentionally removed.",
    "chore": "Use only for small maintenance work that lacks a stronger label.",
    ...TECHNICAL_LABEL_GUIDANCE
  });

  static DEFAULT_ISSUE_LABELS = Object.freeze([...TECHNICAL_DEFAULT_ISSUE_LABELS]);

  static LABEL_DEFINITIONS = Object.freeze({
    bug: Object.freeze({ color: "d73a4a", description: "Something isn't working" }),
    enhancement: Object.freeze({ color: "a2eeef", description: "New feature or request" }),
    improvement: Object.freeze({ color: "a2eeef", description: "Improvement request or capability expansion" }),
    proposal: Object.freeze({ color: "bfd4f2", description: "Proposed future work or product change" }),
    documentation: Object.freeze({ color: "0075ca", description: "Improvements or additions to documentation" }),
    "breaking-change": Object.freeze({ color: "b60205", description: "Incompatible API changes" }),
    ui: Object.freeze({ color: "d4c5f9", description: "Visual or UI/UX improvements" }),
    performance: Object.freeze({ color: "5319e7", description: "Performance improvements" }),
    security: Object.freeze({ color: "e30c0c", description: "Security fixes and updates" }),
    refactor: Object.freeze({ color: "f29513", description: "Code change that neither fixes a bug nor adds a feature" }),
    test: Object.freeze({ color: "cc317c", description: "Adding, missing, or correcting tests" }),
    ci: Object.freeze({ color: "006b75", description: "CI/CD and workflow updates" }),
    dependencies: Object.freeze({ color: "0366d6", description: "Dependency updates" }),
    database: Object.freeze({ color: "fbca04", description: "Database migrations or schema changes" }),
    build: Object.freeze({ color: "89590b", description: "Build system and tooling updates" }),
    accessibility: Object.freeze({ color: "c2e0c6", description: "Accessibility (a11y) improvements" }),
    localization: Object.freeze({ color: "91d674", description: "Localization (i18n) and translation" }),
    api: Object.freeze({ color: "1d76db", description: "API endpoint or schema changes" }),
    infrastructure: Object.freeze({ color: "5e4a80", description: "Cloud infrastructure and IaC changes" }),
    config: Object.freeze({ color: "c5def5", description: "Configuration and environment changes" }),
    types: Object.freeze({ color: "2b67c6", description: "Type definitions and schema changes" }),
    logging: Object.freeze({ color: "bfdadc", description: "Logging, monitoring, and observability" }),
    deprecation: Object.freeze({ color: "ffa500", description: "Deprecated features with migration paths" }),
    chore: Object.freeze({ color: "ededed", description: "Maintenance tasks and cleanup" }),
    dx: Object.freeze({ color: "0e8a16", description: "Developer experience improvements" }),
    release: Object.freeze({ color: "1d76db", description: "Release/versioning/packaging changes" }),
    observability: Object.freeze({ color: "bfe5bf", description: "Metrics/tracing/alerts/monitoring setup" }),
    "docs-site": Object.freeze({ color: "bfd4f2", description: "Documentation site changes" }),
    runtime: Object.freeze({ color: "7057ff", description: "Runtime/platform compatibility changes" }),
    cleanup: Object.freeze({ color: "cfd3d7", description: "Dead code removal and cleanup" }),
    style: Object.freeze({ color: "fef2c0", description: "Formatting-only changes" }),
    lint: Object.freeze({ color: "fbca04", description: "Lint-only changes (rules/config/fixes)" }),
    formatting: Object.freeze({ color: "fef2c0", description: "Formatting changes" }),
    tooling: Object.freeze({ color: "c5def5", description: "Tooling/scripts/editor configuration changes" }),
    "release-notes": Object.freeze({ color: "1d76db", description: "Changelog/release notes updates" }),
    versioning: Object.freeze({ color: "1d76db", description: "Version bumps of the project itself" }),
    packaging: Object.freeze({ color: "1d76db", description: "Packaging/publishing configuration" }),
    workflow: Object.freeze({ color: "006b75", description: "Workflow logic changes beyond basic CI" }),
    automation: Object.freeze({ color: "006b75", description: "Automation/bots/scripts for repo management" }),
    quality: Object.freeze({ color: "d4c5f9", description: "Maintainability/readability improvements" }),
    stability: Object.freeze({ color: "d73a4a", description: "Reliability/flakiness reductions/hardening" }),
    "error-handling": Object.freeze({ color: "d73a4a", description: "Error handling improvements" }),
    validation: Object.freeze({ color: "e99695", description: "Validation/schema/guard changes" }),
    "feature-flag": Object.freeze({ color: "c2e0c6", description: "Feature flags/rollout toggles" }),
    migration: Object.freeze({ color: "fbca04", description: "Migrations/upgrade steps" }),
    compatibility: Object.freeze({ color: "7057ff", description: "Compatibility work, shims, polyfills" }),
    monitoring: Object.freeze({ color: "bfe5bf", description: "Monitoring/alerting/dashboards" }),
    telemetry: Object.freeze({ color: "bfe5bf", description: "Analytics/telemetry instrumentation" }),
    "logging-verbosity": Object.freeze({ color: "bfdadc", description: "Log level/volume/fields changes" }),
    "docs-api": Object.freeze({ color: "0075ca", description: "API documentation changes" }),
    examples: Object.freeze({ color: "bfd4f2", description: "Examples/sample code changes" }),
    devcontainer: Object.freeze({ color: "c5def5", description: "Devcontainer/local env changes" }),
    docker: Object.freeze({ color: "c5def5", description: "Docker/container changes" }),
    kubernetes: Object.freeze({ color: "c5def5", description: "Kubernetes/Helm/Kustomize changes" }),
    terraform: Object.freeze({ color: "c5def5", description: "Terraform changes" }),
    helm: Object.freeze({ color: "c5def5", description: "Helm chart changes" }),
    github: Object.freeze({ color: "0366d6", description: "GitHub templates/settings/codeowners changes" }),
    policy: Object.freeze({ color: "0366d6", description: "Policy/governance/support/security policy changes" }),
    license: Object.freeze({ color: "0366d6", description: "License/legal changes" }),
    "supply-chain": Object.freeze({ color: "e30c0c", description: "Supply chain hardening (SBOM/signing/provenance)" }),
    codegen: Object.freeze({ color: "c5def5", description: "Code generation templates/config/output" }),
    schema: Object.freeze({ color: "2b67c6", description: "Schema changes (proto/graphql/jsonschema)" }),
    serialization: Object.freeze({ color: "2b67c6", description: "Serialization format changes" }),
    ...TECHNICAL_LABEL_DEFINITIONS
  });

  static LABEL_PRIORITY = Object.freeze([
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
    "enhancement",
    "improvement",
    "deprecation",
    "bug",
    "validation",
    "stability",
    "error-handling",
    "ui",
    "accessibility",
    "localization",
    "documentation",
    "docs-site",
    "docs-api",
    "examples",
    "test",
    "workflow",
    "ci",
    "automation",
    "github",
    "config",
    "dependencies",
    "docker",
    "devcontainer",
    "packaging",
    "build",
    "release",
    "release-notes",
    "versioning",
    "tooling",
    "dx",
    "refactor",
    "quality",
    "cleanup",
    "logging",
    "logging-verbosity",
    "observability",
    "monitoring",
    "telemetry",
    "policy",
    "supply-chain",
    "serialization",
    "types",
    "codegen",
    "infrastructure",
    "kubernetes",
    "terraform",
    "helm",
    "proposal",
    "style",
    "formatting",
    "lint",
    "chore",
    ...TECHNICAL_LABELS.filter((label) => !["api", "data", "runtime", "pipeline", "repo"].includes(label))
  ]);

  static URGENT_SYNC_LABELS = Object.freeze(["breaking-change", "security"]);

  static VERSION_SENSITIVE_LABELS = Object.freeze([
    "breaking-change",
    "enhancement",
    "improvement",
    "deprecation",
    "security",
    "bug",
    "performance",
    "api",
    "database",
    "schema",
    "compatibility",
    "migration",
    "feature-flag",
    "runtime"
  ]);

  static RELEASE_CRITICAL_LABELS = Object.freeze([
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
    ...TECHNICAL_RELEASE_RELEVANT_LABELS
  ]);

  static VERSION_LABEL_ALIASES = Object.freeze({});

  static FORCE_RELEASE_TYPES = Object.freeze(["breaking-change", "security"]);

  static RELEASE_RELEVANT_LABELS = Object.freeze([
    "api",
    "breaking-change",
    "bug",
    "compatibility",
    "database",
    "deprecation",
    "enhancement",
    "feature-flag",
    "improvement",
    "migration",
    "performance",
    "runtime",
    "schema",
    "security",
    ...TECHNICAL_RELEASE_RELEVANT_LABELS
  ]);

  static SECONDARY_LABELS = Object.freeze([
    "chore",
    "ci",
    "cleanup",
    "config",
    "dependencies",
    "documentation",
    "dx",
    "formatting",
    "github",
    "lint",
    "quality",
    "refactor",
    "style",
    "test",
    "tooling",
    "workflow",
    ...TECHNICAL_SECONDARY_LABELS
  ]);

  static GENERIC_FALLBACK_LABELS = Object.freeze(["config", "dependencies", "tooling", "dx", "cleanup", "chore"]);

  static MAJOR_VERSION_LABELS = Object.freeze(["breaking-change", ...TECHNICAL_MAJOR_VERSION_LABELS]);

  static MINOR_VERSION_LABELS = Object.freeze(["deprecation", "enhancement", "improvement", ...TECHNICAL_MINOR_VERSION_LABELS]);

  static VALID_LABELS = new Set(Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS));

  static VERSION_BUMP_BY_LABEL = Object.freeze(
    Object.fromEntries(
      Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS).map((label) => {
        if (AutobotLabelRegistry.MAJOR_VERSION_LABELS.includes(label)) {
          return [label, "major"];
        }
        if (AutobotLabelRegistry.MINOR_VERSION_LABELS.includes(label)) {
          return [label, "minor"];
        }
        if (AutobotLabelRegistry.isReleaseRelevantLabel(label)) {
          return [label, "patch"];
        }
        return [label, "none"];
      })
    )
  );

  static normalizeLabelName(label) {
    const normalized = String(label || "").trim().toLowerCase();
    const LABEL_NORMALIZATION_ALIASES = Object.freeze(
      Object.fromEntries(
        Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS).flatMap((label) => getNormalizationEntries(label, label))
      )
    );
    return LABEL_NORMALIZATION_ALIASES[normalized] || AutobotLabelRegistry.VERSION_LABEL_ALIASES[normalized] || normalized;
  }

  static getLabelMetadata(label) {
    return getTechnicalLabelMetadata(AutobotLabelRegistry.normalizeLabelName(label));
  }

  static getLabelDepth(label) {
    return getTechnicalLabelAncestors(AutobotLabelRegistry.normalizeLabelName(label)).length;
  }

  static getLabelAncestors(label) {
    return getTechnicalLabelAncestors(AutobotLabelRegistry.normalizeLabelName(label));
  }

  static isTechnicalLabel(label) {
    return Boolean(AutobotLabelRegistry.getLabelMetadata(label));
  }

  static isDescendantLabel(descendant, ancestor) {
    const normalizedDescendant = AutobotLabelRegistry.normalizeLabelName(descendant);
    const normalizedAncestor = AutobotLabelRegistry.normalizeLabelName(ancestor);
    return isTechnicalDescendant(normalizedDescendant, normalizedAncestor);
  }

  static matchesExpectedLabel(actualLabel, expectedLabel) {
    return matchesTechnicalExpectation(
      AutobotLabelRegistry.normalizeLabelName(actualLabel),
      AutobotLabelRegistry.normalizeLabelName(expectedLabel)
    );
  }

  static collapseHierarchicalLabels(labels) {
    const normalizedLabels = [...new Set(
      (labels || [])
        .map((label) => AutobotLabelRegistry.normalizeLabelName(label))
        .filter((label) => AutobotLabelRegistry.VALID_LABELS.has(label))
    )];
    const technicalLabels = collapseTechnicalLabels(normalizedLabels.filter((label) => Boolean(AutobotLabelRegistry.getLabelMetadata(label))));
    const legacyLabels = normalizedLabels.filter((label) => {
      if (AutobotLabelRegistry.getLabelMetadata(label)) {
        return false;
      }
      return !technicalLabels.some((technicalLabel) => AutobotLabelRegistry.matchesExpectedLabel(technicalLabel, label));
    });
    return [...legacyLabels, ...technicalLabels];
  }

  static isReleaseRelevantLabel(label) {
    const normalized = AutobotLabelRegistry.normalizeLabelName(label);
    const metadata = AutobotLabelRegistry.getLabelMetadata(normalized);
    return Boolean(metadata?.releaseRelevant) || AutobotLabelRegistry.RELEASE_RELEVANT_LABELS.includes(normalized);
  }

  static hasLabelName(labels, expectedLabel) {
    return (labels || []).some((label) => {
      const labelName = typeof label === "string" ? label : label?.name;
      return AutobotLabelRegistry.normalizeLabelName(labelName) === expectedLabel;
    });
  }

  static hasReleaseRelevantLabel(labels) {
    return (labels || []).some((label) => {
      const labelName = typeof label === "string" ? label : label?.name;
      return AutobotLabelRegistry.isReleaseRelevantLabel(labelName);
    });
  }

  static uniqueValidLabels(labels) {
    return AutobotLabelRegistry.collapseHierarchicalLabels(labels);
  }

  static sortLabels(labels) {
    return AutobotLabelRegistry.uniqueValidLabels(labels)
      .sort((left, right) => {
        const leftRank = AutobotLabelRegistry.LABEL_PRIORITY.indexOf(left);
        const rightRank = AutobotLabelRegistry.LABEL_PRIORITY.indexOf(right);
        const normalizedLeftRank = leftRank === -1 ? AutobotLabelRegistry.LABEL_PRIORITY.length : leftRank;
        const normalizedRightRank = rightRank === -1 ? AutobotLabelRegistry.LABEL_PRIORITY.length : rightRank;
        if (normalizedLeftRank !== normalizedRightRank) {
          return normalizedLeftRank - normalizedRightRank;
        }
        const leftMetadata = AutobotLabelRegistry.getLabelMetadata(left);
        const rightMetadata = AutobotLabelRegistry.getLabelMetadata(right);
        const leftFamily = leftMetadata?.family || "zzzz";
        const rightFamily = rightMetadata?.family || "zzzz";
        if (leftFamily !== rightFamily) {
          return leftFamily.localeCompare(rightFamily);
        }
        const depthDelta = AutobotLabelRegistry.getLabelDepth(right) - AutobotLabelRegistry.getLabelDepth(left);
        return depthDelta || left.localeCompare(right);
      });
  }

  static sortTechnicalLabels(labels) {
    const technicalLabels = [...new Set(
      (labels || [])
        .map((label) => AutobotLabelRegistry.normalizeLabelName(label))
        .filter((label) => AutobotLabelRegistry.VALID_LABELS.has(label) && AutobotLabelRegistry.isTechnicalLabel(label))
    )];
    return technicalLabels.sort((left, right) => {
      const leftMetadata = AutobotLabelRegistry.getLabelMetadata(left);
      const rightMetadata = AutobotLabelRegistry.getLabelMetadata(right);
      const releaseRelevantDelta = Number(Boolean(rightMetadata?.releaseRelevant)) - Number(Boolean(leftMetadata?.releaseRelevant));
      if (releaseRelevantDelta !== 0) {
        return releaseRelevantDelta;
      }
      const secondaryDelta = Number(Boolean(leftMetadata?.secondary)) - Number(Boolean(rightMetadata?.secondary));
      if (secondaryDelta !== 0) {
        return secondaryDelta;
      }
      const depthDelta = AutobotLabelRegistry.getLabelDepth(right) - AutobotLabelRegistry.getLabelDepth(left);
      if (depthDelta !== 0) {
        return depthDelta;
      }
      const familyDelta = String(leftMetadata?.family || "zzzz").localeCompare(String(rightMetadata?.family || "zzzz"));
      return familyDelta || left.localeCompare(right);
    });
  }

  static resolveLabelLimit(limitValue) {
    if (limitValue === undefined || limitValue === null || limitValue === "") {
      return Number.POSITIVE_INFINITY;
    }
    const parsedLimit = Number(limitValue);
    if (!Number.isFinite(parsedLimit)) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.max(parsedLimit, 1);
  }

  static technicalLabelsOnly(labels, options = {}) {
    const limit = AutobotLabelRegistry.resolveLabelLimit(options.limit);
    const collapsed = collapseTechnicalLabels(AutobotLabelRegistry.sortTechnicalLabels(labels));
    return Number.isFinite(limit) ? collapsed.slice(0, limit) : collapsed;
  }

  static trimLowSignalLabels(labels, options = {}) {
    const limit = AutobotLabelRegistry.resolveLabelLimit(options.limit);
    const uniqueLabels = AutobotLabelRegistry.uniqueValidLabels(labels);
    if (uniqueLabels.length <= 3) {
      return Number.isFinite(limit) ? uniqueLabels.slice(0, limit) : uniqueLabels;
    }
    const versionCritical = AutobotLabelRegistry.VERSION_SENSITIVE_LABELS.filter((label) => uniqueLabels.includes(label));
    const primary = uniqueLabels.filter((label) => !AutobotLabelRegistry.SECONDARY_LABELS.includes(label) && !versionCritical.includes(label));
    const secondary = uniqueLabels.filter((label) => AutobotLabelRegistry.SECONDARY_LABELS.includes(label));
    if (!Number.isFinite(limit)) {
      return [...versionCritical, ...primary, ...secondary];
    }
    const cappedPrimary = [...versionCritical, ...primary].slice(0, limit);
    const remainingSlots = Math.max(limit - cappedPrimary.length, 0);
    const cappedSecondary = secondary.slice(0, remainingSlots);
    return [...cappedPrimary, ...cappedSecondary].slice(0, limit);
  }

  static parseAutobotLabels(raw) {
    if (!raw) {
      return [];
    }
    try {
      const cleaned = String(raw).replace(/```json\s*/gi, "").replace(/```\s*/gi, "").trim();
      const parsed = JSON.parse(cleaned);
      if (Array.isArray(parsed)) {
        return AutobotLabelRegistry.uniqueValidLabels(parsed);
      }
    } catch (error) {
      return AutobotLabelRegistry.uniqueValidLabels(
        String(raw)
          .replace(/[\[\]"'`]/g, "")
          .split(/[\n,]/)
      );
    }
    return [];
  }

  static labelNamesFromIssue(issue) {
    return AutobotLabelRegistry.uniqueValidLabels(
      (issue?.labels || []).map((label) => typeof label === "string" ? label : label.name)
    );
  }
}

const DEFAULT_ISSUE_LABELS = AutobotLabelRegistry.DEFAULT_ISSUE_LABELS;
const FORCE_RELEASE_TYPES = AutobotLabelRegistry.FORCE_RELEASE_TYPES;
const LABEL_DEFINITIONS = AutobotLabelRegistry.LABEL_DEFINITIONS;
const LABEL_GUIDANCE = AutobotLabelRegistry.LABEL_GUIDANCE;
const LABEL_PRIORITY = AutobotLabelRegistry.LABEL_PRIORITY;
const RELEASE_CRITICAL_LABELS = AutobotLabelRegistry.RELEASE_CRITICAL_LABELS;
const RELEASE_RELEVANT_LABELS = AutobotLabelRegistry.RELEASE_RELEVANT_LABELS;
const SECONDARY_LABELS = AutobotLabelRegistry.SECONDARY_LABELS;
const URGENT_SYNC_LABELS = AutobotLabelRegistry.URGENT_SYNC_LABELS;
const VALID_LABELS = AutobotLabelRegistry.VALID_LABELS;
const VERSION_BUMP_BY_LABEL = AutobotLabelRegistry.VERSION_BUMP_BY_LABEL;
const VERSION_LABEL_ALIASES = AutobotLabelRegistry.VERSION_LABEL_ALIASES;
const VERSION_SENSITIVE_LABELS = AutobotLabelRegistry.VERSION_SENSITIVE_LABELS;

function hasLabelName(labels, expectedLabel) {
  return AutobotLabelRegistry.hasLabelName(labels, expectedLabel);
}

function hasReleaseRelevantLabel(labels) {
  return AutobotLabelRegistry.hasReleaseRelevantLabel(labels);
}

function labelNamesFromIssue(issue) {
  return AutobotLabelRegistry.labelNamesFromIssue(issue);
}

function normalizeLabelName(label) {
  return AutobotLabelRegistry.normalizeLabelName(label);
}

function parseAutobotLabels(raw) {
  return AutobotLabelRegistry.parseAutobotLabels(raw);
}

function sortLabels(labels) {
  return AutobotLabelRegistry.sortLabels(labels);
}

function trimLowSignalLabels(labels, options) {
  return AutobotLabelRegistry.trimLowSignalLabels(labels, options);
}

function uniqueValidLabels(labels) {
  return AutobotLabelRegistry.uniqueValidLabels(labels);
}

function collapseHierarchicalLabels(labels) {
  return AutobotLabelRegistry.collapseHierarchicalLabels(labels);
}

function getLabelMetadata(label) {
  return AutobotLabelRegistry.getLabelMetadata(label);
}

function isDescendantLabel(descendant, ancestor) {
  return AutobotLabelRegistry.isDescendantLabel(descendant, ancestor);
}

function isReleaseRelevantLabel(label) {
  return AutobotLabelRegistry.isReleaseRelevantLabel(label);
}

function isTechnicalLabel(label) {
  return AutobotLabelRegistry.isTechnicalLabel(label);
}

function matchesExpectedLabel(actualLabel, expectedLabel) {
  return AutobotLabelRegistry.matchesExpectedLabel(actualLabel, expectedLabel);
}

function sortTechnicalLabels(labels) {
  return AutobotLabelRegistry.sortTechnicalLabels(labels);
}

function technicalLabelsOnly(labels, options) {
  return AutobotLabelRegistry.technicalLabelsOnly(labels, options);
}

function getNormalizationEntries(sourceLabel, targetLabel) {
  const normalized = String(sourceLabel || "").trim().toLowerCase().replace(/\s+/g, " ");
  const hyphenVariant = normalized.replace(/[\s_]+/g, "-");
  const underscoreVariant = normalized.replace(/[\s-]+/g, "_");
  const spaceVariant = normalized.replace(/[-_]+/g, " ");
  return [
    [normalized, targetLabel],
    [hyphenVariant, targetLabel],
    [underscoreVariant, targetLabel],
    [spaceVariant, targetLabel]
  ];
}

module.exports = {
  AutobotLabelRegistry,
  collapseHierarchicalLabels,
  DEFAULT_ISSUE_LABELS,
  FORCE_RELEASE_TYPES,
  LABEL_DEFINITIONS,
  LABEL_GUIDANCE,
  LABEL_PRIORITY,
  RELEASE_CRITICAL_LABELS,
  RELEASE_RELEVANT_LABELS,
  SECONDARY_LABELS,
  URGENT_SYNC_LABELS,
  VALID_LABELS,
  VERSION_BUMP_BY_LABEL,
  VERSION_LABEL_ALIASES,
  VERSION_SENSITIVE_LABELS,
  getLabelMetadata,
  hasLabelName,
  hasReleaseRelevantLabel,
  isDescendantLabel,
  isReleaseRelevantLabel,
  isTechnicalLabel,
  labelNamesFromIssue,
  matchesExpectedLabel,
  normalizeLabelName,
  parseAutobotLabels,
  sortLabels,
  sortTechnicalLabels,
  technicalLabelsOnly,
  trimLowSignalLabels,
  uniqueValidLabels
};

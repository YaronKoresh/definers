const { LABEL_DEFINITIONS } = require("./autobot_labels");

const MAX_EMITTED_LABELS = 12;

const SCOPE_MULTIPLIERS = Object.freeze({
  private: 0.7,
  subsystem: 1.0,
  public: 1.3,
  repo: 1.5
});

const CONFIDENCE_MULTIPLIERS = Object.freeze({
  structural: 1.0,
  corroborated: 0.8,
  lexical: 0.45,
  title: 0.25
});

const LABEL_THRESHOLDS = Object.freeze({
  retain: 0.35,
  emit: 0.55,
  primary: 0.72
});

const SEMVER_THRESHOLDS = Object.freeze({
  major: 0.72,
  minor: 0.58,
  patch: 0.0
});

const CATEGORY_DEFINITIONS = Object.freeze({
  "public-contract": Object.freeze({ description: "Public import, export, facade, and consumer contract surface." }),
  "api-surface": Object.freeze({ description: "API handlers, endpoints, request, and response contracts." }),
  "schema-contract": Object.freeze({ description: "Shared schema artifacts and contract-bearing fields." }),
  "serialization-format": Object.freeze({ description: "Serialization and wire-format compatibility." }),
  "type-contract": Object.freeze({ description: "Shared type-level contract changes." }),
  "database-persistence": Object.freeze({ description: "Persistence, storage, and database behavior." }),
  "migration-requirement": Object.freeze({ description: "Explicit or implied upgrade and migration work." }),
  "compatibility-interoperability": Object.freeze({ description: "Backward compatibility, shims, adapters, and interop." }),
  "runtime-platform": Object.freeze({ description: "Runtime support, OS, Python, CUDA, and platform policy." }),
  "security-policy": Object.freeze({ description: "Security, auth, permission, and policy surfaces." }),
  "supply-chain": Object.freeze({ description: "Supply-chain hardening and provenance surfaces." }),
  "feature-capability": Object.freeze({ description: "User-visible capability expansion or contraction." }),
  "reliability-bugfix": Object.freeze({ description: "Bug fixes, regression repair, resilience, and stability." }),
  "performance-resource": Object.freeze({ description: "Speed, cache, memory, batching, and resource behavior." }),
  "feature-flag-rollout": Object.freeze({ description: "Rollout toggles, flags, and staged enablement." }),
  "ui-ux": Object.freeze({ description: "Visual, UI, and interaction surface changes." }),
  accessibility: Object.freeze({ description: "Accessibility-specific UI and interaction changes." }),
  localization: Object.freeze({ description: "Localization and i18n surface changes." }),
  "documentation-examples": Object.freeze({ description: "Documentation, examples, and user guidance surfaces." }),
  "docs-site-surface": Object.freeze({ description: "Documentation-site specific behavior or content." }),
  "test-surface": Object.freeze({ description: "Test coverage, harness, fixtures, and assertions." }),
  "workflow-ci": Object.freeze({ description: "Workflow orchestration and CI execution surfaces." }),
  "github-management": Object.freeze({ description: "GitHub templates, metadata, and repository management." }),
  "automation-bot-management": Object.freeze({ description: "Bots and repository automation behavior." }),
  "config-environment": Object.freeze({ description: "Settings, manifests, env, and repository configuration." }),
  "dependencies-packaging": Object.freeze({ description: "Dependencies, lockfiles, packaging, and publish surfaces." }),
  "docker-container": Object.freeze({ description: "Docker image and container runtime surfaces." }),
  "infrastructure-platform": Object.freeze({ description: "Infrastructure and orchestration surfaces." }),
  "tooling-dx": Object.freeze({ description: "Tooling and developer-experience surfaces." }),
  "cleanup-removal": Object.freeze({ description: "Cleanup and obsolete code removal." }),
  "refactor-quality": Object.freeze({ description: "Refactor, maintainability, and code-quality surfaces." }),
  "build-release": Object.freeze({ description: "Build, release, changelog, and packaging workflow surfaces." }),
  "observability-telemetry": Object.freeze({ description: "Logging, telemetry, tracing, and monitoring surfaces." }),
  "validation-guards": Object.freeze({ description: "Validation and guardrail surfaces." }),
  "logging-behavior": Object.freeze({ description: "Logging content and verbosity surfaces." })
});

const SEMVER_CHANNEL_DEFINITIONS = Object.freeze({
  "major-public-contract": Object.freeze({ level: "major", description: "Destructive public import or API contract impact." }),
  "major-data-compatibility": Object.freeze({ level: "major", description: "Destructive schema, serialization, or database compatibility impact." }),
  "major-runtime-support-drop": Object.freeze({ level: "major", description: "Dropped supported runtime or platform policy." }),
  "major-migration-required": Object.freeze({ level: "major", description: "Upgrade now requires migration work." }),
  "minor-additive-capability": Object.freeze({ level: "minor", description: "Additive feature or capability expansion." }),
  "minor-runtime-support-add": Object.freeze({ level: "minor", description: "Added supported runtime or platform target." }),
  "minor-additive-contract": Object.freeze({ level: "minor", description: "Additive API or schema contract expansion." }),
  "minor-operational-capability": Object.freeze({ level: "minor", description: "Operational capability expansion in workflow, Docker, packaging, or tooling." }),
  "patch-maintenance": Object.freeze({ level: "patch", description: "Maintenance-only impact." })
});

const LABEL_PRIORITY = Object.freeze([
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
  "chore"
]);

const EVIDENCE_RULES = Object.freeze({
  "removed-public-export": Object.freeze({
    baseWeight: 1.2,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "public-contract": 1.0,
      "compatibility-interoperability": 0.85,
      "cleanup-removal": 0.2
    }),
    impactWeights: Object.freeze({
      "major-public-contract": 1.0
    }),
    labelBoosts: Object.freeze({
      "breaking-change": 0.95,
      compatibility: 0.55
    }),
    hardSemver: "major"
  }),
  "renamed-public-module": Object.freeze({
    baseWeight: 0.95,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "public-contract": 0.95,
      "compatibility-interoperability": 0.75
    }),
    impactWeights: Object.freeze({
      "major-public-contract": 0.78
    }),
    labelBoosts: Object.freeze({
      "breaking-change": 0.75,
      compatibility: 0.45,
      cleanup: 0.2
    })
  }),
  "added-public-capability": Object.freeze({
    baseWeight: 0.95,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-capability": 0.95,
      "public-contract": 0.6,
      "api-surface": 0.3
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.95,
      "minor-additive-contract": 0.55
    }),
    labelBoosts: Object.freeze({
      enhancement: 0.85,
      api: 0.25
    }),
    hardSemver: "minor"
  }),
  "destructive-api-contract": Object.freeze({
    baseWeight: 1.15,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "api-surface": 1.0,
      "public-contract": 0.85,
      "compatibility-interoperability": 0.7
    }),
    impactWeights: Object.freeze({
      "major-public-contract": 0.95
    }),
    labelBoosts: Object.freeze({
      api: 0.9,
      "breaking-change": 0.85,
      compatibility: 0.45
    }),
    hardSemver: "major"
  }),
  "additive-api-contract": Object.freeze({
    baseWeight: 0.9,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "api-surface": 1.0,
      "feature-capability": 0.75,
      "public-contract": 0.45
    }),
    impactWeights: Object.freeze({
      "minor-additive-contract": 0.9,
      "minor-additive-capability": 0.55
    }),
    labelBoosts: Object.freeze({
      api: 0.85,
      enhancement: 0.55
    }),
    hardSemver: "minor"
  }),
  "destructive-schema-contract": Object.freeze({
    baseWeight: 1.1,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "schema-contract": 1.0,
      "serialization-format": 0.65,
      "public-contract": 0.55
    }),
    impactWeights: Object.freeze({
      "major-data-compatibility": 0.95
    }),
    labelBoosts: Object.freeze({
      schema: 0.85,
      serialization: 0.45,
      "breaking-change": 0.8
    }),
    hardSemver: "major"
  }),
  "additive-schema-contract": Object.freeze({
    baseWeight: 0.85,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "schema-contract": 0.95,
      "serialization-format": 0.45,
      "feature-capability": 0.55
    }),
    impactWeights: Object.freeze({
      "minor-additive-contract": 0.8
    }),
    labelBoosts: Object.freeze({
      schema: 0.8,
      enhancement: 0.45
    }),
    hardSemver: "minor"
  }),
  "serialization-format-change": Object.freeze({
    baseWeight: 0.9,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "serialization-format": 1.0,
      "schema-contract": 0.55
    }),
    impactWeights: Object.freeze({
      "major-data-compatibility": 0.55,
      "minor-additive-contract": 0.3
    }),
    labelBoosts: Object.freeze({
      serialization: 0.9,
      schema: 0.35
    })
  }),
  "type-contract-change": Object.freeze({
    baseWeight: 0.8,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "type-contract": 1.0,
      "schema-contract": 0.4
    }),
    impactWeights: Object.freeze({
      "minor-additive-contract": 0.35,
      "major-data-compatibility": 0.25
    }),
    labelBoosts: Object.freeze({
      types: 0.9,
      schema: 0.2
    })
  }),
  "destructive-database-change": Object.freeze({
    baseWeight: 1.05,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "database-persistence": 1.0,
      "migration-requirement": 0.65,
      "schema-contract": 0.4
    }),
    impactWeights: Object.freeze({
      "major-data-compatibility": 0.85,
      "major-migration-required": 0.6
    }),
    labelBoosts: Object.freeze({
      database: 0.9,
      migration: 0.45,
      "breaking-change": 0.6
    }),
    hardSemver: "major"
  }),
  "additive-database-change": Object.freeze({
    baseWeight: 0.78,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "database-persistence": 1.0,
      "feature-capability": 0.4
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.45
    }),
    labelBoosts: Object.freeze({
      database: 0.75,
      enhancement: 0.2
    })
  }),
  "explicit-migration-marker": Object.freeze({
    baseWeight: 0.95,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "migration-requirement": 1.0,
      "compatibility-interoperability": 0.55
    }),
    impactWeights: Object.freeze({
      "major-migration-required": 0.95
    }),
    labelBoosts: Object.freeze({
      migration: 0.95,
      compatibility: 0.35,
      "breaking-change": 0.45
    })
  }),
  "runtime-support-added": Object.freeze({
    baseWeight: 0.95,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "runtime-platform": 1.0,
      "feature-capability": 0.45,
      "dependencies-packaging": 0.35,
      "docker-container": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-runtime-support-add": 1.0,
      "minor-additive-capability": 0.45
    }),
    labelBoosts: Object.freeze({
      runtime: 0.9,
      enhancement: 0.35,
      dependencies: 0.25
    }),
    hardSemver: "minor"
  }),
  "runtime-support-dropped": Object.freeze({
    baseWeight: 1.15,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "runtime-platform": 1.0,
      "compatibility-interoperability": 0.7,
      "dependencies-packaging": 0.35
    }),
    impactWeights: Object.freeze({
      "major-runtime-support-drop": 1.0
    }),
    labelBoosts: Object.freeze({
      runtime: 0.95,
      compatibility: 0.35,
      "breaking-change": 0.8
    }),
    hardSemver: "major"
  }),
  "runtime-policy-change": Object.freeze({
    baseWeight: 0.85,
    defaultScope: "repo",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "runtime-platform": 1.0,
      "config-environment": 0.45
    }),
    impactWeights: Object.freeze({
      "major-runtime-support-drop": 0.4,
      "minor-runtime-support-add": 0.4
    }),
    labelBoosts: Object.freeze({
      runtime: 0.8,
      config: 0.3
    })
  }),
  "compatibility-shim": Object.freeze({
    baseWeight: 0.72,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "compatibility-interoperability": 1.0,
      "public-contract": 0.25
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      compatibility: 0.85,
      improvement: 0.2
    })
  }),
  "compatibility-drop": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "compatibility-interoperability": 1.0,
      "public-contract": 0.65,
      "runtime-platform": 0.45
    }),
    impactWeights: Object.freeze({
      "major-public-contract": 0.7,
      "major-runtime-support-drop": 0.45
    }),
    labelBoosts: Object.freeze({
      compatibility: 0.95,
      "breaking-change": 0.7
    }),
    hardSemver: "major"
  }),
  "security-auth-change": Object.freeze({
    baseWeight: 0.88,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "security-policy": 1.0,
      "public-contract": 0.25,
      "validation-guards": 0.35
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.25,
      "major-public-contract": 0.25
    }),
    labelBoosts: Object.freeze({
      security: 0.95,
      validation: 0.2
    })
  }),
  "security-secret-handling": Object.freeze({
    baseWeight: 0.92,
    defaultScope: "repo",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "security-policy": 1.0,
      "supply-chain": 0.35,
      "config-environment": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.25
    }),
    labelBoosts: Object.freeze({
      security: 0.95,
      policy: 0.25
    })
  }),
  "supply-chain-hardening": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "repo",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "supply-chain": 1.0,
      "security-policy": 0.45,
      "build-release": 0.35
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.35
    }),
    labelBoosts: Object.freeze({
      "supply-chain": 0.95,
      security: 0.5
    })
  }),
  "feature-flag-added": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-flag-rollout": 1.0,
      "feature-capability": 0.4,
      "config-environment": 0.3
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.4,
      "minor-operational-capability": 0.45
    }),
    labelBoosts: Object.freeze({
      "feature-flag": 0.95,
      enhancement: 0.25
    }),
    hardSemver: "minor"
  }),
  "feature-flag-contract-change": Object.freeze({
    baseWeight: 0.98,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "feature-flag-rollout": 1.0,
      "public-contract": 0.55,
      "compatibility-interoperability": 0.45
    }),
    impactWeights: Object.freeze({
      "major-public-contract": 0.7
    }),
    labelBoosts: Object.freeze({
      "feature-flag": 0.95,
      "breaking-change": 0.55
    })
  }),
  "feature-capability-added": Object.freeze({
    baseWeight: 0.88,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-capability": 1.0,
      "ui-ux": 0.25,
      "tooling-dx": 0.2
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.78
    }),
    labelBoosts: Object.freeze({
      enhancement: 0.9,
      improvement: 0.25
    }),
    hardSemver: "minor"
  }),
  "reliability-bugfix": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "reliability-bugfix": 1.0,
      "validation-guards": 0.25,
      "performance-resource": 0.1
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.8
    }),
    labelBoosts: Object.freeze({
      bug: 0.95,
      stability: 0.25
    })
  }),
  "error-handling-hardening": Object.freeze({
    baseWeight: 0.74,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "reliability-bugfix": 0.7,
      "validation-guards": 0.45
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.55
    }),
    labelBoosts: Object.freeze({
      "error-handling": 0.95,
      stability: 0.35
    })
  }),
  "validation-guard-change": Object.freeze({
    baseWeight: 0.7,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "validation-guards": 1.0,
      "security-policy": 0.2,
      "reliability-bugfix": 0.25
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.4
    }),
    labelBoosts: Object.freeze({
      validation: 0.95
    })
  }),
  "performance-optimization": Object.freeze({
    baseWeight: 0.85,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "performance-resource": 1.0,
      "feature-capability": 0.15,
      "tooling-dx": 0.1
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.35,
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      performance: 0.95,
      improvement: 0.2
    })
  }),
  "ui-surface-change": Object.freeze({
    baseWeight: 0.82,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "ui-ux": 1.0,
      "feature-capability": 0.25,
      accessibility: 0.15,
      localization: 0.1
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.25
    }),
    labelBoosts: Object.freeze({
      ui: 0.95,
      enhancement: 0.2
    })
  }),
  "accessibility-improvement": Object.freeze({
    baseWeight: 0.8,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      accessibility: 1.0,
      "ui-ux": 0.45
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.2
    }),
    labelBoosts: Object.freeze({
      accessibility: 0.95,
      ui: 0.35
    })
  }),
  "localization-change": Object.freeze({
    baseWeight: 0.78,
    defaultScope: "public",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      localization: 1.0,
      "ui-ux": 0.3,
      "documentation-examples": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.2
    }),
    labelBoosts: Object.freeze({
      localization: 0.95,
      ui: 0.15
    })
  }),
  "documentation-surface": Object.freeze({
    baseWeight: 0.95,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "documentation-examples": 1.0
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.65
    }),
    labelBoosts: Object.freeze({
      documentation: 0.95
    })
  }),
  "docs-site-surface": Object.freeze({
    baseWeight: 0.9,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "docs-site-surface": 1.0,
      "documentation-examples": 0.65
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.5
    }),
    labelBoosts: Object.freeze({
      "docs-site": 0.95,
      documentation: 0.45
    })
  }),
  "docs-api-surface": Object.freeze({
    baseWeight: 0.88,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "documentation-examples": 0.9,
      "api-surface": 0.25
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.45
    }),
    labelBoosts: Object.freeze({
      "docs-api": 0.95,
      documentation: 0.45
    })
  }),
  "examples-surface": Object.freeze({
    baseWeight: 0.82,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "documentation-examples": 0.95,
      "feature-capability": 0.15
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.4
    }),
    labelBoosts: Object.freeze({
      examples: 0.95,
      documentation: 0.35
    })
  }),
  "tests-added": Object.freeze({
    baseWeight: 0.88,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "test-surface": 1.0,
      "reliability-bugfix": 0.15
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.6
    }),
    labelBoosts: Object.freeze({
      test: 0.95
    })
  }),
  "tests-removed": Object.freeze({
    baseWeight: 0.82,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "test-surface": 1.0,
      "cleanup-removal": 0.2
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.4
    }),
    labelBoosts: Object.freeze({
      test: 0.9,
      cleanup: 0.25
    })
  }),
  "workflow-orchestration-change": Object.freeze({
    baseWeight: 0.88,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "workflow-ci": 1.0,
      "automation-bot-management": 0.35,
      "build-release": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.45,
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      workflow: 0.95,
      automation: 0.25
    })
  }),
  "ci-execution-change": Object.freeze({
    baseWeight: 0.82,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "workflow-ci": 0.9,
      "build-release": 0.35
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.3,
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      ci: 0.95,
      workflow: 0.45
    })
  }),
  "github-management-change": Object.freeze({
    baseWeight: 0.8,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "github-management": 1.0,
      "workflow-ci": 0.2,
      "security-policy": 0.15
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      github: 0.95
    })
  }),
  "automation-bot-change": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "automation-bot-management": 1.0,
      "workflow-ci": 0.4,
      "github-management": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.35,
      "patch-maintenance": 0.3
    }),
    labelBoosts: Object.freeze({
      automation: 0.95,
      workflow: 0.3,
      github: 0.2
    })
  }),
  "config-surface-change": Object.freeze({
    baseWeight: 0.76,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "config-environment": 1.0,
      "runtime-platform": 0.25,
      "dependencies-packaging": 0.2
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.35,
      "minor-operational-capability": 0.2
    }),
    labelBoosts: Object.freeze({
      config: 0.95
    })
  }),
  "dependency-capability-expansion": Object.freeze({
    baseWeight: 0.82,
    defaultScope: "repo",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "dependencies-packaging": 1.0,
      "feature-capability": 0.35,
      "runtime-platform": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.4,
      "minor-additive-capability": 0.25
    }),
    labelBoosts: Object.freeze({
      dependencies: 0.95,
      enhancement: 0.2,
      packaging: 0.15
    })
  }),
  "dependency-compatibility-tightening": Object.freeze({
    baseWeight: 0.92,
    defaultScope: "repo",
    defaultConfidence: "corroborated",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "dependencies-packaging": 1.0,
      "compatibility-interoperability": 0.55,
      "runtime-platform": 0.35
    }),
    impactWeights: Object.freeze({
      "major-runtime-support-drop": 0.45,
      "major-public-contract": 0.35
    }),
    labelBoosts: Object.freeze({
      dependencies: 0.95,
      compatibility: 0.35,
      packaging: 0.2
    })
  }),
  "docker-runtime-expansion": Object.freeze({
    baseWeight: 0.86,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "docker-container": 1.0,
      "runtime-platform": 0.7,
      "dependencies-packaging": 0.2
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.55,
      "minor-runtime-support-add": 0.4
    }),
    labelBoosts: Object.freeze({
      docker: 0.95,
      runtime: 0.45
    }),
    hardSemver: "minor"
  }),
  "docker-runtime-drop": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "docker-container": 1.0,
      "runtime-platform": 0.75,
      "compatibility-interoperability": 0.3
    }),
    impactWeights: Object.freeze({
      "major-runtime-support-drop": 0.75
    }),
    labelBoosts: Object.freeze({
      docker: 0.95,
      runtime: 0.45,
      "breaking-change": 0.45
    })
  }),
  "tooling-surface-change": Object.freeze({
    baseWeight: 0.76,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "tooling-dx": 1.0,
      "workflow-ci": 0.2
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.2,
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      tooling: 0.95,
      dx: 0.45
    })
  }),
  "devcontainer-surface-change": Object.freeze({
    baseWeight: 0.8,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "tooling-dx": 0.8,
      "docker-container": 0.35,
      "config-environment": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.2,
      "patch-maintenance": 0.25
    }),
    labelBoosts: Object.freeze({
      devcontainer: 0.95,
      tooling: 0.35,
      dx: 0.3
    })
  }),
  "infrastructure-surface-change": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "infrastructure-platform": 1.0,
      "runtime-platform": 0.25,
      "config-environment": 0.2
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.3,
      "patch-maintenance": 0.25
    }),
    labelBoosts: Object.freeze({
      infrastructure: 0.95,
      kubernetes: 0.35,
      terraform: 0.35,
      helm: 0.35
    })
  }),
  "cleanup-removal": Object.freeze({
    baseWeight: 0.75,
    defaultScope: "subsystem",
    defaultConfidence: "structural",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "cleanup-removal": 1.0,
      "refactor-quality": 0.2
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.35
    }),
    labelBoosts: Object.freeze({
      cleanup: 0.95,
      refactor: 0.2
    })
  }),
  "refactor-maintainability": Object.freeze({
    baseWeight: 0.72,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "refactor-quality": 1.0,
      "tooling-dx": 0.15
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.3
    }),
    labelBoosts: Object.freeze({
      refactor: 0.95,
      quality: 0.55
    })
  }),
  "build-release-change": Object.freeze({
    baseWeight: 0.84,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "build-release": 1.0,
      "dependencies-packaging": 0.35,
      "workflow-ci": 0.3
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.35,
      "patch-maintenance": 0.25
    }),
    labelBoosts: Object.freeze({
      build: 0.95,
      release: 0.35,
      packaging: 0.25
    })
  }),
  "release-notes-change": Object.freeze({
    baseWeight: 0.72,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "build-release": 0.7,
      "documentation-examples": 0.45
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.2
    }),
    labelBoosts: Object.freeze({
      "release-notes": 0.95,
      documentation: 0.25
    })
  }),
  "versioning-change": Object.freeze({
    baseWeight: 0.78,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "build-release": 0.85,
      "dependencies-packaging": 0.25
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.25
    }),
    labelBoosts: Object.freeze({
      versioning: 0.95,
      release: 0.2
    })
  }),
  "logging-behavior-change": Object.freeze({
    baseWeight: 0.74,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "logging-behavior": 1.0,
      "observability-telemetry": 0.45
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.25
    }),
    labelBoosts: Object.freeze({
      logging: 0.95,
      "logging-verbosity": 0.45
    })
  }),
  "observability-telemetry-change": Object.freeze({
    baseWeight: 0.8,
    defaultScope: "subsystem",
    defaultConfidence: "corroborated",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "observability-telemetry": 1.0,
      "logging-behavior": 0.25
    }),
    impactWeights: Object.freeze({
      "minor-operational-capability": 0.25,
      "patch-maintenance": 0.2
    }),
    labelBoosts: Object.freeze({
      observability: 0.95,
      monitoring: 0.45,
      telemetry: 0.65
    })
  }),
  "codegen-surface-change": Object.freeze({
    baseWeight: 0.76,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "build-release": 0.45,
      "tooling-dx": 0.45,
      "schema-contract": 0.2
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.25,
      "minor-operational-capability": 0.2
    }),
    labelBoosts: Object.freeze({
      codegen: 0.95,
      tooling: 0.35
    })
  }),
  "license-policy-change": Object.freeze({
    baseWeight: 0.78,
    defaultScope: "repo",
    defaultConfidence: "structural",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "github-management": 0.45,
      "security-policy": 0.3,
      "documentation-examples": 0.2
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.2
    }),
    labelBoosts: Object.freeze({
      license: 0.95,
      policy: 0.45,
      github: 0.2
    })
  }),
  "issue-bug-report": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "destructive",
    categoryWeights: Object.freeze({
      "reliability-bugfix": 1.0
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.55
    }),
    labelBoosts: Object.freeze({
      bug: 0.95
    })
  }),
  "issue-enhancement-request": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-capability": 1.0
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.45
    }),
    labelBoosts: Object.freeze({
      enhancement: 0.95
    })
  }),
  "issue-improvement-request": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-capability": 0.6,
      "tooling-dx": 0.35,
      "refactor-quality": 0.2
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.25
    }),
    labelBoosts: Object.freeze({
      improvement: 0.95,
      enhancement: 0.25
    })
  }),
  "issue-proposal-request": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "feature-capability": 0.5,
      "documentation-examples": 0.15
    }),
    impactWeights: Object.freeze({
      "minor-additive-capability": 0.15
    }),
    labelBoosts: Object.freeze({
      proposal: 1.55,
      enhancement: 0.25
    })
  }),
  "issue-documentation-report": Object.freeze({
    baseWeight: 1.0,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "additive",
    categoryWeights: Object.freeze({
      "documentation-examples": 1.0
    }),
    impactWeights: Object.freeze({
      "patch-maintenance": 0.4
    }),
    labelBoosts: Object.freeze({
      documentation: 0.95
    })
  }),
  "issue-runtime-context": Object.freeze({
    baseWeight: 0.58,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "runtime-platform": 1.0
    }),
    impactWeights: Object.freeze({
      "minor-runtime-support-add": 0.15,
      "major-runtime-support-drop": 0.15
    }),
    labelBoosts: Object.freeze({
      runtime: 0.85
    })
  }),
  "issue-api-context": Object.freeze({
    baseWeight: 0.58,
    defaultScope: "public",
    defaultConfidence: "lexical",
    defaultPolarity: "mixed",
    categoryWeights: Object.freeze({
      "api-surface": 1.0
    }),
    impactWeights: Object.freeze({
      "minor-additive-contract": 0.15,
      "major-public-contract": 0.15
    }),
    labelBoosts: Object.freeze({
      api: 0.85
    })
  })
});

const LABEL_SCORE_RECIPES = Object.freeze({
  "breaking-change": Object.freeze([
    Object.freeze({ type: "impact", key: "major-public-contract", weight: 1.25 }),
    Object.freeze({ type: "impact", key: "major-data-compatibility", weight: 1.1 }),
    Object.freeze({ type: "impact", key: "major-runtime-support-drop", weight: 1.0 }),
    Object.freeze({ type: "impact", key: "major-migration-required", weight: 1.05 }),
    Object.freeze({ type: "category", key: "public-contract", channel: "destructive", weight: 1.1 }),
    Object.freeze({ type: "category", key: "api-surface", channel: "destructive", weight: 0.95 }),
    Object.freeze({ type: "category", key: "schema-contract", channel: "destructive", weight: 0.9 }),
    Object.freeze({ type: "category", key: "database-persistence", channel: "destructive", weight: 0.8 }),
    Object.freeze({ type: "category", key: "compatibility-interoperability", channel: "destructive", weight: 0.85 }),
    Object.freeze({ type: "category", key: "runtime-platform", channel: "destructive", weight: 0.8 }),
    Object.freeze({ type: "category", key: "migration-requirement", weight: 0.7 })
  ]),
  security: Object.freeze([
    Object.freeze({ type: "category", key: "security-policy", weight: 1.0 }),
    Object.freeze({ type: "category", key: "supply-chain", weight: 0.45 })
  ]),
  api: Object.freeze([
    Object.freeze({ type: "category", key: "api-surface", weight: 1.0 }),
    Object.freeze({ type: "category", key: "public-contract", weight: 0.25 })
  ]),
  database: Object.freeze([
    Object.freeze({ type: "category", key: "database-persistence", weight: 1.0 }),
    Object.freeze({ type: "category", key: "migration-requirement", weight: 0.2 })
  ]),
  schema: Object.freeze([
    Object.freeze({ type: "category", key: "schema-contract", weight: 1.0 }),
    Object.freeze({ type: "category", key: "serialization-format", weight: 0.3 }),
    Object.freeze({ type: "category", key: "type-contract", weight: 0.2 })
  ]),
  serialization: Object.freeze([
    Object.freeze({ type: "category", key: "serialization-format", weight: 1.0 })
  ]),
  types: Object.freeze([
    Object.freeze({ type: "category", key: "type-contract", weight: 1.0 })
  ]),
  compatibility: Object.freeze([
    Object.freeze({ type: "category", key: "compatibility-interoperability", weight: 1.0 }),
    Object.freeze({ type: "category", key: "public-contract", channel: "destructive", weight: 0.25 })
  ]),
  migration: Object.freeze([
    Object.freeze({ type: "category", key: "migration-requirement", weight: 1.0 }),
    Object.freeze({ type: "impact", key: "major-migration-required", weight: 0.35 })
  ]),
  "feature-flag": Object.freeze([
    Object.freeze({ type: "category", key: "feature-flag-rollout", weight: 1.0 })
  ]),
  runtime: Object.freeze([
    Object.freeze({ type: "category", key: "runtime-platform", weight: 1.0 }),
    Object.freeze({ type: "impact", key: "major-runtime-support-drop", weight: 0.35 }),
    Object.freeze({ type: "impact", key: "minor-runtime-support-add", weight: 0.35 })
  ]),
  performance: Object.freeze([
    Object.freeze({ type: "category", key: "performance-resource", weight: 1.0 })
  ]),
  bug: Object.freeze([
    Object.freeze({ type: "category", key: "reliability-bugfix", weight: 1.0 }),
    Object.freeze({ type: "impact", key: "patch-maintenance", weight: 0.15 })
  ]),
  enhancement: Object.freeze([
    Object.freeze({ type: "category", key: "feature-capability", channel: "additive", weight: 1.0 }),
    Object.freeze({ type: "impact", key: "minor-additive-capability", weight: 0.7 }),
    Object.freeze({ type: "impact", key: "minor-additive-contract", weight: 0.25 }),
    Object.freeze({ type: "category", key: "ui-ux", channel: "additive", weight: 0.2 })
  ]),
  improvement: Object.freeze([
    Object.freeze({ type: "category", key: "feature-capability", channel: "additive", weight: 0.5 }),
    Object.freeze({ type: "category", key: "tooling-dx", weight: 0.45 }),
    Object.freeze({ type: "category", key: "refactor-quality", weight: 0.35 }),
    Object.freeze({ type: "category", key: "performance-resource", weight: 0.25 })
  ]),
  proposal: Object.freeze([
    Object.freeze({ type: "category", key: "feature-capability", weight: 0.25 })
  ]),
  deprecation: Object.freeze([
    Object.freeze({ type: "category", key: "compatibility-interoperability", channel: "destructive", weight: 0.35 }),
    Object.freeze({ type: "category", key: "migration-requirement", weight: 0.45 })
  ]),
  documentation: Object.freeze([
    Object.freeze({ type: "category", key: "documentation-examples", weight: 1.0 })
  ]),
  "docs-site": Object.freeze([
    Object.freeze({ type: "category", key: "docs-site-surface", weight: 1.0 }),
    Object.freeze({ type: "category", key: "documentation-examples", weight: 0.35 })
  ]),
  "docs-api": Object.freeze([
    Object.freeze({ type: "category", key: "documentation-examples", weight: 0.65 }),
    Object.freeze({ type: "category", key: "api-surface", weight: 0.2 })
  ]),
  examples: Object.freeze([
    Object.freeze({ type: "category", key: "documentation-examples", weight: 0.7 })
  ]),
  test: Object.freeze([
    Object.freeze({ type: "category", key: "test-surface", weight: 1.0 })
  ]),
  workflow: Object.freeze([
    Object.freeze({ type: "category", key: "workflow-ci", weight: 1.0 })
  ]),
  ci: Object.freeze([
    Object.freeze({ type: "category", key: "workflow-ci", weight: 0.75 })
  ]),
  automation: Object.freeze([
    Object.freeze({ type: "category", key: "automation-bot-management", weight: 1.0 }),
    Object.freeze({ type: "category", key: "workflow-ci", weight: 0.25 })
  ]),
  github: Object.freeze([
    Object.freeze({ type: "category", key: "github-management", weight: 1.0 })
  ]),
  config: Object.freeze([
    Object.freeze({ type: "category", key: "config-environment", weight: 1.0 })
  ]),
  dependencies: Object.freeze([
    Object.freeze({ type: "category", key: "dependencies-packaging", weight: 1.0 })
  ]),
  docker: Object.freeze([
    Object.freeze({ type: "category", key: "docker-container", weight: 1.0 }),
    Object.freeze({ type: "category", key: "runtime-platform", weight: 0.25 })
  ]),
  devcontainer: Object.freeze([
    Object.freeze({ type: "category", key: "tooling-dx", weight: 0.45 }),
    Object.freeze({ type: "category", key: "docker-container", weight: 0.25 })
  ]),
  infrastructure: Object.freeze([
    Object.freeze({ type: "category", key: "infrastructure-platform", weight: 1.0 })
  ]),
  kubernetes: Object.freeze([
    Object.freeze({ type: "category", key: "infrastructure-platform", weight: 0.7 })
  ]),
  terraform: Object.freeze([
    Object.freeze({ type: "category", key: "infrastructure-platform", weight: 0.7 })
  ]),
  helm: Object.freeze([
    Object.freeze({ type: "category", key: "infrastructure-platform", weight: 0.7 })
  ]),
  tooling: Object.freeze([
    Object.freeze({ type: "category", key: "tooling-dx", weight: 1.0 })
  ]),
  dx: Object.freeze([
    Object.freeze({ type: "category", key: "tooling-dx", weight: 0.75 }),
    Object.freeze({ type: "category", key: "refactor-quality", weight: 0.2 })
  ]),
  cleanup: Object.freeze([
    Object.freeze({ type: "category", key: "cleanup-removal", weight: 1.0 })
  ]),
  refactor: Object.freeze([
    Object.freeze({ type: "category", key: "refactor-quality", weight: 1.0 })
  ]),
  quality: Object.freeze([
    Object.freeze({ type: "category", key: "refactor-quality", weight: 0.8 })
  ]),
  build: Object.freeze([
    Object.freeze({ type: "category", key: "build-release", weight: 1.0 })
  ]),
  packaging: Object.freeze([
    Object.freeze({ type: "category", key: "dependencies-packaging", weight: 0.65 }),
    Object.freeze({ type: "category", key: "build-release", weight: 0.45 })
  ]),
  versioning: Object.freeze([
    Object.freeze({ type: "category", key: "build-release", weight: 0.7 })
  ]),
  release: Object.freeze([
    Object.freeze({ type: "category", key: "build-release", weight: 0.75 })
  ]),
  "release-notes": Object.freeze([
    Object.freeze({ type: "category", key: "build-release", weight: 0.45 }),
    Object.freeze({ type: "category", key: "documentation-examples", weight: 0.35 })
  ]),
  logging: Object.freeze([
    Object.freeze({ type: "category", key: "logging-behavior", weight: 1.0 })
  ]),
  "logging-verbosity": Object.freeze([
    Object.freeze({ type: "category", key: "logging-behavior", weight: 0.75 })
  ]),
  observability: Object.freeze([
    Object.freeze({ type: "category", key: "observability-telemetry", weight: 1.0 })
  ]),
  monitoring: Object.freeze([
    Object.freeze({ type: "category", key: "observability-telemetry", weight: 0.8 })
  ]),
  telemetry: Object.freeze([
    Object.freeze({ type: "category", key: "observability-telemetry", weight: 0.9 })
  ]),
  validation: Object.freeze([
    Object.freeze({ type: "category", key: "validation-guards", weight: 1.0 })
  ]),
  stability: Object.freeze([
    Object.freeze({ type: "category", key: "reliability-bugfix", weight: 0.7 }),
    Object.freeze({ type: "category", key: "validation-guards", weight: 0.2 })
  ]),
  "error-handling": Object.freeze([
    Object.freeze({ type: "category", key: "reliability-bugfix", weight: 0.45 }),
    Object.freeze({ type: "category", key: "validation-guards", weight: 0.35 })
  ]),
  accessibility: Object.freeze([
    Object.freeze({ type: "category", key: "accessibility", weight: 1.0 }),
    Object.freeze({ type: "category", key: "ui-ux", weight: 0.2 })
  ]),
  localization: Object.freeze([
    Object.freeze({ type: "category", key: "localization", weight: 1.0 })
  ]),
  policy: Object.freeze([
    Object.freeze({ type: "category", key: "security-policy", weight: 0.6 }),
    Object.freeze({ type: "category", key: "github-management", weight: 0.2 })
  ]),
  license: Object.freeze([
    Object.freeze({ type: "category", key: "github-management", weight: 0.35 }),
    Object.freeze({ type: "category", key: "documentation-examples", weight: 0.2 })
  ]),
  "supply-chain": Object.freeze([
    Object.freeze({ type: "category", key: "supply-chain", weight: 1.0 })
  ]),
  codegen: Object.freeze([
    Object.freeze({ type: "category", key: "build-release", weight: 0.3 }),
    Object.freeze({ type: "category", key: "schema-contract", weight: 0.2 })
  ]),
  style: Object.freeze([]),
  formatting: Object.freeze([]),
  lint: Object.freeze([]),
  chore: Object.freeze([])
});

const VALID_LABELS = new Set(Object.keys(LABEL_DEFINITIONS));

function clampScore(value) {
  const numericValue = Number.isFinite(Number(value)) ? Number(value) : 0;
  if (numericValue <= 0) {
    return 0;
  }
  if (numericValue >= 1) {
    return 1;
  }
  return numericValue;
}

function normalizeAccumulatedScore(rawScore) {
  const numericRawScore = Math.max(Number(rawScore) || 0, 0);
  return clampScore(1 - Math.exp(-numericRawScore));
}

function createCategoryAccumulator() {
  return {
    raw: 0,
    score: 0,
    additiveRaw: 0,
    additiveScore: 0,
    destructiveRaw: 0,
    destructiveScore: 0,
    evidenceCount: 0
  };
}

function createImpactAccumulator(level) {
  return {
    level,
    raw: 0,
    score: 0,
    evidenceCount: 0
  };
}

function normalizeEvidenceItem(evidenceInput) {
  const evidence = evidenceInput && typeof evidenceInput === "object" ? evidenceInput : {};
  const ruleId = String(evidence.ruleId || "").trim();
  const rule = EVIDENCE_RULES[ruleId];
  if (!rule) {
    throw new Error(`Unknown deterministic scorer rule: ${ruleId || "(empty)"}`);
  }
  const scope = String(evidence.scope || rule.defaultScope || "subsystem");
  const confidence = String(evidence.confidence || rule.defaultConfidence || "corroborated");
  const polarity = String(evidence.polarity || rule.defaultPolarity || "neutral");
  const scopeMultiplier = SCOPE_MULTIPLIERS[scope] || SCOPE_MULTIPLIERS.subsystem;
  const confidenceMultiplier = CONFIDENCE_MULTIPLIERS[confidence] || CONFIDENCE_MULTIPLIERS.lexical;
  const occurrenceCount = Math.max(Number(evidence.occurrenceCount) || 1, 1);
  const baseWeight = Number(rule.baseWeight || 0) * occurrenceCount;
  const adjustedWeight = baseWeight * scopeMultiplier * confidenceMultiplier;
  return {
    ruleId,
    scope,
    confidence,
    polarity,
    occurrenceCount,
    baseWeight,
    adjustedWeight,
    categoryWeights: rule.categoryWeights || {},
    impactWeights: rule.impactWeights || {},
    labelBoosts: rule.labelBoosts || {},
    hardSemver: rule.hardSemver || null,
    metadata: evidence.metadata || null
  };
}

function readRecipeSourceValue(source, categoryScores, impactScores) {
  if (source.type === "impact") {
    return impactScores[source.key]?.score || 0;
  }
  if (source.type === "category") {
    const category = categoryScores[source.key];
    if (!category) {
      return 0;
    }
    if (source.channel === "additive") {
      return category.additiveScore;
    }
    if (source.channel === "destructive") {
      return category.destructiveScore;
    }
    return category.score;
  }
  return 0;
}

function scoreLabelRecipes(categoryScores, impactScores, directLabelRawScores) {
  const labelScores = {};
  for (const label of Object.keys(LABEL_DEFINITIONS)) {
    const recipe = LABEL_SCORE_RECIPES[label] || [];
    let rawScore = directLabelRawScores[label] || 0;
    for (const source of recipe) {
      rawScore += readRecipeSourceValue(source, categoryScores, impactScores) * source.weight;
    }
    const normalizedScore = normalizeAccumulatedScore(rawScore);
    labelScores[label] = {
      label,
      raw: rawScore,
      score: normalizedScore,
      retained: normalizedScore >= LABEL_THRESHOLDS.retain,
      emitted: normalizedScore >= LABEL_THRESHOLDS.emit,
      primary: normalizedScore >= LABEL_THRESHOLDS.primary
    };
  }
  return labelScores;
}

function rankEmittedLabels(labelScores, options = {}) {
  const maxLabels = Math.max(Number(options.maxLabels) || MAX_EMITTED_LABELS, 1);
  const emittedLabels = Object.values(labelScores)
    .filter((entry) => entry.emitted && VALID_LABELS.has(entry.label))
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      const leftPriority = LABEL_PRIORITY.indexOf(left.label);
      const rightPriority = LABEL_PRIORITY.indexOf(right.label);
      const normalizedLeftPriority = leftPriority === -1 ? LABEL_PRIORITY.length : leftPriority;
      const normalizedRightPriority = rightPriority === -1 ? LABEL_PRIORITY.length : rightPriority;
      return normalizedLeftPriority - normalizedRightPriority || left.label.localeCompare(right.label);
    });
  return emittedLabels.slice(0, maxLabels);
}

function deriveSemverDecision(impactScores, hardSemverSignals) {
  if (hardSemverSignals.major.length > 0) {
    return {
      decision: "major",
      hardRule: true,
      hardSignals: hardSemverSignals.major,
      majorScore: 1,
      minorScore: Math.max(
        impactScores["minor-additive-capability"]?.score || 0,
        impactScores["minor-runtime-support-add"]?.score || 0,
        impactScores["minor-additive-contract"]?.score || 0,
        impactScores["minor-operational-capability"]?.score || 0
      ),
      patchScore: impactScores["patch-maintenance"]?.score || 0
    };
  }
  const majorScore = Math.max(
    impactScores["major-public-contract"]?.score || 0,
    impactScores["major-data-compatibility"]?.score || 0,
    impactScores["major-runtime-support-drop"]?.score || 0,
    impactScores["major-migration-required"]?.score || 0
  );
  if (majorScore >= SEMVER_THRESHOLDS.major) {
    return {
      decision: "major",
      hardRule: false,
      hardSignals: [],
      majorScore,
      minorScore: Math.max(
        impactScores["minor-additive-capability"]?.score || 0,
        impactScores["minor-runtime-support-add"]?.score || 0,
        impactScores["minor-additive-contract"]?.score || 0,
        impactScores["minor-operational-capability"]?.score || 0
      ),
      patchScore: impactScores["patch-maintenance"]?.score || 0
    };
  }
  if (hardSemverSignals.minor.length > 0) {
    return {
      decision: "minor",
      hardRule: true,
      hardSignals: hardSemverSignals.minor,
      majorScore,
      minorScore: 1,
      patchScore: impactScores["patch-maintenance"]?.score || 0
    };
  }
  const minorScore = Math.max(
    impactScores["minor-additive-capability"]?.score || 0,
    impactScores["minor-runtime-support-add"]?.score || 0,
    impactScores["minor-additive-contract"]?.score || 0,
    impactScores["minor-operational-capability"]?.score || 0
  );
  if (minorScore >= SEMVER_THRESHOLDS.minor) {
    return {
      decision: "minor",
      hardRule: false,
      hardSignals: [],
      majorScore,
      minorScore,
      patchScore: impactScores["patch-maintenance"]?.score || 0
    };
  }
  return {
    decision: "patch",
    hardRule: false,
    hardSignals: [],
    majorScore,
    minorScore,
    patchScore: Math.max(impactScores["patch-maintenance"]?.score || 0, 0)
  };
}

function scoreDeterministicEvidence(payload) {
  const evidenceInputs = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.evidenceItems)
      ? payload.evidenceItems
      : [];
  const normalizedEvidence = evidenceInputs.map(normalizeEvidenceItem);
  const categoryScores = Object.fromEntries(
    Object.keys(CATEGORY_DEFINITIONS).map((categoryId) => [categoryId, createCategoryAccumulator()])
  );
  const impactScores = Object.fromEntries(
    Object.entries(SEMVER_CHANNEL_DEFINITIONS).map(([impactId, definition]) => [impactId, createImpactAccumulator(definition.level)])
  );
  const directLabelRawScores = Object.fromEntries(
    Object.keys(LABEL_DEFINITIONS).map((label) => [label, 0])
  );
  const hardSemverSignals = { major: [], minor: [] };

  for (const evidence of normalizedEvidence) {
    for (const [categoryId, weight] of Object.entries(evidence.categoryWeights)) {
      const categoryScore = categoryScores[categoryId];
      if (!categoryScore) {
        continue;
      }
      const contribution = evidence.adjustedWeight * weight;
      categoryScore.raw += contribution;
      categoryScore.evidenceCount += 1;
      if (evidence.polarity === "additive") {
        categoryScore.additiveRaw += contribution;
      } else if (evidence.polarity === "destructive") {
        categoryScore.destructiveRaw += contribution;
      } else {
        categoryScore.additiveRaw += contribution * 0.5;
        categoryScore.destructiveRaw += contribution * 0.5;
      }
    }
    for (const [impactId, weight] of Object.entries(evidence.impactWeights)) {
      const impactScore = impactScores[impactId];
      if (!impactScore) {
        continue;
      }
      const contribution = evidence.adjustedWeight * weight;
      impactScore.raw += contribution;
      impactScore.evidenceCount += 1;
    }
    for (const [label, weight] of Object.entries(evidence.labelBoosts)) {
      if (!Object.prototype.hasOwnProperty.call(directLabelRawScores, label)) {
        continue;
      }
      directLabelRawScores[label] += evidence.adjustedWeight * weight;
    }
    if (evidence.hardSemver === "major") {
      hardSemverSignals.major.push(evidence.ruleId);
    }
    if (evidence.hardSemver === "minor") {
      hardSemverSignals.minor.push(evidence.ruleId);
    }
  }

  for (const categoryScore of Object.values(categoryScores)) {
    categoryScore.score = normalizeAccumulatedScore(categoryScore.raw);
    categoryScore.additiveScore = normalizeAccumulatedScore(categoryScore.additiveRaw);
    categoryScore.destructiveScore = normalizeAccumulatedScore(categoryScore.destructiveRaw);
  }
  for (const impactScore of Object.values(impactScores)) {
    impactScore.score = normalizeAccumulatedScore(impactScore.raw);
  }

  const labelScores = scoreLabelRecipes(categoryScores, impactScores, directLabelRawScores);
  const emittedLabels = rankEmittedLabels(labelScores, payload?.options || {});
  const primaryLabels = emittedLabels.filter((entry) => entry.primary);
  const semver = deriveSemverDecision(impactScores, hardSemverSignals);

  return {
    normalizedEvidence,
    categoryScores,
    impactScores,
    labelScores,
    emittedLabels,
    primaryLabels,
    semver
  };
}

module.exports = {
  CATEGORY_DEFINITIONS,
  CONFIDENCE_MULTIPLIERS,
  EVIDENCE_RULES,
  LABEL_PRIORITY,
  LABEL_SCORE_RECIPES,
  LABEL_THRESHOLDS,
  MAX_EMITTED_LABELS,
  SCOPE_MULTIPLIERS,
  SEMVER_CHANNEL_DEFINITIONS,
  SEMVER_THRESHOLDS,
  clampScore,
  deriveSemverDecision,
  normalizeAccumulatedScore,
  normalizeEvidenceItem,
  rankEmittedLabels,
  scoreDeterministicEvidence
};
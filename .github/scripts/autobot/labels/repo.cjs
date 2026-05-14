const { createNode } = require("./helpers.cjs");

const REPO_TREE = createNode("repo", { color: "0366d6", description: "Repository file changes.", secondary: true }, [
  createNode("docs", { color: "0075ca", description: "Documentation changes.", legacyMatches: ["documentation", "docs-site", "docs-api", "examples", "release-notes"], secondary: true }, [
    createNode("reference", { color: "0075ca", description: "Reference documentation changes.", legacyMatches: ["documentation", "docs-api"], secondary: true }, [
      createNode("api doc", { color: "0075ca", description: "API reference changes.", legacyMatches: ["documentation", "docs-api"], secondary: true }, [
        createNode("endpoint doc", { color: "0075ca", description: "Endpoint documentation changes.", secondary: true }),
        createNode("auth doc", { color: "0075ca", description: "Authentication documentation changes.", secondary: true })
      ]),
      createNode("architecture doc", { color: "0075ca", description: "Architecture documentation changes.", secondary: true }, [
        createNode("module map", { color: "0075ca", description: "Module map changes.", secondary: true }),
        createNode("data flow", { color: "0075ca", description: "Data flow documentation changes.", secondary: true })
      ]),
      createNode("cli doc", { color: "0075ca", description: "CLI reference changes.", secondary: true }),
      createNode("config doc", { color: "0075ca", description: "Configuration reference changes.", secondary: true }),
      createNode("schema doc", { color: "0075ca", description: "Schema reference changes.", secondary: true }),
      createNode("migration doc", { color: "0075ca", description: "Migration reference changes.", secondary: true })
    ]),
    createNode("guide", { color: "0075ca", description: "Guide documentation changes.", legacyMatches: ["documentation"], secondary: true }, [
      createNode("install guide", { color: "0075ca", description: "Installation guide changes.", secondary: true }, [
        createNode("quickstart", { color: "0075ca", description: "Quickstart changes.", secondary: true }),
        createNode("env setup", { color: "0075ca", description: "Environment setup changes.", secondary: true })
      ]),
      createNode("troubleshoot guide", { color: "0075ca", description: "Troubleshooting guide changes.", secondary: true }, [
        createNode("error guide", { color: "0075ca", description: "Error guide changes.", secondary: true }),
        createNode("faq", { color: "0075ca", description: "FAQ changes.", secondary: true })
      ])
    ])
  ]),
  createNode("files", { color: "0366d6", description: "Repository file changes.", secondary: true }, [
    createNode("manifest", { color: "0366d6", description: "Manifest and dependency file changes.", legacyMatches: ["config", "dependencies"], secondary: true }, [
      createNode("pyproject", { color: "0366d6", description: "pyproject changes.", legacyMatches: ["config"], secondary: true }, [
        createNode("dependency group", { color: "0366d6", description: "Dependency group changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true }),
        createNode("task runner", { color: "0366d6", description: "Task runner changes.", legacyMatches: ["tooling", "dx"], secondary: true })
      ]),
      createNode("lockfile", { color: "0366d6", description: "Lockfile changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true }, [
        createNode("package lock", { color: "0366d6", description: "package-lock changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true }),
        createNode("poetry lock", { color: "0366d6", description: "poetry.lock changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true })
      ]),
      createNode("package manifest", { color: "0366d6", description: "package.json and package metadata changes.", legacyMatches: ["config", "dependencies"], secondary: true }),
      createNode("requirement file", { color: "0366d6", description: "requirements file changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true }),
      createNode("constraint file", { color: "0366d6", description: "constraint file changes.", legacyMatches: ["dependencies"], releaseRelevant: true, secondary: true })
    ]),
    createNode("template", { color: "0366d6", description: "Repository template changes.", legacyMatches: ["github"], secondary: true }, [
      createNode("codeowners", { color: "0366d6", description: "CODEOWNERS changes.", legacyMatches: ["github"], secondary: true }, [
        createNode("review rule", { color: "0366d6", description: "Review rule changes.", secondary: true }),
        createNode("path ownership", { color: "0366d6", description: "Path ownership changes.", secondary: true })
      ]),
      createNode("issue form", { color: "0366d6", description: "Issue form changes.", legacyMatches: ["github"], secondary: true }, [
        createNode("bug form", { color: "0366d6", description: "Bug form changes.", secondary: true }),
        createNode("feature form", { color: "0366d6", description: "Feature form changes.", secondary: true })
      ])
    ])
  ])
]);

module.exports = {
  REPO_TREE
};
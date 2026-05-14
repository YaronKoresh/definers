const { createNode } = require("./helpers.cjs");

const PACKAGE_TREE = createNode("package", { color: "1d76db", description: "Package and import-surface changes.", releaseRelevant: true }, [
  createNode("public export", { color: "1d76db", description: "Public package export changes.", legacyMatches: ["api"], releaseRelevant: true }, [
    createNode("facade module", { color: "1d76db", description: "Facade module changes.", legacyMatches: ["api"], releaseRelevant: true, versionBump: "minor" }),
    createNode("shim module", { color: "1d76db", description: "Compatibility shim module changes.", legacyMatches: ["compatibility"], releaseRelevant: true }),
    createNode("import path", { color: "1d76db", description: "Import path changes.", legacyMatches: ["api", "compatibility"], releaseRelevant: true, versionBump: "major" })
  ])
]);

module.exports = {
  PACKAGE_TREE
};
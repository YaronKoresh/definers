const { API_TREE } = require("./api.cjs");
const { DATA_TREE } = require("./data.cjs");
const { INTERFACE_TREE } = require("./interface.cjs");
const { EXTRA_LEAVES } = require("./leaves.cjs");
const { PACKAGE_TREE } = require("./package.cjs");
const { PIPELINE_TREE } = require("./pipeline.cjs");
const { REPO_TREE } = require("./repo.cjs");
const { ROLLOUT_TREE } = require("./rollout.cjs");
const { RUNTIME_TREE } = require("./runtime.cjs");
const {
  buildLegacyMatchMap,
  buildTechnicalLabelIndex,
  collapseTechnicalLabels: collapseTechnicalLabelsWithIndex,
  getTechnicalLabelAncestors: getTechnicalLabelAncestorsWithIndex,
  isTechnicalDescendant: isTechnicalDescendantWithIndex,
  matchesTechnicalExpectation: matchesTechnicalExpectationWithIndex
} = require("./helpers.cjs");

const TECHNICAL_LABEL_TREES = Object.freeze([
  API_TREE,
  DATA_TREE,
  RUNTIME_TREE,
  INTERFACE_TREE,
  PACKAGE_TREE,
  ROLLOUT_TREE,
  PIPELINE_TREE,
  REPO_TREE
]);

const {
  TECHNICAL_LABEL_DEFINITIONS,
  TECHNICAL_LABEL_GUIDANCE,
  TECHNICAL_LABEL_METADATA,
  TECHNICAL_LABELS,
  TECHNICAL_RELEASE_RELEVANT_LABELS,
  TECHNICAL_SECONDARY_LABELS,
  TECHNICAL_MAJOR_VERSION_LABELS,
  TECHNICAL_MINOR_VERSION_LABELS
} = buildTechnicalLabelIndex(TECHNICAL_LABEL_TREES, EXTRA_LEAVES);

const TECHNICAL_DEFAULT_ISSUE_LABELS = Object.freeze([
  "route",
  "request body",
  "response body",
  "jwt",
  "vulnerability",
  "compliance",
  "hardening",
  "pen-test",
  "rbac",
  "json schema",
  "proto schema",
  "migration file",
  "python version",
  "windows",
  "linux",
  "view",
  "github actions",
  "unit test",
  "integration test",
  "public export",
  "pyproject",
  "package lock",
  "api doc",
  "quickstart",
  "codeowners",
  "issue form"
]);

const LEGACY_LABEL_MATCH_MAP = buildLegacyMatchMap(TECHNICAL_LABEL_METADATA);

function getTechnicalLabelMetadata(label) {
  return TECHNICAL_LABEL_METADATA[label] || null;
}

function getTechnicalLabelAncestors(label) {
  return getTechnicalLabelAncestorsWithIndex(label, TECHNICAL_LABEL_METADATA);
}

function isTechnicalDescendant(descendant, ancestor) {
  return isTechnicalDescendantWithIndex(descendant, ancestor, TECHNICAL_LABEL_METADATA, LEGACY_LABEL_MATCH_MAP);
}

function matchesTechnicalExpectation(actualLabel, expectedLabel) {
  return matchesTechnicalExpectationWithIndex(actualLabel, expectedLabel, TECHNICAL_LABEL_METADATA, LEGACY_LABEL_MATCH_MAP);
}

function collapseTechnicalLabels(labels) {
  return collapseTechnicalLabelsWithIndex(labels, TECHNICAL_LABEL_METADATA, LEGACY_LABEL_MATCH_MAP);
}

module.exports = {
  API_TREE,
  DATA_TREE,
  EXTRA_LEAVES,
  INTERFACE_TREE,
  LEGACY_LABEL_MATCH_MAP,
  PACKAGE_TREE,
  PIPELINE_TREE,
  REPO_TREE,
  ROLLOUT_TREE,
  RUNTIME_TREE,
  TECHNICAL_DEFAULT_ISSUE_LABELS,
  TECHNICAL_LABEL_DEFINITIONS,
  TECHNICAL_LABEL_GUIDANCE,
  TECHNICAL_LABEL_METADATA,
  TECHNICAL_LABEL_TREES,
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
};
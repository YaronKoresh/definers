function createNode(label, options = {}, children = []) {
  return Object.freeze({
    children: Object.freeze(children),
    color: options.color || "c5def5",
    description: options.description || `Technical label for ${label}.`,
    legacyMatches: Object.freeze(options.legacyMatches || []),
    releaseRelevant: Boolean(options.releaseRelevant),
    secondary: Boolean(options.secondary),
    versionBump: options.versionBump || "patch",
    label
  });
}

function flattenTree(node, parent = null, family = node.label, inherited = {}) {
  const metadata = {
    color: node.color || inherited.color || "c5def5",
    description: node.description || inherited.description || `Technical label for ${node.label}.`,
    family,
    label: node.label,
    legacyMatches: [...new Set([...(inherited.legacyMatches || []), ...(node.legacyMatches || [])])],
    parent,
    releaseRelevant: node.releaseRelevant === undefined ? Boolean(inherited.releaseRelevant) : Boolean(node.releaseRelevant),
    secondary: node.secondary === undefined ? Boolean(inherited.secondary) : Boolean(node.secondary),
    versionBump: node.versionBump || inherited.versionBump || "patch"
  };
  const children = [metadata];
  for (const child of node.children || []) {
    children.push(...flattenTree(child, node.label, family, metadata));
  }
  return children;
}

function buildLegacyMatchMap(technicalLabelMetadata) {
  const descendants = new Map();
  for (const [label, metadata] of Object.entries(technicalLabelMetadata)) {
    for (const legacyLabel of metadata.legacyMatches || []) {
      if (!descendants.has(legacyLabel)) {
        descendants.set(legacyLabel, new Set());
      }
      descendants.get(legacyLabel).add(label);
    }
  }
  return Object.freeze(
    Object.fromEntries(
      [...descendants.entries()].map(([label, matches]) => [label, Object.freeze([...matches])])
    )
  );
}

function getTechnicalLabelAncestors(label, technicalLabelMetadata) {
  const ancestors = [];
  let current = technicalLabelMetadata[label];
  while (current && current.parent) {
    ancestors.push(current.parent);
    current = technicalLabelMetadata[current.parent] || null;
  }
  return ancestors;
}

function isTechnicalDescendant(descendant, ancestor, technicalLabelMetadata, legacyLabelMatchMap) {
  if (!descendant || !ancestor || descendant === ancestor) {
    return false;
  }
  if (legacyLabelMatchMap[ancestor]?.includes(descendant)) {
    return true;
  }
  return getTechnicalLabelAncestors(descendant, technicalLabelMetadata).includes(ancestor);
}

function matchesTechnicalExpectation(actualLabel, expectedLabel, technicalLabelMetadata, legacyLabelMatchMap) {
  if (!actualLabel || !expectedLabel) {
    return false;
  }
  if (actualLabel === expectedLabel) {
    return true;
  }
  const actualMetadata = technicalLabelMetadata[actualLabel] || null;
  if (actualMetadata && expectedLabel === "breaking-change" && actualMetadata.versionBump === "major") {
    return true;
  }
  if (actualMetadata && ["enhancement", "improvement"].includes(expectedLabel) && actualMetadata.versionBump === "minor") {
    return true;
  }
  if (isTechnicalDescendant(actualLabel, expectedLabel, technicalLabelMetadata, legacyLabelMatchMap)) {
    return true;
  }
  const actualMatches = actualMetadata?.legacyMatches || [];
  return actualMatches.includes(expectedLabel);
}

function collapseTechnicalLabels(labels, technicalLabelMetadata, legacyLabelMatchMap) {
  const uniqueLabels = [...new Set(labels)].filter((label) => technicalLabelMetadata[label]);
  return uniqueLabels.filter((label) => !uniqueLabels.some((other) => other !== label && isTechnicalDescendant(other, label, technicalLabelMetadata, legacyLabelMatchMap)));
}

function buildTechnicalLabelIndex(technicalLabelTrees, extraLeaves) {
  const technicalLabelMetadata = Object.fromEntries(
    [...technicalLabelTrees.flatMap((tree) => flattenTree(tree)), ...(extraLeaves || [])].map((entry) => [entry.label, Object.freeze(entry)])
  );
  const technicalLabels = Object.freeze(Object.keys(technicalLabelMetadata));
  return {
    TECHNICAL_LABEL_DEFINITIONS: Object.freeze(
      Object.fromEntries(
        Object.entries(technicalLabelMetadata).map(([label, metadata]) => [
          label,
          Object.freeze({ color: metadata.color, description: metadata.description })
        ])
      )
    ),
    TECHNICAL_LABEL_GUIDANCE: Object.freeze(
      Object.fromEntries(
        Object.entries(technicalLabelMetadata).map(([label, metadata]) => [label, metadata.description])
      )
    ),
    TECHNICAL_LABEL_METADATA: Object.freeze(technicalLabelMetadata),
    TECHNICAL_LABELS: technicalLabels,
    TECHNICAL_RELEASE_RELEVANT_LABELS: Object.freeze(
      technicalLabels.filter((label) => technicalLabelMetadata[label].releaseRelevant)
    ),
    TECHNICAL_SECONDARY_LABELS: Object.freeze(
      technicalLabels.filter((label) => technicalLabelMetadata[label].secondary)
    ),
    TECHNICAL_MAJOR_VERSION_LABELS: Object.freeze(
      technicalLabels.filter((label) => technicalLabelMetadata[label].versionBump === "major")
    ),
    TECHNICAL_MINOR_VERSION_LABELS: Object.freeze(
      technicalLabels.filter((label) => technicalLabelMetadata[label].versionBump === "minor")
    )
  };
}

module.exports = {
  buildLegacyMatchMap,
  buildTechnicalLabelIndex,
  collapseTechnicalLabels,
  createNode,
  flattenTree,
  getTechnicalLabelAncestors,
  isTechnicalDescendant,
  matchesTechnicalExpectation
};
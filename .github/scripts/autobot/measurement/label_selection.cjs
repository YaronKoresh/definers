const {
  PR_CANONICAL_LABEL_REPLACEMENTS,
  SMALL_PR_FILE_LIMIT,
  SMALL_PR_TOTAL_CHANGE_LIMIT
} = require("../constants.cjs");
const { AutobotLabelRegistry, LABEL_PRIORITY, sortLabels, technicalLabelsOnly } = require("../labels/registry.cjs");

const DEFAULT_MAX_LABEL_WORDS = 2;

function countLabelWords(label) {
  const normalizedLabel = String(label || "").trim().replace(/[_-]+/g, " ");
  if (!normalizedLabel) {
    return 0;
  }
  return normalizedLabel.split(/\s+/).filter(Boolean).length;
}

function compressLabelToWordBudget(label, maxWords = DEFAULT_MAX_LABEL_WORDS) {
  const wordBudget = Math.max(Number(maxWords) || DEFAULT_MAX_LABEL_WORDS, 1);
  let normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  if (!normalizedLabel) {
    return "";
  }
  let iterations = 0;
  while (countLabelWords(normalizedLabel) > wordBudget && iterations < 8) {
    const metadata = AutobotLabelRegistry.getLabelMetadata(normalizedLabel);
    if (!metadata?.parent) {
      break;
    }
    const parentLabel = AutobotLabelRegistry.normalizeLabelName(metadata.parent);
    if (!parentLabel || parentLabel === normalizedLabel) {
      break;
    }
    normalizedLabel = parentLabel;
    iterations += 1;
  }
  return normalizedLabel;
}

function applyLabelWordBudget(labels, options = {}) {
  const maxWords = Math.max(Number(options.maxWords) || DEFAULT_MAX_LABEL_WORDS, 1);
  return (labels || [])
    .map((label) => compressLabelToWordBudget(label, maxWords))
    .filter(Boolean);
}

function compareRankedLabelEntries(left, right) {
  if (right.score !== left.score) {
    return right.score - left.score;
  }
  const leftRank = LABEL_PRIORITY.indexOf(left.label);
  const rightRank = LABEL_PRIORITY.indexOf(right.label);
  const normalizedLeftRank = leftRank === -1 ? LABEL_PRIORITY.length : leftRank;
  const normalizedRightRank = rightRank === -1 ? LABEL_PRIORITY.length : rightRank;
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

function resolveLabelLimit(rawLimit) {
  if (rawLimit === undefined || rawLimit === null || rawLimit === "") {
    return Number.POSITIVE_INFINITY;
  }
  const parsedLimit = Number(rawLimit);
  if (!Number.isFinite(parsedLimit)) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.max(parsedLimit, 1);
}

function isSmallPullRequest(filesChanged, totalChanges) {
  return filesChanged <= SMALL_PR_FILE_LIMIT && totalChanges <= SMALL_PR_TOTAL_CHANGE_LIMIT;
}

function derivePrLabelBudget(smallPullRequest) {
  return smallPullRequest ? 12 : 15;
}

function deriveGenericMaintenanceLabelLimit(smallPullRequest) {
  return smallPullRequest ? 10 : 12;
}

function normalizePrOutputLabel(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  return PR_CANONICAL_LABEL_REPLACEMENTS[normalizedLabel] || normalizedLabel;
}

function applyPrLabelPolicy(labels, options = {}) {
  const limit = resolveLabelLimit(options.limit);
  const maxWords = Math.max(Number(options.maxWords) || DEFAULT_MAX_LABEL_WORDS, 1);
  const normalizedLabels = applyLabelWordBudget((labels || []).map((label) => normalizePrOutputLabel(label)), { maxWords });
  if (!Number.isFinite(limit)) {
    return technicalLabelsOnly(normalizedLabels);
  }
  return technicalLabelsOnly(
    normalizedLabels,
    { limit }
  );
}

function selectDeterministicLabels(orderedSignals, deterministicLabelSet, options = {}) {
  const limit = resolveLabelLimit(options.limit);
  const collectionLimit = Number.isFinite(limit) ? Math.max(limit * 2, 24) : Number.POSITIVE_INFINITY;
  const explicitLabels = sortLabels([...orderedSignals, ...deterministicLabelSet]);
  const fallbackLabels = sortLabels([...deterministicLabelSet]);
  const selectedLabels = [];

  for (const label of explicitLabels) {
    if (selectedLabels.includes(label)) {
      continue;
    }
    selectedLabels.push(label);
    if (selectedLabels.length >= collectionLimit) {
      return applyPrLabelPolicy(selectedLabels, options);
    }
  }

  for (const label of fallbackLabels) {
    if (selectedLabels.includes(label)) {
      continue;
    }
    if (explicitLabels.length > 0 && AutobotLabelRegistry.GENERIC_FALLBACK_LABELS.includes(label) && label !== "dependencies") {
      continue;
    }
    selectedLabels.push(label);
    if (selectedLabels.length >= collectionLimit) {
      break;
    }
  }

  return applyPrLabelPolicy(selectedLabels, options);
}

module.exports = {
  applyPrLabelPolicy,
  compareRankedLabelEntries,
  deriveGenericMaintenanceLabelLimit,
  derivePrLabelBudget,
  applyLabelWordBudget,
  compressLabelToWordBudget,
  countLabelWords,
  isSmallPullRequest,
  mergeRankedLabels,
  normalizePrOutputLabel,
  rankScoredLabels,
  selectDeterministicLabels
};

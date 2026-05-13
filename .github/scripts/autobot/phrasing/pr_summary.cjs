const { LABEL_SUPPORT_PATTERNS } = require("../constants.cjs");
const { AutobotLabelRegistry } = require("../labels.cjs");
const { EVIDENCE_RULES } = require("../measurement/evidence_rules.cjs");
const { escapeRegExp, formatBulletLines, humanizeIdentifier } = require("../utils.cjs");

function isAutobotClassificationInfrastructure(normalizedPath) {
  return /^\.github\/scripts\/(?:(?:autobot|smart_link)[^/]*|(?:autobot|smart_link)\/.+)\.(cjs|mjs|js|ts)$/.test(normalizedPath)
    || /^\.github\/workflows\/(autobot|smart-link)\.ya?ml$/.test(normalizedPath)
    || /^(test|tests)\/(?:(test_)?autobot[^/]*|smart_link[^/]*|test_smart_link[^/]*|deterministic_scenario_engine|javascript_test_harness)(\.test)?\.(cjs|mjs|js|ts|py)$/.test(normalizedPath);
}

function isTestPath(normalizedPath) {
  return /^(test|tests)\//.test(normalizedPath)
    || /(^|\/)test_[^/]+\.py$/.test(normalizedPath)
    || /\.(spec|test)\.(js|jsx|ts|tsx|py|cjs|mjs|cts|mts)$/.test(normalizedPath);
}

function describeEvidenceItem(evidence) {
  const parts = [];
  const evidenceName = humanizeIdentifier(evidence?.ruleId || "evidence");
  const occurrenceCount = Number(evidence?.occurrenceCount || 0);
  parts.push(occurrenceCount > 1 ? `${evidenceName} x${occurrenceCount}` : evidenceName);
  if (evidence?.scope) {
    parts.push(`${evidence.scope} scope`);
  }
  if (evidence?.polarity === "additive" || evidence?.polarity === "destructive") {
    parts.push(evidence.polarity);
  }
  const sampleFiles = Array.isArray(evidence?.metadata?.sampleFiles)
    ? evidence.metadata.sampleFiles.filter(Boolean).slice(0, 3)
    : [];
  if (sampleFiles.length > 0) {
    parts.push(`files: ${sampleFiles.join(", ")}`);
  }
  return parts.join(" | ");
}

function buildDefaultLabelSupportPatterns(label) {
  const normalizedLabel = String(label || "").trim().toLowerCase();
  if (!normalizedLabel) {
    return [];
  }
  return [new RegExp(`(^|[^a-z0-9])${escapeRegExp(normalizedLabel).replace(/\s+/g, "[\\s_-]+") }([^a-z0-9]|$)`)];
}

function getLabelSupportKeys(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  return normalizedLabel ? [normalizedLabel] : [];
}

function collectDirectEvidenceForLabel(label, evidenceItems) {
  const supportKeys = getLabelSupportKeys(label);
  return (evidenceItems || [])
    .map((item) => {
      const ruleId = String(item?.ruleId || "");
      const rule = EVIDENCE_RULES[ruleId];
      const boost = Math.max(...supportKeys.map((supportKey) => Number(rule?.labelBoosts?.[supportKey] || 0)), 0);
      if (boost <= 0) {
        return null;
      }
      return {
        boost,
        occurrenceCount: Math.max(Number(item?.occurrenceCount) || 1, 1),
        sampleFiles: Array.isArray(item?.metadata?.sampleFiles)
          ? item.metadata.sampleFiles.filter(Boolean).slice(0, 3)
          : [],
        ruleId
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.boost - left.boost || right.occurrenceCount - left.occurrenceCount || left.ruleId.localeCompare(right.ruleId));
}

function collectSupportFilesForLabel(label, filesWithContext) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  const patterns = getLabelSupportKeys(label)
    .flatMap((supportKey) => LABEL_SUPPORT_PATTERNS[supportKey] || buildDefaultLabelSupportPatterns(supportKey));
  const matchingFiles = (filesWithContext || [])
    .filter((file) => {
      const text = `${String(file.filename || "").toLowerCase()}\n${String(file.patch || "").toLowerCase()}`;
      return patterns.some((pattern) => pattern.test(text));
    })
    .sort((left, right) => {
      const leftPath = String(left.filename || "").toLowerCase();
      const rightPath = String(right.filename || "").toLowerCase();
      const leftPenalty = Number(isAutobotClassificationInfrastructure(leftPath) || isTestPath(leftPath) || /^docs\//.test(leftPath));
      const rightPenalty = Number(isAutobotClassificationInfrastructure(rightPath) || isTestPath(rightPath) || /^docs\//.test(rightPath));
      return leftPenalty - rightPenalty || right.score - left.score || leftPath.localeCompare(rightPath);
    });
  const preferredFiles = matchingFiles.filter((file) => {
    const normalizedPath = String(file.filename || "").toLowerCase();
    return !isAutobotClassificationInfrastructure(normalizedPath)
      && !isTestPath(normalizedPath)
      && !/^docs\//.test(normalizedPath);
  });

  const securityRelatedLabels = new Set(["security", "vulnerability", "hardening", "pen-test", "compliance"]);
  const supportFiles = preferredFiles.length > 0
    ? preferredFiles
    : securityRelatedLabels.has(normalizedLabel)
      ? []
      : matchingFiles;

  return supportFiles.slice(0, 3).map((file) => file.filename);
}

function deriveLabelConfidence(scoreEntry, directEvidence, supportFiles) {
  if (scoreEntry?.primary || scoreEntry?.emitted) {
    return "high";
  }
  if (scoreEntry?.retained || directEvidence.length > 0 || supportFiles.length > 1) {
    return "medium";
  }
  return supportFiles.length > 0 ? "medium" : "low";
}

const GENERIC_LABEL_MIN_DESCENDANTS = 20;
const TECHNICAL_DESCENDANT_COUNT_CACHE = {};

function getTechnicalDescendantCount(label) {
  const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
  if (Object.prototype.hasOwnProperty.call(TECHNICAL_DESCENDANT_COUNT_CACHE, normalizedLabel)) {
    return TECHNICAL_DESCENDANT_COUNT_CACHE[normalizedLabel];
  }
  const count = Object.keys(AutobotLabelRegistry.LABEL_DEFINITIONS || {}).reduce((total, candidate) => {
    const normalizedCandidate = AutobotLabelRegistry.normalizeLabelName(candidate);
    if (!normalizedCandidate || normalizedCandidate === normalizedLabel) {
      return total;
    }
    return total + Number(AutobotLabelRegistry.isDescendantLabel(normalizedCandidate, normalizedLabel));
  }, 0);
  TECHNICAL_DESCENDANT_COUNT_CACHE[normalizedLabel] = count;
  return count;
}

function isGenericTechnicalLabel(label) {
  const metadata = AutobotLabelRegistry.getLabelMetadata(label);
  if (!metadata) {
    return false;
  }
  return getTechnicalDescendantCount(label) >= GENERIC_LABEL_MIN_DESCENDANTS;
}

function buildLabelConfidenceMap(labelRationaleLines) {
  const confidenceByLabel = {};
  for (const line of labelRationaleLines || []) {
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

function buildLabelRationaleLineMap(labelRationaleLines) {
  const rationaleByLabel = {};
  for (const line of labelRationaleLines || []) {
    const match = String(line || "").match(/^([^:]+):/);
    if (!match) {
      continue;
    }
    const label = String(match[1] || "").trim();
    if (!label) {
      continue;
    }
    rationaleByLabel[label] = String(line);
  }
  return rationaleByLabel;
}

function collectHardSignalEvidenceFiles(hardSignals, evidenceItems) {
  const files = [];
  for (const item of evidenceItems || []) {
    if (!hardSignals.includes(item.ruleId)) continue;
    for (const file of (item.metadata?.sampleFiles || []).slice(0, 2)) {
      if (file && !files.includes(file)) files.push(file);
    }
    if (files.length >= 3) break;
  }
  return files;
}

function describeHardSignalContext(hardSignals, evidenceItems) {
  const parts = [];
  for (const signalId of hardSignals.slice(0, 2)) {
    const matchingEvidence = (evidenceItems || []).find((item) => item.ruleId === signalId);
    const occurrences = matchingEvidence ? Math.max(Number(matchingEvidence.occurrenceCount) || 0, 0) : 0;
    const scope = matchingEvidence?.scope || "";
    const sampleFiles = (matchingEvidence?.metadata?.sampleFiles || []).slice(0, 2);
    let description = humanizeIdentifier(signalId);
    if (occurrences > 1) description += ` x${occurrences}`;
    if (scope) description += ` (${scope} scope)`;
    if (sampleFiles.length > 0) description += ` in ${sampleFiles.join(", ")}`;
    parts.push(description);
  }
  return parts.join("; ");
}

function buildLabelRationale(label, context) {
  const metadata = AutobotLabelRegistry.getLabelMetadata(label) || AutobotLabelRegistry.LABEL_DEFINITIONS[label] || {};
  const scoreEntry = context.labelScores[label];
  const directEvidence = collectDirectEvidenceForLabel(label, context.evidenceItems);
  const supportFiles = collectSupportFilesForLabel(label, context.filesWithContext);
  const confidence = deriveLabelConfidence(scoreEntry, directEvidence, supportFiles);
  const reasonParts = [];
  const directEvidenceFiles = [...new Set(directEvidence.flatMap((item) => item.sampleFiles || []))].slice(0, 3);
  if (scoreEntry?.primary || scoreEntry?.emitted || scoreEntry?.retained) {
    const scoreText = Number.isFinite(Number(scoreEntry.score)) ? `${Math.round(Number(scoreEntry.score) * 100)}%` : "n/a";
    reasonParts.push(`scorer-backed at ${scoreText}`);
  }
  if (directEvidence.length > 0) {
    const evidenceDescriptions = directEvidence.slice(0, 3).map((item) => {
      let desc = humanizeIdentifier(item.ruleId);
      if (item.occurrenceCount > 1) desc += ` x${item.occurrenceCount}`;
      return desc;
    });
    reasonParts.push(`evidence rules: ${evidenceDescriptions.join(", ")}`);
    if (directEvidenceFiles.length > 0) {
      reasonParts.push(`evidence files: ${directEvidenceFiles.join(", ")}`);
    }
  } else if (supportFiles.length > 0) {
    reasonParts.push(`label-specific pattern matches in ${supportFiles.join(", ")}`);
  }
  if (context.hardSignals.length > 0 && AutobotLabelRegistry.isReleaseRelevantLabel(label)) {
    const hardSignalDetail = describeHardSignalContext(context.hardSignals, context.evidenceItems);
    if (hardSignalDetail) {
      reasonParts.push(`correlated with hard release signals: ${hardSignalDetail}`);
    }
  }
  if (reasonParts.length === 0) {
    const nearestEvidence = (context.evidenceItems || [])
      .filter((item) => Number(item.occurrenceCount || 0) > 0)
      .slice(0, 2)
      .map((item) => {
        const sampleFiles = (item.metadata?.sampleFiles || []).slice(0, 1);
        return sampleFiles.length > 0 ? `${humanizeIdentifier(item.ruleId)} in ${sampleFiles.join(", ")}` : humanizeIdentifier(item.ruleId);
      });
    if (nearestEvidence.length > 0) {
      reasonParts.push(`inferred from overall evidence: ${nearestEvidence.join(", ")}`);
    } else {
      reasonParts.push("no direct evidence found for this label");
    }
  }
  return `${label}: ${metadata.description || "Technical label."} Confidence ${confidence}. ${reasonParts.join("; ")}.`;
}

function buildLabelRationaleLines(labels, context) {
  return labels.map((label) => buildLabelRationale(label, context));
}

function buildPrDeterministicSummary(context) {
  const confidenceByLabel = buildLabelConfidenceMap(context.labelRationaleLines);
  const rationaleByLabel = buildLabelRationaleLineMap(context.labelRationaleLines);
  let summaryLabels = (context.deterministicLabels || []).filter((label) => {
    if (isGenericTechnicalLabel(label)) {
      return false;
    }
    return confidenceByLabel[label] !== "low";
  });
  if (summaryLabels.length === 0) {
    summaryLabels = (context.deterministicLabels || [])
      .filter((label) => !isGenericTechnicalLabel(label));
  }
  if (summaryLabels.length === 0) {
    summaryLabels = (context.deterministicLabels || []).slice(0, 3);
  }
  const labelRationaleLines = summaryLabels
    .map((label) => rationaleByLabel[label])
    .filter(Boolean);
  const highestSignalFiles = context.topFiles
    .slice(0, 3)
    .map((file) => `${file.filename} [${file.status.toUpperCase()}] (+${file.additions} -${file.deletions})`);
  const scopeSentences = context.maintenanceOnlyEligible
    ? [
        `This pull request changes ${context.filesWithContext.length} files (+${context.totalAdditions} -${context.totalDeletions}) and stays within narrow maintenance surfaces: ${context.maintenanceCategoriesList.join(", ") || "none"}.`,
        `The highest-signal areas are ${context.topDirectories.join(", ") || "(root)"}.`
      ]
    : [
        `This pull request changes ${context.filesWithContext.length} files (+${context.totalAdditions} -${context.totalDeletions}) across ${context.topDirectories.join(", ") || "(root)"} with category mix ${context.categorySummary || "(none)"}.`,
        context.capabilityExpansionSignal
          ? `Added behavioral source or UI surfaces were detected across ${context.behavioralSurfaceAdditions} path${context.behavioralSurfaceAdditions === 1 ? "" : "s"}.`
          : null,
        context.structuralPublicBreakingSignal
          ? `Public package import surfaces were renamed or removed across ${context.publicContractMoves} path${context.publicContractMoves === 1 ? "" : "s"}, so consumer updates may be required.`
          : null
      ].filter(Boolean);
  const decisionBullets = [
    `Semver: ${context.releaseRelevant ? context.scoring.semver.decision : "none"}.`,
    `Final emitted technical labels: ${summaryLabels.join(", ") || "(none)"}.`,
    context.scoring.semver.hardSignals.length > 0
      ? `Hard release signals: ${context.scoring.semver.hardSignals.join(", ")}.`
      : null,
    context.structuralPublicBreakingSignal
      ? "Structural public package moves indicate a likely breaking import or integration surface."
      : null
  ].filter(Boolean);
  const labelEvidenceBullets = (summaryLabels || []).flatMap((label) => {
    const supportFiles = collectSupportFilesForLabel(label, context.filesWithContext);
    if (supportFiles.length > 0) {
      return [`${label}: support in ${supportFiles.join(", ")}`];
    }
    const directEvidence = collectDirectEvidenceForLabel(label, context.evidenceItems);
    if (directEvidence.length > 0) {
      return directEvidence.slice(0, 2).map((item) => `${label}: ${describeEvidenceItem(item)}`);
    }
    return [];
  });
  const evidenceBullets = labelEvidenceBullets.length > 0
    ? labelEvidenceBullets.slice(0, 4)
    : context.topEvidenceItems.length > 0
      ? context.topEvidenceItems.map((item) => describeEvidenceItem(item))
      : highestSignalFiles.length > 0
        ? highestSignalFiles
        : [
            context.maintenanceOnlyEligible
              ? `The diff stayed inside maintenance surfaces: ${context.maintenanceCategoriesList.join(", ") || "none"}.`
              : "No strong evidence item exceeded the explanation threshold."
          ];
  const reviewBullets = context.maintenanceOnlyEligible
    ? [
        `Review ${context.topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "the touched files"} for wording, workflow, or expectation drift.`,
        context.categoryCounts.get("workflow")
          ? "Validate repository automation behavior because workflow files changed."
          : "Validate that the change stays scoped to the intended maintenance surface."
      ]
    : [
        summaryLabels.length > 0
          ? `Focus review on ${summaryLabels.join(", ")} changes in ${context.topDirectories.join(", ") || "(root)"}.`
          : `Focus review on ${context.topFiles.slice(0, 3).map((file) => file.filename).join(", ") || "the highest-signal files"} because the deterministic scan is only partial.`,
        context.categoryCounts.get("test")
          ? "Run the touched regression surfaces and verify changed tests still reflect product behavior."
          : context.releaseRelevant
            ? "Add or verify targeted checks for the release-relevant technical surfaces."
            : "Validate that the change stays scoped to the touched maintenance surfaces."
      ];
  return [
    "## Autobot Summary",
    "",
    "### Scope",
    scopeSentences.join(" "),
    "",
    "### Decision",
    formatBulletLines(decisionBullets, "No technical labels were selected."),
    "",
    "### Label Rationale",
    formatBulletLines(labelRationaleLines, "No technical labels were selected."),
    "",
    "### Key Evidence",
    formatBulletLines(evidenceBullets, "No evidence items were collected."),
    "",
    "### Review Focus",
    formatBulletLines(reviewBullets, "Review the highest-signal files.")
  ].join("\n");
}

module.exports = {
  buildLabelRationaleLines,
  buildPrDeterministicSummary,
  collectDirectEvidenceForLabel,
  collectSupportFilesForLabel,
  describeEvidenceItem,
  formatBulletLines
};
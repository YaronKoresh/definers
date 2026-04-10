const { scoreDeterministicEvidence } = require("./autobot_deterministic_scorer");

const ISSUE_RELEASE_RELEVANT_LABELS = new Set([
  "bug",
  "runtime",
  "api",
  "database",
  "schema",
  "compatibility",
  "migration",
  "security",
  "performance",
  "breaking-change"
]);

function normalizeIssue(issue) {
  const title = String(issue?.title || "").trim();
  const body = String(issue?.body || "").trim();
  const text = `${title}\n${body}`.toLowerCase();
  return {
    title,
    body,
    text,
    normalizedTitle: title.toLowerCase(),
    normalizedBody: body.toLowerCase()
  };
}

function hasAnyMarker(text, markers) {
  return markers.some((marker) => text.includes(marker));
}

function pushUnique(items, value) {
  if (value && !items.includes(value)) {
    items.push(value);
  }
}

function pushEvidenceItem(items, ruleId) {
  if (ruleId && !items.some((item) => item.ruleId === ruleId)) {
    items.push({ ruleId });
  }
}

function scoredLabelNames(entries) {
  return (entries || []).map((entry) => String(entry?.label || "").trim()).filter(Boolean);
}

function hasScoredLabel(scoring, label, flagName = "emitted") {
  return Boolean(scoring?.labelScores?.[label]?.[flagName]);
}

function analyzeIssueIntake(issue) {
  const normalized = normalizeIssue(issue);
  const evidenceSignals = [];
  const likelyClassification = [];

  const documentationTemplate = hasAnyMarker(normalized.text, [
    "documentation issue report",
    "link(s) to the affected documentation",
    "detailed description of the problem",
    "proposed solution (optional)"
  ]);
  const bugTemplate = hasAnyMarker(normalized.text, [
    "thank you for helping us squash this bug",
    "detailed steps to reproduce",
    "potential causes / workarounds / related issues (optional)",
    "custom modifications / configuration"
  ]);
  const featureTemplate = hasAnyMarker(normalized.text, [
    "proposing an improvement or enhancement",
    "current situation and problem/opportunity",
    "proposed improvement/enhancement",
    "potential costs, challenges, and considerations",
    "alternatives considered (optional)",
    "proposed steps or implementation plan (optional)"
  ]);

  const documentationSignal = documentationTemplate || /\b(docs?|documentation|readme)\b/.test(normalized.normalizedTitle);
  const bugSignal = bugTemplate || /\b(bug|crash|error|failure|broken|regression)\b/.test(normalized.normalizedTitle);
  const enhancementSignal = featureTemplate || /\b(feature request|enhancement)\b/.test(normalized.normalizedTitle);
  const proposalSignal = /\b(proposal|rfc|request for comments|design proposal|roadmap|future work)\b/.test(normalized.normalizedTitle)
    || /\brequest for comments\b/.test(normalized.text);
  const improvementSignal = /\b(improvement|improve|streamline|simplify|quality of life|qol)\b/.test(normalized.normalizedTitle)
    || /\b(pain point|gap closure|quality of life)\b/.test(normalized.text);
  const runtimeSignal = /\b(runtime|windows|linux|macos|cuda|python)\b/.test(normalized.text);
  const apiSignal = /\b(api|endpoint|webhook|request|response)\b/.test(normalized.text);

  if (documentationTemplate) {
    pushUnique(evidenceSignals, "Structured documentation issue template fields are present.");
  }
  if (bugTemplate) {
    pushUnique(evidenceSignals, "Structured bug-report fields are present.");
  }
  if (featureTemplate) {
    pushUnique(evidenceSignals, "Structured feature-request fields are present.");
  }
  if (runtimeSignal) {
    pushUnique(evidenceSignals, "The intake mentions runtime or platform context.");
  }
  if (apiSignal) {
    pushUnique(evidenceSignals, "The intake mentions API context.");
  }

  const labels = [];
  if (documentationSignal) {
    pushUnique(labels, "documentation");
    pushUnique(likelyClassification, "documentation");
  } else if (bugSignal) {
    pushUnique(labels, "bug");
    pushUnique(likelyClassification, "bug");
  } else if (enhancementSignal || proposalSignal || improvementSignal) {
    pushUnique(labels, "enhancement");
    pushUnique(likelyClassification, "enhancement");
    if (proposalSignal) {
      pushUnique(labels, "proposal");
      pushUnique(likelyClassification, "proposal");
    } else if (improvementSignal) {
      pushUnique(labels, "improvement");
      pushUnique(likelyClassification, "improvement");
    }
  }

  if (documentationSignal && /\bproblem|incorrect|missing|unclear|outdated\b/.test(normalized.text)) {
    pushUnique(evidenceSignals, "The intake describes a documentation defect or gap.");
  }
  if (bugSignal) {
    pushUnique(evidenceSignals, "The intake explicitly describes incorrect behavior or a failure.");
  }
  if (enhancementSignal || proposalSignal || improvementSignal) {
    pushUnique(evidenceSignals, "The intake asks for new or improved behavior rather than reporting a confirmed defect.");
  }
  if (proposalSignal) {
    pushUnique(evidenceSignals, "The intake frames the request as a proposal or future design direction.");
  }
  if (improvementSignal && !proposalSignal) {
    pushUnique(evidenceSignals, "The intake frames the request as a targeted improvement or gap closure.");
  }

  const evidenceItems = [];
  if (documentationSignal) {
    pushEvidenceItem(evidenceItems, "issue-documentation-report");
  } else if (bugSignal) {
    pushEvidenceItem(evidenceItems, "issue-bug-report");
  } else if (enhancementSignal || proposalSignal || improvementSignal) {
    pushEvidenceItem(evidenceItems, "issue-enhancement-request");
    if (proposalSignal) {
      pushEvidenceItem(evidenceItems, "issue-proposal-request");
    } else if (improvementSignal) {
      pushEvidenceItem(evidenceItems, "issue-improvement-request");
    }
  }
  if (runtimeSignal) {
    pushEvidenceItem(evidenceItems, "issue-runtime-context");
  }
  if (apiSignal) {
    pushEvidenceItem(evidenceItems, "issue-api-context");
  }

  const deterministicScoring = scoreDeterministicEvidence({ evidenceItems });
  const deterministicLabels = scoredLabelNames(deterministicScoring.emittedLabels);
  const deterministicPrimaryLabels = scoredLabelNames(deterministicScoring.primaryLabels);

  return {
    title: normalized.title,
    body: normalized.body,
    text: normalized.text,
    labels: labels.length > 0
      ? labels
      : [
          ...(hasScoredLabel(deterministicScoring, "runtime", "retained") ? ["runtime"] : []),
          ...(hasScoredLabel(deterministicScoring, "api", "retained") ? ["api"] : [])
        ],
    evidenceSignals,
    evidenceItems,
    likelyClassification,
    deterministicLabels,
    deterministicPrimaryLabels,
    deterministicSemver: deterministicScoring.semver,
    releaseRelevant: deterministicLabels.some((label) => ISSUE_RELEASE_RELEVANT_LABELS.has(label))
      || /\b(bug|regression|security|runtime|breaking|migration|database|schema|api)\b/.test(normalized.text)
  };
}

module.exports = {
  analyzeIssueIntake
};
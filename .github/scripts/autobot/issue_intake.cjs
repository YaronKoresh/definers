const { scoreDeterministicEvidence } = require("./measurement/scorer.cjs");
const { applyLabelWordBudget } = require("./measurement/label_selection.cjs");
const { AutobotLabelRegistry } = require("./labels.cjs");

const MAX_LABEL_WORDS = 2;

class AutobotIssueClassifier {
  static ISSUE_RELEASE_RELEVANT_LABELS = new Set([...AutobotLabelRegistry.RELEASE_CRITICAL_LABELS]);

  static TEMPLATE_MARKERS = Object.freeze({
    documentation: Object.freeze([
      "documentation issue report",
      "link(s) to the affected documentation",
      "detailed description of the problem",
      "proposed solution (optional)"
    ]),
    bug: Object.freeze([
      "thank you for helping us squash this bug",
      "detailed steps to reproduce",
      "potential causes / workarounds / related issues (optional)",
      "custom modifications / configuration"
    ]),
    feature: Object.freeze([
      "proposing an improvement or enhancement",
      "current situation and problem/opportunity",
      "proposed improvement/enhancement",
      "potential costs, challenges, and considerations",
      "alternatives considered (optional)",
      "proposed steps or implementation plan (optional)"
    ])
  });

  static normalizeIssue(issue) {
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

  static hasAnyMarker(text, markers) {
    return markers.some((marker) => text.includes(marker));
  }

  static pushUnique(items, value) {
    if (value && !items.includes(value)) {
      items.push(value);
    }
  }

  static pushEvidenceItem(items, ruleId) {
    if (ruleId && !items.some((item) => item.ruleId === ruleId)) {
      items.push({ ruleId });
    }
  }

  static scoredLabelNames(entries) {
    return (entries || []).map((entry) => String(entry?.label || "").trim()).filter(Boolean);
  }

  static hasScoredLabel(scoring, label, flagName = "emitted") {
    return Boolean(scoring?.labelScores?.[label]?.[flagName]);
  }

  static hasLabelOrAncestor(labels, expectedLabel) {
    return (labels || []).some((label) => {
      const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
      return normalizedLabel === expectedLabel || AutobotLabelRegistry.isDescendantLabel(normalizedLabel, expectedLabel);
    });
  }

  static deriveTechnicalLabels(normalized) {
    const labels = [];
    const text = normalized.text;

    if (/readme/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "readme");
    }
    if (/docs?\/reference|openapi|swagger|api docs|reference docs/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "api doc");
    }
    if (/architecture|module map|data flow/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "architecture doc");
    }
    if (/install|getting started|quickstart/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "install guide");
      if (/quickstart|getting started/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "quickstart");
      }
    }
    if (/troubleshoot|troubleshooting|faq/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "troubleshoot guide");
      if (/faq/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "faq");
      }
    }
    if (/pyproject/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pyproject");
    }
    if (/package-lock|poetry\.lock|pnpm-lock|yarn\.lock|lockfile/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "lockfile");
    }
    if (/codeowners/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "codeowners");
    }
    if (/issue form|issue template/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "issue form");
    }
    if (/pull request template|pr template/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pull template");
    }
    if (/dependabot/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "dependabot");
    }

    if (/workflow|github actions/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "github actions");
      if (/matrix/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "matrix job");
      }
    }
    if (/unit test/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "unit test");
    }
    if (/integration test|scenario|regression/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "integration test");
    }

    if (/api|endpoint|route|router|handler/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "route");
      if (/path param|route param/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "route param");
      }
      if (/query param|querystring|search param/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "query param");
      }
    }
    if (/graphql/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "graphql");
      if (/resolver/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "resolver");
      }
    }
    if (/webhook/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "webhook");
    }
    if (/request body|request payload|input schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "request body");
    }
    if (/response body|response payload|output schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "response body");
    }
    if (/validation|validator|pydantic/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "body validation");
    }
    if (/openapi|swagger/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "openapi spec");
    }

    if (/jwt/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "jwt");
    }
    if (/refresh token/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "refresh token");
    }
    if (/rbac|role-based|role mapping/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "rbac");
    }
    if (/oauth|scope/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "oauth scope");
    }
    if (/csrf/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "csrf");
    }
    if (/authz|authorization/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "authz");
    }
    if (/\b(vulnerab\w*|advisory|cve-\d{4}-\d+|ghsa-[a-z0-9-]+|exploit(?:able)?|security advisory)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "vulnerability");
    }
    if (/\b(compliance|soc ?2|pci(?:-dss)?|gdpr|hipaa|iso ?27001|nist|fedramp)\b/.test(text)
      || /\bsarbanes[-\s]+oxley\b/.test(text)
      || /\bsox\s+(?:compliance|controls?|control matrix|attestation|audit)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "compliance");
    }
    if (/\b(hardening|harden|least privilege|defense in depth|deny by default|sandbox|mitigation|sanitize|sanitiz)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "hardening");
    }
    if (/\b(pen[\s-]?test|penetration test|red team)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pen-test");
    }

    if (/migration|alembic/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "migration file");
      if (/destructive|drop column|drop table/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "destructive migration");
      }
      if (/add column|add table|create table/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "additive migration");
      }
    }
    if (/json schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "json schema");
    }
    if (/proto|protobuf/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "proto schema");
    }
    if (/query|sql|database|postgres|mysql|sqlite/.test(text)) {
      if (/select|read/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "read query");
      }
      if (/insert|update|delete|write|upsert/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "write query");
      }
    }

    if (/python/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "python version");
    }
    if (/windows/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "windows");
      if (/path separator|\\\\|ntpath|posixpath|pathlib|normpath|realpath/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "path separator");
        AutobotIssueClassifier.pushUnique(labels, "path normalization");
      }
      if (/shell command|cmd(?:\.exe)?\s*\/[ck]|powershell(?:\.exe)?|bash\s+-[cl]|\/bin\/(?:sh|bash)/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "shell command");
      }
    }
    if (/linux|ubuntu/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "linux");
    }
    if (/macos/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "macos");
    }
    if (/cuda|nvidia/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "cuda");
    }
    if (/env var|environment variable|getenv|dotenv/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "env var");
    }
    if (/filesystem|file system|path separator|filepath|pathlib|normpath|realpath/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "filesystem");
      AutobotIssueClassifier.pushUnique(labels, "path normalization");
    }
    if (/subprocess|spawn|fork|exec\(|child process|child_process|popen|stdout|stderr|stdin|pipe/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "process");
      AutobotIssueClassifier.pushUnique(labels, "subprocess io");
    }
    if (/memory|heap|buffer/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "heap usage");
    }
    if (/cache/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "cache hit");
    }
    if (/latency|cold start|import time/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "latency");
    }

    return AutobotIssueClassifier.normalizeOutputLabels(labels);
  }

  static normalizeOutputLabels(labels, options = {}) {
    const maxWords = Math.max(Number(options.maxWords) || MAX_LABEL_WORDS, 1);
    const limit = Math.max(Number(options.limit) || AutobotLabelRegistry.MAX_AUTOBOT_LABELS, 1);
    return AutobotLabelRegistry.technicalLabelsOnly(applyLabelWordBudget(labels, { maxWords }), { limit });
  }

  static analyzeIssueIntake(issue) {
    const normalized = AutobotIssueClassifier.normalizeIssue(issue);
    const evidenceSignals = [];
    const likelyClassification = [];
    const technicalLabels = AutobotIssueClassifier.deriveTechnicalLabels(normalized);

    const documentationTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.documentation);
    const bugTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.bug);
    const featureTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.feature);

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
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured documentation issue template fields are present.");
    }
    if (bugTemplate) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured bug-report fields are present.");
    }
    if (featureTemplate) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured feature-request fields are present.");
    }
    if (runtimeSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake mentions runtime or platform context.");
    }
    if (apiSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake mentions API context.");
    }

    const fallbackTechnicalLabels = [];
    if (documentationSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake describes a documentation surface.");
    } else if (bugSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake describes a concrete defect or regression.");
    } else if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake asks for expanded or improved behavior.");
      if (proposalSignal) {
        AutobotIssueClassifier.pushUnique(likelyClassification, "The request is framed as a proposal or future design direction.");
      } else if (improvementSignal) {
        AutobotIssueClassifier.pushUnique(likelyClassification, "The request is framed as a targeted improvement or gap closure.");
      }
    }

    if (documentationSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "docs")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "guide");
    }
    if (runtimeSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "runtime")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "platform");
    }
    if (apiSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "api")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "route");
    }

    if (documentationSignal && /\bproblem|incorrect|missing|unclear|outdated\b/.test(normalized.text)) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake describes a documentation defect or gap.");
    }
    if (bugSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake explicitly describes incorrect behavior or a failure.");
    }
    if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake asks for new or improved behavior rather than reporting a confirmed defect.");
    }
    if (proposalSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake frames the request as a proposal or future design direction.");
    }
    if (improvementSignal && !proposalSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake frames the request as a targeted improvement or gap closure.");
    }

    const evidenceItems = [];
    if (documentationSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-documentation-report");
    } else if (bugSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-bug-report");
    } else if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-enhancement-request");
      if (proposalSignal) {
        AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-proposal-request");
      } else if (improvementSignal) {
        AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-improvement-request");
      }
    }
    if (runtimeSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-runtime-context");
    }
    if (apiSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-api-context");
    }

    const deterministicScoring = scoreDeterministicEvidence({
      evidenceItems,
      options: {
        maxWords: MAX_LABEL_WORDS
      }
    });
    const deterministicLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...AutobotIssueClassifier.scoredLabelNames(deterministicScoring.emittedLabels),
      ...technicalLabels,
      ...fallbackTechnicalLabels
    ]);
    const deterministicPrimaryLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...AutobotIssueClassifier.scoredLabelNames(deterministicScoring.primaryLabels),
      ...technicalLabels,
      ...fallbackTechnicalLabels
    ]);
    const outputLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...technicalLabels,
      ...fallbackTechnicalLabels,
      ...deterministicPrimaryLabels,
      ...deterministicLabels
    ]);

    return {
      title: normalized.title,
      body: normalized.body,
      text: normalized.text,
      labels: outputLabels,
      evidenceSignals,
      evidenceItems,
      likelyClassification,
      deterministicLabels,
      deterministicPrimaryLabels,
      deterministicSemver: deterministicScoring.semver,
      releaseRelevant: outputLabels.some((label) => AutobotLabelRegistry.isReleaseRelevantLabel(label))
        || deterministicLabels.some((label) => AutobotIssueClassifier.ISSUE_RELEASE_RELEVANT_LABELS.has(label))
        || /\b(bug|regression|security|runtime|breaking|migration|database|schema|api|vulnerab\w*|hardening|compliance|pen[\s-]?test)\b/.test(normalized.text)
    };
  }
}

function analyzeIssueIntake(issue) {
  return AutobotIssueClassifier.analyzeIssueIntake(issue);
}

module.exports = {
  AutobotIssueClassifier,
  analyzeIssueIntake
};
const { analyzeIssueIntake: analyzeIssueIntakeFromIssueIntake } = require("../issue_intake.cjs");
const { formatBulletLines } = require("../utils.cjs");

class AutobotPromptBuilder {
  static buildIssueSummaryArtifacts({ issue, analyzeIssueIntake = analyzeIssueIntakeFromIssueIntake }) {
    const existingLabels = (issue.labels || [])
      .map((label) => typeof label === "string" ? label : label.name)
      .filter(Boolean);
    const issueAnalysis = analyzeIssueIntake(issue);
    const evidenceSignals = [...issueAnalysis.evidenceSignals];
    if (existingLabels.length > 0) {
      evidenceSignals.push(`Existing labels: ${existingLabels.join(", ")}.`);
    }

    const likelyClassification = [...issueAnalysis.likelyClassification];
    if (issueAnalysis.deterministicLabels.length > 0) {
      likelyClassification.unshift(`Technical labels: ${issueAnalysis.deterministicLabels.join(", ")}.`);
    }
    if (issueAnalysis.deterministicSemver?.decision && issueAnalysis.deterministicSemver.decision !== "patch") {
      likelyClassification.push(`Deterministic semantic-impact signal: ${issueAnalysis.deterministicSemver.decision}.`);
    }
    if (issueAnalysis.releaseRelevant) {
      likelyClassification.push("release-relevant context");
    }

    const risksOrUnknowns = [];
    if (!String(issue.body || "").trim()) {
      risksOrUnknowns.push("The issue body is empty, so the intake signal is narrow.");
    }
    if (!/repro|steps to reproduce|expected|actual|current situation/.test(issueAnalysis.text)) {
      risksOrUnknowns.push("Reproduction detail or acceptance criteria may still be missing.");
    }

    const fallbackSummary = [
      "## Issue Analysis",
      "",
      "### Intake Summary",
      `Issue #${issue.number} is open and titled \"${String(issue.title || "").trim()}\". This deterministic fallback summary uses the issue title, body, and existing labels directly.`,
      "",
      "### Evidence Signals",
      formatBulletLines(evidenceSignals.slice(0, 5), "The issue text provides limited direct classification signals."),
      "",
      "### Likely Classification",
      formatBulletLines(likelyClassification.slice(0, 5), "No narrow label family is strongly supported by deterministic intake parsing."),
      "",
      "### Risks Or Unknowns",
      formatBulletLines(risksOrUnknowns.slice(0, 4), "Use maintainer triage to confirm severity and scope."),
      "",
      "### Release Relevance",
      issueAnalysis.releaseRelevant
        ? "This issue appears potentially release-relevant because the intake text references a runtime, compatibility, or defect signal."
        : "This issue does not show a strong deterministic release signal from the intake text alone."
    ].join("\n");
    return {
      fallbackSummary,
      ready: "true"
    };
  }
}

function buildIssueSummaryArtifacts(input) {
  return AutobotPromptBuilder.buildIssueSummaryArtifacts(input);
}

module.exports = {
  AutobotPromptBuilder,
  buildIssueSummaryArtifacts
};
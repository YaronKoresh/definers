const { analyzeIssueIntake } = require("./issue_intake.cjs");
const { AutobotPromptBuilder, buildIssueSummaryArtifacts: buildIssueSummaryArtifactsFromPhrasing } = require("./phrasing/issue_summary.cjs");

function buildIssueSummaryArtifacts(input) {
  return buildIssueSummaryArtifactsFromPhrasing({
    ...input,
    analyzeIssueIntake
  });
}

module.exports = {
  AutobotPromptBuilder,
  buildIssueSummaryArtifacts
};
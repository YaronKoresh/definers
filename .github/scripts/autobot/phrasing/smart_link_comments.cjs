const {
  GRAPH_COMMENT_MARKER_PREFIX,
  MANAGED_BODY_END,
  MANAGED_BODY_START,
  MANAGED_COMMENT_MARKER_PREFIX,
  buildDirectiveSummary,
  buildMermaidGraphLines,
  buildResultTableRows,
  getEntityKey
} = require("../smart_link/core.cjs");

function stripManagedBlock(text) {
  const pattern = new RegExp(`${MANAGED_BODY_START}[\\s\\S]*?${MANAGED_BODY_END}`, "g");
  return String(text || "").replace(pattern, "").replace(/\n{3,}/g, "\n\n").trimEnd();
}

function buildManagedBlock(source, analysis) {
  const emittedResults = analysis.emittedResults;
  const directiveSummary = buildDirectiveSummary(emittedResults);
  const lines = [MANAGED_BODY_START, "### Smart Link Intelligence", ""];
  if (directiveSummary.closeIds.length > 0) {
    lines.push(`Closes ${directiveSummary.closeIds.map((id) => `#${id}`).join(", ")}`);
  }
  if (directiveSummary.connectIds.length > 0) {
    lines.push(`Connects to ${directiveSummary.connectIds.map((id) => `#${id}`).join(", ")}`);
  }
  if (directiveSummary.dependencyIds.length > 0) {
    lines.push(`Depends on ${directiveSummary.dependencyIds.map((id) => `#${id}`).join(", ")}`);
  }
  if (directiveSummary.advisoryFixIds.length > 0) {
    lines.push(`Advisory Fix: ${directiveSummary.advisoryFixIds.map((id) => `#${id}`).join(", ")}`);
  }
  if (directiveSummary.relatedIds.length > 0) {
    lines.push(`Related Work: ${directiveSummary.relatedIds.map((id) => `#${id}`).join(", ")}`);
  }
  if (lines[lines.length - 1] !== "") lines.push("");
  lines.push("| Target | Kind | State | Score | Relation | Evidence |", "|---|---|---|---:|---|---|", ...buildResultTableRows(emittedResults), "", `Integrity Fingerprint: \`${analysis.fingerprint}\``, MANAGED_BODY_END);
  return lines.join("\n");
}

function buildManagedCommentBody(source, analysis) {
  const marker = `${MANAGED_COMMENT_MARKER_PREFIX}${getEntityKey(source)} -->`;
  const lines = [marker, "### Smart Link Intelligence", "", "| Target | Kind | State | Score | Relation | Evidence |", "|---|---|---|---:|---|---|", ...buildResultTableRows(analysis.emittedResults), "", `Threshold: ${analysis.threshold}`, `Integrity Fingerprint: \`${analysis.fingerprint}\``];
  const mermaidLines = buildMermaidGraphLines(source, analysis.emittedResults);
  if (mermaidLines.length > 0) {
    lines.push("", "### Relationship Graph", "", "```mermaid", ...mermaidLines, "```");
  }
  return lines.join("\n");
}

function buildGraphCommentBody(source, analysis) {
  const marker = `${GRAPH_COMMENT_MARKER_PREFIX}${getEntityKey(source)} -->`;
  const mermaidLines = buildMermaidGraphLines(source, analysis.emittedResults);
  if (mermaidLines.length === 0) return null;
  return [marker, "### Relationship Graph", "", "```mermaid", ...mermaidLines, "```"].join("\n");
}

module.exports = {
  buildGraphCommentBody,
  buildManagedBlock,
  buildManagedCommentBody,
  stripManagedBlock
};
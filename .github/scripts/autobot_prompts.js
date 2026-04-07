const fs = require("fs");

const { analyzeIssueIntake } = require("./autobot_issue_intake");

const { DEFAULT_ISSUE_LABELS, LABEL_GUIDANCE } = require("./autobot_labels");

function writeText(filePath, content) {
  fs.writeFileSync(filePath, content, "utf8");
}

function stripEndMarker(text) {
  return String(text || "").replace(/\nEND_OF_REPORT\s*$/m, "").trim();
}

function parseJsonArray(raw) {
  try {
    const parsed = JSON.parse(String(raw || "[]"));
    return Array.isArray(parsed) ? parsed.map((value) => String(value || "").trim()).filter(Boolean) : [];
  } catch (error) {
    return [];
  }
}

function clampForPrompt(text, maxChars, suffix) {
  if (text.length <= maxChars) {
    return text;
  }
  if (maxChars <= 0) {
    return "";
  }
  const trimmedLength = Math.max(maxChars - suffix.length, 0);
  return text.slice(0, trimmedLength) + suffix;
}

function buildFinalPrSummaryPrompt({ summaryTier, candidateLabelsRaw, batchSummaries, evidencePath = "/tmp/pr_evidence.txt", singleAiWindowsPath = "/tmp/pr_single_ai_windows.txt", outputPath = "/tmp/final_pr_summary_prompt.txt" }) {
  const candidateLabels = parseJsonArray(candidateLabelsRaw || "[]").filter((label) => LABEL_GUIDANCE[label]);
  const evidence = fs.readFileSync(evidencePath, "utf8");
  const singleAiWindows = fs.readFileSync(singleAiWindowsPath, "utf8");
  const normalizedBatchSummaries = (batchSummaries || []).map(stripEndMarker).filter(Boolean);
  const analysisInput = summaryTier === "capped_batch_ai"
    ? [
        "BATCH ANALYSES",
        normalizedBatchSummaries.length > 0 ? normalizedBatchSummaries.map((summary, index) => `BATCH ${index + 1}/${normalizedBatchSummaries.length}\n${summary}`).join("\n\n═══════════════════════════════════════════\n\n") : "Dense batch analysis was unavailable. Use the deterministic evidence scaffold only."
      ].join("\n")
    : [
        "SELECTED DIFF WINDOWS",
        singleAiWindows
      ].join("\n");
  const finalPrompt = [
    "You are a principal software engineer preparing the final compact analysis for one pull request.",
    "Use only the deterministic evidence scaffold and any supplied batch analyses or selected diff windows.",
    "Do not invent hidden files, motivations, or unsupported behavior.",
    "",
    "OUTPUT REQUIREMENTS:",
    "- Output MUST be valid Markdown.",
    "- Keep the result between 900 and 2200 characters.",
    "- Preserve explicit breaking-change, compatibility, migration, API, database, schema, runtime, security, UI, workflow, tooling, test, and documentation signals when they are supported.",
    "- If the evidence implies incompatible behavior or a major version impact, say that explicitly.",
    "- Use EXACTLY this structure:",
    "",
    "## Autobot Summary",
    "",
    "### What Changed",
    "2-4 sentences.",
    "",
    "### Release Relevance",
    "1-3 bullets.",
    "",
    "### Risks And Testing",
    "2-5 bullets covering the top risks and checks.",
    "",
    "### Classification Signals",
    "2-6 bullets listing direct, compact technical signals that are most relevant for label selection.",
    "",
    `- Use only labels from this candidate set: ${candidateLabels.join(", ") || "(none)"}.`,
    "- Add a final metadata line exactly in this form: AUTOBOT_LABEL_HINTS: [\"label1\",\"label2\"]",
    "- Return 0-6 labels in that metadata line, ordered from most to least relevant.",
    "- Use only direct evidence from the prompt for those label hints.",
    "- End your response with the exact final line: END_OF_REPORT",
    "",
    "LABEL GUIDANCE",
    candidateLabels.length > 0 ? candidateLabels.map((label) => `- ${label} — ${LABEL_GUIDANCE[label]}`).join("\n") : "- (none)",
    "",
    "DETERMINISTIC EVIDENCE",
    evidence,
    "",
    analysisInput
  ].join("\n");

  writeText(outputPath, finalPrompt);
  return {
    promptChars: String(finalPrompt.length),
    ready: "true"
  };
}

function parseFinalPrSummary(raw) {
  const normalizedRaw = String(raw || "").trim();
  const hasExpectedStructure = normalizedRaw.includes("## Autobot Summary")
    && normalizedRaw.includes("### What Changed")
    && /\nEND_OF_REPORT\s*$/m.test(normalizedRaw);
  if (!hasExpectedStructure) {
    return {
      summaryBody: "",
      labelHints: "",
      labelHintsReady: "false"
    };
  }
  const labelHintsMatch = normalizedRaw.match(/^AUTOBOT_LABEL_HINTS:\s*(\[[^\n]*\])$/m);
  const labelHints = labelHintsMatch ? labelHintsMatch[1].trim() : "";
  const summaryBody = normalizedRaw
    .replace(/^AUTOBOT_LABEL_HINTS:\s*\[[^\n]*\]\s*$/m, "")
    .replace(/\nEND_OF_REPORT\s*$/m, "")
    .trim();
  return {
    summaryBody,
    labelHints,
    labelHintsReady: labelHints ? "true" : "false"
  };
}

function formatBulletLines(items, fallback) {
  return items.length > 0 ? items.map((item) => `- ${item}`).join("\n") : `- ${fallback}`;
}

function buildIssueSummaryArtifacts({ issue, outputPath = "/tmp/issue_summary_prompt.txt" }) {
  const MAX_ISSUE_BODY_CHARS = 12000;
  const existingLabels = (issue.labels || [])
    .map((label) => typeof label === "string" ? label : label.name)
    .filter(Boolean);
  const issueAnalysis = analyzeIssueIntake(issue);
  const evidenceSignals = [...issueAnalysis.evidenceSignals];
  if (existingLabels.length > 0) {
    evidenceSignals.push(`Existing labels: ${existingLabels.join(", ")}.`);
  }

  const likelyClassification = [...issueAnalysis.likelyClassification];
  if (issueAnalysis.releaseRelevant) likelyClassification.push("release-relevant context");

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
    `Issue #${issue.number} is open and titled \"${String(issue.title || "").trim()}\". This deterministic fallback summary uses the issue title, body, and existing labels because AI analysis may be unavailable.`,
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

  const prompt = [
    "You are a principal software engineer triaging one GitHub issue.",
    "Analyze the issue as an intake artifact, not as implemented code.",
    "Use only the evidence in the title, body, and existing labels.",
    "Do not invent missing repro steps, architecture, or hidden product behavior.",
    "",
    "OUTPUT REQUIREMENTS:",
    "- Output MUST be valid Markdown.",
    "- Do NOT wrap the report in triple backticks.",
    "- Keep the result between 900 and 2400 characters.",
    "- Make the summary precise enough to support high-quality label classification.",
    "- Distinguish between confirmed defects, enhancement requests, proposals, documentation problems, and operational/runtime concerns.",
    "- End your response with the exact final line: END_OF_REPORT",
    "",
    "Use EXACTLY this structure:",
    "",
    "## Issue Analysis",
    "",
    "### Intake Summary",
    "2-4 sentences describing the core request or problem.",
    "",
    "### Evidence Signals",
    "2-5 bullets citing the strongest concrete signals from the issue text.",
    "",
    "### Likely Classification",
    "2-5 bullets naming the strongest candidate label families and why.",
    "",
    "### Risks Or Unknowns",
    "1-4 bullets covering ambiguity, missing information, or notable impact areas.",
    "",
    "### Release Relevance",
    "State whether the issue looks release-relevant and why.",
    "",
    "ISSUE METRICS",
    `Issue number: ${issue.number}`,
    `State: ${issue.state}`,
    `Existing labels: ${existingLabels.join(", ") || "(none)"}`,
    `Author association: ${issue.author_association || "UNKNOWN"}`,
    "",
    "ISSUE TITLE",
    String(issue.title || ""),
    "",
    "ISSUE BODY",
    clampForPrompt(String(issue.body || ""), MAX_ISSUE_BODY_CHARS, "\n[...truncated for prompt budget...]")
  ].join("\n");

  writeText(outputPath, prompt);
  return {
    fallbackSummary,
    ready: "true"
  };
}

function buildLabelPrompt({ eventName, prSummary, issueSummary, candidateLabelsRaw, outputPath = "/tmp/label_prompt.txt" }) {
  const MAX_AI_LABELS = 6;
  const MAX_SUMMARY_CHARS = 2400;
  const eventKind = eventName === "issues" ? "issue" : "pull request";
  const summaryBody = stripEndMarker(eventKind === "issue" ? issueSummary || "" : prSummary || "");
  const candidateLabels = eventKind === "issue"
    ? DEFAULT_ISSUE_LABELS.filter((label) => LABEL_GUIDANCE[label])
    : parseJsonArray(candidateLabelsRaw || "[]").filter((label) => LABEL_GUIDANCE[label]);
  if (!summaryBody || candidateLabels.length === 0) {
    return {
      promptChars: "0",
      ready: "false"
    };
  }
  const prompt = [
    "You are an expert issue and pull request classifier.",
    eventKind === "issue"
      ? "Classify the reported issue from the structured intake summary."
      : "Classify the pull request from the compact Autobot summary.",
    `Return a valid JSON array with up to ${MAX_AI_LABELS} lowercase label names.`,
    "",
    "RULES:",
    `- Return at most ${MAX_AI_LABELS} labels, ordered from most to least relevant.`,
    "- Use only direct evidence from the summary.",
    "- Prefer version-critical labels first when clearly supported.",
    "- Return ONLY a JSON array of label name strings. No markdown, no explanation, no extra text.",
    "- Return label names exactly as provided in ALLOWED LABELS (lowercase).",
    "- If nothing fits, return an empty array: []",
    "",
    "ALLOWED LABELS",
    "",
    candidateLabels.map((label) => `- ${label} — ${LABEL_GUIDANCE[label]}`).join("\n"),
    "",
    "SUMMARY EVIDENCE",
    clampForPrompt(summaryBody, MAX_SUMMARY_CHARS, "\n[...truncated for label prompt budget...]")
  ].join("\n");
  writeText(outputPath, prompt);
  return {
    promptChars: String(prompt.length),
    ready: "true"
  };
}

module.exports = {
  buildFinalPrSummaryPrompt,
  buildIssueSummaryArtifacts,
  buildLabelPrompt,
  parseFinalPrSummary,
  parseJsonArray,
  stripEndMarker
};
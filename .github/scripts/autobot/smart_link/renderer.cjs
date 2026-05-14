const fs = require("fs");

const {
  buildResultTableRows,
  CHECK_NAME,
  getEntityKey,
  GRAPH_COMMENT_MARKER_PREFIX,
  MANAGED_COMMENT_MARKER_PREFIX
} = require("./core.cjs");
const { buildGraphCommentBody, buildManagedBlock, buildManagedCommentBody, stripManagedBlock } = require("../phrasing/smart_link_comments.cjs");
const { readJson } = require("../utils.cjs");

async function findManagedComment({ github, issueNumber, marker, owner, repo }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    issue_number: issueNumber,
    owner,
    per_page: 100,
    repo
  });
  return comments.find((comment) => String(comment.body || "").includes(marker)) || null;
}

async function syncManagedIssueComment({ github, issueNumber, owner, repo, source, analysis }) {
  const marker = `${MANAGED_COMMENT_MARKER_PREFIX}${getEntityKey(source)} -->`;
  const existing = await findManagedComment({ github, issueNumber, marker, owner, repo });
  if (analysis.emittedResults.length === 0) {
    if (existing) {
      await github.rest.issues.deleteComment({ comment_id: existing.id, owner, repo });
    }
    return;
  }
  const body = buildManagedCommentBody(source, analysis);
  if (existing) {
    if (String(existing.body || "") !== body) {
      await github.rest.issues.updateComment({ body, comment_id: existing.id, owner, repo });
    }
    return;
  }
  await github.rest.issues.createComment({ body, issue_number: issueNumber, owner, repo });
}

async function syncGraphComment({ github, owner, repo, source, analysis }) {
  const issueNumber = source.number;
  const marker = `${GRAPH_COMMENT_MARKER_PREFIX}${getEntityKey(source)} -->`;
  const comments = await github.paginate(github.rest.issues.listComments, {
    issue_number: issueNumber,
    owner,
    per_page: 100,
    repo
  });
  const existing = comments.find((comment) => String(comment.body || "").includes(marker))
    || comments.find((comment) => comment.user && comment.user.login === "github-actions[bot]" && String(comment.body || "").includes("### Relationship Graph"))
    || null;
  const body = buildGraphCommentBody(source, analysis);
  if (!body) {
    if (existing) {
      await github.rest.issues.deleteComment({ comment_id: existing.id, owner, repo });
    }
    return;
  }
  if (existing) {
    if (String(existing.body || "") !== body) {
      await github.rest.issues.updateComment({ body, comment_id: existing.id, owner, repo });
    }
    return;
  }
  await github.rest.issues.createComment({ body, issue_number: issueNumber, owner, repo });
}

async function syncPullRequestBody({ github, owner, repo, source, analysis }) {
  const latest = await github.rest.pulls.get({ owner, pull_number: source.number, repo });
  const latestBody = String(latest.data.body || "");
  const withoutManaged = stripManagedBlock(latestBody);
  const nextBody = analysis.emittedResults.length > 0
    ? `${withoutManaged}\n\n${buildManagedBlock(source, analysis)}`.trim()
    : withoutManaged.trim();
  if (nextBody !== latestBody.trim()) {
    await github.rest.pulls.update({ body: nextBody, owner, pull_number: source.number, repo });
  }
}

function buildDependencyCheckPayload(analysis) {
  const closeFamilyResults = analysis.candidateResults.filter((result) => result.requestedCloseFamily);
  const emittedCloseResults = analysis.emittedResults.filter((result) => result.relationKind === "closes");
  let conclusion = "neutral";
  let title = "No closing dependencies to validate.";
  const summary = [];
  const invalidTargets = closeFamilyResults.filter((result) => result.suppressionReasons.includes("close-target-not-issue") || result.suppressionReasons.includes("missing-target"));
  const closedTargets = closeFamilyResults.filter((result) => result.suppressionReasons.includes("close-target-not-open"));
  if (invalidTargets.length > 0) {
    conclusion = "failure";
    title = "Invalid dependency references found.";
    summary.push(`Action Required: This source contains close-family directives that do not target valid open issues: ${invalidTargets.map((result) => `#${result.candidate.number}`).join(", ")}.`);
  } else if (closedTargets.length > 0) {
    conclusion = "neutral";
    title = "Redundant dependency links found.";
    summary.push(`Warning: Close-family directives point to already closed issues: ${closedTargets.map((result) => `#${result.candidate.number}`).join(", ")}.`);
  } else if (emittedCloseResults.length > 0) {
    conclusion = "success";
    title = "All dependency links are valid and actionable.";
    summary.push(`All ${emittedCloseResults.length} close-family links target open issues and passed deterministic validation.`);
  } else {
    summary.push("No issues were emitted in the close-family at or above the deterministic threshold.");
  }
  summary.push(`A total of ${analysis.emittedResults.length} high-confidence relationships were emitted.`);
  return { conclusion, summary: summary.join("\n"), title };
}

async function syncDependencyCheck({ github, owner, repo, source, analysis }) {
  if (source.kind !== "pull_request") return;
  const payload = buildDependencyCheckPayload(analysis);
  try {
    await github.rest.checks.create({
      head_sha: source.metadata.headSha,
      name: CHECK_NAME,
      output: {
        summary: payload.summary,
        title: payload.title
      },
      owner,
      repo,
      status: "completed",
      conclusion: payload.conclusion
    });
  } catch (error) {
    if (error.status !== 403) throw error;
  }
}

function writeJobSummary(source, analysis) {
  const summaryPath = process.env.GITHUB_STEP_SUMMARY;
  if (!summaryPath) return;
  const lines = ["## Smart Link Intelligence", "", `Source: ${source.kind} ${source.number || source.id || "unknown"}`, `Threshold: ${analysis.threshold}`, "", "| Target | Kind | State | Score | Relation | Evidence |", "|---|---|---|---:|---|---|", ...buildResultTableRows(analysis.emittedResults), ""];
  fs.appendFileSync(summaryPath, `${lines.join("\n")}\n`, "utf8");
}

async function renderSmartLinkOutputs({ analysisFile, github, owner, repo }) {
  const analysis = readJson(analysisFile);
  const source = analysis.source;
  if (source.kind === "pull_request") {
    await syncPullRequestBody({ github, owner, repo, source, analysis });
    await syncGraphComment({ github, owner, repo, source, analysis });
    await syncDependencyCheck({ github, owner, repo, source, analysis });
    return;
  }
  if (source.kind === "issue") {
    await syncManagedIssueComment({ github, issueNumber: source.number, owner, repo, source, analysis });
    return;
  }
  if (source.kind === "security_alert") {
    if (source.renderTarget && source.renderTarget.number > 0) {
      await syncManagedIssueComment({ github, issueNumber: source.renderTarget.number, owner, repo, source, analysis });
      return;
    }
    writeJobSummary(source, analysis);
  }
}

module.exports = {
  renderSmartLinkOutputs,
  stripManagedBlock
};
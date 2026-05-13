const BOT_COMMENT_SIGNATURE = "<!-- autobot-summary -->";

function normalizeLabelEntries(labels) {
  return [...new Set((labels || []).map((label) => typeof label === "string" ? label : label?.name).filter(Boolean))]
    .map((name) => ({ name }));
}

function tokenizeSearchText(value) {
  return String(value || "")
    .toLowerCase()
    .split(/[^a-z0-9#:/._-]+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function cloneMilestone(milestone) {
  return milestone ? { ...milestone } : null;
}

function clonePullRequest(pullRequest) {
  return pullRequest
    ? {
        ...pullRequest,
        base: pullRequest.base ? { ...pullRequest.base } : undefined,
        head: pullRequest.head ? { ...pullRequest.head } : undefined,
        labels: normalizeLabelEntries(pullRequest.labels),
        milestone: cloneMilestone(pullRequest.milestone)
      }
    : null;
}

function cloneIssue(issue) {
  return issue
    ? {
        ...issue,
        labels: normalizeLabelEntries(issue.labels),
        milestone: cloneMilestone(issue.milestone),
        pull_request: issue.pull_request ? { ...issue.pull_request } : undefined
      }
    : null;
}

function createIssueFromPullRequest(pullRequest, repositoryFullName) {
  if (!pullRequest) return null;
  return {
    author_association: pullRequest.author_association,
    body: String(pullRequest.body || ""),
    html_url: pullRequest.html_url,
    id: pullRequest.id,
    labels: normalizeLabelEntries(pullRequest.labels),
    milestone: cloneMilestone(pullRequest.milestone),
    number: pullRequest.number,
    pull_request: { url: pullRequest.html_url },
    repository_full_name: repositoryFullName,
    state: pullRequest.state,
    title: pullRequest.title,
    updated_at: pullRequest.updated_at
  };
}

function extractManagedBotCommentMetadata(body) {
  const match = String(body || "").match(/<!-- autobot-metadata:(.+?) -->/s);
  if (!match) return {};
  try {
    return JSON.parse(match[1].trim());
  } catch {
    return {};
  }
}

function getManagedBotComment(comments) {
  return (comments || []).find((comment) => String(comment.body || "").includes(BOT_COMMENT_SIGNATURE)) || null;
}

function findCommentContaining(comments, fragment) {
  return (comments || []).find((comment) => String(comment.body || "").includes(fragment)) || null;
}

function buildMockFailure(spec, operationName) {
  const failure = spec && typeof spec === "object" ? spec : {};
  const error = new Error(String(failure.message || `Mock failure for ${operationName}`));
  if (failure.code) {
    error.code = String(failure.code);
  }
  if (failure.status !== undefined) {
    error.status = Number(failure.status);
    error.response = {
      data: failure.data !== undefined ? failure.data : { message: error.message },
      headers: failure.headers || {},
      status: error.status
    };
  }
  return error;
}

function normalizeMockFailures(failures) {
  const queues = new Map();
  for (const [operationName, rawSpec] of Object.entries(failures || {})) {
    const queue = [];
    for (const entry of Array.isArray(rawSpec) ? rawSpec : [rawSpec]) {
      const spec = entry && typeof entry === "object" ? { ...entry } : {};
      const repeatCount = Math.max(1, Number(spec.times || 1));
      delete spec.times;
      for (let index = 0; index < repeatCount; index += 1) {
        queue.push({ ...spec });
      }
    }
    if (queue.length > 0) {
      queues.set(operationName, queue);
    }
  }
  return queues;
}

function createWorkflowCoreMock() {
  const outputs = new Map();
  return {
    failedMessages: [],
    outputs,
    setFailed(message) {
      this.failedMessages.push(String(message || ""));
    },
    setOutput(name, value) {
      outputs.set(String(name), String(value));
    }
  };
}

function buildIssueSearchText(issue) {
  return [
    issue.title,
    issue.body,
    ...(issue.labels || []).map((label) => label.name),
    issue.milestone?.title,
    issue.html_url
  ].filter(Boolean).join("\n").toLowerCase();
}

function matchesSearchQuery(issue, query) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  if (!normalizedQuery) return true;
  const repoMatch = normalizedQuery.match(/repo:([^\s]+)/);
  const repoFilter = repoMatch ? repoMatch[1].toLowerCase() : "";
  const labelMatches = [...normalizedQuery.matchAll(/label:"([^"]+)"/g)].map((match) => match[1].toLowerCase());
  const terms = tokenizeSearchText(normalizedQuery.replace(/repo:[^\s]+/g, " ").replace(/label:"[^"]+"/g, " "));
  const issueRepo = String(issue.repository_full_name || "octo/repo").toLowerCase();
  if (repoFilter && issueRepo !== repoFilter) return false;
  const issueLabels = (issue.labels || []).map((label) => label.name.toLowerCase());
  if (labelMatches.some((label) => !issueLabels.includes(label))) return false;
  const searchText = buildIssueSearchText(issue);
  return terms.every((term) => searchText.includes(term));
}

function createAutobotPipelineGithubMock(options = {}) {
  const repositoryFullName = String(options.repositoryFullName || "octo/repo");
  const issueNumber = Number(options.issueNumber || options.pullRequest?.number || 12);
  const issueLabels = normalizeLabelEntries(options.issueLabels || []);
  const issueMilestone = cloneMilestone(options.issueMilestone || null);
  const failureQueues = normalizeMockFailures(options.failures);
  const releases = Array.isArray(options.releases) ? options.releases.map((release) => ({ ...release })) : [];
  const comments = Array.isArray(options.comments) ? options.comments.map((comment) => ({ ...comment })) : [];
  const seededCommentsByIssue = new Map(
    Object.entries(options.commentsByIssue || {}).map(([trackedIssueNumber, issueComments]) => [
      Number(trackedIssueNumber),
      (issueComments || []).map((comment) => ({ ...comment }))
    ])
  );
  const milestones = Array.isArray(options.milestones) ? options.milestones.map((milestone) => ({ ...milestone })) : [];
  const pullRequestHead = {
    ref: String(options.pullRequest?.head?.ref || options.pullRequest?.headRef || `autobot/mock-${issueNumber}`),
    sha: String(options.pullRequest?.head?.sha || `sha-${issueNumber}`)
  };
  const headRepoFullName = options.pullRequest?.head?.repo?.full_name || options.pullRequest?.headRepoFullName;
  if (headRepoFullName) {
    pullRequestHead.repo = { full_name: String(headRepoFullName) };
  }
  const pullRequest = {
    author_association: String(options.pullRequest?.author_association || options.pullRequest?.authorAssociation || "MEMBER"),
    base: { ref: "main" },
    body: String(options.pullRequest?.body || ""),
    head: pullRequestHead,
    html_url: String(options.pullRequest?.html_url || `https://example.invalid/pull/${issueNumber}`),
    id: Number(options.pullRequest?.id || issueNumber * 10),
    labels: normalizeLabelEntries(options.pullRequest?.labels || issueLabels),
    milestone: cloneMilestone(options.pullRequest?.milestone || issueMilestone),
    number: Number(options.pullRequest?.number || issueNumber),
    state: String(options.pullRequest?.state || "open"),
    title: String(options.pullRequest?.title || options.issueTitle || `Mock PR ${issueNumber}`),
    updated_at: String(options.pullRequest?.updated_at || options.pullRequest?.updatedAt || "2026-04-13T00:00:00Z")
  };
  const issue = {
    author_association: String(options.issueAuthorAssociation || pullRequest.author_association || "MEMBER"),
    body: String(options.issueBody || pullRequest.body || ""),
    html_url: `https://example.invalid/issues/${issueNumber}`,
    id: issueNumber * 100,
    labels: issueLabels,
    milestone: issueMilestone,
    number: issueNumber,
    pull_request: { url: pullRequest.html_url },
    repository_full_name: repositoryFullName,
    state: String(options.issueState || options.pullRequest?.state || "open"),
    title: String(options.issueTitle || pullRequest.title),
    updated_at: String(options.issueUpdatedAt || pullRequest.updated_at)
  };
  const additionalPullRequests = (options.additionalPullRequests || []).map((entry) => clonePullRequest(entry));
  const additionalIssues = (options.additionalIssues || []).map((entry) => cloneIssue(entry));
  const mirroredPullIssues = additionalPullRequests
    .map((entry) => createIssueFromPullRequest(entry, repositoryFullName))
    .filter(Boolean);
  if (!seededCommentsByIssue.has(issueNumber)) {
    seededCommentsByIssue.set(issueNumber, comments);
  }
  const state = {
    checks: [],
    commentsByIssue: seededCommentsByIssue,
    createdComments: [],
    createdMilestones: [],
    createdReleases: [],
    deletedReleaseIds: [],
    failureLog: [],
    issueUpdates: [],
    issuesByNumber: new Map([[issueNumber, issue], ...additionalIssues.map((entry) => [entry.number, entry]), ...mirroredPullIssues.map((entry) => [entry.number, entry])]),
    milestones,
    pullFilesByNumber: new Map([[pullRequest.number, (options.pullFiles || []).map((file) => ({ ...file }))]]),
    pullRequestsByNumber: new Map([[pullRequest.number, clonePullRequest(pullRequest)], ...additionalPullRequests.map((entry) => [entry.number, entry])]),
    releases
  };
  let nextMilestoneNumber = Number(options.nextMilestoneNumber || 500);
  let nextCommentId = Number(options.nextCommentId || 900);
  let nextReleaseId = Number(options.nextReleaseId || 1000);

  function maybeFail(operationName) {
    const queue = failureQueues.get(operationName);
    if (!queue || queue.length === 0) {
      return;
    }
    const spec = queue.shift();
    state.failureLog.push({
      message: String(spec.message || `Mock failure for ${operationName}`),
      operationName,
      status: spec.status !== undefined ? Number(spec.status) : null
    });
    throw buildMockFailure(spec, operationName);
  }

  const github = {
    paginate: async (method, params) => {
      if (method === github.rest.issues.listComments) {
        maybeFail("paginate.issues.listComments");
        return (state.commentsByIssue.get(params.issue_number) || []).map((comment) => ({ ...comment }));
      }
      if (method === github.rest.issues.listForRepo) {
        maybeFail("paginate.issues.listForRepo");
        return [...state.issuesByNumber.values()]
          .filter((entry) => params.milestone === undefined || params.milestone === null || entry.milestone?.number === Number(params.milestone))
          .filter((entry) => !params.state || params.state === "all" || entry.state === params.state)
          .map((entry) => cloneIssue(entry));
      }
      if (method === github.rest.pulls.listFiles) {
        maybeFail("paginate.pulls.listFiles");
        return (state.pullFilesByNumber.get(params.pull_number) || []).map((file) => ({ ...file }));
      }
      if (method === github.rest.repos.listReleases) {
        maybeFail("paginate.repos.listReleases");
        return state.releases.map((release) => ({ ...release }));
      }
      if (method === github.rest.search.issuesAndPullRequests) {
        maybeFail("paginate.search.issuesAndPullRequests");
        return [...state.issuesByNumber.values()]
          .filter((entry) => matchesSearchQuery(entry, params.q))
          .map((entry) => cloneIssue(entry));
      }
      throw new Error("Unexpected paginate call");
    },
    rest: {
      checks: {
        create: async (payload) => {
          maybeFail("checks.create");
          state.checks.push({ ...payload });
          return { data: { ...payload } };
        }
      },
      git: {
        deleteRef: async () => {
          maybeFail("git.deleteRef");
          return { data: null };
        }
      },
      issues: {
        addLabels: async ({ issue_number, labels }) => {
          maybeFail("issues.addLabels");
          const issueEntry = state.issuesByNumber.get(issue_number);
          issueEntry.labels = normalizeLabelEntries([...(issueEntry.labels || []), ...(labels || [])]);
          const pullRequestEntry = state.pullRequestsByNumber.get(issue_number);
          if (pullRequestEntry) {
            pullRequestEntry.labels = normalizeLabelEntries([...(pullRequestEntry.labels || []), ...(labels || [])]);
          }
          return { data: cloneIssue(issueEntry) };
        },
        createComment: async ({ issue_number, body }) => {
          maybeFail("issues.createComment");
          const comment = { body, id: nextCommentId += 1, user: { login: "github-actions[bot]", type: "Bot" } };
          const issueComments = state.commentsByIssue.get(issue_number) || [];
          issueComments.push(comment);
          state.commentsByIssue.set(issue_number, issueComments);
          state.createdComments.push(comment);
          return { data: { ...comment } };
        },
        createLabel: async () => {
          maybeFail("issues.createLabel");
          return { data: null };
        },
        createMilestone: async ({ title }) => {
          maybeFail("issues.createMilestone");
          const milestone = {
            closed_issues: 0,
            description: "",
            number: nextMilestoneNumber += 1,
            open_issues: 0,
            state: "open",
            title
          };
          state.milestones.push(milestone);
          state.createdMilestones.push(title);
          return { data: cloneMilestone(milestone) };
        },
        deleteComment: async ({ comment_id }) => {
          maybeFail("issues.deleteComment");
          for (const [trackedIssueNumber, issueComments] of state.commentsByIssue.entries()) {
            state.commentsByIssue.set(trackedIssueNumber, issueComments.filter((comment) => comment.id !== comment_id));
          }
          return { data: null };
        },
        get: async ({ issue_number }) => {
          maybeFail("issues.get");
          return { data: cloneIssue(state.issuesByNumber.get(issue_number)) };
        },
        getLabel: async () => {
          maybeFail("issues.getLabel");
          return { data: null };
        },
        getMilestone: async ({ milestone_number }) => {
          maybeFail("issues.getMilestone");
          return { data: cloneMilestone(state.milestones.find((entry) => entry.number === milestone_number)) };
        },
        listComments: async ({ issue_number }) => {
          maybeFail("issues.listComments");
          return { data: (state.commentsByIssue.get(issue_number) || []).map((comment) => ({ ...comment })) };
        },
        listForRepo: async ({ milestone, state: issueState }) => ({
          ...(() => {
            maybeFail("issues.listForRepo");
            return {};
          })(),
          data: [...state.issuesByNumber.values()]
            .filter((entry) => milestone === undefined || milestone === null || entry.milestone?.number === Number(milestone))
            .filter((entry) => issueState === undefined || issueState === "all" || entry.state === issueState)
            .map((entry) => cloneIssue(entry))
        }),
        listMilestones: async () => {
          maybeFail("issues.listMilestones");
          return { data: state.milestones.map((milestone) => cloneMilestone(milestone)) };
        },
        removeLabel: async ({ issue_number, name }) => {
          maybeFail("issues.removeLabel");
          const issueEntry = state.issuesByNumber.get(issue_number);
          issueEntry.labels = normalizeLabelEntries((issueEntry.labels || []).filter((label) => label.name !== name));
          const pullRequestEntry = state.pullRequestsByNumber.get(issue_number);
          if (pullRequestEntry) {
            pullRequestEntry.labels = normalizeLabelEntries((pullRequestEntry.labels || []).filter((label) => label.name !== name));
          }
          return { data: cloneIssue(issueEntry) };
        },
        update: async ({ issue_number, milestone }) => {
          maybeFail("issues.update");
          const issueEntry = state.issuesByNumber.get(issue_number);
          issueEntry.milestone = milestone == null ? null : cloneMilestone(state.milestones.find((entry) => entry.number === milestone));
          const pullRequestEntry = state.pullRequestsByNumber.get(issue_number);
          if (pullRequestEntry) {
            pullRequestEntry.milestone = cloneMilestone(issueEntry.milestone);
          }
          state.issueUpdates.push({ issueNumber: issue_number, milestone: issueEntry.milestone ? issueEntry.milestone.title : null });
          return { data: cloneIssue(issueEntry) };
        },
        updateComment: async ({ comment_id, body }) => {
          maybeFail("issues.updateComment");
          for (const issueComments of state.commentsByIssue.values()) {
            const comment = issueComments.find((entry) => entry.id === comment_id);
            if (comment) {
              comment.body = body;
              return { data: { ...comment } };
            }
          }
          throw new Error("Missing comment");
        },
        updateMilestone: async ({ milestone_number, ...updates }) => {
          maybeFail("issues.updateMilestone");
          const milestone = state.milestones.find((entry) => entry.number === milestone_number);
          Object.assign(milestone, updates);
          return { data: cloneMilestone(milestone) };
        }
      },
      pulls: {
        get: async ({ pull_number }) => {
          maybeFail("pulls.get");
          return { data: clonePullRequest(state.pullRequestsByNumber.get(pull_number)) };
        },
        listFiles: async ({ pull_number }) => {
          maybeFail("pulls.listFiles");
          return { data: (state.pullFilesByNumber.get(pull_number) || []).map((file) => ({ ...file })) };
        },
        update: async ({ pull_number, body }) => {
          maybeFail("pulls.update");
          const pullRequestEntry = state.pullRequestsByNumber.get(pull_number);
          pullRequestEntry.body = String(body || "");
          return { data: clonePullRequest(pullRequestEntry) };
        }
      },
      repos: {
        createRelease: async ({ tag_name, ...release }) => {
          maybeFail("repos.createRelease");
          const createdRelease = { id: nextReleaseId += 1, tag_name, ...release };
          state.releases.unshift(createdRelease);
          state.createdReleases.push(createdRelease);
          return { data: { ...createdRelease } };
        },
        deleteRelease: async ({ release_id }) => {
          maybeFail("repos.deleteRelease");
          state.releases = state.releases.filter((release) => release.id !== release_id);
          state.deletedReleaseIds.push(release_id);
          return { data: null };
        },
        listReleases: async () => {
          maybeFail("repos.listReleases");
          return { data: state.releases.map((release) => ({ ...release })) };
        }
      },
      search: {
        issuesAndPullRequests: async ({ q }) => ({
          ...(() => {
            maybeFail("search.issuesAndPullRequests");
            return {};
          })(),
          data: {
            items: [...state.issuesByNumber.values()]
              .filter((entry) => matchesSearchQuery(entry, q))
              .map((entry) => cloneIssue(entry))
          }
        })
      }
    },
    state
  };

  return github;
}

module.exports = {
  createAutobotPipelineGithubMock,
  createWorkflowCoreMock,
  extractManagedBotCommentMetadata,
  findCommentContaining,
  getManagedBotComment
};
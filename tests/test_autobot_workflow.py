from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "autobot.yml"
AI_HELPER = ROOT / ".github" / "scripts" / "autobot_ai.js"
LABELS_HELPER = ROOT / ".github" / "scripts" / "autobot_labels.js"
PR_ANALYSIS_HELPER = ROOT / ".github" / "scripts" / "autobot_pr_analysis.js"
PROJECT_MANAGER_HELPER = (
    ROOT / ".github" / "scripts" / "autobot_project_manager.js"
)
PROMPTS_HELPER = ROOT / ".github" / "scripts" / "autobot_prompts.js"


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def read_ai_helper() -> str:
    return AI_HELPER.read_text(encoding="utf-8")


def read_labels_helper() -> str:
    return LABELS_HELPER.read_text(encoding="utf-8")


def read_pr_analysis_helper() -> str:
    return PR_ANALYSIS_HELPER.read_text(encoding="utf-8")


def read_project_manager_helper() -> str:
    return PROJECT_MANAGER_HELPER.read_text(encoding="utf-8")


def read_prompts_helper() -> str:
    return PROMPTS_HELPER.read_text(encoding="utf-8")


def test_major_release_alert_is_scoped_to_breaking_prs() -> None:
    helper = read_project_manager_helper()

    assert (
        'const currentPrIsBreaking = currentLabelNames.includes("breaking-change");'
        in helper
    )
    assert "if (currentPrIsBreaking && isPR) {" in helper
    assert "} else if (existingMajorAlertComment) {" in helper


def test_release_relevance_checks_normalize_label_names() -> None:
    helper = read_project_manager_helper()

    assert "function hasLabelName(labels, expectedLabel) {" in helper
    assert (
        "return RELEASE_RELEVANT_LABELS.some((label) => hasLabelName(labels, label));"
        in helper
    )
    assert (
        "const currentLabelNames = freshIssue.data.labels.map((label) => normalizeLabelName(label.name));"
        in helper
    )


def test_workflow_uses_pr_scoped_concurrency() -> None:
    workflow = read_workflow()

    assert "concurrency:" in workflow
    assert (
        "group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.event.pull_request.number || github.event.issue.number || github.ref }}"
        in workflow
    )
    assert "cancel-in-progress: true" in workflow


def test_pr_summary_uses_tiered_budget_routing() -> None:
    workflow = read_workflow()
    helper = read_pr_analysis_helper()

    assert "collectPullRequestSnapshot" in workflow
    assert "analyzePullRequestSnapshot" in workflow
    assert "const MAX_BATCH_SUMMARY_REQUESTS = 2;" in helper
    assert "function buildSummaryBatches(entries) {" in helper
    assert (
        "writeText(`/tmp/summary_batch_${batchIndex + 1}.txt`, prompt);"
        in helper
    )
    assert (
        'const summaryTier = zeroAiEligible ? "zero_ai" : highVolume ? "capped_batch_ai" : "single_ai";'
        in helper
    )
    assert "estimated_ai_requests: String(estimatedAiRequests)," in helper
    assert "deterministic_summary: deterministicSummary," in helper
    assert "const titleSuggestsBug =" in helper
    assert "const titleSuggestsEnhancement =" in helper


def test_large_pr_path_caps_batch_summaries_and_merges_final_summary() -> None:
    workflow = read_workflow()
    helper = read_prompts_helper()

    assert "- name: AI — Generate Batch Summary 1" in workflow
    assert "- name: AI — Generate Batch Summary 2" in workflow
    assert "- name: AI — Generate Batch Summary 10" not in workflow
    assert "- name: Build Final PR Summary Prompt" in workflow
    assert "- name: AI — Generate Final PR Summary" in workflow
    assert "- name: Parse Final PR Summary" in workflow
    assert (
        "- Add a final metadata line exactly in this form: AUTOBOT_LABEL_HINTS:"
        in helper
    )
    assert 'labelHintsReady: labelHints ? "true" : "false"' in helper


def test_workflow_uses_helper_backed_ai_steps_and_cooldown_resolution() -> None:
    workflow = read_workflow()

    assert "autobot_ai.js" in workflow
    assert "runManagedInference" in workflow
    assert "- name: Resolve AI Cooldown State" in workflow
    assert "ai-inference@v2" not in workflow
    assert (
        "AI_COOLDOWN_UNTIL: ${{ steps.ai-cooldown-state.outputs.until }}"
        in workflow
    )


def test_zero_ai_prs_skip_final_pr_summary_step() -> None:
    workflow = read_workflow()
    helper = read_pr_analysis_helper()

    assert "steps.collect.outputs.summary_tier != 'zero_ai'" in workflow
    assert (
        "deterministic_labels_json: JSON.stringify(deterministicLabels),"
        in helper
    )
    assert "summary_tier: summaryTier," in helper


def test_fallback_label_prompt_uses_parsed_pr_summary_and_candidate_labels() -> (
    None
):
    workflow = read_workflow()
    helper = read_prompts_helper()

    assert (
        "AI_PR_SUMMARY: ${{ steps.parse-pr-summary.outputs.summary_body }}"
        in workflow
    )
    assert (
        "CANDIDATE_LABELS: ${{ steps.collect.outputs.candidate_labels_json }}"
        in workflow
    )
    assert (
        "AI_COMPACT_SUMMARY: ${{ steps.ai-compact.outputs.response }}"
        not in workflow
    )
    assert (
        "steps.parse-pr-summary.outputs.label_hints_ready != 'true'" in workflow
    )
    assert (
        '"Classify the pull request from the compact Autobot summary."'
        in helper
    )


def test_final_summary_prompt_preserves_classification_signals() -> None:
    helper = read_prompts_helper()

    assert "### Classification Signals" in helper
    assert (
        "Preserve explicit breaking-change, compatibility, migration, API, database, schema, runtime, security, workflow, tooling, test, and documentation signals"
        in helper
    )
    assert (
        "If the evidence implies incompatible behavior or a major version impact, say that explicitly."
        in helper
    )


def test_synchronize_without_previous_bot_comment_keeps_predicted_labels() -> (
    None
):
    helper = read_project_manager_helper()

    assert (
        "const hasFreshPrLabels = isPR && parsedAiLabels.length > 0;" in helper
    )
    assert (
        'if (payload.action === "synchronize" && previousBotLabels.length > 0) {'
        in helper
    )
    assert "nextAiLabels = predictedLabels;" in helper
    assert (
        '} else if (isPR && payload.action === "synchronize" && previousBotLabels.length > 0) {'
        in helper
    )
    assert (
        "labelsToRemove = isPR && hasFreshPrLabels ? previousBotLabels.filter((label) => !nextAiLabels.includes(label)) : [];"
        in helper
    )


def test_issue_label_fallback_uses_reduced_default_label_set() -> None:
    labels = read_labels_helper()
    prompts = read_prompts_helper()

    assert "const DEFAULT_ISSUE_LABELS = [" in labels
    assert '"improvement",' in labels
    assert '"proposal",' in labels
    assert '"workflow",' in labels
    assert (
        "DEFAULT_ISSUE_LABELS.filter((label) => LABEL_GUIDANCE[label])"
        in prompts
    )


def test_issue_summary_exposes_deterministic_fallback() -> None:
    workflow = read_workflow()
    helper = read_prompts_helper()

    assert "const fallbackSummary = [" in helper
    assert "return {" in helper
    assert "fallbackSummary," in helper
    assert (
        "AI_ISSUE_SUMMARY: ${{ steps.ai-issue-summary.outputs.response || steps.issue-summary-prompt.outputs.fallback_summary }}"
        in workflow
    )


def test_summary_comment_is_source_agnostic() -> None:
    helper = read_project_manager_helper()

    assert (
        "<summary><strong>🏷️ Label Classification</strong></summary>" in helper
    )
    assert (
        "Labels were determined from the code diff evidence collected by Autobot, not from the PR title or description alone."
        in helper
    )


def test_summary_comment_includes_cached_cooldown_wait_notice() -> None:
    workflow = read_workflow()
    helper = read_project_manager_helper()

    assert (
        "AI_COOLDOWN_UNTIL: ${{ steps.ai-cooldown-state.outputs.until }}"
        in workflow
    )
    assert (
        'const cooldownNotice = aiCooldownActive && prSummaryTier && prSummaryTier !== "zero_ai"'
        in helper
    )
    assert "Try again after that time." in helper


def test_ai_helper_tracks_48_hour_rate_limit_cooldown() -> None:
    helper = read_ai_helper()

    assert "const DEFAULT_COOLDOWN_HOURS = 48;" in helper
    assert 'const COOLDOWN_LABEL_NAME = "autobot-ai-cooldown";' in helper
    assert (
        'const AI_ENDPOINT = "https://models.github.ai/inference/chat/completions";'
        in helper
    )
    assert "if (response.status === 429) {" in helper
    assert "await upsertCooldownLabel({" in helper
    assert "Try again after that time." in helper


def test_stale_major_release_alert_is_removed_when_pr_is_not_breaking() -> None:
    helper = read_project_manager_helper()

    assert (
        "async function getExistingMajorAlertComment({ github, owner, repo, issueNumber }) {"
        in helper
    )
    assert "} else if (existingMajorAlertComment) {" in helper
    assert (
        "await github.rest.issues.deleteComment({ owner, repo, comment_id: existingMajorAlertComment.id });"
        in helper
    )


def test_workflow_uses_smaller_helper_backed_project_steps() -> None:
    workflow = read_workflow()

    assert "- name: Prepare Project State" in workflow
    assert "- name: Sync Prepared Project State" in workflow
    assert "- name: Sync Project Milestone" in workflow
    assert "- name: Finalize Closed Pull Request Release" in workflow
    assert "- name: Manage Project" not in workflow
    assert "autobot_project_manager.js" in workflow


def test_workflow_uses_helper_backed_prompt_steps() -> None:
    workflow = read_workflow()

    assert "buildFinalPrSummaryPrompt" in workflow
    assert "parseFinalPrSummary" in workflow
    assert "buildIssueSummaryArtifacts" in workflow
    assert "buildLabelPrompt" in workflow
    assert "autobot_prompts.js" in workflow

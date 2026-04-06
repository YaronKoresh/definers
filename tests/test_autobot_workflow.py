from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "autobot.yml"


def test_major_release_alert_is_scoped_to_breaking_prs() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "const currentPrIsBreaking = currentIssueLabels.includes('breaking-change');"
        in workflow
    )
    assert "if (currentPrIsBreaking && isPR) {" in workflow
    assert (
        "const isBreaking = releaseItems.some(item => item.labels.some(label => label.name === 'breaking-change'));"
        not in workflow
    )


def test_release_relevance_checks_normalize_label_names() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "function hasLabelName(labels, expectedLabel) {" in workflow
    assert (
        "return RELEASE_RELEVANT_LABELS.some((label) => hasLabelName(labels, label));"
        in workflow
    )
    assert (
        "const currentLabelNames = freshIssue.data.labels.map(l => normalizeLabelName(l.name));"
        in workflow
    )


def test_pr_summary_uses_hierarchical_batch_reduction() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "const MAX_SUMMARY_BATCHES = 10;" in workflow
    assert "function buildSummaryBatches(entries) {" in workflow
    assert (
        "fs.writeFileSync(`/tmp/summary_batch_${batchIndex + 1}.txt`, prompt);"
        in workflow
    )
    assert "- name: Build Aggregated Summary Prompt" in workflow
    assert "BATCH SUMMARIES" in workflow


def test_pr_summary_runs_per_batch_ai_steps_before_final_summary() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "- name: AI — Generate Batch Summary 1" in workflow
    assert "- name: AI — Generate Batch Summary 10" in workflow
    assert (
        "AI_BATCH_SUMMARY_10: ${{ steps.ai-summary-batch-10.outputs.response }}"
        in workflow
    )
    assert (
        "if: steps.aggregate-summary-prompt.outputs.ready == 'true'" in workflow
    )


def test_label_prompt_uses_only_compact_summary() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "AI_COMPACT_SUMMARY: ${{ steps.ai-compact.outputs.response }}"
        in workflow
    )
    assert (
        "AI_DETAILED_SUMMARY: ${{ steps.ai-summary.outputs.response }}"
        not in workflow.split("- name: Build Label Prompt", 1)[1].split(
            "- name: AI — Classify Labels", 1
        )[0]
    )
    assert 'const summarySections = eventKind === "issue"' in workflow
    assert '"COMPACT SUMMARY"' in workflow
    assert (
        '"You will receive an AI-generated compact technical summary of code changes."'
        in workflow
    )


def test_compact_summary_prompt_preserves_classification_signals() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "### Classification Signals" in workflow
    assert (
        "Preserve explicit breaking-change, compatibility, migration, API, database, schema, runtime, security, workflow, tooling, test, and documentation signals"
        in workflow
    )
    assert (
        "If the evidence implies incompatible behavior or a major version impact, say that explicitly."
        in workflow
    )


def test_synchronize_without_previous_bot_comment_keeps_predicted_labels() -> (
    None
):
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "if (payload.action === 'synchronize' && previousBotLabels.length > 0) {"
        in workflow
    )
    assert "nextAiLabels = predictedLabels;" in workflow


def test_label_prompt_expects_multiple_supported_labels() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "Returning 2-6 labels is normal for multi-area pull requests or issues"
        in workflow
    )
    assert (
        "major version bump, incompatibility, required migration, or consumer adaptation"
        in workflow
    )


def test_stale_major_release_alert_is_removed_when_pr_is_not_breaking() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "async function getExistingMajorAlertComment() {" in workflow
    assert "} else if (existingMajorAlertComment) {" in workflow
    assert (
        "await github.rest.issues.deleteComment({ owner, repo, comment_id: existingMajorAlertComment.id });"
        in workflow
    )

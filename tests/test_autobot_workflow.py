import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "autobot.yml"
LABELS_HELPER = ROOT / ".github" / "scripts" / "autobot_labels.js"
ISSUE_INTAKE_HELPER = ROOT / ".github" / "scripts" / "autobot_issue_intake.js"
SCORER_HELPER = ROOT / ".github" / "scripts" / "autobot_deterministic_scorer.js"
PR_ANALYSIS_HELPER = ROOT / ".github" / "scripts" / "autobot_pr_analysis.js"
PROJECT_MANAGER_HELPER = (
    ROOT / ".github" / "scripts" / "autobot_project_manager.js"
)
PROMPTS_HELPER = ROOT / ".github" / "scripts" / "autobot_prompts.js"


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def read_labels_helper() -> str:
    return LABELS_HELPER.read_text(encoding="utf-8")


def read_scorer_helper() -> str:
    return SCORER_HELPER.read_text(encoding="utf-8")


def read_pr_analysis_helper() -> str:
    return PR_ANALYSIS_HELPER.read_text(encoding="utf-8")


def read_project_manager_helper() -> str:
    return PROJECT_MANAGER_HELPER.read_text(encoding="utf-8")


def read_prompts_helper() -> str:
    return PROMPTS_HELPER.read_text(encoding="utf-8")


def build_snapshot_file(
    filename: str,
    patch: str,
    *,
    status: str = "modified",
    additions: int = 1,
    deletions: int = 0,
) -> dict[str, object]:
    return {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
        "patch": patch,
        "rawPatchAvailable": True,
    }


def build_snapshot(
    *,
    title: str,
    files: list[dict[str, object]],
    body: str = "",
    head_ref: str = "feature/autobot-test",
) -> dict[str, object]:
    total_additions = sum(int(file["additions"]) for file in files)
    total_deletions = sum(int(file["deletions"]) for file in files)
    return {
        "pullRequest": {
            "number": 1,
            "title": title,
            "body": body,
            "headRef": head_ref,
        },
        "totals": {
            "filesChanged": len(files),
            "additions": total_additions,
            "deletions": total_deletions,
            "totalChanges": total_additions + total_deletions,
        },
        "files": files,
    }


def run_pr_analysis(snapshot: dict[str, object]) -> dict[str, object]:
    node_path = shutil.which("node") or shutil.which("node.exe")
    if node_path is None:
        pytest.skip("node is required for autobot helper behavior tests")
    script = """
const fs = require("fs");
const helper = require(process.argv[1]);
const snapshot = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const result = helper.analyzePullRequestSnapshotData(snapshot, { writeArtifacts: false });
process.stdout.write(JSON.stringify(result));
""".strip()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(snapshot, handle)
        snapshot_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                node_path,
                "-e",
                script,
                str(PR_ANALYSIS_HELPER),
                str(snapshot_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        snapshot_path.unlink(missing_ok=True)
    return json.loads(completed.stdout)


def run_js_helper_function(
    helper_path: Path, function_name: str, payload: object
) -> object:
    node_path = shutil.which("node") or shutil.which("node.exe")
    if node_path is None:
        pytest.skip("node is required for autobot helper behavior tests")
    script = """
const fs = require("fs");
const helper = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const result = helper[process.argv[3]](payload);
process.stdout.write(JSON.stringify(result));
""".strip()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                node_path,
                "-e",
                script,
                str(helper_path),
                str(payload_path),
                function_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        payload_path.unlink(missing_ok=True)
    return json.loads(completed.stdout)


def read_js_module_export(helper_path: Path, export_name: str) -> object:
    node_path = shutil.which("node") or shutil.which("node.exe")
    if node_path is None:
        pytest.skip("node is required for autobot helper behavior tests")
    script = """
const helper = require(process.argv[1]);
process.stdout.write(JSON.stringify(helper[process.argv[2]]));
""".strip()
    completed = subprocess.run(
        [
            node_path,
            "-e",
            script,
            str(helper_path),
            export_name,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def run_js_async_script(
    helper_path: Path, script: str, payload: object
) -> object:
    node_path = shutil.which("node") or shutil.which("node.exe")
    if node_path is None:
        pytest.skip("node is required for autobot helper behavior tests")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                node_path,
                "-e",
                script,
                str(helper_path),
                str(payload_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        payload_path.unlink(missing_ok=True)
    return json.loads(completed.stdout)


def parse_result_labels(result: dict[str, object], key: str) -> set[str]:
    return set(json.loads(str(result[key])))


def test_major_release_alert_is_scoped_to_breaking_prs() -> None:
    helper = read_project_manager_helper()

    assert (
        'const currentPrRequiresMajorRelease = currentSemverDecision === "major" || currentLabelNames.includes("breaking-change");'
        in helper
    )
    assert "if (currentPrRequiresMajorRelease && isPR) {" in helper
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
    assert "function buildSummaryBatches(entries) {" not in helper
    assert "writeText(`/tmp/summary_batch_" not in helper
    assert "estimated_ai_requests: String(estimatedAiRequests)," not in helper
    assert "summary_tier: summaryTier," not in helper
    assert "deterministic_summary: deterministicSummary," in helper
    assert "function deriveTitleSignals(prSignalText) {" in helper
    assert "const titleSignals = deriveTitleSignals(prSignalText);" in helper


def test_large_pr_path_caps_batch_summaries_and_merges_final_summary() -> None:
    workflow = read_workflow()

    assert "- name: AI — Generate Batch Summary 1" not in workflow
    assert "- name: AI — Generate Batch Summary 2" not in workflow
    assert "- name: Build Final PR Summary Prompt" not in workflow
    assert "- name: AI — Generate Final PR Summary" not in workflow
    assert "- name: Parse Final PR Summary" not in workflow
    assert (
        "AUTOBOT_SUMMARY_TEXT: ${{ steps.collect.outputs.deterministic_summary || steps.issue-summary-prompt.outputs.fallback_summary }}"
        in workflow
    )


def test_workflow_removes_live_ai_steps_and_models_permissions() -> None:
    workflow = read_workflow()

    assert "autobot_ai.js" not in workflow
    assert "runManagedInference" not in workflow
    assert "- name: Resolve AI Cooldown State" not in workflow
    assert "ai-inference@v2" not in workflow
    assert "models: read" not in workflow


def test_zero_ai_prs_skip_final_pr_summary_step() -> None:
    workflow = read_workflow()
    helper = read_pr_analysis_helper()

    assert "steps.collect.outputs.summary_tier != 'zero_ai'" not in workflow
    assert (
        "deterministic_labels_json: JSON.stringify(deterministicLabels),"
        in helper
    )
    assert "summary_tier: summaryTier," not in helper


def test_workflow_passes_deterministic_semver_into_project_state() -> None:
    workflow = read_workflow()

    assert (
        "PR_DETERMINISTIC_SEMVER: ${{ steps.collect.outputs.deterministic_semver_json }}"
        in workflow
    )
    assert (
        "deterministicSemverRaw: process.env.PR_DETERMINISTIC_SEMVER,"
        in workflow
    )


def test_workflow_uses_deterministic_labels_without_ai_prompt_layer() -> None:
    workflow = read_workflow()

    assert (
        "AUTOBOT_LABELS_JSON: ${{ steps.collect.outputs.deterministic_labels_json }}"
        in workflow
    )
    assert "Build Label Prompt" not in workflow
    assert "AI — Classify Labels" not in workflow
    assert "parse-pr-summary" not in workflow


def test_final_summary_prompt_preserves_classification_signals() -> None:
    helper = read_prompts_helper()

    assert "### Likely Classification" in helper
    assert "Deterministic scorer labels:" in helper
    assert "Deterministic semantic-impact signal:" in helper


def test_synchronize_with_fresh_analysis_reconciles_against_live_pr_labels() -> (
    None
):
    helper = read_project_manager_helper()

    assert "function resolvePrLabelDelta(input) {" in helper
    assert (
        "const { action, previousBotLabels, currentPrLabels, autobotLabelsRaw } = input;"
        in helper
    )
    assert "const currentLabels = uniqueValidLabels(currentPrLabels);" in helper
    assert "deriveSummarySignalLabels" not in helper
    assert 'const hasFreshPrLabelResult = rawAutobotLabels !== "";' in helper
    assert "if (hasFreshPrLabelResult) {" in helper
    assert "nextAutobotLabels = parsedAutobotLabels;" in helper
    assert (
        '} else if (action === "synchronize" && previousLabels.length > 0) {'
        in helper
    )
    assert "const labelsToRemove = hasFreshPrLabelResult" in helper
    assert (
        "const labelsToAdd = nextAutobotLabels.filter((label) => !currentLabels.includes(label));"
        in helper
    )
    assert (
        "const livePrIssue = isPR ? await github.rest.issues.get({ owner, repo, issue_number: issueNumber }) : null;"
        in helper
    )
    assert (
        "const currentPrLabels = isPR ? labelNamesFromIssue(livePrIssue?.data) : [];"
        in helper
    )


def test_issue_label_fallback_uses_reduced_default_label_set() -> None:
    labels = read_labels_helper()

    assert "const DEFAULT_ISSUE_LABELS = [" in labels
    assert '"improvement",' in labels
    assert '"proposal",' in labels
    assert '"workflow",' in labels


def test_issue_summary_exposes_deterministic_fallback() -> None:
    workflow = read_workflow()
    helper = read_prompts_helper()

    assert "const fallbackSummary = [" in helper
    assert "return {" in helper
    assert "fallbackSummary," in helper
    assert (
        "AUTOBOT_SUMMARY_TEXT: ${{ steps.collect.outputs.deterministic_summary || steps.issue-summary-prompt.outputs.fallback_summary }}"
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


def test_summary_comment_no_longer_uses_workflow_cooldown_path() -> None:
    workflow = read_workflow()
    helper = read_project_manager_helper()

    assert (
        "AI_COOLDOWN_UNTIL: ${{ steps.ai-cooldown-state.outputs.until }}"
        not in workflow
    )
    assert (
        'const cooldownNotice = aiCooldownActive && prSummaryTier && prSummaryTier !== "zero_ai"'
        not in helper
    )
    assert "GitHub Models returned temporary throttling" not in helper


def test_ai_helper_file_is_removed_from_repo() -> None:
    assert not (ROOT / ".github" / "scripts" / "autobot_ai.js").exists()


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


def test_workflow_uses_deterministic_labels_as_only_live_source() -> None:
    workflow = read_workflow()

    assert (
        "AUTOBOT_LABELS_JSON: ${{ steps.collect.outputs.deterministic_labels_json }}"
        in workflow
    )


def test_version_bump_map_covers_every_supported_label() -> None:
    label_definitions = read_js_module_export(
        LABELS_HELPER, "LABEL_DEFINITIONS"
    )
    version_bumps = read_js_module_export(
        LABELS_HELPER, "VERSION_BUMP_BY_LABEL"
    )

    assert set(version_bumps) == set(label_definitions)
    assert version_bumps["breaking-change"] == "major"
    assert version_bumps["enhancement"] == "minor"
    assert version_bumps["improvement"] == "minor"
    assert version_bumps["deprecation"] == "minor"
    assert version_bumps["bug"] == "patch"
    assert version_bumps["documentation"] == "patch"


def test_autobot_uses_expanded_label_cap() -> None:
    project_helper = read_project_manager_helper()
    analysis_helper = read_pr_analysis_helper()

    assert "const MAX_AUTOBOT_LABELS = 12;" in project_helper
    assert "const MAX_AUTOBOT_LABELS = 12;" in analysis_helper


def test_pr_analysis_exports_deterministic_scored_outputs() -> None:
    helper = read_pr_analysis_helper()

    assert (
        'const { scoreDeterministicEvidence } = require("./autobot_deterministic_scorer");'
        in helper
    )
    assert (
        "function buildDeterministicEvidence(filesWithContext, context) {"
        in helper
    )
    assert (
        "deterministic_evidence_json: JSON.stringify(evidenceItems)," in helper
    )
    assert (
        "deterministic_category_scores_json: JSON.stringify(scoring.categoryScores),"
        in helper
    )
    assert (
        "deterministic_impact_scores_json: JSON.stringify(scoring.impactScores),"
        in helper
    )
    assert (
        "deterministic_label_scores_json: JSON.stringify(scoring.labelScores),"
        in helper
    )
    assert (
        "deterministic_semver_json: JSON.stringify(scoring.semver)," in helper
    )


def test_deterministic_scorer_exports_core_contract() -> None:
    helper = read_scorer_helper()
    thresholds = read_js_module_export(SCORER_HELPER, "LABEL_THRESHOLDS")
    semver_thresholds = read_js_module_export(
        SCORER_HELPER, "SEMVER_THRESHOLDS"
    )
    categories = read_js_module_export(SCORER_HELPER, "CATEGORY_DEFINITIONS")
    rules = read_js_module_export(SCORER_HELPER, "EVIDENCE_RULES")

    assert "scoreDeterministicEvidence" in helper
    assert thresholds == {"retain": 0.35, "emit": 0.55, "primary": 0.72}
    assert semver_thresholds == {"major": 0.72, "minor": 0.58, "patch": 0}
    assert "public-contract" in categories
    assert "runtime-platform" in categories
    assert "removed-public-export" in rules
    assert "runtime-support-added" in rules


def test_deterministic_scorer_marks_removed_public_export_as_major() -> None:
    result = run_js_helper_function(
        SCORER_HELPER,
        "scoreDeterministicEvidence",
        {"evidenceItems": [{"ruleId": "removed-public-export"}]},
    )

    emitted_labels = {entry["label"] for entry in result["emittedLabels"]}

    assert result["semver"]["decision"] == "major"
    assert result["semver"]["hardRule"] is True
    assert "breaking-change" in emitted_labels
    assert result["impactScores"]["major-public-contract"]["score"] >= 0.72


def test_deterministic_scorer_marks_runtime_support_add_as_minor() -> None:
    result = run_js_helper_function(
        SCORER_HELPER,
        "scoreDeterministicEvidence",
        {"evidenceItems": [{"ruleId": "runtime-support-added"}]},
    )

    emitted_labels = {entry["label"] for entry in result["emittedLabels"]}

    assert result["semver"]["decision"] == "minor"
    assert "runtime" in emitted_labels
    assert result["impactScores"]["minor-runtime-support-add"]["score"] > 0


def test_deterministic_scorer_keeps_docs_and_tests_patch_only() -> None:
    result = run_js_helper_function(
        SCORER_HELPER,
        "scoreDeterministicEvidence",
        {
            "evidenceItems": [
                {"ruleId": "documentation-surface"},
                {"ruleId": "tests-added"},
            ]
        },
    )

    emitted_labels = {entry["label"] for entry in result["emittedLabels"]}

    assert result["semver"]["decision"] == "patch"
    assert "documentation" in emitted_labels
    assert "test" in emitted_labels
    assert "breaking-change" not in emitted_labels


def test_pr_label_delta_retains_more_than_six_labels_when_supported() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "resolvePrLabelDelta",
        {
            "action": "opened",
            "currentPrLabels": [],
            "previousBotLabels": [],
            "autobotLabelsRaw": json.dumps(
                [
                    "breaking-change",
                    "enhancement",
                    "security",
                    "api",
                    "database",
                    "schema",
                    "compatibility",
                    "migration",
                    "feature-flag",
                    "runtime",
                    "performance",
                    "workflow",
                ]
            ),
        },
    )

    assert len(result["nextAutobotLabels"]) == 12
    assert result["nextAutobotLabels"] == [
        "breaking-change",
        "enhancement",
        "security",
        "performance",
        "api",
        "database",
        "schema",
        "compatibility",
        "migration",
        "feature-flag",
        "runtime",
        "workflow",
    ]


def test_pr_label_delta_does_not_backfill_from_summary_text() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "resolvePrLabelDelta",
        {
            "action": "opened",
            "currentPrLabels": ["bug"],
            "previousBotLabels": [],
            "autobotLabelsRaw": json.dumps(["bug"]),
            "summaryText": "## Autobot Summary\n\n### What Changed\nThis adds a new feature and introduces breaking changes for existing consumers.\n\n### Release Relevance\n- The summary explicitly calls out a minor version impact for the new capability.\n- The old contract is backward-incompatible and requires migration, so this is a major version impact.\n\n### Risks And Testing\n- Consumers must update.\n\n### Classification Signals\n- New feature added.\n- Breaking change for consumers.",
        },
    )

    assert result["nextAutobotLabels"] == ["bug"]
    assert result["labelsToAdd"] == []


def test_sync_prepared_project_state_retries_invalid_new_label_addition() -> (
    None
):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(
            {
                "action": "opened",
                "isPR": True,
                "labelsToRemove": [],
                "labelsToAdd": ["docker", "workflow"],
                "commentBody": "",
                "nextAutobotLabels": ["docker", "workflow"],
            },
            handle,
        )
        state_path = Path(handle.name)

    script = """
const fs = require("fs");
const helper = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const knownLabels = new Set(payload.knownLabels || []);
const addAttempts = new Map();
const calls = { getLabel: [], createLabel: [], addLabels: [] };
const github = {
    rest: {
        issues: {
            getLabel: async ({ name }) => {
                calls.getLabel.push(name);
                if (knownLabels.has(name)) {
                    return { data: { name } };
                }
                const error = new Error("Not Found");
                error.status = 404;
                throw error;
            },
            createLabel: async ({ name }) => {
                calls.createLabel.push(name);
                knownLabels.add(name);
                return { data: { name } };
            },
            addLabels: async ({ labels }) => {
                calls.addLabels.push(labels);
                const label = labels[0];
                const attempt = (addAttempts.get(label) || 0) + 1;
                addAttempts.set(label, attempt);
                if (label === "workflow" && attempt === 1) {
                    const error = new Error("Validation Failed");
                    error.status = 422;
                    error.response = {
                        data: {
                            errors: [
                                { value: "workflow", resource: "Label", field: "name", code: "invalid" }
                            ]
                        }
                    };
                    throw error;
                }
                return { data: labels };
            },
            removeLabel: async () => {
                throw new Error("removeLabel should not be called");
            },
            updateComment: async () => ({ data: {} }),
            createComment: async () => ({ data: {} }),
        }
    }
};

(async () => {
    const result = await helper.syncPreparedProjectState({
        github,
        owner: "owner",
        repo: "repo",
        issueNumber: 101,
        stateFile: payload.stateFile,
    });
    process.stdout.write(JSON.stringify({ result, calls, knownLabels: [...knownLabels] }));
})().catch((error) => {
    process.stderr.write(String(error && error.stack || error));
    process.exit(1);
});
""".strip()

    try:
        result = run_js_async_script(
            PROJECT_MANAGER_HELPER,
            script,
            {"stateFile": str(state_path), "knownLabels": ["docker"]},
        )
    finally:
        state_path.unlink(missing_ok=True)

    assert result["result"]["labelsToAdd"] == ["docker", "workflow"]
    assert result["calls"]["createLabel"] == ["workflow"]
    assert result["calls"]["addLabels"] == [
        ["docker"],
        ["workflow"],
        ["workflow"],
    ]
    assert ["docker", "workflow"] not in result["calls"]["addLabels"]
    assert "workflow" in result["knownLabels"]


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
    helper = read_prompts_helper()

    assert "buildIssueSummaryArtifacts" in workflow
    assert "autobot_prompts.js" in workflow
    assert "buildFinalPrSummaryPrompt" not in helper
    assert "parseFinalPrSummary" not in helper
    assert "buildLabelPrompt" not in helper


def test_prompt_helper_exports_only_deterministic_issue_summary_builder() -> (
    None
):
    helper = read_prompts_helper()

    assert "module.exports = {" in helper
    assert "buildIssueSummaryArtifacts" in helper
    assert "parseFinalPrSummary" not in helper


def test_pr_comment_body_stays_deterministic_and_has_no_cooldown_notice() -> (
    None
):
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "buildPrCommentBody",
        {
            "summaryForComment": "## Autobot Summary\n\n### What Changed\nDeterministic fallback.",
            "pullRequest": {
                "number": 12,
                "head": {"ref": "feature/refine-autobot"},
                "base": {"ref": "main"},
            },
            "nextAutobotLabels": ["workflow"],
        },
    )

    assert "GitHub Models returned temporary throttling" not in result
    assert "Deterministic fallback." in result


def test_parse_deterministic_semver_accepts_json_payload() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "parseDeterministicSemver",
        json.dumps(
            {
                "decision": "major",
                "hardRule": True,
                "hardSignals": ["removed-public-export"],
                "majorScore": 1,
                "minorScore": 0.2,
                "patchScore": 0,
            }
        ),
    )

    assert result == {
        "decision": "major",
        "hardRule": True,
        "hardSignals": ["removed-public-export"],
        "majorScore": 1,
        "minorScore": 0.2,
        "patchScore": 0,
    }


def test_resolve_required_bump_prefers_metadata_semver_over_labels() -> None:
    script = """
const fs = require("fs");
const helper = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const commentsByIssue = payload.commentsByIssue || {};
const github = {
    paginate: async (_method, params) => commentsByIssue[String(params.issue_number)] || [],
    rest: { issues: { listComments: async () => ({ data: [] }) } }
};

(async () => {
    const result = await helper.resolveRequiredBump({
        github,
        owner: "owner",
        repo: "repo",
        releaseItems: payload.releaseItems,
        currentLabelNames: payload.currentLabelNames,
        currentIssueNumber: payload.currentIssueNumber,
        currentSemverDecision: payload.currentSemverDecision,
        metadataCache: new Map(),
    });
    process.stdout.write(JSON.stringify(result));
})().catch((error) => {
    process.stderr.write(String(error && error.stack || error));
    process.exit(1);
});
""".strip()

    result = run_js_async_script(
        PROJECT_MANAGER_HELPER,
        script,
        {
            "currentLabelNames": ["documentation"],
            "currentIssueNumber": 10,
            "currentSemverDecision": "patch",
            "releaseItems": [
                {
                    "number": 11,
                    "labels": [{"name": "documentation"}],
                }
            ],
            "commentsByIssue": {
                "11": [
                    {
                        "user": {"type": "Bot"},
                        "body": '<!-- autobot-ai-summary -->\n<!-- autobot-metadata:{"semverDecision":"major"} -->\n# Autobot',
                    }
                ]
            },
        },
    )

    assert result == "major"


def test_sync_prepared_project_state_persists_semver_metadata() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(
            {
                "action": "opened",
                "isPR": True,
                "labelsToRemove": [],
                "labelsToAdd": [],
                "commentBody": "# Autobot — Changes Analysis",
                "nextAutobotLabels": ["breaking-change"],
                "semverDecision": "major",
                "deterministicSemver": {
                    "decision": "major",
                    "hardRule": True,
                    "hardSignals": ["removed-public-export"],
                    "majorScore": 1,
                    "minorScore": 0,
                    "patchScore": 0,
                },
            },
            handle,
        )
        state_path = Path(handle.name)

    script = """
const fs = require("fs");
const helper = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let createdCommentBody = "";
const github = {
    rest: {
        issues: {
            listComments: async () => ({ data: [] }),
            createComment: async ({ body }) => {
                createdCommentBody = body;
                return { data: {} };
            },
            updateComment: async () => ({ data: {} }),
            addLabels: async () => ({ data: [] }),
            getLabel: async () => ({ data: { name: "breaking-change" } }),
            createLabel: async () => ({ data: { name: "breaking-change" } }),
            removeLabel: async () => ({ data: {} }),
        }
    },
    paginate: async () => []
};

(async () => {
    await helper.syncPreparedProjectState({
        github,
        owner: "owner",
        repo: "repo",
        issueNumber: 12,
        stateFile: payload.stateFile,
    });
    process.stdout.write(JSON.stringify(createdCommentBody));
})().catch((error) => {
    process.stderr.write(String(error && error.stack || error));
    process.exit(1);
});
""".strip()

    try:
        result = run_js_async_script(
            PROJECT_MANAGER_HELPER,
            script,
            {"stateFile": str(state_path)},
        )
    finally:
        state_path.unlink(missing_ok=True)

    assert result.startswith("<!-- autobot-summary -->")
    assert '"semverDecision":"major"' in result
    assert '"deterministicSemver":{"decision":"major"' in result


def test_pr_label_delta_replaces_stale_labels_on_synchronize() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "resolvePrLabelDelta",
        {
            "action": "synchronize",
            "currentPrLabels": [
                "breaking-change",
                "security",
                "database",
                "schema",
                "compatibility",
            ],
            "previousBotLabels": [
                "breaking-change",
                "security",
                "api",
                "database",
                "schema",
                "compatibility",
            ],
            "autobotLabelsRaw": json.dumps(
                ["api", "ui", "workflow", "automation", "github"]
            ),
        },
    )

    assert set(result["nextAutobotLabels"]) == {
        "api",
        "ui",
        "workflow",
        "automation",
        "github",
    }
    assert result["labelsToRemove"] == [
        "breaking-change",
        "security",
        "database",
        "schema",
        "compatibility",
    ]
    assert set(result["labelsToAdd"]) == {
        "api",
        "ui",
        "workflow",
        "automation",
        "github",
    }


def test_pr_label_delta_clears_previous_labels_when_fresh_result_is_empty() -> (
    None
):
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "resolvePrLabelDelta",
        {
            "action": "synchronize",
            "currentPrLabels": ["breaking-change", "security"],
            "previousBotLabels": ["breaking-change", "security", "api"],
            "autobotLabelsRaw": "[]",
        },
    )

    assert result["hasFreshPrLabelResult"] is True
    assert result["nextAutobotLabels"] == []
    assert result["labelsToRemove"] == ["breaking-change", "security"]
    assert result["labelsToAdd"] == []


def test_pr_label_delta_readds_label_missing_from_live_pr_state() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "resolvePrLabelDelta",
        {
            "action": "synchronize",
            "currentPrLabels": [
                "ui",
                "automation",
                "test",
                "workflow",
                "github",
            ],
            "previousBotLabels": [
                "api",
                "ui",
                "automation",
                "test",
                "workflow",
                "github",
            ],
            "autobotLabelsRaw": json.dumps(
                ["api", "ui", "automation", "test", "workflow", "github"]
            ),
        },
    )

    assert set(result["nextAutobotLabels"]) == {
        "api",
        "ui",
        "automation",
        "test",
        "workflow",
        "github",
    }
    assert result["labelsToRemove"] == []
    assert result["labelsToAdd"] == ["api"]


def test_issue_fallback_labels_keep_feature_requests_conservative() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "inferIssueLabels",
        {
            "title": "Feature request: add guided setup wizard",
            "body": "## Proposing an Improvement or Enhancement\n\nCurrent Situation and Problem/Opportunity\nManual setup is slow.\n\nProposed Improvement/Enhancement\nAdd a guided setup wizard.",
        },
    )

    assert result == ["enhancement"]


def test_issue_fallback_labels_keep_explicit_proposals_specific() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "inferIssueLabels",
        {
            "title": "Proposal: RFC for plugin API roadmap",
            "body": "This proposal outlines a future design direction and requests feedback before implementation.",
        },
    )

    assert result == ["enhancement", "proposal"]


def test_issue_intake_exposes_scorer_backed_evidence_and_semver() -> None:
    result = run_js_helper_function(
        ISSUE_INTAKE_HELPER,
        "analyzeIssueIntake",
        {
            "title": "Proposal: RFC for plugin API roadmap",
            "body": "This proposal outlines a future API roadmap and requests feedback before implementation.",
        },
    )

    evidence_rules = {item["ruleId"] for item in result["evidenceItems"]}

    assert {
        "issue-enhancement-request",
        "issue-proposal-request",
        "issue-api-context",
    }.issubset(evidence_rules)
    assert "proposal" in result["deterministicLabels"]
    assert result["deterministicSemver"]["minorScore"] > 0


def test_issue_fallback_labels_keep_documentation_issues_narrow() -> None:
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "inferIssueLabels",
        {
            "title": "Documentation issue report",
            "body": "Link(s) to the Affected Documentation\nREADME.md\n\nDetailed Description of the Problem\nThe install instructions are outdated.",
        },
    )

    assert result == ["documentation"]


def test_issue_fallback_labels_surface_runtime_context_without_primary_family() -> (
    None
):
    result = run_js_helper_function(
        PROJECT_MANAGER_HELPER,
        "inferIssueLabels",
        {
            "title": "Windows CUDA runtime support matrix",
            "body": "Need clarity on Python 3.12, Windows, and CUDA compatibility for local installs.",
        },
    )

    assert result == ["runtime"]


def test_pr_analysis_detects_ui_without_version_false_positives() -> None:
    result = run_pr_analysis(
        build_snapshot(
            title="Polish chat UI layout",
            files=[
                build_snapshot_file(
                    "src/definers/presentation/apps/chat_app.py",
                    """
+with gr.Blocks() as app:
+    send_button = gr.Button(\"Send\")
+    request_handler = build_request_handler()
+    response_body_preview = \"ok\"
+    db_snapshot = {}
+    schema_name = \"chat_layout\"
+    permission_note = \"visible\"
+    breaking_change_banner = False
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")

    assert "ui" in deterministic_labels
    assert "ui" in candidate_labels
    assert "api" not in deterministic_labels
    assert "database" not in deterministic_labels
    assert "schema" not in deterministic_labels
    assert "security" not in deterministic_labels
    assert "breaking-change" not in deterministic_labels


def test_pr_analysis_detects_workflow_outside_github_workflows() -> None:
    result = run_pr_analysis(
        build_snapshot(
            title="Tighten autobot release pipeline",
            files=[
                build_snapshot_file(
                    "scripts/autobot_release_pipeline.py",
                    """
+WORKFLOW_DISPATCH = True
+DEFAULT_CRON = \"0 5 * * *\"
+TRIAGE_LABELS = [\"bug\", \"security\"]
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )

    assert "workflow" in deterministic_labels
    assert "automation" in deterministic_labels
    assert "ci" in deterministic_labels


def test_pr_analysis_detects_direct_security_and_api_evidence() -> None:
    result = run_pr_analysis(
        build_snapshot(
            title="Harden GitHub models API auth",
            files=[
                build_snapshot_file(
                    ".github/scripts/github_models_api.js",
                    """
+const AI_ENDPOINT = \"https://models.github.ai/inference/chat/completions\";
+const headers = { Authorization: `Bearer ${token}` };
+const permissions = { contents: \"read\", issues: \"write\" };
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )

    assert "api" in deterministic_labels
    assert "security" in deterministic_labels
    assert "github" in deterministic_labels


def test_pr_analysis_does_not_infer_enhancement_from_generic_action_title() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Implement safer logging around training flows",
            files=[
                build_snapshot_file(
                    "src/definers/application_ml/answer_service.py",
                    """
+logger.info("Collected training flow telemetry")
+return build_answer_result(payload)
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")

    assert "enhancement" not in deterministic_labels
    assert "enhancement" not in candidate_labels
    assert "bug" not in deterministic_labels
    assert "bug" not in candidate_labels


def test_pr_analysis_ignores_generic_request_response_text_outside_api_surfaces() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Refine chat context assembly",
            files=[
                build_snapshot_file(
                    "src/definers/application_chat/request_context_assembler.py",
                    """
+request_body = normalize_payload(raw_message)
+response_body = build_context_summary(request_body)
+return response_body
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")

    assert "api" not in deterministic_labels
    assert "api" not in candidate_labels


def test_pr_analysis_ignores_runtime_words_in_documentation_only_changes() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Clarify Linux and Windows setup docs",
            files=[
                build_snapshot_file(
                    "README.md",
                    """
+The runtime examples below were validated on Windows, Linux, and macOS.
+Python 3.11 remains the recommended baseline for local setup.
""".strip(),
                )
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")

    assert deterministic_labels == {"documentation"}
    assert "runtime" not in candidate_labels


def test_pr_analysis_detects_structural_package_reorg_as_breaking_change() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Reorganize package layout and update docs",
            files=[
                build_snapshot_file(
                    "src/definers/__init__.py",
                    """
-from .presentation import launch_chat
+from .ui import launch_chat
""".strip(),
                    status="renamed",
                    additions=24,
                    deletions=24,
                ),
                build_snapshot_file(
                    "src/definers/presentation/__init__.py",
                    """
-from .apps.chat_app import build_chat_app
""".strip(),
                    status="removed",
                    additions=0,
                    deletions=42,
                ),
                build_snapshot_file(
                    "src/definers/application_data.py",
                    """
-LEGACY_LAYOUT = True
""".strip(),
                    status="removed",
                    additions=0,
                    deletions=76,
                ),
                build_snapshot_file(
                    "src/definers/ui/apps/chat_app.py",
                    """
+def build_chat_app():
+    return object()
""".strip(),
                    status="added",
                    additions=120,
                    deletions=0,
                ),
                build_snapshot_file(
                    "README.md",
                    """
+Updated package layout notes.
""".strip(),
                    status="modified",
                    additions=18,
                    deletions=2,
                ),
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")

    assert "breaking-change" in deterministic_labels
    assert "enhancement" in deterministic_labels
    assert "breaking-change" in candidate_labels
    assert "enhancement" in candidate_labels
    assert "documentation" in candidate_labels


def test_pr_analysis_deterministic_fallback_preserves_more_than_six_labels() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Reorganize package layout and tighten release automation",
            files=[
                build_snapshot_file(
                    ".github/workflows/autobot.yml",
                    """
+on:
+  pull_request:
+    types: [opened, synchronize]
+jobs:
+  autobot:
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v6
""".strip(),
                    status="modified",
                    additions=44,
                    deletions=10,
                ),
                build_snapshot_file(
                    "docker/audio/Dockerfile",
                    """
+FROM python:3.14-slim
+RUN apt-get update && apt-get install -y ffmpeg
""".strip(),
                    status="modified",
                    additions=16,
                    deletions=3,
                ),
                build_snapshot_file(
                    "README.md",
                    """
+Document the new package layout.
""".strip(),
                    status="modified",
                    additions=22,
                    deletions=1,
                ),
                build_snapshot_file(
                    "tests/test_public_module_splits.py",
                    """
+def test_public_imports_remain_stable():
+    assert True
""".strip(),
                    status="modified",
                    additions=20,
                    deletions=0,
                ),
                build_snapshot_file(
                    "src/definers/__init__.py",
                    """
-from .presentation import launch_chat
+from .ui import launch_chat
""".strip(),
                    status="renamed",
                    additions=18,
                    deletions=18,
                ),
                build_snapshot_file(
                    "src/definers/presentation/__init__.py",
                    """
-from .apps.chat_app import build_chat_app
""".strip(),
                    status="removed",
                    additions=0,
                    deletions=38,
                ),
                build_snapshot_file(
                    "src/definers/application_data.py",
                    """
-LEGACY_LAYOUT = True
""".strip(),
                    status="removed",
                    additions=0,
                    deletions=72,
                ),
                build_snapshot_file(
                    "src/definers/ui/apps/chat_app.py",
                    """
+def build_chat_app():
+    return object()
""".strip(),
                    status="added",
                    additions=108,
                    deletions=0,
                ),
            ],
        )
    )

    deterministic_labels = json.loads(str(result["deterministic_labels_json"]))

    assert len(deterministic_labels) >= 7
    assert {
        "breaking-change",
        "enhancement",
        "runtime",
        "documentation",
        "test",
        "workflow",
        "docker",
    }.issubset(set(deterministic_labels))


def test_pr_analysis_emits_scored_major_outputs_for_public_reorg() -> None:
    result = run_pr_analysis(
        build_snapshot(
            title="Reorganize package layout and update docs",
            files=[
                build_snapshot_file(
                    "src/definers/__init__.py",
                    """
-from .presentation import launch_chat
+from .ui import launch_chat
""".strip(),
                    status="renamed",
                    additions=24,
                    deletions=24,
                ),
                build_snapshot_file(
                    "src/definers/presentation/__init__.py",
                    """
-from .apps.chat_app import build_chat_app
""".strip(),
                    status="removed",
                    additions=0,
                    deletions=42,
                ),
                build_snapshot_file(
                    "src/definers/ui/apps/chat_app.py",
                    """
+def build_chat_app():
+    return object()
""".strip(),
                    status="added",
                    additions=120,
                    deletions=0,
                ),
            ],
        )
    )

    evidence_items = json.loads(str(result["deterministic_evidence_json"]))
    impact_scores = json.loads(str(result["deterministic_impact_scores_json"]))
    semver = json.loads(str(result["deterministic_semver_json"]))
    primary_labels = json.loads(
        str(result["deterministic_primary_labels_json"])
    )
    evidence_rules = {item["ruleId"] for item in evidence_items}

    assert semver["decision"] == "major"
    assert "removed-public-export" in evidence_rules
    assert "renamed-public-module" in evidence_rules
    assert "added-public-capability" in evidence_rules
    assert impact_scores["major-public-contract"]["score"] >= 0.72
    assert "breaking-change" in primary_labels


def test_pr_analysis_emits_patch_scored_outputs_for_docs_and_tests_only() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Refresh docs and regression coverage",
            files=[
                build_snapshot_file(
                    "README.md",
                    """
+Document the refreshed local setup flow.
""".strip(),
                    additions=14,
                ),
                build_snapshot_file(
                    "tests/test_setup_docs.py",
                    """
+def test_docs_example_stays_current():
+    assert True
""".strip(),
                    additions=22,
                ),
            ],
        )
    )

    evidence_items = json.loads(str(result["deterministic_evidence_json"]))
    category_scores = json.loads(
        str(result["deterministic_category_scores_json"])
    )
    semver = json.loads(str(result["deterministic_semver_json"]))
    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    evidence_rules = {item["ruleId"] for item in evidence_items}

    assert semver["decision"] == "patch"
    assert "documentation-surface" in evidence_rules
    assert "tests-added" in evidence_rules
    assert category_scores["documentation-examples"]["score"] > 0
    assert category_scores["test-surface"]["score"] > 0
    assert deterministic_labels == {"documentation", "test"}


def test_pr_analysis_ignores_version_critical_label_words_inside_autobot_infrastructure() -> (
    None
):
    result = run_pr_analysis(
        build_snapshot(
            title="Refine autobot fallback labeling",
            files=[
                build_snapshot_file(
                    ".github/scripts/autobot_prompts.js",
                    """
+const supportedSignals = ["breaking-change", "compatibility", "migration", "api", "database", "schema", "runtime", "security"];
+const reminder = "Preserve explicit breaking-change, compatibility, migration, API, database, schema, runtime, security, UI, workflow, tooling, test, and documentation signals when they are supported.";
""".strip(),
                ),
                build_snapshot_file(
                    ".github/workflows/autobot.yml",
                    """
+permissions:
+  contents: write
+  pull-requests: write
+  issues: write
+  models: read
""".strip(),
                ),
                build_snapshot_file(
                    "tests/test_autobot_workflow.py",
                    """
+assert "breaking-change" not in deterministic_labels
+assert "database" not in deterministic_labels
+assert "schema" not in deterministic_labels
""".strip(),
                ),
            ],
        )
    )

    deterministic_labels = parse_result_labels(
        result, "deterministic_labels_json"
    )
    candidate_labels = parse_result_labels(result, "candidate_labels_json")
    forbidden_labels = {
        "breaking-change",
        "security",
        "api",
        "database",
        "schema",
        "compatibility",
        "migration",
        "runtime",
        "performance",
        "feature-flag",
    }

    assert forbidden_labels.isdisjoint(deterministic_labels)
    assert forbidden_labels.isdisjoint(candidate_labels)
    assert "github" in deterministic_labels
    assert "workflow" in deterministic_labels
    assert "test" in candidate_labels

const fs = require("fs");

const DEFAULT_COOLDOWN_HOURS = 48;
const COOLDOWN_LABEL_NAME = "autobot-ai-cooldown";
const COOLDOWN_LABEL_COLOR = "8b949e";
const AI_API_VERSION = "2026-03-10";
const AI_ENDPOINT = "https://models.github.ai/inference/chat/completions";

function safeIsoTimestamp(value) {
  const text = String(value || "").trim();
  const timestamp = Date.parse(text);
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : "";
}

function parseCooldownDescription(description) {
  const text = String(description || "").trim();
  const untilMatch = text.match(/(?:^|\s)until=([^\s]+)/i);
  const setMatch = text.match(/(?:^|\s)set=([^\s]+)/i);
  const sourceMatch = text.match(/(?:^|\s)source=([^\s]+)/i);
  const statusMatch = text.match(/(?:^|\s)status=(\d+)/i);
  return {
    until: safeIsoTimestamp(untilMatch ? untilMatch[1] : ""),
    setAt: safeIsoTimestamp(setMatch ? setMatch[1] : ""),
    source: sourceMatch ? sourceMatch[1] : "",
    status: statusMatch ? Number(statusMatch[1]) : 0
  };
}

function formatCooldownDescription({ until, setAt, source, status }) {
  const safeUntil = safeIsoTimestamp(until);
  const safeSetAt = safeIsoTimestamp(setAt) || new Date().toISOString();
  const safeSource = String(source || "github-models").trim() || "github-models";
  const safeStatus = Number.isFinite(Number(status)) && Number(status) > 0 ? Number(status) : 429;
  return `until=${safeUntil} set=${safeSetAt} source=${safeSource} status=${safeStatus}`;
}

function formatCooldownMessage(until) {
  const safeUntil = safeIsoTimestamp(until);
  return safeUntil
    ? `AI inference is paused until ${safeUntil} after a recent 429 rate limit response. Try again after that time.`
    : "AI inference is temporarily paused after a recent 429 rate limit response.";
}

async function getCooldownLabel({ github, owner, repo }) {
  try {
    const response = await github.rest.issues.getLabel({ owner, repo, name: COOLDOWN_LABEL_NAME });
    return response.data;
  } catch (error) {
    if (error.status === 404) {
      return null;
    }
    throw error;
  }
}

async function upsertCooldownLabel({ github, owner, repo, until, setAt, source, status }) {
  const description = formatCooldownDescription({ until, setAt, source, status });
  const existing = await getCooldownLabel({ github, owner, repo });
  if (existing) {
    await github.rest.issues.updateLabel({
      owner,
      repo,
      name: COOLDOWN_LABEL_NAME,
      new_name: COOLDOWN_LABEL_NAME,
      color: existing.color || COOLDOWN_LABEL_COLOR,
      description
    });
    return;
  }
  await github.rest.issues.createLabel({
    owner,
    repo,
    name: COOLDOWN_LABEL_NAME,
    color: COOLDOWN_LABEL_COLOR,
    description
  });
}

async function clearCooldownLabel({ github, owner, repo }) {
  const existing = await getCooldownLabel({ github, owner, repo });
  if (!existing || !String(existing.description || "").trim()) {
    return;
  }
  await github.rest.issues.updateLabel({
    owner,
    repo,
    name: COOLDOWN_LABEL_NAME,
    new_name: COOLDOWN_LABEL_NAME,
    color: existing.color || COOLDOWN_LABEL_COLOR,
    description: ""
  });
}

async function getCooldownState({ github, owner, repo, clearExpired = true }) {
  const label = await getCooldownLabel({ github, owner, repo });
  if (!label) {
    return { active: false, until: "", setAt: "", source: "", status: 0, message: "" };
  }
  const parsed = parseCooldownDescription(label.description || "");
  if (!parsed.until) {
    return { active: false, until: "", setAt: parsed.setAt, source: parsed.source, status: parsed.status, message: "" };
  }
  if (Date.parse(parsed.until) <= Date.now()) {
    if (clearExpired) {
      await clearCooldownLabel({ github, owner, repo });
    }
    return { active: false, until: parsed.until, setAt: parsed.setAt, source: parsed.source, status: parsed.status, message: "" };
  }
  return {
    active: true,
    until: parsed.until,
    setAt: parsed.setAt,
    source: parsed.source,
    status: parsed.status,
    message: formatCooldownMessage(parsed.until)
  };
}

function normalizeMessageContent(content) {
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object") {
          return String(item.text || item.content || "");
        }
        return "";
      })
      .join("")
      .trim();
  }
  return String(content || "").trim();
}

function extractResponseContent(rawBody) {
  const rawText = String(rawBody || "");
  try {
    const parsed = JSON.parse(rawText);
    return normalizeMessageContent(parsed?.choices?.[0]?.message?.content);
  } catch (error) {
    return rawText.trim();
  }
}

async function runInferenceRequest({ token, promptFile, model, maxTokens, systemPrompt, temperature = 0, topP = 1, seed = 1 }) {
  const prompt = fs.readFileSync(promptFile, "utf8");
  const messages = [];
  if (String(systemPrompt || "").trim()) {
    messages.push({ role: "system", content: String(systemPrompt).trim() });
  }
  messages.push({ role: "user", content: prompt });
  const requestBody = {
    model,
    messages,
    max_tokens: maxTokens,
    temperature,
    top_p: topP,
    seed,
    stream: false
  };
  const response = await fetch(AI_ENDPOINT, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": AI_API_VERSION
    },
    body: JSON.stringify(requestBody)
  });
  const rawBody = await response.text();
  return {
    ok: response.ok,
    status: response.status,
    raw: rawBody,
    content: extractResponseContent(rawBody)
  };
}

async function runManagedInference({ github, owner, repo, token, promptFile, model, maxTokens, systemPrompt, source, cooldownHours = DEFAULT_COOLDOWN_HOURS }) {
  try {
    const cooldown = await getCooldownState({ github, owner, repo, clearExpired: true });
    if (cooldown.active) {
      return {
        ok: false,
        skipped: true,
        rateLimited: true,
        status: 429,
        raw: "",
        content: "",
        cooldownUntil: cooldown.until,
        cooldownMessage: cooldown.message,
        errorMessage: cooldown.message
      };
    }

    const response = await runInferenceRequest({ token, promptFile, model, maxTokens, systemPrompt });
    if (response.status === 429) {
      const until = new Date(Date.now() + Number(cooldownHours) * 60 * 60 * 1000).toISOString();
      await upsertCooldownLabel({
        github,
        owner,
        repo,
        until,
        setAt: new Date().toISOString(),
        source: String(source || "github-models"),
        status: 429
      });
      return {
        ...response,
        skipped: false,
        rateLimited: true,
        cooldownUntil: until,
        cooldownMessage: formatCooldownMessage(until),
        errorMessage: formatCooldownMessage(until)
      };
    }

    return {
      ...response,
      skipped: false,
      rateLimited: false,
      cooldownUntil: "",
      cooldownMessage: "",
      errorMessage: response.ok ? "" : `AI inference failed with status ${response.status}.`
    };
  } catch (error) {
    return {
      ok: false,
      skipped: false,
      rateLimited: false,
      status: 0,
      raw: "",
      content: "",
      cooldownUntil: "",
      cooldownMessage: "",
      errorMessage: error instanceof Error ? error.message : String(error)
    };
  }
}

module.exports = {
  AI_ENDPOINT,
  AI_API_VERSION,
  COOLDOWN_LABEL_NAME,
  DEFAULT_COOLDOWN_HOURS,
  formatCooldownMessage,
  getCooldownState,
  runManagedInference
};
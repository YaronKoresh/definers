const fs = require("fs");

function asString(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatBulletLines(items, fallback) {
  return items.length > 0 ? items.map((item) => `- ${item}`).join("\n") : `- ${fallback}`;
}

function humanizeIdentifier(value) {
  return String(value || "").trim().replace(/[-_]+/g, " ");
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(filePath, content) {
  fs.writeFileSync(filePath, content, "utf8");
}

module.exports = {
  asString,
  escapeRegExp,
  formatBulletLines,
  humanizeIdentifier,
  readJson,
  writeJson,
  writeText
};
function resolveVitestApi() {
  try {
    const v = require("vitest");
    if (v && (v.test || v.it)) return v;
  } catch {
    if (typeof globalThis.test === "function") {
      return {
        test: globalThis.test,
        describe: globalThis.describe,
        beforeEach: globalThis.beforeEach,
        afterEach: globalThis.afterEach,
        beforeAll: globalThis.beforeAll || globalThis.before,
        afterAll: globalThis.afterAll || globalThis.after,
      };
    }
  }
  return null;
}

const vitestApi = resolveVitestApi();
const nodeTestApi = vitestApi ? null : require("node:test");

function createSeededRandom(seed) {
  let state = Number(seed) >>> 0;
  return function nextRandom() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function chance(random, probability) {
  return random() < Math.max(0, Math.min(1, Number(probability || 0)));
}

function randomInt(random, min, max) {
  return Math.floor(random() * (max - min + 1)) + min;
}

function pick(random, values) {
  return values[randomInt(random, 0, values.length - 1)];
}

function pickWeighted(random, values) {
  if (!Array.isArray(values) || values.length === 0) {
    return undefined;
  }
  const weightedValues = values.map((entry) => {
    if (entry && typeof entry === "object" && Object.prototype.hasOwnProperty.call(entry, "weight")) {
      return {
        value: entry,
        weight: Math.max(0, Number(entry.weight || 0))
      };
    }
    return {
      value: entry,
      weight: 1
    };
  });
  const totalWeight = weightedValues.reduce((sum, entry) => sum + entry.weight, 0);
  if (totalWeight <= 0) {
    return weightedValues[weightedValues.length - 1].value;
  }
  let remaining = random() * totalWeight;
  for (const entry of weightedValues) {
    remaining -= entry.weight;
    if (remaining <= 0) {
      return entry.value;
    }
  }
  return weightedValues[weightedValues.length - 1].value;
}

function shuffle(random, values) {
  const cloned = values.slice();
  for (let index = cloned.length - 1; index > 0; index -= 1) {
    const swapIndex = randomInt(random, 0, index);
    const next = cloned[index];
    cloned[index] = cloned[swapIndex];
    cloned[swapIndex] = next;
  }
  return cloned;
}

function randomWords(random, count) {
  const dictionary = [
    "adapter",
    "advisory",
    "autobot",
    "balancer",
    "bridge",
    "bundle",
    "cache",
    "compat",
    "contract",
    "definer",
    "dependency",
    "engine",
    "guard",
    "handoff",
    "integrity",
    "link",
    "mermaid",
    "milestone",
    "orchestrator",
    "package",
    "patch",
    "policy",
    "prompt",
    "release",
    "runtime",
    "security",
    "signal",
    "summary",
    "surface",
    "triage",
    "validator",
    "workflow"
  ];
  const words = [];
  while (words.length < count) {
    const word = pick(random, dictionary);
    if (!words.includes(word)) {
      words.push(word);
    }
  }
  return words;
}

module.exports = {
  after: vitestApi?.after || nodeTestApi?.after,
  afterEach: vitestApi?.afterEach || nodeTestApi?.afterEach,
  before: vitestApi?.before || nodeTestApi?.before,
  beforeEach: vitestApi?.beforeEach || nodeTestApi?.beforeEach,
  chance,
  createSeededRandom,
  describe: vitestApi?.describe || nodeTestApi?.describe || ((name, fn) => nodeTestApi.test(name, fn)),
  isVitestRuntime: Boolean(vitestApi),
  pick,
  pickWeighted,
  randomInt,
  randomWords,
  shuffle,
  test: vitestApi?.test || nodeTestApi?.test,
};
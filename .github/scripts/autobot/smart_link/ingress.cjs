const { analyzeCandidatePair, buildResultFingerprint, selectSmartLinkResults } = require("./core.cjs");
const { clampThreshold, collectSmartLinkSource: collectGatheredSmartLinkSource, readStoredSource, retrieveCandidateSnapshots } = require("../gather_info.cjs");
const { writeJson } = require("../utils.cjs");

async function collectSmartLinkSource(input) {
  return collectGatheredSmartLinkSource(input);
}

async function analyzeSmartLinkSource({ analysisFile, core, github, owner, repo, sourceFile }) {
  const stored = readStoredSource(sourceFile);
  const source = stored.source;
  const threshold = clampThreshold(stored.threshold);
  const candidates = await retrieveCandidateSnapshots({ github, owner, repo, source });
  const candidateResults = candidates
    .filter((candidate) => candidate.number !== source.number)
    .map((candidate) => analyzeCandidatePair({ candidate, source, threshold }));
  const emittedResults = selectSmartLinkResults({ candidateResults, threshold });
  const analysis = {
    candidateResults,
    emittedResults,
    fingerprint: buildResultFingerprint(source, emittedResults),
    source,
    stats: {
      candidateCount: candidateResults.length,
      emittedCount: emittedResults.length,
      suppressedCount: candidateResults.filter((result) => result.suppressionReasons.length > 0 || result.emittedScore < threshold).length
    },
    threshold
  };
  writeJson(analysisFile, analysis);
  const outputs = {
    emitted_count: String(analysis.stats.emittedCount),
    ready: "true",
    total_candidates: String(analysis.stats.candidateCount)
  };
  for (const [name, value] of Object.entries(outputs)) {
    core.setOutput(name, value);
  }
  return outputs;
}

module.exports = {
  analyzeSmartLinkSource,
  collectSmartLinkSource
};
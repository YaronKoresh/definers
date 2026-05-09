const { evaluateScenarioLane } = require("../tests/deterministic_scenario_lanes.cjs");

function parseSuiteKeys(rawValue) {
  return String(rawValue || "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

async function main() {
  const suiteKeys = parseSuiteKeys(process.env.SCENARIO_SUITES);
  const result = await evaluateScenarioLane({
    lane: process.env.SCENARIO_LANE || "exploratory",
    rotationKey: process.env.SCENARIO_ROTATION_KEY,
    suiteKeys: suiteKeys.length > 0 ? suiteKeys : undefined
  });

  console.log(result.reportText);
  if (String(process.env.SCENARIO_PRINT_CAPTURE_BUNDLE || "").toLowerCase() === "true") {
    console.log("");
    console.log(JSON.stringify({
      captureCandidates: result.captureCandidates,
      lane: result.lane,
      rotationKey: result.rotationKey,
      summary: result.summary
    }, null, 2));
  }
  process.exit(result.failed ? 1 : 0);
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
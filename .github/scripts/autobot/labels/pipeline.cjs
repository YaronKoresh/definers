const { createNode } = require("./helpers.cjs");

const PIPELINE_TREE = createNode("pipeline", { color: "006b75", description: "CI and release pipeline changes.", legacyMatches: ["workflow", "ci", "automation"], secondary: true }, [
  createNode("ci", { color: "006b75", description: "Continuous integration changes.", legacyMatches: ["workflow", "ci", "automation"], secondary: true }, [
    createNode("workflow file", { color: "006b75", description: "Workflow definition changes.", legacyMatches: ["workflow", "ci", "automation"], secondary: true }, [
      createNode("github actions", { color: "006b75", description: "GitHub Actions changes.", legacyMatches: ["workflow", "ci", "automation"], secondary: true }, [
        createNode("step order", { color: "006b75", description: "Workflow step ordering changes.", secondary: true }),
        createNode("action pin", { color: "006b75", description: "Pinned action version changes.", secondary: true })
      ]),
      createNode("matrix job", { color: "006b75", description: "Matrix workflow changes.", secondary: true }, [
        createNode("python matrix", { color: "006b75", description: "Python matrix changes.", secondary: true }),
        createNode("os matrix", { color: "006b75", description: "OS matrix changes.", secondary: true })
      ]),
      createNode("security scan", { color: "e30c0c", description: "Security scan workflow changes.", legacyMatches: ["security", "workflow", "ci", "automation"], secondary: true }, [
        createNode("secret scan", { color: "e30c0c", description: "Secret scanning changes.", legacyMatches: ["security"], secondary: true }),
        createNode("codeql", { color: "e30c0c", description: "CodeQL scanning changes.", legacyMatches: ["security"], secondary: true })
      ])
    ]),
    createNode("test job", { color: "cc317c", description: "Pipeline test job changes.", legacyMatches: ["test"], secondary: true }, [
      createNode("unit test", { color: "cc317c", description: "Unit test changes.", legacyMatches: ["test"], secondary: true }, [
        createNode("test fixture", { color: "cc317c", description: "Test fixture changes.", secondary: true }),
        createNode("mock setup", { color: "cc317c", description: "Mock setup changes.", secondary: true })
      ]),
      createNode("integration test", { color: "cc317c", description: "Integration test changes.", legacyMatches: ["test"], secondary: true }, [
        createNode("scenario lane", { color: "cc317c", description: "Scenario lane changes.", secondary: true }),
        createNode("replay seed", { color: "cc317c", description: "Replay seed changes.", secondary: true })
      ])
    ])
  ]),
  createNode("release flow", { color: "1d76db", description: "Build and deployment changes.", secondary: true }, [
    createNode("build job", { color: "1d76db", description: "Build job changes.", legacyMatches: ["build", "packaging", "release"], secondary: true }, [
      createNode("wheel", { color: "1d76db", description: "Wheel build changes.", secondary: true }, [
        createNode("wheel publish", { color: "1d76db", description: "Wheel publishing changes.", secondary: true }),
        createNode("package metadata", { color: "1d76db", description: "Package metadata changes.", secondary: true })
      ]),
      createNode("image", { color: "1d76db", description: "Container image changes.", legacyMatches: ["docker"], secondary: true }, [
        createNode("image tag", { color: "1d76db", description: "Image tag changes.", secondary: true }),
        createNode("layer cache", { color: "1d76db", description: "Image layer cache changes.", secondary: true })
      ])
    ]),
    createNode("deploy job", { color: "1d76db", description: "Deployment job changes.", secondary: true }, [
      createNode("staging deploy", { color: "1d76db", description: "Staging deployment changes.", secondary: true }, [
        createNode("rollout gate", { color: "1d76db", description: "Rollout gate changes.", secondary: true }),
        createNode("smoke check", { color: "1d76db", description: "Smoke check changes.", secondary: true })
      ]),
      createNode("prod deploy", { color: "1d76db", description: "Production deployment changes.", secondary: true }, [
        createNode("release gate", { color: "1d76db", description: "Release gate changes.", secondary: true }),
        createNode("rollback hook", { color: "1d76db", description: "Rollback hook changes.", secondary: true })
      ])
    ])
  ])
]);

module.exports = {
  PIPELINE_TREE
};
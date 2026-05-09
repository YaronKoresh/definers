const { createNode } = require("./helpers.cjs");

const RUNTIME_TREE = createNode("runtime", { color: "7057ff", description: "Runtime and execution changes.", legacyMatches: ["runtime"], releaseRelevant: true }, [
  createNode("platform", { color: "7057ff", description: "Platform support changes.", legacyMatches: ["runtime"], releaseRelevant: true }, [
    createNode("python", { color: "7057ff", description: "Python runtime changes.", releaseRelevant: true }, [
      createNode("python version", { color: "7057ff", description: "Python version support changes.", releaseRelevant: true }, [
        createNode("support matrix", { color: "7057ff", description: "Supported version matrix changes.", releaseRelevant: true }),
        createNode("version pin", { color: "7057ff", description: "Pinned runtime version changes.", legacyMatches: ["compatibility"], releaseRelevant: true })
      ]),
      createNode("env var", { color: "7057ff", description: "Environment variable handling changes.", secondary: true }, [
        createNode("env loading", { color: "7057ff", description: "Environment loading changes.", secondary: true }),
        createNode("env default", { color: "7057ff", description: "Environment default changes.", secondary: true })
      ])
    ]),
    createNode("os", { color: "7057ff", description: "Operating system compatibility changes.", releaseRelevant: true }, [
      createNode("windows", { color: "7057ff", description: "Windows support changes.", releaseRelevant: true }, [
        createNode("path separator", { color: "7057ff", description: "Path separator handling changes.", releaseRelevant: true }),
        createNode("shell command", { color: "7057ff", description: "Shell command compatibility changes.", releaseRelevant: true })
      ]),
      createNode("linux", { color: "7057ff", description: "Linux support changes.", releaseRelevant: true }, [
        createNode("package install", { color: "7057ff", description: "Package installation changes.", releaseRelevant: true }),
        createNode("shared lib", { color: "7057ff", description: "Shared library changes.", releaseRelevant: true }),
        createNode("systemd", { color: "7057ff", description: "systemd unit changes.", releaseRelevant: true }),
        createNode("glibc", { color: "7057ff", description: "glibc or musl compatibility changes.", releaseRelevant: true })
      ]),
      createNode("macos", { color: "7057ff", description: "macOS support changes.", releaseRelevant: true, legacyMatches: ["runtime"] })
    ]),
    createNode("container", { color: "7057ff", description: "Container runtime changes.", releaseRelevant: true }),
    createNode("shell", { color: "7057ff", description: "Shell environment changes.", releaseRelevant: true }),
    createNode("filesystem", { color: "7057ff", description: "Filesystem behavior changes.", releaseRelevant: true }),
    createNode("process", { color: "7057ff", description: "Process execution changes.", releaseRelevant: true }),
    createNode("locale", { color: "7057ff", description: "Locale handling changes.", releaseRelevant: true }),
    createNode("timezone", { color: "7057ff", description: "Timezone handling changes.", releaseRelevant: true }),
    createNode("architecture", { color: "7057ff", description: "CPU architecture compatibility changes.", releaseRelevant: true })
  ]),
  createNode("performance", { color: "5319e7", description: "Runtime performance changes.", legacyMatches: ["performance"], releaseRelevant: true }, [
    createNode("memory", { color: "5319e7", description: "Memory behavior changes.", releaseRelevant: true }, [
      createNode("heap usage", { color: "5319e7", description: "Heap usage changes.", releaseRelevant: true }, [
        createNode("peak memory", { color: "5319e7", description: "Peak memory changes.", releaseRelevant: true }),
        createNode("buffer copy", { color: "5319e7", description: "Buffer copy behavior changes.", releaseRelevant: true })
      ]),
      createNode("leak risk", { color: "5319e7", description: "Leak risk changes.", releaseRelevant: true }, [
        createNode("object retention", { color: "5319e7", description: "Object retention changes.", releaseRelevant: true }),
        createNode("cache leak", { color: "5319e7", description: "Cache leak changes.", releaseRelevant: true })
      ])
    ]),
    createNode("latency", { color: "5319e7", description: "Latency changes.", releaseRelevant: true }, [
      createNode("cold start", { color: "5319e7", description: "Cold start changes.", releaseRelevant: true }, [
        createNode("init time", { color: "5319e7", description: "Initialization time changes.", releaseRelevant: true }),
        createNode("import time", { color: "5319e7", description: "Import time changes.", releaseRelevant: true })
      ]),
      createNode("cache hit", { color: "5319e7", description: "Cache hit behavior changes.", releaseRelevant: true }, [
        createNode("cache key", { color: "5319e7", description: "Cache key changes.", releaseRelevant: true }),
        createNode("cache invalidation", { color: "5319e7", description: "Cache invalidation changes.", releaseRelevant: true })
      ])
    ])
  ])
]);

module.exports = {
  RUNTIME_TREE
};
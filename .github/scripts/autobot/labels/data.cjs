const { createNode } = require("./helpers.cjs");

const DATA_TREE = createNode("data", { color: "fbca04", description: "Data and storage changes.", releaseRelevant: true }, [
  createNode("schema", { color: "2b67c6", description: "Schema and type changes.", legacyMatches: ["schema"], releaseRelevant: true }, [
    createNode("format", { color: "2b67c6", description: "Serialization format changes.", legacyMatches: ["serialization"], releaseRelevant: true }, [
      createNode("json", { color: "2b67c6", description: "JSON format changes.", releaseRelevant: true }, [
        createNode("json schema", { color: "2b67c6", description: "JSON schema changes.", releaseRelevant: true }, [
          createNode("field rule", { color: "2b67c6", description: "Field validation rule changes.", releaseRelevant: true }),
          createNode("enum value", { color: "2b67c6", description: "Enum value changes.", releaseRelevant: true })
        ])
      ]),
      createNode("yaml", { color: "2b67c6", description: "YAML format changes.", releaseRelevant: true }),
      createNode("xml", { color: "2b67c6", description: "XML format changes.", releaseRelevant: true }),
      createNode("csv", { color: "2b67c6", description: "CSV format changes.", releaseRelevant: true }),
      createNode("tsv", { color: "2b67c6", description: "TSV format changes.", releaseRelevant: true }),
      createNode("toml", { color: "2b67c6", description: "TOML format changes.", releaseRelevant: true }),
      createNode("ini", { color: "2b67c6", description: "INI format changes.", releaseRelevant: true }),
      createNode("proto", { color: "2b67c6", description: "Proto format changes.", releaseRelevant: true }, [
        createNode("proto schema", { color: "2b67c6", description: "Proto schema changes.", releaseRelevant: true }, [
          createNode("wire format", { color: "2b67c6", description: "Wire format changes.", legacyMatches: ["compatibility"], releaseRelevant: true, versionBump: "major" }),
          createNode("field number", { color: "2b67c6", description: "Proto field number changes.", releaseRelevant: true, versionBump: "major" })
        ])
      ])
    ]),
    createNode("type", { color: "2b67c6", description: "Type system changes.", legacyMatches: ["types", "compatibility"], releaseRelevant: true }, [
      createNode("public type", { color: "2b67c6", description: "Public type changes.", releaseRelevant: true }, [
        createNode("type export", { color: "2b67c6", description: "Type export changes.", releaseRelevant: true }),
        createNode("type alias", { color: "2b67c6", description: "Type alias or shim changes.", legacyMatches: ["compatibility"], releaseRelevant: true })
      ]),
      createNode("generated type", { color: "2b67c6", description: "Generated type changes.", releaseRelevant: true }, [
        createNode("codegen output", { color: "2b67c6", description: "Code generation output changes.", secondary: true }),
        createNode("sdk type", { color: "2b67c6", description: "SDK type changes.", releaseRelevant: true })
      ])
    ])
  ]),
  createNode("storage", { color: "fbca04", description: "Persistence and query changes.", legacyMatches: ["database"], releaseRelevant: true }, [
    createNode("migration file", { color: "fbca04", description: "Migration changes.", legacyMatches: ["migration", "database"], releaseRelevant: true }, [
      createNode("destructive migration", { color: "fbca04", description: "Destructive migration changes.", releaseRelevant: true, versionBump: "major" }, [
        createNode("column drop", { color: "fbca04", description: "Dropped column changes.", releaseRelevant: true, versionBump: "major" }),
        createNode("table drop", { color: "fbca04", description: "Dropped table changes.", releaseRelevant: true, versionBump: "major" })
      ]),
      createNode("additive migration", { color: "fbca04", description: "Additive migration changes.", releaseRelevant: true, versionBump: "minor" }, [
        createNode("column add", { color: "fbca04", description: "Added column changes.", releaseRelevant: true, versionBump: "minor" }),
        createNode("table add", { color: "fbca04", description: "Added table changes.", releaseRelevant: true, versionBump: "minor" })
      ])
    ]),
    createNode("query", { color: "fbca04", description: "Query behavior changes.", legacyMatches: ["database"], releaseRelevant: true }, [
      createNode("read query", { color: "fbca04", description: "Read query changes.", releaseRelevant: true }, [
        createNode("result shape", { color: "fbca04", description: "Read result shape changes.", releaseRelevant: true }),
        createNode("index usage", { color: "fbca04", description: "Query index usage changes.", releaseRelevant: true })
      ]),
      createNode("write query", { color: "fbca04", description: "Write query changes.", releaseRelevant: true }, [
        createNode("upsert", { color: "fbca04", description: "Upsert changes.", releaseRelevant: true }),
        createNode("bulk write", { color: "fbca04", description: "Bulk write changes.", releaseRelevant: true })
      ])
    ])
  ])
]);

module.exports = {
  DATA_TREE
};
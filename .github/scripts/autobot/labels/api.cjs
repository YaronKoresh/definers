const { createNode } = require("./helpers.cjs");

const API_TREE = createNode("api", { color: "1d76db", description: "API surface changes.", releaseRelevant: true }, [
  createNode("http", { color: "1d76db", description: "HTTP API surface.", releaseRelevant: true }, [
    createNode("route", { color: "1d76db", description: "Route and handler changes.", legacyMatches: ["api"], releaseRelevant: true }, [
      createNode("rest", { color: "1d76db", description: "REST route changes.", releaseRelevant: true }, [
        createNode("route param", { color: "1d76db", description: "Route parameter contract changes.", releaseRelevant: true, versionBump: "major" }),
        createNode("query param", { color: "1d76db", description: "Query parameter contract changes.", releaseRelevant: true, versionBump: "minor" })
      ]),
      createNode("graphql", { color: "1d76db", description: "GraphQL API changes.", releaseRelevant: true }, [
        createNode("resolver", { color: "1d76db", description: "GraphQL resolver changes.", releaseRelevant: true, versionBump: "minor" }),
        createNode("schema field", { color: "1d76db", description: "GraphQL field contract changes.", releaseRelevant: true, versionBump: "major" })
      ]),
      createNode("webhook", { color: "1d76db", description: "Webhook surface changes.", releaseRelevant: true }, [
        createNode("webhook payload", { color: "1d76db", description: "Webhook payload changes.", releaseRelevant: true, versionBump: "major" }),
        createNode("webhook retry", { color: "1d76db", description: "Webhook retry behavior changes.", releaseRelevant: true })
      ])
    ]),
    createNode("payload", { color: "1d76db", description: "Request and response contract changes.", legacyMatches: ["api"], releaseRelevant: true }, [
      createNode("request body", { color: "1d76db", description: "Request body contract changes.", releaseRelevant: true, versionBump: "minor" }, [
        createNode("input schema", { color: "1d76db", description: "Input schema changes.", releaseRelevant: true, versionBump: "minor" }),
        createNode("body validation", { color: "1d76db", description: "Request validation changes.", releaseRelevant: true })
      ]),
      createNode("response body", { color: "1d76db", description: "Response payload changes.", releaseRelevant: true, versionBump: "minor" }, [
        createNode("output schema", { color: "1d76db", description: "Output schema changes.", releaseRelevant: true, versionBump: "minor" }),
        createNode("error payload", { color: "1d76db", description: "Error payload changes.", releaseRelevant: true })
      ])
    ]),
    createNode("rate limit", { color: "e30c0c", description: "Rate limiting changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("cors", { color: "e30c0c", description: "CORS policy changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("csp", { color: "e30c0c", description: "Content Security Policy changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("openapi", { color: "1d76db", description: "OpenAPI and API reference surface.", releaseRelevant: true }, [
      createNode("openapi spec", { color: "1d76db", description: "OpenAPI spec changes.", legacyMatches: ["api", "schema"], releaseRelevant: true }),
      createNode("scope doc", { color: "0075ca", description: "OAuth scope documentation changes.", secondary: true }),
      createNode("token doc", { color: "0075ca", description: "Token flow documentation changes.", secondary: true }),
      createNode("release note", { color: "0075ca", description: "API release note changes.", secondary: true })
    ])
  ]),
  createNode("auth", { color: "e30c0c", description: "Authentication and authorization changes.", legacyMatches: ["security"], releaseRelevant: true }, [
    createNode("token", { color: "e30c0c", description: "Token handling changes.", legacyMatches: ["security"], releaseRelevant: true }, [
      createNode("jwt", { color: "e30c0c", description: "JWT changes.", releaseRelevant: true }, [
        createNode("jwt expiry", { color: "e30c0c", description: "JWT expiration changes.", releaseRelevant: true }),
        createNode("jwt claim", { color: "e30c0c", description: "JWT claim changes.", releaseRelevant: true })
      ]),
      createNode("refresh token", { color: "e30c0c", description: "Refresh token changes.", releaseRelevant: true }, [
        createNode("token rotation", { color: "e30c0c", description: "Token rotation changes.", releaseRelevant: true }),
        createNode("token revocation", { color: "e30c0c", description: "Token revocation changes.", releaseRelevant: true })
      ]),
      createNode("api key", { color: "e30c0c", description: "API key handling changes.", legacyMatches: ["security"], releaseRelevant: true })
    ]),
    createNode("session", { color: "e30c0c", description: "Session handling changes.", legacyMatches: ["security"], releaseRelevant: true }, [
      createNode("cookie", { color: "e30c0c", description: "Cookie handling changes.", legacyMatches: ["security"], releaseRelevant: true })
    ]),
    createNode("mfa", { color: "e30c0c", description: "Multi-factor authentication changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("permission", { color: "e30c0c", description: "Permission model changes.", legacyMatches: ["security"], releaseRelevant: true }, [
      createNode("rbac", { color: "e30c0c", description: "Role-based access control changes.", releaseRelevant: true }, [
        createNode("role map", { color: "e30c0c", description: "Role mapping changes.", releaseRelevant: true }),
        createNode("role check", { color: "e30c0c", description: "Role enforcement changes.", releaseRelevant: true })
      ]),
      createNode("oauth scope", { color: "e30c0c", description: "OAuth scope changes.", releaseRelevant: true }, [
        createNode("scope map", { color: "e30c0c", description: "Scope mapping changes.", releaseRelevant: true }),
        createNode("scope check", { color: "e30c0c", description: "Scope enforcement changes.", releaseRelevant: true })
      ]),
      createNode("csrf", { color: "e30c0c", description: "CSRF protection changes.", legacyMatches: ["security"], releaseRelevant: true }),
      createNode("authz", { color: "e30c0c", description: "Authorization rule changes.", legacyMatches: ["security"], releaseRelevant: true })
    ]),
    createNode("tls", { color: "e30c0c", description: "TLS and certificate handling changes.", legacyMatches: ["security"], releaseRelevant: true })
  ]),
  createNode("security posture", { color: "e30c0c", description: "Security posture and assurance changes.", legacyMatches: ["security"], releaseRelevant: true }, [
    createNode("vulnerability", { color: "e30c0c", description: "Vulnerability discovery or remediation changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("compliance", { color: "e30c0c", description: "Compliance control and policy evidence changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("hardening", { color: "e30c0c", description: "Security hardening and mitigation changes.", legacyMatches: ["security"], releaseRelevant: true }),
    createNode("pen-test", { color: "e30c0c", description: "Penetration-test findings and remediation changes.", legacyMatches: ["security"], releaseRelevant: true })
  ])
]);

module.exports = {
  API_TREE
};
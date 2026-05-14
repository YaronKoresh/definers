const { createNode } = require("./helpers.cjs");

const INTERFACE_TREE = createNode("interface", { color: "d4c5f9", description: "User-facing interface changes.", legacyMatches: ["ui"], secondary: true }, [
  createNode("view", { color: "d4c5f9", description: "View and screen changes.", legacyMatches: ["ui"], secondary: true }, [
    createNode("template", { color: "d4c5f9", description: "Template or markup changes.", legacyMatches: ["ui"], secondary: true }),
    createNode("form control", { color: "d4c5f9", description: "Form control and input changes.", legacyMatches: ["ui"], secondary: true }),
    createNode("theme token", { color: "d4c5f9", description: "Theme token and styling changes.", legacyMatches: ["ui"], secondary: true })
  ]),
  createNode("navigation", { color: "c2e0c6", description: "Navigation and accessibility changes.", legacyMatches: ["ui", "accessibility"], secondary: true }, [
    createNode("keyboard nav", { color: "c2e0c6", description: "Keyboard navigation changes.", legacyMatches: ["accessibility"], secondary: true }),
    createNode("focus order", { color: "c2e0c6", description: "Focus order changes.", legacyMatches: ["accessibility"], secondary: true }),
    createNode("screen reader", { color: "c2e0c6", description: "Screen reader support changes.", legacyMatches: ["accessibility"], secondary: true })
  ]),
  createNode("localization", { color: "91d674", description: "Localization and translated copy changes.", legacyMatches: ["localization"], secondary: true }, [
    createNode("translation catalog", { color: "91d674", description: "Translation catalog changes.", legacyMatches: ["localization"], secondary: true }),
    createNode("locale string", { color: "91d674", description: "Locale string changes.", legacyMatches: ["localization"], secondary: true }),
    createNode("rtl layout", { color: "91d674", description: "Right-to-left layout changes.", legacyMatches: ["localization"], secondary: true })
  ])
]);

module.exports = {
  INTERFACE_TREE
};
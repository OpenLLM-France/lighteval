// Jest config for the WebApp1K runner.
// The generated component + its test are written into a throwaway dir under this project and
// run with an explicit test path, so testMatch stays permissive. jsdom provides the DOM that
// @testing-library/react renders into.
module.exports = {
  testEnvironment: "jsdom",
  transform: {
    "^.+\\.[jt]sx?$": "babel-jest",
  },
  testMatch: ["**/*.test.js"],
  // The .webapp1k_runs scratch dir lives inside the project; don't let watch/scan choke on it.
  modulePathIgnorePatterns: [],
  testTimeout: 30000,
};

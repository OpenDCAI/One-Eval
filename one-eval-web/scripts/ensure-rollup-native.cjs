const { execSync } = require("node:child_process");

const platformToPackage = {
  linux: {
    x64: "@rollup/rollup-linux-x64-gnu",
    arm64: "@rollup/rollup-linux-arm64-gnu",
  },
  win32: {
    x64: "@rollup/rollup-win32-x64-msvc",
    arm64: "@rollup/rollup-win32-arm64-msvc",
  },
  darwin: {
    x64: "@rollup/rollup-darwin-x64",
    arm64: "@rollup/rollup-darwin-arm64",
  },
};

const packageName = platformToPackage?.[process.platform]?.[process.arch];

if (!packageName) {
  process.exit(0);
}

try {
  require.resolve(packageName);
} catch {
  execSync(`npm install ${packageName} --no-save --no-package-lock`, { stdio: "inherit" });
}

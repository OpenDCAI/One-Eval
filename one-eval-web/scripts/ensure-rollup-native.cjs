const { execSync } = require("node:child_process");

const platformToPackage = {
  linux: "@rollup/rollup-linux-x64-gnu",
  win32: "@rollup/rollup-win32-x64-msvc",
};

const packageName = platformToPackage[process.platform];

if (!packageName) {
  process.exit(0);
}

try {
  require.resolve(packageName);
} catch {
  execSync(`npm install ${packageName} --no-save --no-package-lock`, { stdio: "inherit" });
}

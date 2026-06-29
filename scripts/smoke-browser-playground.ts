import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";

const host = "127.0.0.1";
const port = 4173;
const baseUrl = `http://${host}:${port}`;
const viteBin = fileURLToPath(new URL("../node_modules/vite/bin/vite.js", import.meta.url));

const server = spawn(process.execPath, [
  viteBin,
  "preview",
  "--config",
  "examples/browser-playground/vite.config.ts",
  "--host",
  host,
  "--port",
  String(port),
  "--strictPort"
], {
  stdio: ["ignore", "pipe", "pipe"]
});

let output = "";
server.stdout.on("data", (chunk) => {
  output += chunk.toString();
});
server.stderr.on("data", (chunk) => {
  output += chunk.toString();
});

try {
  await waitForServer(server);
  const html = await fetchText(`${baseUrl}/`);
  assertIncludes(html, "SymTorch Agent Policy Playground", "HTML title");
  assertIncludes(html, "stateBuffer", "state import/export surface");
  assertIncludes(html, "Train High Risk", "training panel");

  const assets = assetUrls(html);
  if (assets.length < 2) throw new Error(`Expected CSS and JS assets, found ${assets.length}.`);
  const loaded = await Promise.all(assets.map((asset) => fetchText(asset)));
  if (!loaded.some((asset) => asset.includes("symtorch.playground.v1"))) {
    throw new Error("Built JS asset does not include the playground state schema marker.");
  }

  console.log("SymTorch Browser Playground Smoke");
  console.log("preview server: PASS");
  console.log("html shell: PASS");
  console.log("built assets: PASS");
} finally {
  stop(server);
}

async function waitForServer(process: ChildProcessWithoutNullStreams): Promise<void> {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (process.exitCode !== null) {
      throw new Error(`Preview server exited early with ${process.exitCode}.\n${output}`);
    }
    try {
      await fetchText(`${baseUrl}/`);
      return;
    } catch {
      await sleep(250);
    }
  }
  throw new Error(`Timed out waiting for Vite preview.\n${output}`);
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`GET ${url} failed with ${response.status}.`);
  return response.text();
}

function assetUrls(html: string): string[] {
  const urls: string[] = [];
  for (const match of html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)) {
    const path = match[1];
    if (path) urls.push(new URL(path, baseUrl).toString());
  }
  return urls;
}

function assertIncludes(value: string, expected: string, label: string): void {
  if (!value.includes(expected)) throw new Error(`Missing ${label}: ${expected}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stop(process: ChildProcessWithoutNullStreams): void {
  if (process.exitCode === null) process.kill();
}

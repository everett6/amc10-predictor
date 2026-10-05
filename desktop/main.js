// Desktop shell: starts the local Python prediction engine on a free
// port, waits until it answers, then shows its UI in a window. Nothing
// leaves the machine; the engine listens on 127.0.0.1 only.
const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");
const fs = require("node:fs");

const REPO_ROOT = path.join(__dirname, "..");
let backend = null;
let backendLog = "";
let quitting = false;

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function startBackend(port) {
  const userData = app.getPath("userData");
  const env = {
    ...process.env,
    AMC10_USER_DATA_DIR: userData,
    // Versioned so a new release rebuilds the database from its own seed data.
    AMC10_DB_PATH: path.join(userData, `amc10-${app.getVersion()}.sqlite3`),
  };
  let command, args, cwd;
  if (app.isPackaged) {
    const res = process.resourcesPath;
    const exe = process.platform === "win32" ? "amc10-backend.exe" : "amc10-backend";
    command = path.join(res, "backend", exe);
    args = ["--port", String(port)];
    cwd = res;
    env.AMC10_FRONTEND_DIST = path.join(res, "frontend-dist");
    env.AMC10_SEED_DIR = path.join(res, "seed");
  } else {
    // Development: run the engine from the repository with the system Python.
    command = process.env.AMC10_PYTHON || (process.platform === "win32" ? "python" : "python3");
    args = [path.join(REPO_ROOT, "src", "api", "app.py"), "--port", String(port)];
    cwd = REPO_ROOT;
    if (!fs.existsSync(path.join(REPO_ROOT, "frontend", "dist", "index.html"))) {
      throw new Error("frontend/dist is missing. Run `npm run build` in frontend/ first.");
    }
  }
  backend = spawn(command, args, {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true, // no stray console window for the engine on Windows
  });
  const keep = (chunk) => { backendLog = (backendLog + chunk.toString()).slice(-4000); };
  backend.stdout.on("data", keep);
  backend.stderr.on("data", keep);
  backend.on("exit", (code) => {
    backend = null;
    if (!quitting) {
      dialog.showErrorBox("The prediction engine stopped", `Exit code ${code}.\n\n${backendLog}`);
      app.quit();
    }
  });
  backend.on("error", (err) => { backendLog += `\n${err.message}`; });
}

function waitUntilReady(port, timeoutMs = 60000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      if (!backend) return reject(new Error(`The engine exited during startup.\n\n${backendLog}`));
      const req = http.get({ host: "127.0.0.1", port, path: "/api/health", timeout: 1500 }, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else retry();
      });
      req.on("error", retry);
      req.on("timeout", () => { req.destroy(); });
    };
    const retry = () => {
      if (Date.now() - started > timeoutMs) reject(new Error(`The engine did not start in time.\n\n${backendLog}`));
      else setTimeout(attempt, 250);
    };
    attempt();
  });
}

async function createWindow() {
  const port = await freePort();
  startBackend(port);
  await waitUntilReady(port);
  const origin = `http://127.0.0.1:${port}`;
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    minWidth: 720,
    minHeight: 560,
    title: "AMC 10 Predictor",
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  // Links to the papers open in the user's browser; the window itself never leaves the local UI.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(origin)) {
      event.preventDefault();
      if (url.startsWith("https://")) shell.openExternal(url);
    }
  });
  await win.loadURL(origin);
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const [win] = BrowserWindow.getAllWindows();
    if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
  });
  app.whenReady().then(() =>
    createWindow().catch((err) => {
      dialog.showErrorBox("AMC 10 Predictor could not start", String(err.message || err));
      app.quit();
    }),
  );
  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", () => {
    quitting = true;
    if (backend) backend.kill();
  });
}

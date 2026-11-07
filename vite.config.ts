import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "child_process";
import { createInterface } from "node:readline";
import path from "path";

function vitePythonBackendPlugin(): Plugin {
  let activeTrainingClients: any[] = [];
  let activeTrainingProcess: ReturnType<typeof spawn> | null = null;
  const python = process.env.MODELBUILDER_PYTHON || "python3";
  const publishTrainingEvent = (line: string) => {
    if (!line.trim()) return;
    activeTrainingClients.forEach(client => {
      if (!client.destroyed) client.write(`data: ${line.trim()}\n\n`);
    });
  };

  return {
    name: "vite-python-backend",
    configureServer(server) {
      server.middlewares.use("/api/backend", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method Not Allowed");
          return;
        }
        let body = "";
        req.on("data", chunk => { body += chunk; });
        req.on("end", () => {
          const py = spawn(python, ["backend/modelbuilder/engine.py"], {
            cwd: process.cwd()
          });
          let stdout = "";
          let stderr = "";
          py.stdout.on("data", data => { stdout += data; });
          py.stderr.on("data", data => { stderr += data; });
          py.on("close", code => {
            if (code !== 0 && !stdout) {
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: false, error: { message: stderr || `Python exited with code ${code}` } }));
              return;
            }
            const lines = stdout.trim().split("\n");
            const lastLine = lines[lines.length - 1];
            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.end(lastLine);
          });
          py.stdin.write(body);
          py.stdin.end();
        });
      });

      server.middlewares.use("/api/training-events", (req, res) => {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        if (res.flushHeaders) res.flushHeaders();
        activeTrainingClients.push(res);
        req.on("close", () => {
          activeTrainingClients = activeTrainingClients.filter(c => c !== res);
        });
      });

      server.middlewares.use("/api/start-training", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method Not Allowed");
          return;
        }
        let body = "";
        req.on("data", chunk => { body += chunk; });
        req.on("end", () => {
          if (activeTrainingProcess && activeTrainingProcess.exitCode === null) {
            res.statusCode = 409;
            res.end("Ya hay un entrenamiento en curso.");
            return;
          }
          const py = spawn(python, ["backend/modelbuilder/engine.py"], {
            cwd: process.cwd()
          });
          activeTrainingProcess = py;
          let stderr = "";
          let terminalEvent = false;
          const lines = createInterface({ input: py.stdout });
          lines.on("line", line => {
            try {
              const event = JSON.parse(line);
              if (event.type === "complete" || event.type === "error") {
                terminalEvent = true;
                if (activeTrainingProcess === py) activeTrainingProcess = null;
              }
              publishTrainingEvent(line);
            } catch {
              publishTrainingEvent(JSON.stringify({ type: "log", message: line }));
            }
          });
          py.stderr.on("data", data => { stderr += data.toString(); });
          py.on("error", error => {
            terminalEvent = true;
            publishTrainingEvent(JSON.stringify({ type: "error", message: error.message }));
          });
          py.on("close", code => {
            if (activeTrainingProcess === py) activeTrainingProcess = null;
            if (code !== 0 && !terminalEvent) {
              publishTrainingEvent(JSON.stringify({ type: "error", message: stderr.trim() || `Python terminó con código ${code}` }));
            }
          });
          py.stdin.write(body);
          py.stdin.end();
          res.statusCode = 200;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ started: true }));
        });
      });
    }
  };
}

export default defineConfig({
  plugins: [react(), vitePythonBackendPlugin()],
  root: "frontend",
  clearScreen: false,
  server: { port: Number(process.env.PORT || 1420), strictPort: true },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: { output: { manualChunks: { charts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"], flow: ["@xyflow/react"], react: ["react", "react-dom"] } } }
  }
});

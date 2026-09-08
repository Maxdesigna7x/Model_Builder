import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "child_process";
import path from "path";

function vitePythonBackendPlugin(): Plugin {
  let activeTrainingClients: any[] = [];

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
          const py = spawn("python3", ["backend/modelbuilder/engine.py"], {
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
          const py = spawn("python3", ["backend/modelbuilder/engine.py"], {
            cwd: process.cwd()
          });
          py.stdout.on("data", data => {
            const lines = data.toString().split("\n");
            lines.forEach(line => {
              if (line.trim()) {
                activeTrainingClients.forEach(client => {
                  client.write(`data: ${line.trim()}\n\n`);
                });
              }
            });
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
  server: { port: 1420, strictPort: true },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: { output: { manualChunks: { charts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"], flow: ["@xyflow/react"], react: ["react", "react-dom"] } } }
  }
});

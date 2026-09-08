import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { BackendResponse } from "./types";

const isTauri = () => "__TAURI_INTERNALS__" in window;

export async function backend<T>(action: string, payload: Record<string, unknown> = {}): Promise<T> {
  if (isTauri()) {
    const response = await invoke<BackendResponse<T>>("backend_request", { request: { action, ...payload } });
    if (!response.ok) throw new Error(response.error?.message || "Error del backend");
    return response.result as T;
  }

  // Web Browser dev mode proxy to Vite server plugin
  const res = await fetch("/api/backend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...payload })
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || "Error al conectar con el servidor web backend");
  }

  const response: BackendResponse<T> = await res.json();
  if (!response.ok) throw new Error(response.error?.message || "Error del backend");
  return response.result as T;
}

export async function startTraining(payload: Record<string, unknown>): Promise<void> {
  if (isTauri()) {
    await invoke("start_training", { request: { action: "train", ...payload } });
    return;
  }

  await fetch("/api/start-training", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "train", ...payload })
  });
}

export async function pickDatasetDirectory(): Promise<string | null> {
  if (!isTauri()) return null;
  return invoke<string | null>("pick_directory");
}

export async function onTrainingEvent(callback: (event: Record<string, unknown>) => void): Promise<UnlistenFn> {
  if (isTauri()) {
    return listen<Record<string, unknown>>("training-event", event => callback(event.payload));
  }

  const es = new EventSource("/api/training-events");
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      callback(data);
    } catch {
      // ignore
    }
  };
  return () => es.close();
}

export { isTauri };

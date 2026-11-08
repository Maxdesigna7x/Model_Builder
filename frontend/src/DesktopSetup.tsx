import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import App from "./App";
import { isTauri } from "./bridge";

export default function DesktopSetup() {
  const [ready, setReady] = useState(!isTauri());
  const [message, setMessage] = useState("Preparando ModelBuilder…");
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!isTauri()) return;
    let active = true;
    let unlisten: (() => void) | undefined;
    (async () => {
      try {
        const stop = await listen<string>("setup-progress", event => {
          if (active) setMessage(event.payload);
        });
        if (!active) { stop(); return; }
        unlisten = stop;
        await invoke("prepare_runtime");
        if (active) setReady(true);
      } catch (reason) {
        if (active) setError(String(reason));
      }
    })();
    return () => { active = false; unlisten?.(); };
  }, [attempt]);

  if (ready) return <App />;
  return <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", background: "#12151a", color: "#f4f4f4", fontFamily: "system-ui, sans-serif", padding: 24 }}>
    <section style={{ maxWidth: 580, width: "100%" }} aria-live="polite">
      <h1>ModelBuilder</h1>
      {error ? <><h2>No se pudo completar la preparación</h2><p>{error}</p><p>Comprueba tu conexión y el espacio disponible, y vuelve a intentarlo.</p><button style={{ background: "#f28c28", color: "#151515", border: 0, borderRadius: 6, padding: "10px 18px", cursor: "pointer" }} onClick={() => { setError(""); setMessage("Preparando ModelBuilder…"); setAttempt(n => n + 1); }}>Reintentar</button></> : <><h2>Preparando la aplicación</h2><p>La primera vez se descargan Python y las dependencias necesarias. Puede tardar varios minutos y requiere conexión a Internet.</p><progress style={{ width: "100%" }} /><p style={{ overflowWrap: "anywhere" }}>{message}</p></>}
    </section>
  </main>;
}

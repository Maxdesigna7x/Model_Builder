import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { isTauri } from "./bridge";

type Available = { version: string; notes?: string; canInstall: boolean };
export default function UpdateNotice() {
  const [update, setUpdate] = useState<Available | null>(null);
  const [message, setMessage] = useState("");
  const [opened, setOpened] = useState(false);
  const [installing, setInstalling] = useState(false);
  async function check(manual = false) {
    if (manual) { setOpened(true); setMessage("Buscando actualizaciones…"); }
    try {
      const result = await invoke<Available | null>("app_update_check");
      setUpdate(result);
      if (result) setOpened(true);
      setMessage(result ? "" : "Ya tienes la versión más reciente.");
    } catch { if (manual) setMessage("No se pudo consultar la actualización. Comprueba tu conexión e inténtalo de nuevo."); }
  }
  useEffect(() => { if (isTauri()) void check(); }, []);
  if (!isTauri()) return null;
  async function install() {
    setInstalling(true); setMessage("Descargando y verificando la actualización…");
    try { await invoke("app_update_install"); }
    catch (error) { setInstalling(false); setMessage(String(error)); }
  }
  return <>
    <button onClick={() => void check(true)} style={{position:"fixed",bottom:12,right:18,zIndex:150,padding:"7px 12px",background:"var(--surface)",color:"var(--text)",border:"1px solid var(--border)",borderRadius:8,fontSize:11}}>Actualizaciones{update ? " •" : ""}</button>
    {opened && <div role="dialog" aria-modal="true" aria-label="Actualizar ModelBuilder" style={{position:"fixed",inset:0,zIndex:10000,background:"#0009",display:"grid",placeItems:"center"}}><section style={{width:440,maxWidth:"90vw",padding:26,borderRadius:14,background:"var(--surface)",border:"1px solid var(--border)",color:"var(--text)"}}>
      <h2>{update ? `Nueva versión ${update.version}` : "Actualizaciones"}</h2>
      {update && <p style={{lineHeight:1.6,fontSize:13}}>Guarda tus cambios y termina tus tareas antes de actualizar. Tus proyectos y el entorno compatible se conservarán.</p>}
      {message && <p role="status" style={{overflowWrap:"anywhere"}}>{message}</p>}
      <div style={{display:"flex",gap:10,justifyContent:"flex-end",marginTop:20}}>
      {!installing && <button className="secondary" onClick={() => setOpened(false)}>Más tarde</button>}
      {update && <button className="primary" disabled={installing} onClick={() => update.canInstall ? void install() : void openUrl("https://github.com/Maxdesigna7x/Model_Builder/releases/latest")}>{installing ? "Actualizando…" : update.canInstall ? "Instalar y reiniciar" : "Descargar actualización"}</button>}
      </div>
    </section></div>}
  </>;
}

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ArrowRight, Check, CheckCircle2, CircleAlert, Cpu, Download, HardDrive, Layers3, LoaderCircle, LockKeyhole, RotateCcw, Sparkles, Wifi } from "lucide-react";
import App from "./App";
import { isTauri } from "./bridge";
import "./desktop-setup.css";
const nvidiaLogo = new URL("./assets/nvidia.svg", import.meta.url).href;

type Backend = "cpu" | "nvidia";
type SetupProgress = { stage: string; message: string; completed: number; total: number; current: string };
const logo = new URL("../../assets/Logo_Icon.png", import.meta.url).href;

export default function DesktopSetup() {
  const [state, setState] = useState<"checking" | "select" | "install" | "error" | "ready">(isTauri() ? "checking" : "ready");
  const [backend, setBackend] = useState<Backend | null>(null);
  const [progress, setProgress] = useState<SetupProgress>({ stage: "python", message: "Preparando Python privado", completed: 0, total: 0, current: "Python 3.12" });
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isTauri()) return;
    let active = true;
    invoke<{ready: boolean}>("runtime_status").then(result => {
      if (active) setState(result.ready ? "ready" : "select");
    }).catch(reason => { if (active) { setError(String(reason)); setState("error"); } });
    return () => { active = false; };
  }, []);

  async function startSetup() {
    if (!backend) return;
    setError("");
    setProgress({ stage: "python", message: "Preparando Python privado", completed: 0, total: 0, current: "Python 3.12" });
    setState("install");
    let stop: (() => void) | undefined;
    try {
      stop = await listen<SetupProgress>("setup-progress", event => setProgress(event.payload));
      await invoke("prepare_runtime", { backend });
      setState("ready");
    } catch (reason) {
      setError(String(reason));
      setState("error");
    } finally { stop?.(); }
  }

  if (state === "ready") return <App />;
  const installing = state === "install";
  const percentage = progress.total > 0 ? Math.round(progress.completed / progress.total * 100) : null;
  const steps = ["Preparar Python", "Instalar componentes", "Comprobar motor"];
  const step = progress.stage === "verify" ? 2 : progress.stage === "install" ? 1 : 0;

  return <main className="setup-shell">
    <header className="setup-header"><div className="setup-brand"><span className="setup-brand-logo" aria-hidden="true" style={{maskImage:`url(${logo})`,WebkitMaskImage:`url(${logo})`}} /><div><strong>ModelBuilder</strong><span>Tu espacio para crear modelos</span></div></div><span className="setup-header-tag"><span /> CONFIGURACIÓN INICIAL</span></header>
    <div className="setup-layout">
      <aside className="setup-intro"><div className="setup-intro-icon"><Sparkles size={22} /></div><div className="setup-overline">TODO LISTO PARA EMPEZAR</div><h1>Construye tus ideas.<br/><em>Nosotros preparamos<br/>el entorno.</em></h1><p>Un espacio para diseñar, entrenar y experimentar con tus modelos, directamente en tu equipo.</p><div className="setup-intro-rule"/><div className="setup-feature"><LockKeyhole size={17}/><span>Tu trabajo permanece en tu equipo</span></div><div className="setup-feature"><Layers3 size={17}/><span>Instalación sencilla, solo la primera vez</span></div><div className="setup-orbit" aria-hidden="true"><span/><span/><span/><i/><b/></div></aside>
      <section className="setup-panel" aria-live="polite">
        {state === "checking" ? <div className="setup-checking"><LoaderCircle className="setup-spin" size={28}/><p>Comprobando el entorno…</p></div> : state === "select" ? <>
          <div className="setup-kicker"><span>01</span> ELIGE TU EQUIPO</div><h2>¿Cómo quieres entrenar?</h2><p className="setup-description">Elige el dispositivo que usará ModelBuilder. Prepararemos las herramientas adecuadas para ti.</p>
          <div className="setup-options" role="radiogroup" aria-label="Dispositivo de entrenamiento">
            <button type="button" role="radio" aria-checked={backend === "nvidia"} onClick={() => setBackend("nvidia")} className={`setup-option ${backend === "nvidia" ? "selected" : ""}`}><div className="setup-option-icon nvidia"><img src={nvidiaLogo} alt=""/><span>NVIDIA</span></div><div className="setup-option-copy"><strong>GPU NVIDIA</strong><span>Entrenamiento acelerado con CUDA</span><small>Necesitas una GPU y un controlador NVIDIA compatibles</small></div><span className="setup-radio">{backend === "nvidia" && <Check size={14}/>}</span></button>
            <button type="button" role="radio" aria-checked={backend === "cpu"} onClick={() => setBackend("cpu")} className={`setup-option ${backend === "cpu" ? "selected" : ""}`}><div className="setup-option-icon cpu"><Cpu size={26}/></div><div className="setup-option-copy"><strong>Procesador · CPU</strong><span>Compatible con la mayoría de los equipos</span><small>La instalación puede ocupar menos espacio</small></div><span className="setup-radio">{backend === "cpu" && <Check size={14}/>}</span></button>
          </div>
          <div className="setup-note"><Wifi size={16}/><span>La primera preparación necesita conexión a Internet y varios GB de espacio libre.</span></div>
          <button className="setup-primary" disabled={!backend} onClick={startSetup}>Preparar ModelBuilder <ArrowRight size={17}/></button><div className="setup-footnote">El tiempo dependerá de tu conexión y de los componentes necesarios.</div>
        </> : state === "error" ? <><div className="setup-kicker error"><CircleAlert size={16}/> NECESITAMOS TU ATENCIÓN</div><h2>No se pudo completar</h2><p className="setup-description">Comprueba tu conexión, el espacio disponible y, si elegiste NVIDIA, que la GPU y su controlador sean compatibles.</p><div className="setup-error">{error}</div><div className="setup-error-actions"><button className="setup-secondary" onClick={() => setState("select")}>Cambiar dispositivo</button><button className="setup-primary" disabled={!backend} onClick={startSetup}><RotateCcw size={16}/> Reintentar</button></div></> : <>
          <div className="setup-kicker"><span>02</span> PREPARANDO TU ESPACIO</div><h2>Todo está tomando forma<span className="setup-title-dot">.</span></h2><p className="setup-description">Estamos preparando {backend === "nvidia" ? "tu entorno con NVIDIA CUDA" : "tu entorno para CPU"}. Puedes dejar esta ventana abierta mientras terminamos.</p>
          <div className="setup-progress-card"><div className="setup-progress-top"><div className="setup-download-icon"><Download size={20}/></div><div><strong>{progress.message}</strong><span>{percentage === null ? "Esto puede tardar unos minutos" : `${progress.completed} de ${progress.total} componentes preparados`}</span></div>{percentage !== null && <b>{percentage}%</b>}</div><div className={`setup-progress-track ${percentage === null ? "indeterminate" : ""}`} role="progressbar" aria-label="Componentes preparados" aria-valuemin={0} aria-valuemax={100} {...(percentage !== null ? {"aria-valuenow": percentage} : {})}><i style={percentage === null ? undefined : {width:`${percentage}%`}}/></div><div className="setup-current"><span className="setup-current-dot"/><span>{progress.current ? `Ahora: ${progress.current}` : "Preparando componentes"}</span><span className="setup-current-status">EN PROCESO</span></div>{progress.total > 0 && <p className="setup-progress-explain">El porcentaje cuenta componentes preparados. El progreso individual de descarga no está disponible.</p>}</div>
          <div className="setup-steps">{steps.map((label, index) => <div key={label} className={`setup-step ${index < step ? "done" : index === step ? "active" : ""}`}><span>{index < step ? <Check size={13}/> : index === step ? <LoaderCircle className="setup-spin" size={14}/> : index + 1}</span>{label}</div>)}</div>
          <div className="setup-bottom-note"><HardDrive size={16}/> Se guardará en tu espacio de usuario, sin alterar tu Python.</div>
        </>}
      </section>
    </div>
    <footer className="setup-footer"><span>MODELBUILDER <b>·</b> HECHO PARA CREAR</span><span><CheckCircle2 size={13}/> LOCAL Y PRIVADO</span></footer>
  </main>;
}

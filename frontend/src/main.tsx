import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
// Version the style entry points so the desktop WebView never reuses a partial
// development response after these large stylesheets change.
import "./styles.css?layout-v2";
import "./ui-polish.css?layout-v4";
// Vite emits this as an independent stylesheet so the desktop WebView receives it intact.
// @ts-expect-error CSS modules are handled by Vite at runtime.
void import("./inference-layout.css?layout-v2");
import "./revision14.css";
import "./revision15.css";
import "./revision16.css";
import "./i18n";
import DesktopSetup from "./DesktopSetup";

createRoot(document.getElementById("root")!).render(
  <StrictMode><DesktopSetup /></StrictMode>
);

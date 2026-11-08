import { useState } from "react";
import { Cloud, Download, Search } from "lucide-react";
import { useTranslation } from "react-i18next";
import { backend } from "./bridge";
import { translateBackendError } from "./i18n/errors";
import type { DatasetOptions, KaggleImportReport } from "./types";

export function KaggleReport({ report }: { report: KaggleImportReport }) {
  const { t } = useTranslation("app");
  return <div className="kaggle-report" aria-live="polite">
    <strong>{t(`kaggle.layout.${report.layout}`)}</strong>
    <p>{t("kaggle.summary", { samples: report.samples, classes: Object.keys(report.classes).length })}</p>
    <p>{report.originalSplits.length ? t("kaggle.merged", { splits: report.originalSplits.join(", ") }) : t("kaggle.unsplit")}</p>
    <div className="class-tags">{Object.entries(report.classes).slice(0, 20).map(([name, count]) => <span key={name}>{name}: {count}</span>)}</div>
    <p>{t("kaggle.splitPolicy")}</p>
  </div>;
}

export function KaggleImport({ onImport }: { onImport: (handle: string, options: DatasetOptions) => Promise<void> }) {
  const { t } = useTranslation("app");
  const [handle, setHandle] = useState("");
  const [subdirectory, setSubdirectory] = useState("");
  const [labelFile, setLabelFile] = useState("");
  const [report, setReport] = useState<KaggleImportReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const options = { subdirectory, label_file: labelFile };
  const inspect = async () => {
    setBusy(true); setError(""); setReport(null);
    try { setReport(await backend<KaggleImportReport>("data.kaggle.inspect", { handle, options })); }
    catch (error) { setError(translateBackendError(String(error))); }
    finally { setBusy(false); }
  };
  const load = async () => {
    setBusy(true); setError("");
    try { await onImport(handle, options); }
    catch (error) { setError(translateBackendError(String(error))); }
    finally { setBusy(false); }
  };
  return <article className="source-card kaggle-source" aria-busy={busy}>
    <Cloud/><h3>Kaggle</h3><p>{t("kaggle.description")}</p>
    <label className="app-field"><span>{t("kaggle.handle")}</span><input disabled={busy} value={handle} placeholder="owner/dataset" onChange={event => { setHandle(event.target.value); setReport(null); }}/></label>
    <details className="kaggle-advanced"><summary>{t("kaggle.structureOptions")}</summary>
    <label className="app-field"><span>{t("kaggle.subdirectory")}</span><input disabled={busy} value={subdirectory} placeholder={t("kaggle.optional")} onChange={event => { setSubdirectory(event.target.value); setReport(null); }}/></label>
    <label className="app-field"><span>{t("kaggle.labelFile")}</span><input disabled={busy} value={labelFile} placeholder={t("kaggle.automatic")} onChange={event => { setLabelFile(event.target.value); setReport(null); }}/></label>
    </details>
    <p>{t("kaggle.auth")}</p>
    {error && <p role="alert" className="catalog-error">{error}</p>}
    {report && <KaggleReport report={report}/>}
    <button className="secondary" disabled={busy || !handle.trim()} onClick={inspect}><Search size={15}/>{busy ? t("kaggle.working") : t("kaggle.inspect")}</button>
    {report && <button className="primary" disabled={busy} onClick={load}><Download size={15}/>{t("kaggle.import")}</button>}
  </article>;
}

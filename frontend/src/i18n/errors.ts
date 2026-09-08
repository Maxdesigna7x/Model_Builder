import i18n from "i18next";

/**
 * Translate a backend error code into a human-readable message.
 *
 * The backend serializes errors as either a plain code or `CODE|detail`.
 * If a translation is missing, the code (and detail, if present) is returned.
 */
export function translateBackendError(code: string, detail?: string): string {
  if (!code) return "";
  code = code.replace(/^Error:\s*/, "");

  let resolvedDetail = detail;
  if (resolvedDetail === undefined && code.includes("|")) {
    const [head, ...tail] = code.split("|");
    code = head;
    resolvedDetail = tail.join("|");
  }

  const translated = i18n.t(code, {
    ns: "errors",
    defaultValue: code,
    detail: resolvedDetail,
  });

  if (translated === code && resolvedDetail) {
    return `${code}: ${resolvedDetail}`;
  }
  return translated;
}

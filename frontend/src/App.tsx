import { useCallback, useEffect, useState, useRef } from "react";
import init, { NeuralEngine } from "./wasm/wasm_engine";
import type {
  DeleteSamplesResponse,
  DatasetStatsResponse,
  InferenceResult,
  LabelSampleResponse,
  ModelApiResponse,
} from "./types";
import { NetworkVisualizer } from "./NetworkVisualizer";
import { DrawingCanvas } from "./DrawingCanvas";
import { useTranslation } from "./translationContext";

type DeleteScope = "all" | "label";

function App() {
  const { language, toggleLanguage, t } = useTranslation();
  const [isWasmReady, setIsWasmReady] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [lastInputTensor, setLastInputTensor] = useState<Float32Array | null>(
    null,
  );
  const [isCanvasEmpty, setIsCanvasEmpty] = useState(true);
  const [modelSource, setModelSource] = useState<"backend" | "local-fallback">(
    "backend",
  );
  const [sampleLabel, setSampleLabel] = useState<string>("0");
  const [sampleStatus, setSampleStatus] = useState<string | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStatsResponse | null>(
    null,
  );
  const [statsStatus, setStatsStatus] = useState<string | null>(null);
  const [lastSavedSampleId, setLastSavedSampleId] = useState<number | null>(
    null,
  );
  const [adminUsername, setAdminUsername] = useState("");
  const [adminPassword, setAdminPassword] = useState("");
  const [authHeader, setAuthHeader] = useState<string | null>(null);
  const [authStatus, setAuthStatus] = useState<string | null>(null);
  const [isAdminPanelOpen, setIsAdminPanelOpen] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [deleteScope, setDeleteScope] = useState<DeleteScope | null>(null);

  const engineRef = useRef<NeuralEngine | null>(null);
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  const modelVersion = import.meta.env.VITE_MODEL_VERSION ?? "v1.0";

  const readErrorMessage = useCallback(async (response: Response) => {
    const rawBody = await response.text();
    if (!rawBody.trim()) {
      return `Request failed with status ${response.status}.`;
    }

    try {
      const payload = JSON.parse(rawBody) as { error?: string };
      return payload.error ?? rawBody;
    } catch {
      return rawBody;
    }
  }, []);

  const readDeleteResponse = useCallback(async (response: Response) => {
    const rawBody = await response.text();
    if (!rawBody.trim()) {
      return { deleted_count: 0 } as DeleteSamplesResponse;
    }

    try {
      return JSON.parse(rawBody) as DeleteSamplesResponse;
    } catch {
      return { deleted_count: 0 } as DeleteSamplesResponse;
    }
  }, []);

  const authorizedFetch = useCallback(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      if (!authHeader) {
        throw new Error("Missing admin authorization.");
      }

      const headers = new Headers(init?.headers);
      headers.set("Authorization", authHeader);
      return fetch(input, { ...init, headers });
    },
    [authHeader],
  );

  useEffect(() => {
    const loadWasm = async () => {
      try {
        setIsWasmReady(false);
        await init();
        const modelResponse = await fetch(
          `${apiBaseUrl}/model/${encodeURIComponent(modelVersion)}`,
        );
        let jsonText = "";
        if (modelResponse.ok) {
          const payload = (await modelResponse.json()) as ModelApiResponse;
          jsonText = JSON.stringify(payload.model);
          setModelSource("backend");
        } else if (modelResponse.status === 404) {
          const localFallback = await fetch("/model.json");
          if (!localFallback.ok) throw new Error("model.json not found");
          jsonText = await localFallback.text();
          setModelSource("local-fallback");
        } else {
          throw new Error(await readErrorMessage(modelResponse));
        }

        engineRef.current = new NeuralEngine(jsonText);
        setIsWasmReady(true);
      } catch (err) {
        console.error("Error loading WASM or Model:", err);
      }
    };
    loadWasm();
  }, [apiBaseUrl, modelVersion, readErrorMessage]);

  const fetchDatasetStats = useCallback(async () => {
    if (!authHeader) return;

    try {
      setStatsStatus(t("loadingStats"));
      const response = await authorizedFetch(`${apiBaseUrl}/datasets/stats`);
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = (await response.json()) as DatasetStatsResponse;
      setDatasetStats(payload);
      setStatsStatus(null);
    } catch (error) {
      console.error("Error loading stats:", error);
      setStatsStatus(
        error instanceof Error ? error.message : t("statsUnavailable"),
      );
    }
  }, [apiBaseUrl, authHeader, authorizedFetch, readErrorMessage, t]);

  useEffect(() => {
    if (!authHeader || !isAdminPanelOpen) return;
    fetchDatasetStats();
  }, [authHeader, fetchDatasetStats, isAdminPanelOpen]);

  const handleLogin = async () => {
    if (!adminUsername || !adminPassword) {
      setAuthStatus(t("insertUsernamePassword"));
      return;
    }

    const token = btoa(`${adminUsername}:${adminPassword}`);
    const candidateAuthHeader = `Basic ${token}`;

    try {
      setAuthStatus(t("checkingCredentials"));
      const response = await fetch(`${apiBaseUrl}/datasets/stats`, {
        headers: { Authorization: candidateAuthHeader },
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      setAuthHeader(candidateAuthHeader);
      setAuthStatus(null);
      setIsAuthModalOpen(false);
      setIsAdminPanelOpen(true);
    } catch (error) {
      console.error("Auth error:", error);
      setAuthStatus(error instanceof Error ? error.message : t("loginFailed"));
      setAuthHeader(null);
    }
  };

  const handleLogout = () => {
    setAuthHeader(null);
    setDatasetStats(null);
    setSampleStatus(null);
    setStatsStatus(null);
    setLastSavedSampleId(null);
    setAuthStatus(null);
    setIsAdminPanelOpen(false);
    setIsAuthModalOpen(false);
  };

  // This function receives the normalized 28x28 tensor from the Canvas
  const handleInference = (tensor: Float32Array) => {
    if (!engineRef.current || !isWasmReady) return;
    const empty = tensor.every((value) => value === 0);
    setIsCanvasEmpty(empty);

    try {
      setLastInputTensor(Float32Array.from(tensor));
      const rawResult = engineRef.current.predict(tensor);
      setResult(rawResult as InferenceResult);
    } catch (err) {
      console.error("Error during inference:", err);
    }
  };

  const resetNetworkState = useCallback(() => {
    if (!engineRef.current || !isWasmReady) return;
    const zeroTensor = new Float32Array(28 * 28);
    setIsCanvasEmpty(true);
    try {
      setLastInputTensor(Float32Array.from(zeroTensor));
      const zeroResult = engineRef.current.predict(zeroTensor);
      setResult(zeroResult as InferenceResult);
    } catch (err) {
      console.error("Error resetting neural state:", err);
    }
  }, [isWasmReady]);

  useEffect(() => {
    if (!isWasmReady || result) return;
    resetNetworkState();
  }, [isWasmReady, resetNetworkState, result]);

  const handleClear = () => {
    resetNetworkState();
    setSampleStatus(null);
  };

  const submitLabeledSample = async () => {
    if (!lastInputTensor || isCanvasEmpty) {
      setSampleStatus(t("drawBeforeSending"));
      return;
    }
    const numericLabel = Number(sampleLabel);
    if (
      !Number.isInteger(numericLabel) ||
      numericLabel < 0 ||
      numericLabel > 9
    ) {
      setSampleStatus(t("labelMustBeInteger"));
      return;
    }

    try {
      setSampleStatus(t("sendingSample"));
      const response = await authorizedFetch(`${apiBaseUrl}/datasets/labels`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pixels: Array.from(lastInputTensor),
          label: numericLabel,
          source: "frontend-manual-correction",
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = (await response.json()) as LabelSampleResponse;
      setSampleStatus(t("sampleStored", { id: payload.sample_id }));
      setLastSavedSampleId(payload.sample_id);
      fetchDatasetStats();
    } catch (error) {
      console.error("Error sending labeled sample:", error);
      setSampleStatus(
        error instanceof Error ? error.message : t("failedStoreSample"),
      );
    }
  };

  const deleteAllSamples = async () => {
    try {
      setSampleStatus(t("deletingAllSamples"));
      const response = await authorizedFetch(`${apiBaseUrl}/datasets/labels`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = await readDeleteResponse(response);
      setSampleStatus(t("deletedSamples", { count: payload.deleted_count }));
      setLastSavedSampleId(null);
      fetchDatasetStats();
    } catch (error) {
      console.error("Error deleting all samples:", error);
      setSampleStatus(
        error instanceof Error ? error.message : t("deleteFailed"),
      );
    }
  };

  const deleteSamplesForSelectedLabel = async () => {
    try {
      setSampleStatus(t("deletingLabelSamples", { label: sampleLabel }));
      const response = await authorizedFetch(
        `${apiBaseUrl}/datasets/labels/${sampleLabel}`,
        {
          method: "DELETE",
        },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = await readDeleteResponse(response);
      setSampleStatus(
        t("deletedLabelSamples", {
          count: payload.deleted_count,
          label: sampleLabel,
        }),
      );
      setLastSavedSampleId(null);
      fetchDatasetStats();
    } catch (error) {
      console.error("Error deleting label samples:", error);
      setSampleStatus(
        error instanceof Error ? error.message : t("deleteFailed"),
      );
    }
  };

  const undoLastLabel = async () => {
    if (!lastSavedSampleId) {
      return;
    }

    try {
      setSampleStatus(t("undoingSample", { id: lastSavedSampleId }));
      const response = await authorizedFetch(
        `${apiBaseUrl}/datasets/samples/${lastSavedSampleId}`,
        {
          method: "DELETE",
        },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      await readDeleteResponse(response);
      setSampleStatus(t("undoDone"));
      setLastSavedSampleId(null);
      fetchDatasetStats();
    } catch (error) {
      console.error("Error undoing label operation:", error);
      setSampleStatus(error instanceof Error ? error.message : t("undoFailed"));
    }
  };

  const openAdminPanel = () => {
    if (authHeader) {
      setIsAdminPanelOpen((value) => !value);
      return;
    }
    setIsAuthModalOpen(true);
  };

  const confirmDelete = async () => {
    if (deleteScope === "all") {
      await deleteAllSamples();
    } else if (deleteScope === "label") {
      await deleteSamplesForSelectedLabel();
    }
    setDeleteScope(null);
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center gap-8 font-sans bg-slate-900 text-white">
      <header className="text-center">
        <h1 className="text-4xl font-bold bg-linear-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          NeuralCanvas WASM!
        </h1>
        <p className="mt-2 text-sm text-slate-300">{t("heroDescription")}</p>
        <div className="mt-3 flex items-center justify-center gap-2 text-xs text-slate-300">
          <span className="font-semibold text-cyan-300">{t("english")}</span>
          <button
            type="button"
            role="switch"
            aria-checked={language === "it"}
            onClick={toggleLanguage}
            className="relative inline-flex h-6 w-12 items-center rounded-full border border-slate-600 bg-slate-800 transition-colors"
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-cyan-400 transition-transform ${
                language === "it" ? "translate-x-7" : "translate-x-1"
              }`}
            />
          </button>
          <span className="font-semibold text-cyan-300">{t("italian")}</span>
        </div>
        <p className="text-slate-400 mt-2">
          {t("engineStatus")}:{" "}
          {isWasmReady ? (
            <span className="text-emerald-400">{t("online")}</span>
          ) : (
            <span className="text-amber-400">{t("booting")}</span>
          )}
        </p>
        <p className="text-xs text-slate-500 mt-1">
          {t("modelSource")}:{" "}
          {modelSource === "backend" ? t("backendApi") : t("localFallback")}
        </p>
        <button
          type="button"
          onClick={openAdminPanel}
          className="mt-3 px-3 py-1 rounded bg-indigo-700 hover:bg-indigo-600 text-xs font-semibold text-white transition-colors"
        >
          {isAdminPanelOpen ? t("closeLabelManager") : t("manageLabelsMetrics")}
        </button>
      </header>

      <main className="w-full max-w-4xl flex flex-col gap-8">
        <div className="bg-slate-800 p-8 rounded-xl border border-slate-700 flex flex-col items-center justify-center gap-4 shadow-lg">
          <DrawingCanvas
            onDrawEnd={handleInference}
            onClear={handleClear}
            clearLabel={t("clearDrawing")}
          />
          <p className="text-xs text-slate-400 text-center">{t("drawHint")}</p>

          {!isWasmReady && (
            <div className="text-slate-400 text-sm animate-pulse">
              {t("wasmInitialization")}
            </div>
          )}
        </div>

        {result && (
          <div className="flex flex-col gap-4 w-full">
            <h2 className="text-xl font-semibold text-slate-200 flex items-center justify-between">
              <span>{t("internalNetworkState")}</span>

              {/* Prediction and Probability box*/}
              <div className="grid grid-cols-[auto_auto_auto] items-center gap-3 bg-slate-900 px-4 py-2 rounded-lg border border-slate-700 shadow-inner font-mono">
                <div className="flex items-baseline gap-2 leading-none">
                  <span className="text-sm text-slate-400">
                    {t("confidence")}:
                  </span>
                  {isCanvasEmpty ? (
                    <strong className="text-slate-500 text-xl">-</strong>
                  ) : (
                    <strong
                      className={
                        result.probabilities[result.predicted_label] > 0.8
                          ? "text-emerald-400 text-xl"
                          : result.probabilities[result.predicted_label] > 0.5
                            ? "text-amber-400 text-xl"
                            : "text-red-400 text-xl"
                      }
                    >
                      {(
                        result.probabilities[result.predicted_label] * 100
                      ).toFixed(1)}
                      %
                    </strong>
                  )}
                </div>
                <div className="w-px h-7 bg-slate-700" />
                <div className="flex items-baseline gap-2 leading-none">
                  <span className="text-sm text-cyan-300">
                    {t("prediction")}:
                  </span>
                  {isCanvasEmpty ? (
                    <strong className="text-slate-500 text-2xl">-</strong>
                  ) : (
                    <strong className="text-cyan-400 text-2xl">
                      {result.predicted_label}
                    </strong>
                  )}
                </div>
              </div>
            </h2>

            <NetworkVisualizer
              inputTensor={lastInputTensor}
              activations={result.layer_activations}
              probabilities={result.probabilities}
            />

            {isAdminPanelOpen && authHeader && (
              <>
                <section className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-200">
                      {t("datasetCollector")}
                    </h3>
                    <button
                      type="button"
                      onClick={handleLogout}
                      className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-xs font-medium text-slate-100 transition-colors"
                    >
                      {t("logoutAdmin")}
                    </button>
                  </div>
                  <p className="text-xs text-slate-400">
                    {t("datasetCollectorHint")}
                  </p>
                  <div className="flex flex-wrap items-center gap-3">
                    <label
                      htmlFor="sample-label"
                      className="text-sm text-slate-300"
                    >
                      {t("correctLabel")}
                    </label>
                    <select
                      id="sample-label"
                      className="bg-slate-900 border border-slate-600 rounded px-3 py-1 text-slate-200"
                      value={sampleLabel}
                      onChange={(event) => setSampleLabel(event.target.value)}
                    >
                      {Array.from({ length: 10 }, (_, value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={submitLabeledSample}
                      disabled={isCanvasEmpty}
                      title={isCanvasEmpty ? t("drawBeforeSaving") : undefined}
                      className="px-4 py-1.5 rounded bg-cyan-600 enabled:hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
                    >
                      {t("saveSample")}
                    </button>
                    <button
                      type="button"
                      onClick={() => setDeleteScope("label")}
                      className="px-4 py-1.5 rounded bg-amber-700 hover:bg-amber-600 text-white text-sm font-medium transition-colors"
                    >
                      {t("deleteLabel", { label: sampleLabel })}
                    </button>
                    <button
                      type="button"
                      onClick={undoLastLabel}
                      disabled={lastSavedSampleId === null}
                      className="px-4 py-1.5 rounded bg-slate-700 enabled:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
                    >
                      {t("undoLastLabel")}
                    </button>
                    <button
                      type="button"
                      onClick={() => setDeleteScope("all")}
                      className="ml-auto px-4 py-1.5 rounded bg-red-700 hover:bg-red-600 text-white text-sm font-semibold transition-colors border border-red-500/80"
                    >
                      {t("deleteAllSamples")}
                    </button>
                  </div>
                  {sampleStatus && (
                    <p className="text-xs text-slate-300 font-mono">
                      {sampleStatus}
                    </p>
                  )}
                </section>

                <section className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-200">
                      {t("datasetMetrics")}
                    </h3>
                    <button
                      type="button"
                      onClick={fetchDatasetStats}
                      className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-xs font-medium text-slate-100 transition-colors"
                    >
                      {t("refresh")}
                    </button>
                  </div>
                  {statsStatus && (
                    <p className="text-xs text-slate-400 font-mono">
                      {statsStatus}
                    </p>
                  )}
                  {datasetStats && (
                    <>
                      <p className="text-sm text-slate-300">
                        {t("totalSamples")}:{" "}
                        <span className="font-mono text-cyan-300">
                          {datasetStats.total_samples}
                        </span>
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                        {Array.from({ length: 10 }, (_, label) => {
                          const hit = datasetStats.counts_by_label.find(
                            (item) => item.label === label,
                          );
                          return (
                            <div
                              key={label}
                              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-mono flex items-center justify-between"
                            >
                              <span className="text-slate-400">{label}</span>
                              <span className="text-slate-200">
                                {hit?.count ?? 0}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}
                </section>
              </>
            )}
          </div>
        )}
      </main>

      {isAuthModalOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
          <section className="w-full max-w-md rounded-xl border border-slate-700 bg-slate-800 p-6 flex flex-col gap-4 shadow-xl">
            <h2 className="text-lg font-bold text-cyan-300">
              {t("adminAccessRequired")}
            </h2>
            <p className="text-sm text-slate-400">{t("adminAccessHint")}</p>
            <input
              type="text"
              placeholder={t("username")}
              value={adminUsername}
              onChange={(event) => setAdminUsername(event.target.value)}
              className="bg-slate-900 border border-slate-600 rounded px-3 py-2 text-slate-200"
            />
            <input
              type="password"
              placeholder={t("password")}
              value={adminPassword}
              onChange={(event) => setAdminPassword(event.target.value)}
              className="bg-slate-900 border border-slate-600 rounded px-3 py-2 text-slate-200"
            />
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setIsAuthModalOpen(false)}
                className="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm transition-colors"
              >
                {t("cancel")}
              </button>
              <button
                type="button"
                onClick={handleLogin}
                className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors"
              >
                {t("login")}
              </button>
            </div>
            {authStatus && (
              <p className="text-xs text-amber-300 font-mono">{authStatus}</p>
            )}
          </section>
        </div>
      )}

      {deleteScope && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
          <section className="w-full max-w-md rounded-xl border border-red-500/50 bg-slate-900 p-6 flex flex-col gap-4 shadow-xl">
            <h2 className="text-lg font-bold text-red-300">
              {t("confirmDeletingAction")}
            </h2>
            <p className="text-sm text-slate-300">
              {deleteScope === "all"
                ? t("deletingAllSamplesWarning")
                : t("deletingByLabelWarning", { label: sampleLabel })}
            </p>
            <p className="text-xs text-slate-400">
              {t("operationCannotBeUndone")}
            </p>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteScope(null)}
                className="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm transition-colors"
              >
                {t("cancel")}
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                className="px-4 py-2 rounded bg-red-700 hover:bg-red-600 text-white text-sm font-semibold transition-colors border border-red-500/80"
              >
                {t("confirmDelete")}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;

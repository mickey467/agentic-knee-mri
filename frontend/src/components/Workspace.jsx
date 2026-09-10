import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "../api";

const SUGGESTIONS = [
  "Explain the highest prediction",
  "Show me the sagittal series",
  "What does the medical evidence say?",
];

export default function Workspace({ study, session, onBack }) {
  const [meta, setMeta] = useState(null);
  const [seriesId, setSeriesId] = useState(null);
  const [slice, setSlice] = useState(1);
  const [probs, setProbs] = useState({ state: "loading" });
  const [report, setReport] = useState({ state: "loading", started: Date.now() });

  const series = meta?.series ?? [];
  const current = series.find((s) => s.series_id === seriesId) ?? series[0];
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    api.metadata(study.study_id).then(setMeta).catch(() => setMeta({ series: [] }));
    api.analyze(study.study_id)
      .then((d) => setProbs({ state: "ok", data: d }))
      .catch((e) => setProbs({ state: "error", error: e.message }));
    runReport(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (current) {
      setSeriesId((id) => id ?? preferSeries(series).series_id);
      setSlice(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  const [vizNote, setVizNote] = useState("");

  const selectSeries = (id) => {
    setSeriesId(id);
    setSlice(1);
  };

  const applyViz = useCallback(
    (commands) => {
      const cmd = (commands ?? [])[0];
      if (!cmd) return;
      const match =
        series.find((s) => s.series_number === cmd.series_number) ??
        (cmd.orientation
          ? series.find((s) => s.orientation.toLowerCase() === cmd.orientation.toLowerCase())
          : null);
      if (match) {
        setSeriesId(match.series_id);
        setSlice(Math.max(1, Math.ceil(match.slice_count / 2)));
        setVizNote(`Viewer → ${label(match)} (slice ${Math.max(1, Math.ceil(match.slice_count / 2))}/${match.slice_count})`);
      } else {
        setVizNote("No matching series found for that view.");
      }
    },
    [series]
  );

  const runReport = (payload) => {
    setReport({ state: "loading", started: Date.now(), progress: [] });
    api.reportStream(study.study_id, session, payload, (event, data) => {
      if (event === "tool") {
        setReport((r) =>
          r.state === "loading" ? { ...r, progress: [...r.progress, data] } : r
        );
      } else if (event === "model" && data?.model === "fallback") {
        setReport((r) =>
          r.state === "loading"
            ? { ...r, progress: [...r.progress, { tool: "model_fallback", phase: "start" }] }
            : r
        );
      }
    })
      .then((d) => setReport(finishReport(d)))
      .catch((e) => setReport({ state: "error", error: e.message }));
  };

  const resume = (payload) => runReport(payload);

  return (
    <div className="min-h-screen bg-base text-ink flex flex-col">
      <header className="border-b border-line px-5 py-2.5 flex items-center justify-between no-print">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="font-mono text-xs text-dim hover:text-ink">
            ← Studies
          </button>
          <span className="font-semibold">{study.name}</span>
          <span className="font-mono text-xs text-faint">{study.study_id}</span>
        </div>
        {report.model && (
          <span className="font-mono text-xs text-faint">
            model: <span className="text-accent">{report.model}</span>
          </span>
        )}
      </header>

      <div className="flex-1 grid lg:grid-cols-[1fr_380px]">
        <div className="p-4 border-r border-line">
          <Viewer
            studyId={study.study_id}
            series={series}
            current={current}
            slice={slice}
            onSeries={selectSeries}
            onSlice={setSlice}
            vizNote={vizNote}
          />
          <ProbBars probs={probs} />
        </div>

        <div className="p-4 no-print">
          <ReportView report={report} onResume={resume} />
        </div>

        <div className="print-only-report hidden print:block p-4">
          {report.state === "ok" && (
            <div className="md"><ReactMarkdown>{report.data.report}</ReactMarkdown></div>
          )}
        </div>
      </div>

      <div className="border-t border-line p-4 no-print">
        <Chat studyId={study.study_id} session={session} reportReady={report.state === "ok"} onViz={applyViz} />
      </div>

      <footer className="border-t border-line px-5 py-2 font-mono text-xs text-faint no-print">
        Research Use Only. Not for standalone diagnostic decision-making.
      </footer>
    </div>
  );
}

function preferSeries(series) {
  return series.find((s) => s.orientation === "Sagittal") ?? series[0];
}

function label(s) {
  return s.display_label || s.series_description;
}

function finishReport(d) {
  if (d.status === "paused_for_demographics") return { state: "paused", questions: d.questions };
  return { state: "ok", data: d, model: d.model };
}

/* ------------------------------- Viewer ---------------------------------- */

function Viewer({ studyId, series, current, slice, onSeries, onSlice, vizNote }) {
  // Race guard: overlapping slice requests resolve out of order while
  // scrubbing, so only display the image if it is still the requested one.
  const reqId = useRef(0);
  const [src, setSrc] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") onSlice(Math.min(slice + 1, current?.slice_count ?? 1));
      if (e.key === "ArrowLeft") onSlice(Math.max(1, slice - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const want = current ? api.sliceUrl(studyId, current.series_id, slice) : "";
  useEffect(() => {
    if (!want) return;
    const my = ++reqId.current;
    setFailed(false);
    const im = new Image();
    im.onload = () => {
      if (reqId.current === my) setSrc(want);
    };
    im.onerror = () => {
      if (reqId.current === my) setFailed(true);
    };
    im.src = want;
  }, [want]);

  if (!current) return <p className="text-dim text-sm">Loading viewer…</p>;

  return (
    <div>
      <div className="mb-3">
        <select
          value={current.series_id}
          onChange={(e) => e.target.value && onSeries(e.target.value)}
          className="w-full bg-card border border-line rounded text-sm px-2 py-1.5 text-ink"
        >
          {[...new Set(series.map((s) => s.orientation))].map((ori) => (
            <optgroup key={ori} label={ori}>
              {series
                .filter((s) => s.orientation === ori)
                .map((s) => (
                  <option key={s.series_id} value={s.series_id}>
                    {label(s)} · {s.slice_count} slices
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        {vizNote && <p className="mt-1 font-mono text-xs text-accent">{vizNote}</p>}
      </div>

      <div className="relative bg-black rounded-lg overflow-hidden">
        {src && !failed ? (
          <img
            src={src}
            alt={`slice ${slice}`}
            className="w-full max-h-[60vh] object-contain"
            draggable={false}
          />
        ) : (
          <div className="w-full h-[40vh] flex items-center justify-center font-mono text-xs text-faint">
            {failed ? "slice failed to load" : "loading slice…"}
          </div>
        )}
        <div className="absolute top-2 left-2 font-mono text-xs bg-black/70 px-2 py-0.5 rounded">
          {label(current)}
        </div>
        <div className="absolute top-2 right-2 font-mono text-xs bg-black/70 px-2 py-0.5 rounded">
          Im: {slice}/{current.slice_count}
        </div>
        <div className="absolute bottom-2 left-2 font-mono text-xs bg-black/70 px-2 py-0.5 rounded">
          {current.dimensions} · {current.slice_thickness ?? "?"}mm
        </div>
      </div>

      <div className="flex items-center gap-3 mt-3">
        <button onClick={() => onSlice(Math.max(1, slice - 1))} className="px-3 py-1.5 bg-card border border-line rounded text-sm">←</button>
        <input
          type="range" min={1} max={current.slice_count} value={slice}
          onChange={(e) => onSlice(Number(e.target.value))}
          className="flex-1"
        />
        <button onClick={() => onSlice(Math.min(current.slice_count, slice + 1))} className="px-3 py-1.5 bg-card border border-line rounded text-sm">→</button>
      </div>
    </div>
  );
}

/* ------------------------------ Prob bars -------------------------------- */

function ProbBars({ probs }) {
  if (probs.state === "loading") return <p className="mt-4 text-dim text-sm">Running inference…</p>;
  if (probs.state === "error")
    return <p className="mt-4 text-bad text-sm">Inference failed: {probs.error}</p>;
  const rows = Object.entries(probs.data.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="mt-5">
      <h3 className="font-mono text-xs tracking-widest text-faint mb-2">AI ANALYSIS · 12 TARGETS</h3>
      <div className="space-y-1.5">
        {rows.map(([label, p]) => (
          <div key={label} className="flex items-center gap-2 text-sm">
            <span className="w-36 truncate text-dim">{label}</span>
            <div className="flex-1 h-2 bg-card rounded overflow-hidden">
              <div
                className={`h-full rounded ${p >= 0.5 ? "bg-accent" : "bg-[#3a3a40]"}`}
                style={{ width: `${Math.round(p * 100)}%` }}
              />
            </div>
            <span className={`font-mono text-xs w-10 text-right ${p >= 0.5 ? "text-accent" : "text-faint"}`}>
              {Math.round(p * 100)}%
            </span>
          </div>
        ))}
      </div>
      <p className="mt-1 font-mono text-xs text-faint">Threshold 0.50 · DINOv2 + LoRA, 5-fold</p>
    </div>
  );
}

/* -------------------------------- Report --------------------------------- */

const TOOL_LABELS = {
  get_study_info: "Reading study info",
  get_predictions: "Running inference (DINOv2 ×5 folds)",
  search_medical_knowledge: "Retrieving evidence",
  list_series: "Listing series",
  model_fallback: "Primary overloaded → fallback model",
};

function ProgressList({ progress }) {
  const steps = [];
  for (const p of progress) {
    const label = TOOL_LABELS[p.tool] ?? p.tool;
    if (p.phase === "start") steps.push({ label, done: false });
    else {
      const open = [...steps].reverse().find((s) => s.label === label && !s.done);
      if (open) open.done = true;
    }
  }
  if (steps.length === 0) {
    return (
      <div className="animate-pulse space-y-2">
        <div className="h-4 bg-card rounded w-1/3" />
        <div className="h-3 bg-card rounded" />
        <div className="h-3 bg-card rounded w-2/3" />
      </div>
    );
  }
  return (
    <ul className="space-y-1.5 font-mono text-xs">
      {steps.map((s, i) => (
        <li key={i} className={s.done ? "text-faint" : "text-accent"}>
          {s.done ? "✓" : "…"} {s.label}
        </li>
      ))}
    </ul>
  );
}

function ReportView({ report, onResume }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (report.state !== "loading") return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - report.started) / 1000)), 1000);
    return () => clearInterval(t);
  }, [report]);

  if (report.state === "loading")
    return (
      <div className="border border-line rounded-lg p-4">
        <ProgressList progress={report.progress ?? []} />
        <p className="mt-3 font-mono text-xs text-faint">Generating report… {elapsed}s</p>
      </div>
    );

  if (report.state === "paused")
    return <DemoForm questions={report.questions} onSubmit={onResume} />;

  if (report.state === "error")
    return (
      <div className="border border-bad rounded-lg p-4 text-sm">
        <p className="text-bad font-semibold">Report failed</p>
        <p className="mt-1 text-dim">{report.error}</p>
        <p className="mt-1 font-mono text-xs text-faint">If the model is overloaded, wait a minute and use chat below — or reload.</p>
      </div>
    );

  const { data } = report;
  return (
    <div className="border border-line rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">AI Report</h3>
        <button onClick={() => window.print()} className="text-sm bg-card border border-line rounded px-3 py-1.5 hover:border-accent">
          Print
        </button>
      </div>
      {(data.warnings ?? []).map((w) => (
        <p key={w} className="mb-2 text-xs bg-[#1a1405] border border-warn text-warn rounded px-2 py-1.5">{w}</p>
      ))}
      <div className="md"><ReactMarkdown>{data.report}</ReactMarkdown></div>
      {(data.tool_trace ?? []).length > 0 && (
        <details className="mt-3 font-mono text-xs text-faint">
          <summary className="cursor-pointer hover:text-dim">tool trace ({data.tool_trace.length})</summary>
          <ul className="mt-1 space-y-0.5">
            {data.tool_trace.map((t, i) => (
              <li key={i}>→ {t.tool}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function DemoForm({ questions, onSubmit }) {
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  return (
    <div className="border border-accent rounded-lg p-4">
      <p className="text-sm font-semibold">One thing missing from the scan headers</p>
      <p className="text-dim text-sm mt-1">The agent needs demographics before writing the report.</p>
      <div className="mt-3 space-y-2">
        {questions.includes("patient_age") && (
          <label className="block text-sm">
            <span className="text-dim">Age</span>
            <input
              type="number" min={0} max={120} value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="e.g. 45"
              className="mt-1 w-full bg-card border border-line rounded px-2 py-1.5 text-ink"
            />
          </label>
        )}
        {questions.includes("patient_sex") && (
          <label className="block text-sm">
            <span className="text-dim">Sex</span>
            <select
              value={sex} onChange={(e) => setSex(e.target.value)}
              className="mt-1 w-full bg-card border border-line rounded px-2 py-1.5 text-ink"
            >
              <option value="">Select…</option>
              <option value="M">M</option>
              <option value="F">F</option>
              <option value="unknown">Unknown</option>
            </select>
          </label>
        )}
        <button
          onClick={() => onSubmit({ patient_age: age || "unknown", patient_sex: sex || "unknown" })}
          className="bg-accent text-black font-semibold rounded px-4 py-1.5 text-sm"
        >
          Continue report
        </button>
      </div>
    </div>
  );
}

/* --------------------------------- Chat ---------------------------------- */

function Chat({ studyId, session, reportReady, onViz }) {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const send = async (text) => {
    const q = (text ?? input).trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", text: q }]);
    const t0 = Date.now();
    try {
      const d = await api.chat(studyId, q, session);
      const ms = Date.now() - t0;
      setMsgs((m) => [...m, { role: "agent", text: d.reply, trace: d.tool_trace, model: d.model, ms }]);
      onViz(d.viz_commands);
    } catch (e) {
      setMsgs((m) => [...m, { role: "agent", text: `Error: ${e.message}`, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h3 className="font-semibold mb-1">AI Clinical Agent</h3>
      <p className="text-dim text-sm mb-3">
        {reportReady ? "Ask about this study, or ask to see a series." : "Chat unlocks once the report is ready."}
      </p>
      <div className="space-y-3 max-h-[40vh] overflow-y-auto pr-1">
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            <div
              className={`inline-block max-w-[80%] text-left rounded-lg px-3 py-2 text-sm ${
                m.role === "user" ? "bg-card border border-line" : m.error ? "border border-bad" : "bg-panel border border-line"
              }`}
            >
              {m.role === "agent" ? (
                <div className="md"><ReactMarkdown>{m.text}</ReactMarkdown></div>
              ) : (
                m.text
              )}
              {m.trace?.length > 0 && (
                <details className="mt-2 font-mono text-xs text-faint">
                  <summary className="cursor-pointer">tools · {m.model} · {(m.ms / 1000).toFixed(1)}s</summary>
                  <ul className="mt-1">{m.trace.map((t, j) => <li key={j}>→ {t.tool}</li>)}</ul>
                </details>
              )}
            </div>
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <div className="flex gap-2 flex-wrap mt-3">
        {SUGGESTIONS.map((s) => (
          <button key={s} onClick={() => send(s)} className="font-mono text-xs bg-card border border-line rounded-full px-3 py-1 hover:border-accent">
            {s}
          </button>
        ))}
      </div>

      <div className="flex gap-2 mt-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask about this MRI study…"
          className="flex-1 bg-card border border-line rounded px-3 py-2 text-sm placeholder:text-faint focus:outline-none focus:border-accent"
        />
        <button
          onClick={() => send()}
          disabled={busy}
          className="bg-accent text-black font-semibold rounded px-5 text-sm disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}

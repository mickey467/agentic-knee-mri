import { useEffect, useState } from "react";
import { api } from "./api";
import Workspace from "./components/Workspace";

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

export default function App() {
  const [studies, setStudies] = useState(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(null); // {study, session}

  useEffect(() => {
    (async () => {
      try {
        const list = await api.studies();
        const enriched = await Promise.all(
          list.map(async (s) => {
            try {
              const meta = await api.metadata(s.study_id);
              const first = meta.series[0];
              const mid = Math.max(1, Math.ceil((first?.slice_count ?? 1) / 2));
              return { ...s, thumb: api.sliceUrl(s.study_id, first.series_id, mid) };
            } catch {
              return { ...s, thumb: "" };
            }
          })
        );
        setStudies(enriched);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  if (open) {
    return <Workspace study={open.study} session={open.session} onBack={() => setOpen(null)} />;
  }

  return (
    <div className="min-h-screen bg-base text-ink">
      <header className="border-b border-line px-5 py-3 flex items-center justify-between">
        <div>
          <span className="font-bold tracking-tight">KneeAI</span>
          <span className="ml-2 font-mono text-xs text-faint">AGENTIC KNEE MRI · ACADEMIC PROTOTYPE</span>
        </div>
        <HealthDot />
      </header>

      <main className="max-w-5xl mx-auto px-5 py-10">
        <h1 className="text-3xl font-semibold tracking-tight">AI-Assisted Knee MRI Analysis</h1>
        <p className="mt-2 font-mono text-xs tracking-widest text-faint">CREATED BY MICHEAL EHAB</p>
        <p className="mt-2 text-dim max-w-2xl text-sm leading-relaxed">
          Computer vision, medical knowledge retrieval, and an AI agent analyze knee MRI
          studies and explain the predictions. Research use only — not a medical device.
        </p>

        <h2 className="mt-10 mb-4 font-mono text-xs tracking-widest text-faint">HOW IT WORKS</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="bg-panel border border-line rounded-lg p-4">
            <div className="font-mono text-xs text-accent">01 · VISION MODEL</div>
            <h3 className="font-semibold mt-1">DINOv2 + LoRA, 5-fold ensemble</h3>
            <ul className="mt-2 text-sm text-dim space-y-1.5 leading-relaxed">
              <li>DINOv2 ViT backbone at 336px, adapted with LoRA (rank 8) on attention projections — only a fraction of weights trained.</li>
              <li>Model-G head fuses 2.5D slice embeddings with scan metadata into 12 abnormality scores (ACL, menisci, OA, effusion…).</li>
              <li>5-fold ensemble averaged at 0.50 threshold; ~0.80 macro ROC-AUC out-of-fold. Runs on CPU.</li>
            </ul>
          </div>
          <div className="bg-panel border border-line rounded-lg p-4">
            <div className="font-mono text-xs text-accent">02 · MCP SERVERS</div>
            <h3 className="font-semibold mt-1">Four tool servers</h3>
            <ul className="mt-2 text-sm text-dim space-y-1.5 leading-relaxed">
              <li><span className="text-ink font-mono text-xs">DICOM</span> — study info, series listing from real headers.</li>
              <li><span className="text-ink font-mono text-xs">Model</span> — wraps the ensemble inference.</li>
              <li><span className="text-ink font-mono text-xs">RAG</span> — FAISS search over 916 chunks from 9 knee PDFs.</li>
              <li><span className="text-ink font-mono text-xs">Visualization</span> — finds series by orientation for the viewer.</li>
            </ul>
          </div>
          <div className="bg-panel border border-line rounded-lg p-4">
            <div className="font-mono text-xs text-accent">03 · AGENT</div>
            <h3 className="font-semibold mt-1">Report, then chat</h3>
            <ul className="mt-2 text-sm text-dim space-y-1.5 leading-relaxed">
              <li>On study open the agent gathers metadata → predictions → cited evidence and writes a 4-section report.</li>
              <li>Missing age/sex pauses the run and asks you, then resumes with your answer.</li>
              <li>Afterwards, chat over the report: RAG answers plus viewer commands (“show me sagittal”).</li>
            </ul>
          </div>
        </div>

        <h2 className="mt-10 mb-4 font-mono text-xs tracking-widest text-faint">CHOOSE A TEST STUDY</h2>
        {error && <p className="text-bad text-sm">Backend unreachable: {error}</p>}
        {!studies && !error && <p className="text-dim text-sm">Loading studies…</p>}
        <div className="grid gap-4 md:grid-cols-3">
          {(studies ?? []).map((s) => (
            <button
              key={s.study_id}
              onClick={() => setOpen({ study: s, session: uid() })}
              className="text-left bg-panel border border-line rounded-lg overflow-hidden hover:border-accent transition-colors"
            >
              {s.thumb ? (
                <img
                  src={s.thumb}
                  alt={s.name}
                  loading="lazy"
                  className="w-full aspect-square object-cover bg-black"
                />
              ) : (
                <div className="w-full aspect-square bg-black flex items-center justify-center font-mono text-xs text-faint">
                  no preview
                </div>
              )}
              <div className="p-4">
                <div className="font-mono text-[11px] text-faint">{s.study_id.toUpperCase()}</div>
                <div className="font-semibold mt-0.5">{s.name}</div>
                <div className="mt-2 font-mono text-xs text-dim">
                  {s.series_count} series · {s.total_files} slices
                </div>
                <div className="mt-3 inline-block bg-accent text-black text-sm font-semibold rounded px-4 py-1.5">
                  Analyze Study
                </div>
              </div>
            </button>
          ))}
        </div>
      </main>

      <footer className="border-t border-line px-5 py-3 font-mono text-xs text-faint flex justify-between">
        <span>Research Use Only. Not for standalone diagnostic decision-making.</span>
        <span>DINOv2 + LoRA · 12 targets</span>
      </footer>
    </div>
  );
}

function HealthDot() {
  const [ok, setOk] = useState(null);
  useEffect(() => {
    api.health().then(() => setOk(true)).catch(() => setOk(false));
  }, []);
  return (
    <span className="font-mono text-xs text-dim flex items-center gap-2">
      <span
        className={`inline-block w-2 h-2 rounded-full ${ok === null ? "bg-faint" : ok ? "bg-good" : "bg-bad"}`}
      />
      {ok === null ? "…" : ok ? "Backend connected" : "Backend offline"}
    </span>
  );
}

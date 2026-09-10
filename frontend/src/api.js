async function req(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () => req("/health"),
  studies: () => req("/api/studies"),
  metadata: (id) => req(`/api/studies/${id}/metadata`),
  series: (id, seriesId) => req(`/api/studies/${id}/series/${seriesId}`),
  sliceUrl: (id, seriesId, i) => `/api/studies/${id}/series/${seriesId}/slices/${i}/image`,
  // Background inference + poll: each HTTP round-trip is short, so slow
  // hosted links (tunnels/proxies) can't time out mid-inference.
  analyze: async (id, onStatus) => {
    const job = await req(`/api/studies/${id}/analyze`, {
      method: "POST",
      body: JSON.stringify({ background: true }),
    });
    const deadline = Date.now() + 15 * 60 * 1000;
    for (;;) {
      await new Promise((r) => setTimeout(r, 3000));
      const s = await req(`/api/studies/${id}/jobs/${job.job_id}`);
      onStatus?.(s.status);
      if (s.status === "done") return s.result;
      if (Date.now() > deadline) throw new Error("Inference timed out after 15 minutes");
      // failed jobs raise via req() on the failed payload automatically
    }
  },
  report: (id, sessionId, resume) =>
    req(`/api/studies/${id}/report`, {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, resume_payload: resume ?? null }),
    }),
  // Streaming variant: emits {event, data} as SSE blocks arrive; resolves
  // with the final report/paused payload (same shape as report()).
  reportStream: async (id, sessionId, resume, onEvent) => {
    const res = await fetch(`/api/studies/${id}/report/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, resume_payload: resume ?? null }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let final = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const blocks = buf.split("\n\n");
      buf = blocks.pop();
      for (const block of blocks) {
        let event = null;
        let data = null;
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data = JSON.parse(line.slice(5).trim());
        }
        if (!event) continue;
        onEvent?.(event, data);
        if (event === "paused") final = { status: "paused_for_demographics", ...data };
        else if (event === "report") final = data;
        else if (event === "error") throw new Error(data.detail || "Stream failed");
      }
    }
    if (!final) throw new Error("Stream ended without a report");
    return final;
  },
  chat: (id, message, sessionId) =>
    req(`/api/studies/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId }),
    }),
};

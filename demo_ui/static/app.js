const state = {
  running: false,
  startedAt: null,
  timer: null,
};

const elements = {
  answer: document.querySelector("#answer"),
  clear: document.querySelector("#clearButton"),
  latency: document.querySelector("#latency"),
  question: document.querySelector("#question"),
  run: document.querySelector("#runButton"),
  runStatus: document.querySelector("#runStatus"),
  timeline: document.querySelector("#timeline"),
  topology: document.querySelector("#topology"),
  traceId: document.querySelector("#traceId"),
};

const agentLabels = {
  registry: "Registry",
  customer: "Customer Agent",
  law: "Law Agent",
  tax: "Tax Agent",
  compliance: "Compliance Agent",
};

function resetAgents() {
  document.querySelectorAll(".agent-card").forEach((card) => {
    card.classList.remove("running", "completed", "skipped", "available", "failed");
    card.querySelector(".agent-state").textContent = "Standby";
  });
}

function setAgentStatus(agent, status) {
  const card = document.querySelector(`[data-agent="${agent}"]`);
  if (!card) return;
  card.classList.remove("running", "completed", "skipped", "available", "failed");
  card.classList.add(status);
  const labels = {
    running: "Processing",
    completed: "Complete",
    skipped: "Skipped",
    available: "Available",
    failed: "Failed",
  };
  card.querySelector(".agent-state").textContent = labels[status] || status;
}

function setRunStatus(status, label) {
  elements.runStatus.className = `status-badge ${status}`;
  elements.runStatus.textContent = label;
}

function formatDetails(details) {
  if (!details) return "";
  return Object.entries(details)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => {
      const rendered = Array.isArray(value)
        ? value.join(", ")
        : typeof value === "object"
          ? JSON.stringify(value)
          : String(value);
      return `<div class="log-field"><code>${escapeHtml(key)}</code><span>${escapeHtml(rendered)}</span></div>`;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addTimeline(title, detail = "", kind = "", details = null) {
  if (elements.timeline.querySelector(".empty-state")) elements.timeline.innerHTML = "";
  const row = document.createElement("div");
  row.className = `timeline-item ${kind}`;
  const time = state.startedAt
    ? `${((performance.now() - state.startedAt) / 1000).toFixed(1)}s`
    : "0.0s";
  row.innerHTML = `
    <span class="timeline-time">${time}</span>
    <span class="timeline-dot"></span>
    <span class="timeline-copy"><strong></strong><span></span><div class="log-details"></div></span>
  `;
  row.querySelector("strong").textContent = title;
  row.querySelector(".timeline-copy span").textContent = detail;
  row.querySelector(".log-details").innerHTML = formatDetails(details);
  elements.timeline.appendChild(row);
  elements.timeline.scrollTop = elements.timeline.scrollHeight;
}

function startTimer() {
  clearInterval(state.timer);
  state.startedAt = performance.now();
  state.timer = setInterval(() => {
    elements.latency.textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(1)}s`;
  }, 100);
}

function finishTimer(latency) {
  clearInterval(state.timer);
  elements.latency.textContent = `${Number(latency).toFixed(2)}s`;
}

function handleEvent(event) {
  if (event.type === "run_started") {
    elements.traceId.textContent = `Trace ${event.trace_id.slice(0, 13)}...`;
    addTimeline("Stage 5 A2A started", "Khởi tạo request mới", "info", {
      trace_id: event.trace_id,
      context_id: event.context_id,
      delegation_depth: event.delegation_depth,
      question_length: event.question_length,
    });
  }

  if (event.type === "log") {
    addTimeline(
      `[${event.phase}] ${event.message}`,
      event.level.toUpperCase(),
      event.level,
      event.details,
    );
  }

  if (event.type === "agent_status") {
    setAgentStatus(event.agent, event.status);
    const action = event.status === "running" ? "started" : event.status;
    addTimeline(`${agentLabels[event.agent] || event.agent} ${action}`, event.preview?.slice(0, 130) || "");
  }

  if (event.type === "routing") {
    const routes = Object.entries(event)
      .filter(([key, value]) => key !== "type" && value)
      .map(([key]) => agentLabels[key] || key);
    addTimeline("Routing decision", routes.length ? `Dispatch: ${routes.join(", ")}` : "No specialist required");
  }

  if (event.type === "registry") {
    const online = event.agents.filter((agent) => agent.online).length;
    addTimeline(
      "Registry discovery",
      `${online}/${event.agents.length} services online`,
      "info",
      {
        services: event.agents.map((agent) =>
          `${agent.name}:${agent.online ? "online" : "offline"}@${agent.latency_ms}ms`
        ),
      },
    );
    event.agents.filter((agent) => !agent.online).forEach((agent) => {
      const key = agent.name.replace("-agent", "");
      setAgentStatus(key, "failed");
    });
  }

  if (event.type === "result") {
    elements.answer.textContent = event.answer || "No answer returned.";
    addTimeline("Final answer received", `${(event.answer || "").length} characters`);
  }

  if (event.type === "done") {
    finishTimer(event.latency);
    setRunStatus("completed", "Completed");
    elements.topology.classList.remove("processing");
    addTimeline("Execution completed", `Total latency: ${event.latency}s`);
  }

  if (event.type === "error") {
    clearInterval(state.timer);
    setRunStatus("failed", "Failed");
    elements.topology.classList.remove("processing");
    elements.answer.textContent = `Không thể hoàn tất request:\n\n${event.message}`;
    addTimeline("Execution failed", event.message, "error", {
      error_type: event.error_type,
      elapsed_s: event.elapsed,
      trace_id: event.trace_id,
      context_id: event.context_id,
    });
  }
}

async function runDemo() {
  const question = elements.question.value.trim();
  if (!question || state.running) return;

  state.running = true;
  elements.run.disabled = true;
  elements.answer.innerHTML = '<div class="empty-state">Agents are preparing the analysis...</div>';
  elements.timeline.innerHTML = "";
  elements.latency.textContent = "0.0s";
  elements.traceId.textContent = "Creating trace...";
  resetAgents();
  setRunStatus("running", "Running");
  elements.topology.classList.add("processing");
  startTimer();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      lines.filter(Boolean).forEach((line) => handleEvent(JSON.parse(line)));
      if (done) break;
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch (error) {
    handleEvent({ type: "error", message: error.message });
  } finally {
    state.running = false;
    elements.run.disabled = false;
  }
}

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => { elements.question.value = button.dataset.question; });
});
elements.run.addEventListener("click", runDemo);
elements.clear.addEventListener("click", () => {
  elements.timeline.innerHTML = '<div class="empty-state">Run a request to inspect agent events.</div>';
  elements.answer.innerHTML = '<div class="empty-state">The aggregated answer will appear here.</div>';
  elements.latency.textContent = "--";
  elements.traceId.textContent = "Chưa có trace";
  setRunStatus("idle", "Idle");
  resetAgents();
});

resetAgents();

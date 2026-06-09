const state = { running: false, started: 0, workers: {}, activeWorker: null };
const $ = (selector) => document.querySelector(selector);
const labels = {
  supervisor: "Supervisor",
  legal_worker: "Legal Worker",
  news_worker: "News Worker",
  verification_worker: "Verification Worker",
  synthesizer: "Synthesizer",
};

function reset() {
  state.workers = {};
  state.activeWorker = null;
  document.querySelectorAll(".node").forEach((node) => {
    node.classList.remove("running", "completed");
    node.querySelector("em").textContent = "Standby";
  });
  $("#logs").innerHTML = "";
  $("#tabs").innerHTML = "";
  $("#evidence").innerHTML = '<p class="empty">Evidence của workers sẽ xuất hiện tại đây.</p>';
  $("#answer").innerHTML = '<p class="empty">Câu trả lời cuối sẽ xuất hiện tại đây.</p>';
  $("#evidenceCount").textContent = "0";
  $("#sourceCount").textContent = "0";
}

function log(message) {
  const row = document.createElement("div");
  row.className = "log";
  row.innerHTML = `<time>${((performance.now() - state.started) / 1000).toFixed(2)}s</time><span></span>`;
  row.querySelector("span").textContent = message;
  $("#logs").appendChild(row);
  $("#logs").scrollTop = $("#logs").scrollHeight;
}

function setNode(node, status) {
  const card = document.querySelector(`[data-node="${node}"]`);
  if (!card) return;
  card.classList.remove("running", "completed");
  card.classList.add(status);
  card.querySelector("em").textContent = status === "running" ? "Running" : "Done";
}

function updateMetrics() {
  const results = Object.values(state.workers);
  const evidence = results.flatMap((item) => item.evidence || []);
  $("#evidenceCount").textContent = evidence.length;
  $("#sourceCount").textContent = new Set(evidence.map((item) => item.source)).size;
}

function renderTabs() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  Object.entries(state.workers).forEach(([key, result], index) => {
    const button = document.createElement("button");
    button.textContent = `${labels[key]} (${result.evidence.length})`;
    button.className = key === state.activeWorker || (!state.activeWorker && index === 0) ? "active" : "";
    button.onclick = () => {
      state.activeWorker = key;
      renderTabs();
      renderEvidence(result);
    };
    tabs.appendChild(button);
    if (!state.activeWorker && index === 0) {
      state.activeWorker = key;
      renderEvidence(result);
    }
  });
}

function renderEvidence(result) {
  const container = $("#evidence");
  container.innerHTML = "";
  const summary = document.createElement("p");
  summary.textContent = result.summary;
  container.appendChild(summary);
  (result.warnings || []).forEach((warning) => {
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = warning;
    container.appendChild(item);
  });
  (result.evidence || []).forEach((evidence, index) => {
    const item = document.createElement("article");
    item.className = "evidence-item";
    item.innerHTML = `<strong></strong><p></p><small></small>`;
    item.querySelector("strong").textContent = `${index + 1}. ${evidence.source}`;
    item.querySelector("p").textContent = evidence.content.slice(0, 520);
    item.querySelector("small").textContent = `${evidence.document_type} · score ${Number(evidence.score).toFixed(3)}`;
    container.appendChild(item);
  });
}

function handle(event) {
  if (event.type === "run_started") {
    $("#mode").textContent = event.mode.toUpperCase();
    log(`Workflow started · mode=${event.mode}`);
  }
  if (event.type === "configuration_warning") {
    log(`WARNING: ${event.message}`);
  }
  if (event.type === "node_status") {
    setNode(event.node, event.status);
    log(`${labels[event.node]} → ${event.status}`);
  }
  if (event.type === "supervisor_plan") {
    event.logs.forEach(log);
    log(`Plan: ${event.plan.join(", ")}`);
  }
  if (event.type === "worker_result") {
    const result = event.result;
    state.workers[event.node] = result;
    event.logs.forEach(log);
    updateMetrics();
    renderTabs();
  }
  if (event.type === "final_answer") {
    event.logs.forEach(log);
    $("#answer").textContent = event.answer;
  }
  if (event.type === "done") {
    $("#latency").textContent = `${event.latency.toFixed(3)}s`;
    $("#graph").classList.remove("processing");
    log(`Workflow completed in ${event.latency}s`);
  }
  if (event.type === "error") {
    $("#graph").classList.remove("processing");
    $("#answer").textContent = `${event.error_type}: ${event.message}`;
    log(`ERROR: ${event.message}`);
  }
}

async function run() {
  const question = $("#question").value.trim();
  if (!question || state.running) return;
  state.running = true;
  state.started = performance.now();
  $("#run").disabled = true;
  $("#latency").textContent = "...";
  reset();
  $("#graph").classList.add("processing");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, use_llm: $("#useLlm").checked }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      lines.filter(Boolean).forEach((line) => handle(JSON.parse(line)));
      if (done) break;
    }
  } catch (error) {
    handle({ type: "error", error_type: error.name, message: error.message });
  } finally {
    state.running = false;
    $("#run").disabled = false;
  }
}

$("#run").addEventListener("click", run);
reset();

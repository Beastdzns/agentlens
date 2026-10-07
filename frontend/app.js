/**
 * AgentLens – app.js
 * Pure Vanilla JS. No build step required.
 *
 * To point at a different backend, change the constant below:
 */
const API_BASE_URL = 'http://localhost:8000';

// ── State ────────────────────────────────────────────────────
let isLoading  = false;
let isMockMode = false;
let activeFilter = 'all';
let lastData   = null;

// ── DOM refs ────────────────────────────────────────────────
const queryInput    = document.getElementById('queryInput');
const runBtn        = document.getElementById('runBtn');
const runIcon       = document.getElementById('runIcon');
const runLabel      = document.getElementById('runLabel');
const mockModeBtn   = document.getElementById('mockModeBtn');
const mockLabel     = document.getElementById('mockLabel');
const mockPanel     = document.getElementById('mockPanel');
const loadingState  = document.getElementById('loadingState');
const emptyState    = document.getElementById('emptyState');
const errorBanner   = document.getElementById('errorBanner');
const errorMessage  = document.getElementById('errorMessage');
const dismissError  = document.getElementById('dismissError');
const resultsView   = document.getElementById('resultsView');
const loadingStep   = document.getElementById('loadingStep');
const apiStatus     = document.getElementById('apiStatus');
const apiStatusText = document.getElementById('apiStatusText');

// stat card refs
const statTraceId    = document.getElementById('statTraceId');
const statLatency    = document.getElementById('statLatency');
const statEventCount = document.getElementById('statEventCount');
const statStatus     = document.getElementById('statStatus');
const responseText   = document.getElementById('responseText');
const eventList      = document.getElementById('eventList');
const filterBtns     = document.querySelectorAll('.filter-btn');

// sidebar meta refs
const traceMeta      = document.getElementById('traceMeta');
const metaTraceId    = document.getElementById('metaTraceId');
const metaStatus     = document.getElementById('metaStatus');
const metaLatency    = document.getElementById('metaLatency');
const metaEventCount = document.getElementById('metaEventCount');

// ── Helpers ─────────────────────────────────────────────────
function fmtLatency(ms) {
  if (ms == null) return 'N/A';
  if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
  return ms.toFixed(1) + ' ms';
}

function getEventLatency(ev) {
  // Handle both latency_ms and latency field names
  return ev.latency_ms ?? ev.latency ?? null;
}

function statusClass(status) {
  const s = (status || '').toLowerCase();
  if (s === 'success' || s === 'completed') return 'success';
  if (s === 'failure' || s === 'failed' || s === 'error') return 'failure';
  if (s === 'partial') return 'partial';
  return 'pending';
}

function badgeHTML(status) {
  const cls = statusClass(status);
  return `<span class="badge badge-${cls}">${status || 'unknown'}</span>`;
}

function jsonPretty(obj) {
  if (obj == null) return 'null';
  if (typeof obj === 'string') return obj;
  return JSON.stringify(obj, null, 2);
}

function shortId(id) {
  if (!id) return '—';
  const s = String(id);
  if (s.length > 16) return s.slice(0, 8) + '…' + s.slice(-4);
  return s;
}

// ── API health check ────────────────────────────────────────
async function checkApiHealth() {
  try {
    const r = await fetch(API_BASE_URL + '/health', { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      apiStatus.className = 'api-status online';
      apiStatusText.textContent = 'API Online';
    } else {
      apiStatus.className = 'api-status offline';
      apiStatusText.textContent = 'API Error';
    }
  } catch {
    // Try root endpoint as fallback
    try {
      await fetch(API_BASE_URL + '/', { signal: AbortSignal.timeout(3000) });
      apiStatus.className = 'api-status online';
      apiStatusText.textContent = 'API Online';
    } catch {
      apiStatus.className = 'api-status offline';
      apiStatusText.textContent = 'API Offline';
    }
  }
}

// ── UI State controllers ────────────────────────────────────
function setLoading(v) {
  isLoading = v;
  queryInput.disabled = v;
  runBtn.disabled = v;

  if (v) {
    runIcon.innerHTML = `<circle cx="12" cy="12" r="3"/>
      <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>`;
    runLabel.textContent = 'Running…';
    loadingState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsView.classList.add('hidden');
    errorBanner.classList.add('hidden');
  } else {
    runIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"/>`;
    runLabel.textContent = 'Run Agent';
    loadingState.classList.add('hidden');
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.classList.remove('hidden');
  resultsView.classList.add('hidden');
  emptyState.classList.add('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
}

function animateLoadingSteps() {
  const steps = [
    'Initializing trace…',
    'Sending query to agent…',
    'Agent reasoning (LLM call)…',
    'Executing tool calls…',
    'Collecting telemetry…',
    'Finalizing trace…',
  ];
  let i = 0;
  loadingStep.textContent = steps[0];
  return setInterval(() => {
    i = (i + 1) % steps.length;
    loadingStep.textContent = steps[i];
  }, 1500);
}

// ── Render results ──────────────────────────────────────────
function renderResults(data) {
  lastData = data;

  // Stat cards
  statTraceId.textContent    = data.trace_id || 'N/A';
  statLatency.textContent    = fmtLatency(data.total_latency_ms);
  statEventCount.textContent = (data.events || []).length;
  statStatus.innerHTML       = badgeHTML(data.status);

  // Response
  responseText.textContent = data.response || '(no response)';

  // Sidebar meta
  metaTraceId.textContent    = shortId(data.trace_id);
  metaStatus.innerHTML       = badgeHTML(data.status);
  metaLatency.textContent    = fmtLatency(data.total_latency_ms);
  metaEventCount.textContent = (data.events || []).length + ' events';
  traceMeta.classList.remove('hidden');

  renderEventList(data.events || []);

  resultsView.classList.remove('hidden');
  emptyState.classList.add('hidden');
}

function renderEventList(events) {
  const filtered = activeFilter === 'all'
    ? events
    : events.filter(ev => {
        const sc = statusClass(ev.status);
        if (activeFilter === 'success') return sc === 'success';
        if (activeFilter === 'failure') return sc === 'failure' || sc === 'error';
        return true;
      });

  if (filtered.length === 0) {
    eventList.innerHTML = `<p style="color:var(--text-muted);font-size:12px;padding:14px 6px;">No events match this filter.</p>`;
    return;
  }

  eventList.innerHTML = filtered.map((ev, idx) => {
    const lat    = getEventLatency(ev);
    const sc     = statusClass(ev.status);
    const typeStr = (ev.event_type || 'unknown').replace(/_/g, ' ');
    const hasBody = ev.input_data || ev.output_data || ev.parent_id || ev.metadata;

    return `
      <div class="event-card status-${sc}" id="ev-card-${idx}">
        <div class="event-header" onclick="toggleEvent(${idx})">
          <span class="event-index">#${idx + 1}</span>
          <span class="event-type-badge">${typeStr}</span>
          <span class="event-id-cell" title="${ev.event_id || ''}">${shortId(ev.event_id)}</span>
          <span class="event-latency">${fmtLatency(lat)}</span>
          <span class="event-status-badge">${badgeHTML(ev.status)}</span>
          <svg class="event-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </div>
        ${hasBody ? `
        <div class="event-body">
          <div class="event-meta-row">
            ${ev.event_id ? `<div class="event-meta-pair"><span class="event-meta-label">Event ID</span><span class="event-meta-value">${ev.event_id}</span></div>` : ''}
            ${ev.parent_id ? `<div class="event-meta-pair"><span class="event-meta-label">Parent ID</span><span class="event-meta-value">${ev.parent_id}</span></div>` : ''}
            ${ev.agent_id  ? `<div class="event-meta-pair"><span class="event-meta-label">Agent ID</span><span class="event-meta-value">${ev.agent_id}</span></div>` : ''}
            ${ev.timestamp ? `<div class="event-meta-pair"><span class="event-meta-label">Timestamp</span><span class="event-meta-value">${ev.timestamp}</span></div>` : ''}
          </div>
          ${ev.input_data  ? `<div class="event-data-block"><div class="event-data-label">Input Data</div><pre class="event-data-pre">${escHtml(jsonPretty(ev.input_data))}</pre></div>` : ''}
          ${ev.output_data ? `<div class="event-data-block"><div class="event-data-label">Output Data</div><pre class="event-data-pre">${escHtml(jsonPretty(ev.output_data))}</pre></div>` : ''}
          ${ev.metadata    ? `<div class="event-data-block"><div class="event-data-label">Metadata</div><pre class="event-data-pre">${escHtml(jsonPretty(ev.metadata))}</pre></div>` : ''}
        </div>` : ''}
      </div>`;
  }).join('');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

window.toggleEvent = function(idx) {
  const card = document.getElementById(`ev-card-${idx}`);
  if (card) card.classList.toggle('open');
};

// ── Filter bar ───────────────────────────────────────────────
filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    if (lastData) renderEventList(lastData.events || []);
  });
});

// ── Mock Scenarios ───────────────────────────────────────────
const MOCK_DATA = {
  success: {
    trace_id: 'aabbcc11-0000-0000-0000-112233445566',
    response: 'AI agents can be debugged through structured observability: trace every LLM call, tool invocation, and decision with unique IDs and parent links. Replay failed traces with deterministic seeds. Inspect input/output deltas at each step to localize the fault.',
    total_latency_ms: 342.7,
    status: 'success',
    events: [
      { event_id: 'ev-001', event_type: 'llm_call', status: 'success', latency_ms: 102.3, parent_id: null, input_data: { prompt: 'How can AI agents be debugged?' }, output_data: { text: 'Reasoning step 1...' }, agent_id: 'debug-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-002', event_type: 'tool_call', status: 'success', latency_ms: 55.1,  parent_id: 'ev-001', input_data: { tool: 'web_search', query: 'AI agent debugging techniques' }, output_data: { results: ['result A', 'result B'] }, agent_id: 'debug-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-003', event_type: 'retrieval',  status: 'success', latency_ms: 38.4,  parent_id: 'ev-001', input_data: { query: 'observability patterns' }, output_data: { docs: ['doc-1', 'doc-2'] }, agent_id: 'debug-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-004', event_type: 'decision',   status: 'success', latency_ms: 12.0,  parent_id: 'ev-003', input_data: { context: 'retrieved docs' }, output_data: { decision: 'synthesize' }, agent_id: 'debug-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-005', event_type: 'llm_call',   status: 'success', latency_ms: 134.9, parent_id: 'ev-004', input_data: { prompt: 'Synthesize findings' }, output_data: { text: 'Final answer...' }, agent_id: 'debug-agent', timestamp: new Date().toISOString() },
    ]
  },
  partial: {
    trace_id: 'ddee1122-0000-0000-0000-aabbccddeeff',
    response: 'Partial response generated — one tool call failed but LLM recovered.',
    total_latency_ms: 891.2,
    status: 'partial',
    events: [
      { event_id: 'ev-101', event_type: 'llm_call',  status: 'success', latency_ms: 212.0, parent_id: null, input_data: { prompt: 'Research AI debugging' }, output_data: { text: 'Plan...' }, agent_id: 'research-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-102', event_type: 'tool_call', status: 'failure', latency_ms: 400.0, parent_id: 'ev-101', input_data: { tool: 'api_call', endpoint: '/external' }, output_data: { error: 'Connection timed out after 400ms' }, agent_id: 'research-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-103', event_type: 'retry',     status: 'success', latency_ms: 180.0, parent_id: 'ev-102', input_data: { retry_count: 1 }, output_data: { status: 'success on retry' }, agent_id: 'research-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-104', event_type: 'final_response', status: 'success', latency_ms: 99.2, parent_id: 'ev-103', input_data: { context: '...' }, output_data: { response: 'Partial answer' }, agent_id: 'research-agent', timestamp: new Date().toISOString() },
    ]
  },
  full_failure: {
    trace_id: 'fail9999-0000-0000-0000-000000000000',
    response: '',
    total_latency_ms: 1203.5,
    status: 'failure',
    events: [
      { event_id: 'ev-200', event_type: 'llm_call', status: 'success', latency_ms: 300.0, parent_id: null, input_data: { prompt: 'Research topic' }, output_data: { text: 'Planning...' }, agent_id: 'fail-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-201', event_type: 'tool_call', status: 'failure', latency_ms: 550.0, parent_id: 'ev-200', input_data: { tool: 'db_query', query: 'SELECT * FROM data' }, output_data: { error: 'Database connection refused' }, agent_id: 'fail-agent', timestamp: new Date().toISOString() },
      { event_id: 'ev-202', event_type: 'error',    status: 'failure', latency_ms: 2.5,   parent_id: 'ev-201', input_data: { context: 'db failure' }, output_data: { error: 'Unrecoverable: cascading failure in data pipeline' }, agent_id: 'fail-agent', timestamp: new Date().toISOString() },
    ]
  },
  backend_error: null  // special: simulates HTTP error
};

// ── Mock Mode Toggle ─────────────────────────────────────────
mockModeBtn.addEventListener('click', () => {
  isMockMode = !isMockMode;
  mockLabel.textContent = isMockMode ? 'Mock Mode' : 'Live Mode';
  mockModeBtn.classList.toggle('active', isMockMode);
  mockPanel.classList.toggle('hidden', !isMockMode);
});

document.querySelectorAll('.mock-scenario-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const scenario = btn.dataset.scenario;
    if (isLoading) return;

    hideError();
    const stepTimer = animateLoadingSteps();
    setLoading(true);

    await new Promise(r => setTimeout(r, 1400)); // simulated delay

    clearInterval(stepTimer);
    setLoading(false);

    if (scenario === 'backend_error') {
      showError('HTTP 500 Internal Server Error — POST /api/run returned a server error. Check that the FastAPI backend is running and healthy.');
      return;
    }

    const data = MOCK_DATA[scenario];
    if (data) renderResults(data);
  });
});

// ── Live Run ─────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }
  if (isLoading) return;

  hideError();
  const stepTimer = animateLoadingSteps();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE_URL}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    clearInterval(stepTimer);
    setLoading(false);

    if (!res.ok) {
      let detail = `HTTP ${res.status} ${res.statusText}`;
      try {
        const errBody = await res.json();
        detail += ` — ${errBody.detail || JSON.stringify(errBody)}`;
      } catch {}
      showError(detail);
      return;
    }

    const data = await res.json();

    // Treat a failed trace status as a visual error banner too
    if (data.status && (data.status === 'failure' || data.status === 'failed' || data.status === 'error')) {
      showError(`Agent trace completed with status: ${data.status}. See event timeline below for details.`);
    }

    renderResults(data);

  } catch (err) {
    clearInterval(stepTimer);
    setLoading(false);
    showError(`Network error: ${err.message}. Is the backend running at ${API_BASE_URL}?`);
  }
});

// ── Allow Ctrl+Enter to submit ────────────────────────────────
queryInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    runBtn.click();
  }
});

// ── Dismiss error ─────────────────────────────────────────────
dismissError.addEventListener('click', hideError);

// ── Boot ──────────────────────────────────────────────────────
checkApiHealth();
setInterval(checkApiHealth, 30_000);

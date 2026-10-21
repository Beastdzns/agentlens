/**
 * AgentLens – app.js  (feature/timeline_intf)
 *
 * Change API_BASE_URL to point at a different backend:
 */
const API_BASE_URL = (() => {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  const configured = new URLSearchParams(window.location.search).get('api');
  if (configured) return configured.replace(/\/+$/, '');
  if (window.location.pathname.startsWith('/app')) return window.location.origin;
  return 'http://localhost:8000';
})();

// ── Pure utility functions (exported via window for test.js) ──────────────

/**
 * Format a millisecond value for display.
 * @param {number|null|undefined} ms
 * @returns {string}
 */
function formatLatency(ms) {
  if (ms == null) return 'N/A';
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(1)} ms`;
}
window.formatLatency = formatLatency;

/**
 * Normalise a raw status string into one of the four canonical classes:
 * success | failure | partial | pending
 * @param {string|null|undefined} status
 * @returns {string}
 */
function statusClass(status) {
  const s = (status || '').toLowerCase();
  if (s === 'success' || s === 'completed') return 'success';
  if (s === 'failure' || s === 'failed' || s === 'error') return 'failure';
  if (s === 'partial') return 'partial';
  return 'pending';
}
window.statusClass = statusClass;

/**
 * Build an eventsById map and compute the hierarchy depth of every event
 * by walking up the parent chain.
 * @param {Array} events
 * @returns {Map<string, number>}  eventId -> depth (root = 0)
 */
function computeDepths(events) {
  const byId = new Map(events.map(e => [e.event_id, e]));
  const depthCache = new Map();

  function depth(id) {
    if (id == null) return 0;
    if (depthCache.has(id)) return depthCache.get(id);
    const ev = byId.get(id);
    if (!ev || ev.parent_id == null) {
      depthCache.set(id, 0);
      return 0;
    }
    const d = 1 + depth(ev.parent_id);
    depthCache.set(id, d);
    return d;
  }

  events.forEach(e => depth(e.event_id));
  return depthCache;
}
window.computeDepths = computeDepths;

/**
 * Extract the human-readable prompt from event input_data defensively.
 * @param {object|null} input_data
 * @returns {string|null}
 */
function extractPrompt(input_data) {
  if (!input_data) return null;
  return input_data.prompt ?? input_data.query ?? null;
}
window.extractPrompt = extractPrompt;

/**
 * Extract the human-readable response from event output_data defensively.
 * @param {object|null} output_data
 * @returns {string|null}
 */
function extractResponse(output_data) {
  if (!output_data) return null;
  return output_data.response ?? output_data.answer ?? output_data.text ?? null;
}
window.extractResponse = extractResponse;

/**
 * Sort events chronologically by timestamp (safe: falls back to original order).
 * @param {Array} events
 * @returns {Array} new sorted array (original untouched)
 */
function sortEvents(events) {
  return [...events].sort((a, b) => {
    const ta = a.timestamp ? new Date(a.timestamp) : 0;
    const tb = b.timestamp ? new Date(b.timestamp) : 0;
    return ta - tb;
  });
}
window.sortEvents = sortEvents;

/**
 * Filter events by the active filter key.
 * Failed events are ALWAYS shown (bypassed by 'success'/'partial' filters) –
 * they must never disappear.
 * @param {Array} events
 * @param {string} filter  'all' | 'success' | 'failure' | 'partial'
 * @returns {Array}
 */
function filterEvents(events, filter) {
  if (filter === 'all') return events;
  return events.filter(ev => {
    const sc = statusClass(ev.status);
    // Failed events are always visible
    if (sc === 'failure') return true;
    if (filter === 'success') return sc === 'success';
    if (filter === 'failure') return sc === 'failure';
    if (filter === 'partial') return sc === 'partial';
    return true;
  });
}
window.filterEvents = filterEvents;

// ── State ─────────────────────────────────────────────────────────────────
let isLoading    = false;
let isMockMode   = false;
let activeFilter = 'all';
let lastData     = null;   // full API response
let sortedEvents = [];     // chronologically sorted events from lastData

// ── DOM refs ──────────────────────────────────────────────────────────────
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

const statTraceId    = document.getElementById('statTraceId');
const statLatency    = document.getElementById('statLatency');
const statEventCount = document.getElementById('statEventCount');
const statStatus     = document.getElementById('statStatus');
const responseText   = document.getElementById('responseText');
const eventList      = document.getElementById('eventList');
const filterBtns     = document.querySelectorAll('.filter-btn');

const traceMeta      = document.getElementById('traceMeta');
const failureSummary = document.getElementById('failureSummary');
const metaTraceId    = document.getElementById('metaTraceId');
const metaStatus     = document.getElementById('metaStatus');
const metaLatency    = document.getElementById('metaLatency');
const metaEventCount = document.getElementById('metaEventCount');

// ── Helpers ───────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function jsonPretty(obj) {
  if (obj == null) return 'null';
  if (typeof obj === 'string') return obj;
  return JSON.stringify(obj, null, 2);
}

function shortId(id) {
  if (!id) return '—';
  const s = String(id);
  return s.length > 16 ? s.slice(0, 8) + '…' + s.slice(-4) : s;
}

function badgeHTML(status) {
  const cls = statusClass(status);
  return `<span class="badge badge-${cls}">${escHtml(status || 'unknown')}</span>`;
}

function isLlmEvent(ev) {
  const t = (ev.event_type || '').toLowerCase();
  return t === 'llm_call' || t === 'final_response' || t === 'decision';
}

// ── API health check ──────────────────────────────────────────────────────
async function checkApiHealth() {
  try {
    const r = await fetch(API_BASE_URL + '/health', { signal: AbortSignal.timeout(3000) });
    if (r.ok) { apiStatus.className = 'api-status online'; apiStatusText.textContent = 'API Online'; return; }
  } catch {}
  try {
    await fetch(API_BASE_URL + '/', { signal: AbortSignal.timeout(3000) });
    apiStatus.className = 'api-status online'; apiStatusText.textContent = 'API Online';
  } catch {
    apiStatus.className = 'api-status offline'; apiStatusText.textContent = 'API Offline';
  }
}

// ── Loading state ─────────────────────────────────────────────────────────
function setLoading(v) {
  isLoading = v;
  queryInput.disabled = v;
  runBtn.disabled = v;
  if (v) {
    runIcon.innerHTML = '<circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>';
    runLabel.textContent = 'Running…';
    loadingState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsView.classList.add('hidden');
    errorBanner.classList.add('hidden');
  } else {
    runIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    runLabel.textContent = 'Run Agent';
    loadingState.classList.add('hidden');
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.classList.remove('hidden');
  emptyState.classList.add('hidden');
}

function hideError() { errorBanner.classList.add('hidden'); }

function animateLoadingSteps() {
  const steps = ['Initializing trace…','Sending query to agent…','Agent reasoning (LLM call)…','Executing tool calls…','Collecting telemetry…','Finalizing trace…'];
  let i = 0;
  loadingStep.textContent = steps[0];
  return setInterval(() => { i = (i + 1) % steps.length; loadingStep.textContent = steps[i]; }, 1500);
}

// ── Render full results ───────────────────────────────────────────────────
/**
 * Decide the default filter tab for a trace.
 *
 * Returns `'failure'` when the trace contains any failed events (either via
 * the API's `failure_count` summary or by scanning the event statuses),
 * otherwise `'all'`.  Pure helper — kept separate so it can be unit-tested.
 *
 * @param {Object} data    Trace response payload (may contain failure_count).
 * @param {Array}  events  Events belonging to the trace.
 * @returns {'failure'|'all'}
 */
function defaultFilterFor(data, events) {
  const hasFailures = (data && data.failure_count > 0) ||
    (events || []).some(e => statusClass(e.status) === 'failure');
  return hasFailures ? 'failure' : 'all';
}
window.defaultFilterFor = defaultFilterFor;

function renderResults(data) {
  lastData     = data;
  sortedEvents = sortEvents(data.events || []);

  // Stat cards
  statTraceId.textContent    = data.trace_id || 'N/A';
  statLatency.textContent    = formatLatency(data.total_latency_ms);
  statEventCount.textContent = (data.events || []).length;
  statStatus.innerHTML       = badgeHTML(data.status);

  // Agent response (multi-field fallback)
  const resp = data.response
    || (sortedEvents.length ? extractResponse(sortedEvents.slice().reverse().find(e => e.event_type === 'final_response')?.output_data) : null)
    || '';
  responseText.textContent = resp || '(no response)';

  // Sidebar meta
  metaTraceId.textContent    = shortId(data.trace_id);
  metaStatus.innerHTML       = badgeHTML(data.status);
  metaLatency.textContent    = formatLatency(data.total_latency_ms);
  metaEventCount.textContent = sortedEvents.length + ' events';
  traceMeta.classList.remove('hidden');


  // -- Smart filter: default to 'failure' tab when trace has failures
  const defaultFilter = defaultFilterFor(data, sortedEvents);
  if (activeFilter !== defaultFilter) {
    activeFilter = defaultFilter;
    filterBtns.forEach(b => {
      b.classList.toggle('active', b.dataset.filter === activeFilter);
    });
  }

  // -- Failure summary banner (shows above timeline for failed traces)
  renderFailureSummary(data, sortedEvents);

  renderEventList();

  resultsView.classList.remove('hidden');
  emptyState.classList.add('hidden');
}

// ── Render failure summary banner ────────────────────────────────────────
/**
 * Builds and shows the red diagnostic strip above the event timeline.
 * Only visible when the trace contains at least one failed event.
 * Uses data.failure_event_ids when provided; falls back to scanning events.
 */
function renderFailureSummary(data, events) {
  const eventsById = new Map(events.map(e => [e.event_id, e]));

  // Resolve the set of failed events to highlight in the banner
  let failedEvents = [];
  if (Array.isArray(data.failure_event_ids) && data.failure_event_ids.length) {
    failedEvents = data.failure_event_ids
      .map(id => eventsById.get(id))
      .filter(Boolean);
  }
  // Fall back to scanning all events when the field isn't present
  if (failedEvents.length === 0) {
    failedEvents = events.filter(e => statusClass(e.status) === 'failure');
  }

  if (failedEvents.length === 0) {
    failureSummary.classList.add('hidden');
    failureSummary.innerHTML = '';
    return;
  }

  const traceStatusCls = statusClass(data.status);
  const rows = failedEvents.map(ev => {
    const typeLabel = (ev.event_type || 'unknown').replace(/_/g, ' ').toUpperCase();
    const errMsg    = ev.output_data?.error ?? ev.output_data?.message
                      ?? ev.metadata?.exception_message ?? '(no error message)';
    const excType   = ev.metadata?.exception_type ?? null;
    const parentEv  = ev.parent_id ? eventsById.get(ev.parent_id) : null;
    const parentLbl = parentEv
      ? (parentEv.event_type || 'unknown').replace(/_/g, ' ').toUpperCase()
      : null;

    return `
      <div class="fs-row">
        <div class="fs-step-badge">${escHtml(typeLabel)}</div>
        <div class="fs-detail">
          <div class="fs-error">${escHtml(String(errMsg))}${excType ? ` <span class="fs-exc">(${escHtml(excType)})</span>` : ''}</div>
          ${parentLbl ? `<div class="fs-crumb">↳ triggered inside <span class="fs-crumb-type">${escHtml(parentLbl)}</span> · parent <code class="fs-crumb-id">${escHtml(shortId(ev.parent_id))}</code></div>` : ''}
        </div>
        <div class="fs-id"><code>${escHtml(shortId(ev.event_id))}</code></div>
      </div>`;
  }).join('');

  failureSummary.innerHTML = `
    <div class="fs-header">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>
        <line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <span class="fs-header-text">
        Trace <span class="fs-status-word">${escHtml(data.status || 'failed')}</span>
        &mdash; ${failedEvents.length} failed step${failedEvents.length > 1 ? 's' : ''} detected
      </span>
      <button class="fs-all-btn filter-btn" data-filter="all"
              onclick="document.querySelector('.filter-btn[data-filter=all]').click()">
        Show full trace
      </button>
    </div>
    <div class="fs-rows">${rows}</div>`;
  failureSummary.classList.remove('hidden');
}
window.renderFailureSummary = renderFailureSummary;

// ── Adaptive observability: observation level + importance ────────────────
// Observation levels (Phase 10): FULL / COMPRESSED / METADATA_ONLY / DROP.
// Events without a level (e.g. traces recorded before Phase 10) are FULL.

const OBS_LEVEL_LABELS = {
  full: 'FULL',
  compressed: 'COMP',
  metadata_only: 'META',
  drop: 'DROP',
};

function buildObservationBadge(ev) {
  const level = String(ev.observation_level || 'full').toLowerCase();
  const label = OBS_LEVEL_LABELS[level] || level.toUpperCase();
  return `<span class="obs-level-badge obs-level-${level}" title="Observation level: ${level}">${label}</span>`;
}

function buildImportanceBar(ev) {
  const score = typeof ev.importance_score === 'number' ? ev.importance_score : null;
  if (score === null) return '';
  const pct = Math.max(0, Math.min(100, score * 100)).toFixed(0);
  return `<span class="importance-bar-wrap" title="Importance score: ${score.toFixed(2)}">
           <span class="importance-bar-track"><span class="importance-bar-fill" style="width:${pct}%"></span></span>
           <span class="importance-bar-label">${score.toFixed(2)}</span>
         </span>`;
}

window.buildObservationBadge = buildObservationBadge;
window.buildImportanceBar = buildImportanceBar;

// ── Render the event timeline ─────────────────────────────────────────────
function renderEventList() {
  const events   = filterEvents(sortedEvents, activeFilter);
  const depthMap = computeDepths(sortedEvents);  // always compute on full set

  if (events.length === 0) {
    eventList.innerHTML = '<p class="tl-empty">No events match this filter.</p>';
    return;
  }

  eventList.innerHTML = events.map((ev, idx) => {
    const sc       = statusClass(ev.status);
    const depth    = Math.min(depthMap.get(ev.event_id) ?? 0, 6); // cap at 6
    const isFailed = sc === 'failure';
    const isLlm    = isLlmEvent(ev);
    const typeStr  = (ev.event_type || 'unknown').replace(/_/g, ' ');
    const lat      = ev.latency_ms ?? ev.latency ?? null;

    // Indent connector lines
    const connectors = depth > 0
      ? `<div class="tl-connectors">${'<div class="tl-connector"></div>'.repeat(depth)}</div>`
      : '';

    // Failure details block
    const failureBlock = isFailed ? buildFailureBlock(ev) : '';

    // Data display section
    const dataSection = isLlm ? buildLlmBlock(ev) : buildJsonBlock(ev);

    // Observation level badge + importance score bar
    const obsLevelBadge = buildObservationBadge(ev);
    const importanceBar = buildImportanceBar(ev);

    return `
      <div class="tl-row" style="--depth:${depth}" id="tl-row-${idx}">
        ${connectors}
        <div class="tl-card event-card status-${sc}${isFailed ? ' is-failure' : ''}" id="ev-card-${idx}">
          <div class="event-header" onclick="toggleEvent(${idx})">
            <span class="event-index">#${String(idx + 1).padStart(2, '0')}</span>
            <span class="event-type-badge event-type-${ev.event_type}">${escHtml(typeStr)}</span>
            <span class="event-id-cell" title="${escHtml(ev.event_id || '')}">
              ${escHtml(shortId(ev.event_id))}
            </span>
            <span class="event-latency">${escHtml(formatLatency(lat))}</span>
            ${ev.timestamp ? `<span class="event-timestamp">${escHtml(new Date(ev.timestamp).toLocaleTimeString())}</span>` : ''}
            <span class="event-status-badge">${badgeHTML(ev.status)}</span>
            ${obsLevelBadge}
            ${importanceBar}
            <svg class="event-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
          </div>

          <div class="event-body">
            ${failureBlock}

            <div class="event-meta-row">
              ${ev.event_id  ? `<div class="event-meta-pair"><span class="event-meta-label">Event ID</span><span class="event-meta-value">${escHtml(ev.event_id)}</span></div>` : ''}
              ${ev.parent_id ? `<div class="event-meta-pair"><span class="event-meta-label">Parent ID</span><span class="event-meta-value">${escHtml(ev.parent_id)}</span></div>` : ''}
              ${ev.agent_id  ? `<div class="event-meta-pair"><span class="event-meta-label">Agent</span><span class="event-meta-value">${escHtml(ev.agent_id)}</span></div>` : ''}
              ${ev.timestamp ? `<div class="event-meta-pair"><span class="event-meta-label">Timestamp</span><span class="event-meta-value">${escHtml(ev.timestamp)}</span></div>` : ''}
            </div>

            ${dataSection}

            ${ev.metadata && Object.keys(ev.metadata).length
              ? `<div class="event-data-block">
                   <div class="event-data-label">Metadata</div>
                   <pre class="event-data-pre">${escHtml(jsonPretty(ev.metadata))}</pre>
                 </div>`
              : ''}
          </div>
        </div>
      </div>`;
  }).join('');
}

// ── Build failure highlight block ─────────────────────────────────────────
function buildFailureBlock(ev) {
  const err      = ev.output_data?.error ?? ev.output_data?.message ?? null;
  const excType  = ev.metadata?.exception_type ?? null;
  const excMsg   = ev.metadata?.exception_message ?? ev.metadata?.exc_message ?? null;
  const retryOf  = ev.metadata?.retry_of ?? null;
  const retriedBy = ev.metadata?.retried_by ?? null;

  // Breadcrumb: look up parent event type for context
  const eventsById = new Map((sortedEvents || []).map(e => [e.event_id, e]));
  const parentEv   = ev.parent_id ? eventsById.get(ev.parent_id) : null;
  const parentType = parentEv
    ? (parentEv.event_type || '').replace(/_/g, ' ').toUpperCase()
    : null;

  let html = '<div class="failure-highlight">';
  html += '<div class="failure-title"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> Failure Details</div>';

  if (err)     html += `<div class="failure-row"><span class="failure-key">Error</span><span class="failure-val">${escHtml(String(err))}</span></div>`;
  if (excType) html += `<div class="failure-row"><span class="failure-key">Exception</span><span class="failure-val">${escHtml(excType)}</span></div>`;
  if (excMsg)  html += `<div class="failure-row"><span class="failure-key">Message</span><span class="failure-val">${escHtml(excMsg)}</span></div>`;

  // Breadcrumb row
  if (parentType) {
    html += `<div class="failure-row">
      <span class="failure-key">Parent</span>
      <span class="failure-val">
        <span class="failure-breadcrumb">
          <span class="fb-type">${escHtml(parentType)}</span>
          <span class="fb-sep">·</span>
          <code class="fb-id">${escHtml(shortId(ev.parent_id))}</code>
        </span>
      </span>
    </div>`;
  } else if (ev.parent_id) {
    html += `<div class="failure-row"><span class="failure-key">Parent ID</span><span class="failure-val">${escHtml(ev.parent_id)}</span></div>`;
  }

  // Retry linkage badges
  if (retryOf) {
    html += `<div class="failure-row">
      <span class="failure-key">Retry of</span>
      <span class="failure-val"><span class="retry-badge retry-badge-of">↩ retry of <code>${escHtml(shortId(retryOf))}</code></span></span>
    </div>`;
  }
  if (retriedBy) {
    html += `<div class="failure-row">
      <span class="failure-key">Retried by</span>
      <span class="failure-val"><span class="retry-badge retry-badge-by">↻ retried by <code>${escHtml(shortId(retriedBy))}</code></span></span>
    </div>`;
  }

  html += '</div>';
  return html;
}
window.buildFailureBlock = buildFailureBlock;

// ── LLM-specific display block ────────────────────────────────────────────
function buildLlmBlock(ev) {
  const prompt   = extractPrompt(ev.input_data);
  const response = extractResponse(ev.output_data);
  let html = '';

  if (prompt != null) {
    html += `<div class="event-data-block llm-block">
      <div class="event-data-label llm-label llm-label-prompt">Prompt / Query</div>
      <div class="llm-text">${escHtml(String(prompt))}</div>
    </div>`;
  } else if (ev.input_data && Object.keys(ev.input_data).length) {
    html += `<div class="event-data-block">
      <div class="event-data-label">Input</div>
      <pre class="event-data-pre">${escHtml(jsonPretty(ev.input_data))}</pre>
    </div>`;
  }

  if (response != null) {
    html += `<div class="event-data-block llm-block">
      <div class="event-data-label llm-label llm-label-response">Response</div>
      <div class="llm-text">${escHtml(String(response))}</div>
    </div>`;
  } else if (ev.output_data && Object.keys(ev.output_data).length) {
    html += `<div class="event-data-block">
      <div class="event-data-label">Output</div>
      <pre class="event-data-pre">${escHtml(jsonPretty(ev.output_data))}</pre>
    </div>`;
  }

  return html;
}

// ── Generic JSON display block ────────────────────────────────────────────
function buildJsonBlock(ev) {
  let html = '';
  if (ev.input_data && Object.keys(ev.input_data).length) {
    html += `<div class="event-data-block">
      <div class="event-data-label">Input</div>
      <pre class="event-data-pre">${escHtml(jsonPretty(ev.input_data))}</pre>
    </div>`;
  }
  if (ev.output_data && Object.keys(ev.output_data).length) {
    html += `<div class="event-data-block">
      <div class="event-data-label">Output</div>
      <pre class="event-data-pre">${escHtml(jsonPretty(ev.output_data))}</pre>
    </div>`;
  }
  return html;
}

window.toggleEvent = function(idx) {
  const card = document.getElementById(`ev-card-${idx}`);
  if (card) card.classList.toggle('open');
};

// ── Filter bar ────────────────────────────────────────────────────────────
filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    if (lastData) renderEventList();
  });
});

// ── Mock data ─────────────────────────────────────────────────────────────
const T0 = new Date('2026-10-08T12:00:00Z');
const ts  = i => new Date(T0.getTime() + i * 1200).toISOString();

const MOCK_DATA = {
  success: {
    trace_id: 'aabbcc11-0000-0000-0000-112233445566',
    response: 'AI agents can be debugged through structured observability: trace every LLM call, tool invocation, and decision with unique IDs and parent links. Replay failed traces with deterministic seeds. Inspect input/output deltas at each step to localise the fault.',
    total_latency_ms: 342.7,
    status: 'success',
    events: [
      { event_id: 'ev-001', event_type: 'decision',  status: 'success', latency_ms: 25.0,  parent_id: null,    input_data: { query: 'How can AI agents be debugged?' }, output_data: { plan: 'search → retrieve → synthesise' }, agent_id: 'debug-agent', timestamp: ts(0), metadata: {} },
      { event_id: 'ev-002', event_type: 'tool_call', status: 'success', latency_ms: 55.1,  parent_id: 'ev-001', input_data: { tool: 'web_search', query: 'AI agent debugging techniques' }, output_data: { results: ['paper-A', 'paper-B'] }, agent_id: 'debug-agent', timestamp: ts(1), metadata: { tool: 'web_search' } },
      { event_id: 'ev-003', event_type: 'retrieval', status: 'success', latency_ms: 38.4,  parent_id: 'ev-002', input_data: { query: 'observability patterns', top_k: 3 }, output_data: { docs: ['doc-1', 'doc-2', 'doc-3'] }, agent_id: 'debug-agent', timestamp: ts(2), metadata: {} },
      { event_id: 'ev-004', event_type: 'memory_write', status: 'success', latency_ms: 4.0, parent_id: 'ev-001', input_data: { key: 'search_results' }, output_data: { written: true }, agent_id: 'debug-agent', timestamp: ts(3), metadata: {} },
      { event_id: 'ev-005', event_type: 'llm_call', status: 'success', latency_ms: 134.9, parent_id: null,    input_data: { prompt: 'Synthesise findings about AI agent debugging', plan: 'search → retrieve → synthesise' }, output_data: { response: 'Structured observability with parent-linked traces enables fast root-cause localisation.' }, agent_id: 'debug-agent', timestamp: ts(4), metadata: { model: 'gemini-2.5-flash', tokens: 820 } },
      { event_id: 'ev-006', event_type: 'final_response', status: 'success', latency_ms: 1.0, parent_id: 'ev-005', input_data: {}, output_data: { answer: 'AI agents can be debugged through structured observability: trace every LLM call, tool invocation, and decision with unique IDs and parent links. Replay failed traces with deterministic seeds.' }, agent_id: 'debug-agent', timestamp: ts(5), metadata: {} },
    ]
  },
  partial: {
    trace_id: 'ddee1122-0000-0000-0000-aabbccddeeff',
    failure_count: 1,
    failure_event_ids: ['ev-102'],
    response: 'Partial response generated — one tool call failed but LLM recovered.',
    total_latency_ms: 891.2,
    status: 'partial',
    events: [
      { event_id: 'ev-101', event_type: 'llm_call',  status: 'success', latency_ms: 212.0, parent_id: null,     input_data: { prompt: 'Research AI debugging methods' }, output_data: { text: 'Planning search strategy…' }, agent_id: 'research-agent', timestamp: ts(0), metadata: { model: 'gemini-2.5-flash' } },
      { event_id: 'ev-102', event_type: 'tool_call', status: 'failure', latency_ms: 400.0, parent_id: 'ev-101', input_data: { tool: 'api_call', endpoint: '/external' }, output_data: { error: 'Connection timed out after 400 ms' }, agent_id: 'research-agent', timestamp: ts(1), metadata: { exception_type: 'TimeoutError', exception_message: 'Read timed out (connect=5, read=5)' } },
      { event_id: 'ev-103', event_type: 'retry',     status: 'success', latency_ms: 180.0, parent_id: 'ev-102', input_data: { retry_count: 1 }, output_data: { status: 'success on retry' }, agent_id: 'research-agent', timestamp: ts(2), metadata: { retry_of: 'ev-102' } },
      { event_id: 'ev-104', event_type: 'final_response', status: 'success', latency_ms: 99.2, parent_id: 'ev-103', input_data: {}, output_data: { answer: 'Partial answer retrieved after retry.' }, agent_id: 'research-agent', timestamp: ts(3), metadata: { retry_of: 'ev-102' } },
    ]
  },
  full_failure: {
    trace_id: 'fail9999-0000-0000-0000-000000000000',
    failure_count: 2,
    failure_event_ids: ['ev-201', 'ev-202'],
    response: '',
    total_latency_ms: 1203.5,
    status: 'failure',
    events: [
      { event_id: 'ev-200', event_type: 'llm_call', status: 'success', latency_ms: 300.0, parent_id: null,     input_data: { prompt: 'Retrieve metrics from DB and summarise' }, output_data: { text: 'Planning DB query…' }, agent_id: 'fail-agent', timestamp: ts(0), metadata: {} },
      { event_id: 'ev-201', event_type: 'tool_call', status: 'failure', latency_ms: 550.0, parent_id: 'ev-200', input_data: { tool: 'db_query', query: 'SELECT * FROM metrics' }, output_data: { error: 'Database connection refused' }, agent_id: 'fail-agent', timestamp: ts(1), metadata: { exception_type: 'ConnectionRefusedError', exception_message: 'Connection refused on port 5432' } },
      { event_id: 'ev-202', event_type: 'error', status: 'failure', latency_ms: 2.5, parent_id: 'ev-201', input_data: { context: 'db failure' }, output_data: { error: 'Unrecoverable: cascading failure in data pipeline' }, agent_id: 'fail-agent', timestamp: ts(2), metadata: { exception_type: 'RuntimeError' } },
    ]
  },
  backend_error: null
};

// ── Mock Mode toggle ──────────────────────────────────────────────────────
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
    await new Promise(r => setTimeout(r, 1200));
    clearInterval(stepTimer);
    setLoading(false);
    if (scenario === 'backend_error') {
      showError('HTTP 500 Internal Server Error — POST /api/run returned a server error. Ensure the FastAPI backend is running and healthy.');
      return;
    }
    const data = MOCK_DATA[scenario];
    if (data) renderResults(data);
  });
});

// ── Live run ──────────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  const query = queryInput.value.trim();
  if (!query) { queryInput.focus(); return; }
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
      try { const b = await res.json(); detail += ` — ${b.detail || JSON.stringify(b)}`; } catch {}
      showError(detail);
      return;
    }

    const data = await res.json();
    if (statusClass(data.status) === 'failure') {
      showError(`Agent trace completed with status: ${data.status}. See the failure events highlighted below.`);
    }
    renderResults(data);

  } catch (err) {
    clearInterval(stepTimer);
    setLoading(false);
    showError(`Network error: ${err.message}. Is the backend running at ${API_BASE_URL}?`);
  }
});

queryInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runBtn.click();
});

dismissError.addEventListener('click', hideError);

// ── Boot ──────────────────────────────────────────────────────────────────
checkApiHealth();
setInterval(checkApiHealth, 30_000);

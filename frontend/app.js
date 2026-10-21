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
let lastCausalGraph = null;   // graph payload of the last rendered panel
let causalSelectedId = null;  // event id currently selected in the graph

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

const causalSection  = document.getElementById('causalSection');
const causalPanel    = document.getElementById('causalPanel');
const causalFullscreenBtn = document.getElementById('causalFullscreenBtn');

const cfsOverlay     = document.getElementById('cfsOverlay');
const cfsClose       = document.getElementById('cfsClose');
const cfsContent     = document.getElementById('cfsContent');

let cfsActive = false;

const runFailure     = document.getElementById('runFailure');

const historySection = document.getElementById('historySection');
const historyRefresh = document.getElementById('historyRefresh');
const historyStatus  = document.getElementById('historyStatus');
const historyFrom    = document.getElementById('historyFrom');
const historyTo      = document.getElementById('historyTo');
const historyMeta    = document.getElementById('historyMeta');
const historyList    = document.getElementById('historyList');
const historyMore    = document.getElementById('historyMore');

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

/**
 * Minimal markdown-to-HTML renderer (no dependencies).
 *
 * Handles: fenced code blocks, headers (h1-h6), blockquotes, horizontal
 * rules, ordered / unordered lists, links, inline code, bold, italic and
 * paragraphs.  All user text is HTML-escaped so the output is safe to inject
 * with innerHTML.
 *
 * @param {string|null|undefined} text
 * @returns {string} HTML markup
 */
function renderMarkdown(text) {
  if (text == null || String(text).trim() === '') {
    return '<p class="md-empty">(no response)</p>';
  }

  let s = String(text);

  // --- Protect fenced code blocks -------------------------------------------
  const codeBlocks = [];
  s = s.replace(/```([\s\S]*?)```/g, (_m, code) => {
    codeBlocks.push(escHtml(code.trim()));
    return `\x00CB${codeBlocks.length - 1}\x00`;
  });

  // --- Structural pass -------------------------------------------------------
  const lines = s.split('\n');
  const out = [];
  let inList = false;
  let listType = null;

  const flushList = () => {
    if (inList) {
      out.push(listType === 'ul' ? '</ul>' : '</ol>');
      inList = false;
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(trimmed)) {
      flushList();
      out.push('<hr class="md-hr">');
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      flushList();
      out.push(`<blockquote class="md-quote">${renderInline(trimmed.slice(2))}</blockquote>`);
      continue;
    }

    // Blank line
    if (trimmed === '') {
      flushList();
      out.push('');
      continue;
    }

    // Header h1-h6
    const h = trimmed.match(/^(#{1,6})\s(.*)/);
    if (h) {
      flushList();
      out.push(`<h${h[1].length} class="md-h${h[1].length}">${renderInline(h[2])}</h${h[1].length}>`);
      continue;
    }

    // Unordered list
    const ul = trimmed.match(/^([*+-])\s(.*)/);
    if (ul) {
      if (!inList || listType !== 'ul') {
        flushList();
        out.push('<ul class="md-ul">');
        inList = true;
        listType = 'ul';
      }
      out.push(`<li class="md-li">${renderInline(ul[2])}</li>`);
      continue;
    }

    // Ordered list
    const ol = trimmed.match(/^(\d+)\.\s(.*)/);
    if (ol) {
      if (!inList || listType !== 'ol') {
        flushList();
        out.push('<ol class="md-ol">');
        inList = true;
        listType = 'ol';
      }
      out.push(`<li class="md-li">${renderInline(ol[2])}</li>`);
      continue;
    }

    // Paragraph
    flushList();
    out.push(`<p class="md-p">${renderInline(trimmed)}</p>`);
  }
  flushList();

  // --- Restore protected code blocks -----------------------------------------
  return out.join('\n').replace(/\x00CB(\d+)\x00/g, (_m, i) => {
    return `<pre class="md-code"><code>${codeBlocks[+i]}</code></pre>`;
  });
}

/**
 * Inline markdown: links, code, bold, italic.  Uses placeholder tokens so
 * that generated HTML tags are not re-escaped by escHtml.
 * @param {string} text  raw (unescaped) inline text
 * @returns {string} HTML with safe text and protected tags
 */
function renderInline(text) {
  const protectedSpans = [];
  const protect = (html) => {
    protectedSpans.push(html);
    return `\x01PROT${protectedSpans.length - 1}\x01`;
  };

  // Links: [label](url)
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, label, url) => {
    return protect(`<a href="${escHtml(url)}" target="_blank" rel="noopener">${escHtml(label)}</a>`);
  });

  // Inline code: `code`
  text = text.replace(/`([^`]+)`/g, (_m, inner) => {
    return protect(`<code class="md-code-inline">${escHtml(inner)}</code>`);
  });

  // Bold: **text** / __text__
  text = text.replace(/\*\*(.+?)\*\*/g, (_m, inner) => {
    return protect(`<strong>${escHtml(inner)}</strong>`);
  });
  text = text.replace(/__(.+?)__/g, (_m, inner) => {
    return protect(`<strong>${escHtml(inner)}</strong>`);
  });

  // Italic: *text* / _text_
  text = text.replace(/\*(.+?)\*/g, (_m, inner) => {
    return protect(`<em>${escHtml(inner)}</em>`);
  });
  text = text.replace(/_(.+?)_/g, (_m, inner) => {
    return protect(`<em>${escHtml(inner)}</em>`);
  });

  // Escape everything that remains, then restore protected tags.
  text = escHtml(text);
  return text.replace(/\x01PROT(\d+)\x01/g, (_m, i) => protectedSpans[+i]);
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
  responseText.innerHTML = renderMarkdown(resp || '(no response)');

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

  // -- Live-run failure state (why the run failed + pointer to the analysis)
  renderRunFailure(data);

  // -- Causal analysis panel (Phase 11) — hidden when the trace has no graph
  renderCausalSection(data);

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

window.toggleEvent = function(idx, forceOpen) {
  const card = document.getElementById(`ev-card-${idx}`);
  if (card) card.classList.toggle('open', forceOpen === undefined ? !card.classList.contains('open') : forceOpen);
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

// ── Causal analysis panel (Phase 11) ───────────────────────────────────────
/**
 * Renders the causal graph of a trace and the root-cause diagnosis on top of
 * it.
 *
 * The layout is a deterministic layered graph: every node is placed in the
 * column of its parent (`parent_of` edges), and the rows inside a column are
 * ordered by (timestamp, event_id).  The same graph therefore always draws
 * the same picture — no physics, no randomness, no new dependency.
 */

const CAUSAL_LAYOUT = {
  nodeWidth:  210,
  nodeHeight:  54,
  layerGap:    92,
  rowGap:      18,
  padding:     28,
};

const CAUSAL_EDGE_CLASS = {
  parent_of:  'cg-edge-parent',
  retry_of:   'cg-edge-retry',
  depends_on: 'cg-edge-depends',
  influenced: 'cg-edge-influenced',
};

const CAUSAL_VERDICT_LABEL = {
  root_cause:            'Root cause identified',
  no_failure:            'No failure recorded',
  recovered:             'Failure recovered',
  insufficient_evidence: 'Insufficient evidence',
};

let _causalSvgSeq = 0;

function causalTruncate(text, max) {
  const s = String(text == null ? '' : text);
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

/**
 * Assign every node of the graph to a column (layer) and a row.
 *
 * @param {object} graph  the `causal_graph` payload of the API
 * @returns {object}      {positions, byId, edges, width, height, layers}
 */
function layoutCausalGraph(graph) {
  const nodes = (graph && Array.isArray(graph.nodes)) ? graph.nodes.slice() : [];
  const edges = (graph && Array.isArray(graph.edges))
    ? graph.edges.filter(e => e && e.source != null && e.target != null)
    : [];

  const byId = new Map();
  nodes.forEach(n => byId.set(String(n.event_id), n));

  // The column of a node is decided by its parent.  `parent_id` is the
  // authoritative link; the `parent_of` edge is the fallback for payloads
  // that only carry the edges.  A parent that is not part of the graph is
  // ignored — an unresolvable reference must not invent a relation.
  const parentOf = new Map();
  nodes.forEach(n => parentOf.set(String(n.event_id), null));
  edges.forEach(e => {
    if (e.relation === 'parent_of' && byId.has(String(e.target))) {
      parentOf.set(String(e.target), String(e.source));
    }
  });
  nodes.forEach(n => {
    const declared = n.parent_id == null ? null : String(n.parent_id);
    if (declared != null && byId.has(declared)) {
      parentOf.set(String(n.event_id), declared);
    }
  });

  // Layer of a node = its distance to a root.  Memoised and cycle-safe.
  const layers = new Map();
  const resolveLayer = (id) => {
    if (layers.has(id)) return layers.get(id);
    const chain = [];
    const seen = new Set();
    let current = id;
    while (current != null && !layers.has(current) && !seen.has(current)) {
      seen.add(current);
      chain.push(current);
      current = parentOf.has(current) ? parentOf.get(current) : null;
    }
    let base = layers.has(current) ? layers.get(current) + 1 : 0;
    for (let i = chain.length - 1; i >= 0; i--) {
      layers.set(chain[i], base);
      base += 1;
    }
    return layers.get(id) || 0;
  };

  // Group the nodes by layer, ordered deterministically inside a layer.
  const columns = new Map();
  nodes.forEach(n => {
    const layer = resolveLayer(String(n.event_id));
    if (!columns.has(layer)) columns.set(layer, []);
    columns.get(layer).push(n);
  });
  columns.forEach(list => list.sort((a, b) => {
    const ta = String(a.timestamp || '');
    const tb = String(b.timestamp || '');
    if (ta !== tb) return ta < tb ? -1 : 1;
    const ia = String(a.event_id);
    const ib = String(b.event_id);
    return ia < ib ? -1 : (ia > ib ? 1 : 0);
  }));

  const layerKeys = Array.from(columns.keys()).sort((a, b) => a - b);
  const maxRows = layerKeys.reduce((max, k) => Math.max(max, columns.get(k).length), 0);
  const contentHeight = maxRows * CAUSAL_LAYOUT.nodeHeight
                     + Math.max(0, maxRows - 1) * CAUSAL_LAYOUT.rowGap;
  const width = layerKeys.length * CAUSAL_LAYOUT.nodeWidth
              + Math.max(0, layerKeys.length - 1) * CAUSAL_LAYOUT.layerGap
              + CAUSAL_LAYOUT.padding * 2;
  const height = contentHeight + CAUSAL_LAYOUT.padding * 2;

  const positions = [];
  const positionById = new Map();
  layerKeys.forEach((layer, column) => {
    const list = columns.get(layer);
    const columnHeight = list.length * CAUSAL_LAYOUT.nodeHeight
                       + Math.max(0, list.length - 1) * CAUSAL_LAYOUT.rowGap;
    const yOffset = CAUSAL_LAYOUT.padding + (contentHeight - columnHeight) / 2;
    list.forEach((node, row) => {
      const pos = {
        id: String(node.event_id),
        x: CAUSAL_LAYOUT.padding
           + column * (CAUSAL_LAYOUT.nodeWidth + CAUSAL_LAYOUT.layerGap),
        y: yOffset + row * (CAUSAL_LAYOUT.nodeHeight + CAUSAL_LAYOUT.rowGap),
        node,
        layer,
        row,
      };
      positions.push(pos);
      positionById.set(pos.id, pos);
    });
  });

  return { positions, byId: positionById, edges, width, height, layers: layerKeys.length };
}

/**
 * Draw the causal graph as an SVG.
 *
 * @param {object} graph      the `causal_graph` payload
 * @param {object} rootCause  the `root_cause` payload (may be null)
 * @returns {string}          the SVG markup (empty when there is no graph)
 */
function renderCausalGraph(graph, rootCause) {
  const layout = layoutCausalGraph(graph);
  if (!layout.positions.length) return '';

  const rootId = rootCause && rootCause.root_cause_event_id
    ? String(rootCause.root_cause_event_id)
    : null;
  const evidence = new Set(((rootCause && rootCause.evidence_path) || []).map(String));
  const affected = new Set(((rootCause && rootCause.affected_events) || []).map(String));

  const uid = `cg${++_causalSvgSeq}`;
  const parts = [];

  parts.push(
    `<svg class="causal-svg" viewBox="0 0 ${layout.width} ${layout.height}" ` +
    `preserveAspectRatio="xMinYMin meet" role="img" ` +
    `aria-label="Causal graph of the trace" data-testid="causal-svg">`
  );
  parts.push(
    '<defs>' +
    `<marker id="${uid}-arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" ` +
    'orient="auto"><path d="M0,0 L0,6 L9,3 z" class="cg-arrow-head" /></marker>' +
    `<marker id="${uid}-arrow-hot" markerWidth="9" markerHeight="9" refX="8" refY="3" ` +
    'orient="auto"><path d="M0,0 L0,6 L9,3 z" class="cg-arrow-head cg-arrow-head-hot" /></marker>' +
    '</defs>'
  );

  // -- edges ---------------------------------------------------------------
  parts.push('<g class="cg-edges">');
  layout.edges.forEach(edge => {
    const from = layout.byId.get(String(edge.source));
    const to = layout.byId.get(String(edge.target));
    if (!from || !to) return;
    const x1 = from.x + CAUSAL_LAYOUT.nodeWidth;
    const y1 = from.y + CAUSAL_LAYOUT.nodeHeight / 2;
    const x2 = to.x;
    const y2 = to.y + CAUSAL_LAYOUT.nodeHeight / 2;
    const mid = (x1 + x2) / 2;
    const d = `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
    const hot = evidence.has(String(edge.source)) && evidence.has(String(edge.target));
    const cls = CAUSAL_EDGE_CLASS[edge.relation] || 'cg-edge-parent';
    parts.push(
      `<path class="cg-edge ${cls}${hot ? ' cg-edge-hot' : ''}" d="${d}" ` +
      `marker-end="url(#${uid}-arrow${hot ? '-hot' : ''})" ` +
      `data-relation="${escHtml(edge.relation || '')}" ` +
      `data-source="${escHtml(String(edge.source))}" ` +
      `data-target="${escHtml(String(edge.target))}" />`
    );
  });
  parts.push('</g>');

  // -- nodes ---------------------------------------------------------------
  parts.push('<g class="cg-nodes">');
  layout.positions.forEach(pos => {
    const node = pos.node || {};
    const isRoot = rootId != null && pos.id === rootId;
    const isFailure = node.is_failure === true || statusClass(node.status) === 'failure';
    const onEvidence = evidence.has(pos.id);
    const isAffected = affected.has(pos.id);
    const classes = ['cg-node'];
    if (isFailure) classes.push('cg-node-failure');
    if (isRoot) classes.push('cg-node-root');
    if (onEvidence) classes.push('cg-node-evidence');
    if (isAffected) classes.push('cg-node-affected');

    const label = causalTruncate(node.label || node.event_type || node.event_id, 30);
    const type = causalTruncate(node.event_type || '', 22);

    parts.push(
      `<g class="${classes.join(' ')}" transform="translate(${pos.x},${pos.y})" ` +
      `data-event-id="${escHtml(pos.id)}" data-root-cause="${isRoot ? 'true' : 'false'}">`
    );
    parts.push(
      `<rect class="cg-node-box" width="${CAUSAL_LAYOUT.nodeWidth}" ` +
      `height="${CAUSAL_LAYOUT.nodeHeight}" rx="9" ry="9" />`
    );
    parts.push(
      `<circle class="cg-node-dot" cx="14" cy="${CAUSAL_LAYOUT.nodeHeight / 2}" r="5" />`
    );
    parts.push(
      `<text class="cg-node-label" x="28" y="22">${escHtml(label)}</text>`
    );
    parts.push(
      `<text class="cg-node-type" x="28" y="39">${escHtml(type)}</text>`
    );
    if (isRoot) {
      parts.push(
        `<text class="cg-node-tag" x="${CAUSAL_LAYOUT.nodeWidth - 10}" y="16" ` +
        `text-anchor="end">ROOT CAUSE</text>`
      );
    }
    parts.push('</g>');
  });
  parts.push('</g>');
  parts.push('</svg>');

  return parts.join('');
}

/**
 * Render the root-cause diagnosis.
 *
 * @param {object} rootCause  the `root_cause` payload (may be null)
 * @param {object} graph      the `causal_graph` payload (used for the labels)
 * @returns {string}          the markup (empty when there is no diagnosis)
 */
function renderRootCausePanel(rootCause, graph) {
  if (!rootCause) return '';
  const verdict = String(rootCause.verdict || 'no_failure');
  const label = CAUSAL_VERDICT_LABEL[verdict] || verdict;
  const confidence = typeof rootCause.confidence === 'number' ? rootCause.confidence : 0;

  const nodesById = new Map(((graph && graph.nodes) || []).map(n => [String(n.event_id), n]));
  const pathText = (rootCause.evidence_path || []).map(id => {
    const n = nodesById.get(String(id));
    return n ? (n.label || n.event_type || id) : String(id);
  }).join(' → ');

  const parts = [];
  parts.push(`<div class="causal-diagnosis causal-verdict-${escHtml(verdict)}">`);
  parts.push(
    '<div class="cd-head">' +
    `<span class="cd-verdict">${escHtml(label)}</span>` +
    `<span class="cd-confidence" title="Confidence">${confidence.toFixed(2)}</span>` +
    `<span class="cd-category">${escHtml(rootCause.category || 'none')}</span>` +
    '</div>'
  );
  if (rootCause.explanation) {
    parts.push(`<p class="cd-explanation">${escHtml(rootCause.explanation)}</p>`);
  }
  if (pathText) {
    parts.push(`<div class="cd-path"><span class="cd-path-label">Evidence path</span>${escHtml(pathText)}</div>`);
  }
  const affectedCount = (rootCause.affected_events || []).length;
  const secondaryCount = (rootCause.secondary_failures || []).length;
  const recoveredCount = (rootCause.recovered_failures || []).length;
  parts.push(
    '<div class="cd-stats">' +
    `<span class="cd-stat"><b>${affectedCount}</b> affected</span>` +
    `<span class="cd-stat"><b>${secondaryCount}</b> secondary failures</span>` +
    `<span class="cd-stat"><b>${recoveredCount}</b> recovered</span>` +
    '</div>'
  );
  parts.push('</div>');
  return parts.join('');
}

/**
 * The complete causal panel: diagnosis + graph + legend.
 *
 * @param {object} data  the API response of a trace
 * @returns {string}     the markup
 */
function renderCausalPanel(data) {
  const graph = data && data.causal_graph;
  const rootCause = data && data.root_cause;
  if (!graph || !Array.isArray(graph.nodes) || !graph.nodes.length) return '';

  lastCausalGraph = graph;

  const layout = layoutCausalGraph(graph);
  const parts = [];
  parts.push('<div class="causal-inner">');
  parts.push(renderRootCausePanel(rootCause, graph));
  parts.push(
    '<div class="causal-canvas"><div class="causal-scroll">' +
    renderCausalGraph(graph, rootCause) +
    '</div></div>'
  );
  parts.push('<div id="causalTooltip" class="causal-tooltip hidden"></div>');
  parts.push('<div id="causalNodeDetails" class="causal-node-details hidden"></div>');
  parts.push(
    '<div class="causal-legend">' +
    '<span class="cg-legend-item"><i class="cg-legend-dot cg-legend-failure"></i>failed step</span>' +
    '<span class="cg-legend-item"><i class="cg-legend-box cg-legend-root"></i>root cause</span>' +
    '<span class="cg-legend-item"><i class="cg-legend-line cg-legend-parent"></i>parent → child</span>' +
    '<span class="cg-legend-item"><i class="cg-legend-line cg-legend-influenced"></i>failure propagation</span>' +
    `<span class="cg-legend-count">${layout.positions.length} events · ${layout.layers} layer(s)</span>` +
    '</div>'
  );
  parts.push('</div>');
  return parts.join('');
}

window.layoutCausalGraph = layoutCausalGraph;
window.renderCausalGraph = renderCausalGraph;
window.renderRootCausePanel = renderRootCausePanel;
window.renderCausalPanel = renderCausalPanel;
window.CAUSAL_LAYOUT = CAUSAL_LAYOUT;

/**
 * Show or hide the causal-analysis section of the dashboard.
 *
 * A trace recorded before Phase 11 has no graph — the section is hidden
 * instead of showing an empty box (graceful degradation).
 */
function renderCausalSection(data) {
  if (!causalSection || !causalPanel) return;
  const markup = renderCausalPanel(data);
  if (!markup) {
    causalSection.classList.add('hidden');
    causalPanel.innerHTML = '';
    lastCausalGraph = null;
    causalSelectedId = null;
    return;
  }
  causalPanel.innerHTML = markup;
  lastCausalGraph = data && data.causal_graph;
  if (causalSelectedId && lastCausalGraph.nodes.some(n => String(n.event_id) === causalSelectedId)) {
    selectCausal(causalSelectedId);
  }
  causalSection.classList.remove('hidden');
}
window.renderCausalSection = renderCausalSection;

// ── Causal graph interactivity (Phase 12) ─────────────────────
/**
 * Ancestors (upstream parents) of an event, cycle-safe.
 * @param {Array} edges  {source, target} pairs
 * @param {string} id
 * @returns {Set<string>}
 */
function causalAncestors(edges, id) {
  const parents = new Map();
  for (const e of edges || []) {
    const list = parents.get(e.target) || [];
    list.push(e.source);
    parents.set(e.target, list);
  }
  const out = new Set();
  const stack = [id];
  while (stack.length) {
    const cur = stack.pop();
    for (const p of parents.get(cur) || []) {
      if (p !== id && !out.has(p)) { out.add(p); stack.push(p); }
    }
  }
  return out;
}
window.causalAncestors = causalAncestors;

/**
 * Descendants (downstream children) of an event, cycle-safe.
 * @param {Array} edges  {source, target} pairs
 * @param {string} id
 * @returns {Set<string>}
 */
function causalDescendants(edges, id) {
  const children = new Map();
  for (const e of edges || []) {
    const list = children.get(e.source) || [];
    list.push(e.target);
    children.set(e.source, list);
  }
  const out = new Set();
  const stack = [id];
  while (stack.length) {
    const cur = stack.pop();
    for (const c of children.get(cur) || []) {
      if (c !== id && !out.has(c)) { out.add(c); stack.push(c); }
    }
  }
  return out;
}
window.causalDescendants = causalDescendants;

/**
 * Highlight the direct neighbours of a node on hover.
 * @param {string} id  event id hovered
 */
function highlightCausal(id) {
  if (!lastCausalGraph || !causalPanel) return;
  const edges = lastCausalGraph.edges || [];
  const related = new Set([id]);
  for (const e of edges) {
    if (e.source === id) related.add(e.target);
    if (e.target === id) related.add(e.source);
  }
  causalPanel.querySelectorAll('.cg-node').forEach(n => {
    n.classList.toggle('cg-dim', !related.has(n.dataset.eventId));
  });
  causalPanel.querySelectorAll('.cg-edge').forEach(e => {
    e.classList.toggle('cg-dim', !(related.has(e.dataset.source) && related.has(e.dataset.target)));
  });
}
window.highlightCausal = highlightCausal;

/**
 * Pure tooltip markup for a node.
 * @param {object} node  causal graph node
 * @returns {string}
 */
function causalTooltipHtml(node) {
  const lines = [
    `<div class="tt-label">${escHtml(String(node.label || node.event_type || node.event_id))}</div>`,
    `<span class="tt-type">${escHtml(node.event_type)}</span>`,
    `<div class="tt-meta">` +
    `<span class="tt-key">Status</span>${badgeHTML(node.status)}` +
    `<span class="tt-key">Latency</span>${formatLatency(node.latency_ms)}` +
    '</div>',
  ];
  if (node.error_message) {
    lines.push(`<div class="tt-error">${escHtml(node.error_message)}</div>`);
  }
  return lines.join('');
}
window.causalTooltipHtml = causalTooltipHtml;

/**
 * Select a node: highlight its full upstream/downstream path, dim the rest,
 * and render the node details card.
 * @param {string|null} id  event id, or null to deselect
 */
function selectCausal(id) {
  causalSelectedId = id;
  if (!lastCausalGraph || !causalPanel) return;

  const edges = lastCausalGraph.edges || [];
  const path = new Set(id != null ? [id] : []);
  if (id != null) {
    for (const a of causalAncestors(edges, id)) path.add(a);
    for (const d of causalDescendants(edges, id)) path.add(d);
  }

  causalPanel.querySelectorAll('.cg-node').forEach(n => {
    const isSelected = id != null && n.dataset.eventId === id;
    n.classList.toggle('cg-selected', isSelected);
    n.classList.toggle('cg-dim', id != null && !path.has(n.dataset.eventId));
  });
  causalPanel.querySelectorAll('.cg-edge').forEach(e => {
    e.classList.toggle('cg-dim', id != null && !(path.has(e.dataset.source) && path.has(e.dataset.target)));
  });

  const details = document.getElementById('causalNodeDetails');
  if (!nodeOf(id)) {
    if (details) details.classList.add('hidden');
    return;
  }
  if (details) {
    details.innerHTML = renderCausalNodeDetails(nodeOf(id), lastCausalGraph);
    details.classList.remove('hidden');
  }
}
window.selectCausal = selectCausal;

function nodeOf(id) {
  return lastCausalGraph && lastCausalGraph.nodes.find(n => String(n.event_id) === id);
}

/**
 * Render the details card of a graph node.
 * @param {object|null} node
 * @param {object} graph
 * @returns {string}
 */
function renderCausalNodeDetails(node, graph) {
  if (!node) return '<div class="causal-node-details hidden"></div>';
  const err = node.error_message
    ? String(node.error_message)
    : (node.exception_type ? `${node.exception_type}${node.exception_message ? ': ' + node.exception_message : ''}` : '');
  const chip = statusClass(node.status);
  return `
<div class="causal-node-details">
  <div class="cnd-header">
    <span class="cnd-title">${escHtml(String(node.label || node.event_type || node.event_id))}</span>
    <span class="cnd-chip ${chip}">${escHtml(chip)}</span>
  </div>
  <div class="cnd-body">
    <div class="cnd-row"><span class="cnd-key">Type</span><span class="cnd-val mono">${escHtml(node.event_type)}</span></div>
    <div class="cnd-row"><span class="cnd-key">Status</span><span class="cnd-val">${badgeHTML(node.status)}</span></div>
    <div class="cnd-row"><span class="cnd-key">Latency</span><span class="cnd-val">${formatLatency(node.latency_ms)}</span></div>
    <div class="cnd-row"><span class="cnd-key">Time</span><span class="cnd-val">${node.timestamp ? new Date(node.timestamp).toLocaleTimeString() : '—'}</span></div>
    <div class="cnd-row"><span class="cnd-key">Importance</span><span class="cnd-val">${typeof node.importance_score === 'number' ? node.importance_score.toFixed(2) : '—'}</span></div>
    ${err ? `<div class="cnd-row"><span class="cnd-key">Error</span><span class="cnd-val mono">${escHtml(err)}</span></div>` : ''}
  </div>
  <div class="cnd-actions">
    <button type="button" class="cnd-btn" data-action="show-in-timeline">Show in timeline →</button>
  </div>
</div>`;
}
window.renderCausalNodeDetails = renderCausalNodeDetails;

/**
 * Show the tooltip for a hovered node, following the pointer.
 */
function showCausalTooltip(ev, id) {
  const tooltip = document.getElementById('causalTooltip');
  if (!tooltip || !lastCausalGraph) return;
  const node = nodeOf(id);
  if (!node) return;
  tooltip.innerHTML = causalTooltipHtml(node);
  const rect = causalPanel.getBoundingClientRect();
  let x = Math.min(Math.max(ev.clientX - rect.left + 14, 8), rect.width - 10);
  let y = Math.min(Math.max(ev.clientY - rect.top + 14, 8), rect.height - 10);
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
  tooltip.classList.remove('hidden');
}
window.showCausalTooltip = showCausalTooltip;

function clearCausalHighlight() {
  if (!causalPanel) return;
  causalPanel.querySelectorAll('.cg-node').forEach(n => n.classList.remove('cg-dim'));
  causalPanel.querySelectorAll('.cg-edge').forEach(e => e.classList.remove('cg-dim'));
  const tooltip = document.getElementById('causalTooltip');
  if (tooltip) tooltip.classList.add('hidden');
}
window.clearCausalHighlight = clearCausalHighlight;

/**
 * Open the matching event card in the timeline for the selected node.
 */
function focusEventInTimeline(eventId) {
  if (!lastData || !eventId) return;
  const idx = sortedEvents.findIndex(e => e.event_id === eventId);
  if (idx < 0) return;
  toggleEvent(idx, true);
  const card = document.getElementById(`ev-card-${idx}`);
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  activeFilter = 'all';
  filterBtns.forEach(b => b.classList.toggle('active', b.dataset.filter === 'all'));
  if (lastData) renderEventList();
}
window.focusEventInTimeline = focusEventInTimeline;

if (causalPanel) {
  causalPanel.addEventListener('mouseover', ev => {
    if (causalSelectedId != null) return;
    const node = ev.target.closest('.cg-node[data-event-id]');
    if (!node) return;
    highlightCausal(node.dataset.eventId);
    showCausalTooltip(ev, node.dataset.eventId);
  });
  causalPanel.addEventListener('mouseout', ev => {
    if (ev.target.closest('.cg-node[data-event-id], [data-action="show-in-timeline"]')) return;
    clearCausalHighlight();
  });
  causalPanel.addEventListener('mousemove', ev => {
    if (causalSelectedId != null) return;
    const node = ev.target.closest('.cg-node[data-event-id]');
    if (!node) return;
    showCausalTooltip(ev, node.dataset.eventId);
  });
  causalPanel.addEventListener('click', ev => {
    const node = ev.target.closest('.cg-node[data-event-id]');
    if (node) { selectCausal(node.dataset.eventId); return; }
    if (ev.target.closest('[data-action="show-in-timeline"]')) { focusEventInTimeline(causalSelectedId); return; }
    if (ev.target.closest('.causal-canvas')) selectCausal(null);
  });
}

// ── Live-run failure state (Phase 12) ─────────────────────────────────────
/**
 * Build the failure state of a run.
 *
 * A failing run is *not* a request error: the API returns the failed trace
 * (status="failure", the error message, the causal graph and the root-cause
 * diagnosis).  This panel explains what went wrong and points at the causal
 * analysis instead of leaving the user with a bare error banner.
 *
 * @param {object} data  the API response of the run
 * @returns {string}     the markup ('' when the run did not fail)
 */
function renderRunFailureMarkup(data) {
  if (!data || statusClass(data.status) !== 'failure') return '';

  const rootCause = data.root_cause || null;
  const graph = data.causal_graph || null;
  const hasGraph = !!(graph && Array.isArray(graph.nodes) && graph.nodes.length);
  const verdict = rootCause
    ? (CAUSAL_VERDICT_LABEL[rootCause.verdict] || rootCause.verdict)
    : null;

  const parts = [];
  parts.push('<div class="rf-head">');
  parts.push(
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">' +
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>' +
    '<line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
  );
  parts.push('<span class="rf-title">Run failed</span>');
  if (verdict) parts.push(`<span class="rf-verdict">${escHtml(verdict)}</span>`);
  parts.push('</div>');

  if (data.error) {
    parts.push(`<p class="rf-error">${escHtml(String(data.error))}</p>`);
  }
  if (rootCause && rootCause.explanation) {
    parts.push(`<p class="rf-explanation">${escHtml(String(rootCause.explanation))}</p>`);
  }
  if (rootCause && typeof rootCause.confidence === 'number' && rootCause.confidence > 0) {
    parts.push(
      '<p class="rf-confidence">Diagnosis confidence ' +
      `<b>${rootCause.confidence.toFixed(2)}</b> · category ` +
      `<code>${escHtml(rootCause.category || 'none')}</code></p>`
    );
  }

  parts.push('<div class="rf-actions">');
  if (hasGraph) {
    parts.push(
      '<button type="button" class="rf-btn rf-btn-primary" data-action="show-causal">' +
      'Show causal analysis →</button>'
    );
  }
  parts.push(
    '<button type="button" class="rf-btn" data-action="show-failures">Show failed steps</button>'
  );
  parts.push('</div>');
  return parts.join('');
}
window.renderRunFailureMarkup = renderRunFailureMarkup;

/**
 * Show or hide the failure state of the current run.
 *
 * @param {object} data  the API response of the run
 */
function renderRunFailure(data) {
  if (!runFailure) return;
  const markup = renderRunFailureMarkup(data);
  if (!markup) {
    runFailure.classList.add('hidden');
    runFailure.innerHTML = '';
    return;
  }
  runFailure.innerHTML = markup;
  runFailure.classList.remove('hidden');
}
window.renderRunFailure = renderRunFailure;

/**
 * Scroll to the causal analysis and highlight it.
 */
function focusCausalAnalysis() {
  if (!causalSection) return;
  if (causalSection.scrollIntoView) {
    causalSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  causalSection.classList.add('causal-flash');
  window.setTimeout(() => causalSection.classList.remove('causal-flash'), 1800);
}
window.focusCausalAnalysis = focusCausalAnalysis;

/**
 * Toggle full-screen mode for the causal graph.
 *
 * Copies the rendered graph into an overlay that fills the viewport so the
 * graph can be inspected at a comfortable size.  Press Escape or click the
 * close button to exit.
 */
function toggleCausalFullscreen() {
  if (!causalPanel || !cfsOverlay || !cfsContent) return;

  if (!cfsActive) {
    // Enter fullscreen
    cfsContent.innerHTML = causalPanel.innerHTML;
    cfsOverlay.classList.remove('hidden');
    cfsActive = true;
    document.body.style.overflow = 'hidden';
    // Make the graph SVG fill the available width inside the overlay.
    const svg = cfsContent.querySelector('.causal-svg');
    if (svg) {
      svg.style.width = '100%';
      svg.style.maxWidth = 'none';
    }
  } else {
    // Exit fullscreen
    cfsOverlay.classList.add('hidden');
    cfsContent.innerHTML = '';
    cfsActive = false;
    document.body.style.overflow = '';
  }
}
window.toggleCausalFullscreen = toggleCausalFullscreen;

if (cfsClose) {
  cfsClose.addEventListener('click', toggleCausalFullscreen);
}
if (causalFullscreenBtn) {
  causalFullscreenBtn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleCausalFullscreen();
  });
}

// Close on Escape key
window.addEventListener('keydown', (ev) => {
  if (cfsActive && ev.key === 'Escape') toggleCausalFullscreen();
});


/**
 * Jump to the event timeline, filtered down to the failed steps.
 */
function focusFailedSteps() {
  activeFilter = 'failure';
  filterBtns.forEach(b => b.classList.toggle('active', b.dataset.filter === activeFilter));
  if (lastData) renderEventList();
  const timeline = document.getElementById('timelineSection');
  if (timeline && timeline.scrollIntoView) {
    timeline.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}
window.focusFailedSteps = focusFailedSteps;

if (runFailure) {
  runFailure.addEventListener('click', ev => {
    const btn = ev.target.closest('button[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'show-causal') { focusCausalAnalysis(); return; }
    if (btn.dataset.action === 'show-failures') { focusFailedSteps(); }
  });
}

// ── Trace history (Phase 12) ──────────────────────────────────────────────
/**
 * The history panel browses `GET /api/traces`: a keyset-paginated list of
 * trace summaries (no event payloads).  Clicking a row loads that trace —
 * including its causal graph and root-cause diagnosis — into the results
 * pane, so a demo can jump between runs.
 */
const HISTORY_PAGE_SIZE = 20;

let historyItems   = [];
let historyCursor  = null;
let historyCount   = 0;
let historyLoading = false;

/**
 * Build the query string of a `GET /api/traces` request.
 *
 * @param {object} filters        {status, startDate, endDate}
 * @param {string|null} cursor     keyset cursor of the next page
 * @param {number} limit           page size
 * @returns {string}               the URL-encoded query string
 */
function buildTraceHistoryQuery(filters, cursor, limit) {
  const params = new URLSearchParams();
  const f = filters || {};
  const status = f.status || 'all';
  if (status && status !== 'all') params.set('status', status);
  if (f.startDate) params.set('start_date', f.startDate);
  if (f.endDate) params.set('end_date', f.endDate);
  if (cursor) params.set('cursor', cursor);
  params.set('limit', String(limit || HISTORY_PAGE_SIZE));
  return params.toString();
}
window.buildTraceHistoryQuery = buildTraceHistoryQuery;

/**
 * Read the filter values out of the DOM.
 * @returns {{status: string, startDate: string, endDate: string}}
 */
function readHistoryFilters() {
  return {
    status: (historyStatus && historyStatus.value) || 'all',
    startDate: (historyFrom && historyFrom.value) || '',
    endDate: (historyTo && historyTo.value) || '',
  };
}
window.readHistoryFilters = readHistoryFilters;

/**
 * Format the start time of a trace for the history list.
 * @param {string|null} iso
 * @returns {string}
 */
function formatTraceTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString();
}
window.formatTraceTime = formatTraceTime;

/**
 * Render the history rows.
 * @param {Array} items  trace summaries
 * @returns {string}     the markup
 */
function renderHistoryItems(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return '<p class="history-empty">No traces match the current filters.</p>';
  }
  return items.map(t => {
    const cls  = statusClass(t.status);
    const when = formatTraceTime(t.start_time);
    const count = t.event_count == null ? 0 : t.event_count;
    return `<button type="button" class="history-item status-${cls}" ` +
           `data-trace-id="${escHtml(t.trace_id || '')}" title="${escHtml(t.trace_id || '')}">` +
           `<span class="history-row">` +
           `<span class="history-dot status-${cls}"></span>` +
           `<code class="history-id">${escHtml(shortId(t.trace_id))}</code>` +
           `<span class="history-status">${escHtml(t.status || 'unknown')}</span>` +
           `</span>` +
           `<span class="history-row history-row-meta">` +
           `<span class="history-agent">${escHtml(t.agent_id || 'unknown agent')}</span>` +
           `<span class="history-time">${escHtml(when)}</span>` +
           `<span class="history-events">${escHtml(String(count))} ev</span>` +
           `<span class="history-latency">${escHtml(formatLatency(t.total_latency_ms))}</span>` +
           `</span>` +
           `</button>`;
  }).join('');
}
window.renderHistoryItems = renderHistoryItems;

/**
 * Paint the history panel from the current state.
 */
function renderHistory() {
  if (historyList) historyList.innerHTML = renderHistoryItems(historyItems);
  if (historyMeta) {
    historyMeta.textContent = historyCount
      ? `${historyItems.length} of ${historyCount} trace${historyCount === 1 ? '' : 's'}`
      : '';
  }
  if (historyMore) historyMore.classList.toggle('hidden', !historyCursor);
}
window.renderHistory = renderHistory;

/**
 * Paint an error in the history panel (the rest of the dashboard keeps
 * working — the history is an extra, not a prerequisite).
 */
function renderHistoryError(err) {
  const message = err && err.message ? err.message : String(err);
  if (historyList) {
    historyList.innerHTML = `<p class="history-empty history-error">Could not load traces: ${escHtml(message)}</p>`;
  }
  if (historyMeta) historyMeta.textContent = '';
  if (historyMore) historyMore.classList.add('hidden');
}
window.renderHistoryError = renderHistoryError;

/**
 * Load a page of traces.
 *
 * @param {boolean} append  true → append the next page (keyset pagination)
 */
async function loadHistory(append) {
  if (historyLoading) return;
  historyLoading = true;
  try {
    const cursor = append ? historyCursor : null;
    const query = buildTraceHistoryQuery(readHistoryFilters(), cursor, HISTORY_PAGE_SIZE);
    const res = await fetch(`${API_BASE_URL}/api/traces?${query}`);
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];
    historyItems  = append ? historyItems.concat(items) : items;
    historyCursor = data.next_cursor || null;
    historyCount  = typeof data.count === 'number' ? data.count : historyItems.length;
    renderHistory();
  } catch (err) {
    renderHistoryError(err);
  } finally {
    historyLoading = false;
  }
}
window.loadHistory = loadHistory;

/**
 * Load a stored trace into the results pane.
 * @param {string} traceId
 */
async function openTrace(traceId) {
  if (!traceId) return;
  hideError();
  try {
    const res = await fetch(`${API_BASE_URL}/api/traces/${encodeURIComponent(traceId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const trace = await res.json();
    renderResults(trace);
    if (resultsView && resultsView.scrollIntoView) {
      resultsView.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  } catch (err) {
    showError(`Could not load trace ${traceId}: ${err.message}`);
  }
}
window.openTrace = openTrace;

if (historyList) {
  historyList.addEventListener('click', ev => {
    const item = ev.target.closest('.history-item');
    if (!item) return;
    openTrace(item.dataset.traceId);
  });
}
if (historyMore) {
  historyMore.addEventListener('click', () => loadHistory(true));
}
if (historyRefresh) {
  historyRefresh.addEventListener('click', () => loadHistory(false));
}
if (historyStatus) {
  historyStatus.addEventListener('change', () => loadHistory(false));
}
if (historyFrom) {
  historyFrom.addEventListener('change', () => loadHistory(false));
}
if (historyTo) {
  historyTo.addEventListener('change', () => loadHistory(false));
}

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
    ],
    causal_graph: {
      trace_id: 'ddee1122-0000-0000-0000-aabbccddeeff',
      nodes: [
        { event_id: 'ev-101', parent_id: null,     event_type: 'llm_call',       status: 'success', timestamp: ts(0), label: 'llm_call',            is_failure: false, suspicious_output: false, importance_score: 0.42, observation_level: 'full' },
        { event_id: 'ev-102', parent_id: 'ev-101', event_type: 'tool_call',      status: 'failure', timestamp: ts(1), label: 'tool_call: api_call', is_failure: true,  suspicious_output: true,  importance_score: 0.95, observation_level: 'full', error_message: 'Connection timed out after 400 ms', exception_type: 'TimeoutError' },
        { event_id: 'ev-103', parent_id: 'ev-102', event_type: 'retry',          status: 'success', timestamp: ts(2), label: 'retry: attempt 2',     is_failure: false, suspicious_output: false, importance_score: 0.60, observation_level: 'full' },
        { event_id: 'ev-104', parent_id: 'ev-103', event_type: 'final_response', status: 'success', timestamp: ts(3), label: 'final_response',       is_failure: false, suspicious_output: false, importance_score: 0.90, observation_level: 'full' },
      ],
      edges: [
        { source: 'ev-101', target: 'ev-102', relation: 'parent_of' },
        { source: 'ev-102', target: 'ev-103', relation: 'parent_of' },
        { source: 'ev-103', target: 'ev-104', relation: 'parent_of' },
        { source: 'ev-102', target: 'ev-103', relation: 'retry_of' },
        { source: 'ev-102', target: 'ev-103', relation: 'influenced' },
      ],
    },
    root_cause: {
      trace_id: 'ddee1122-0000-0000-0000-aabbccddeeff',
      verdict: 'recovered',
      root_cause_event_id: null,
      confidence: 0.0,
      category: 'none',
      explanation: 'The failure in tool_call: api_call (event_type=tool_call, status=failure) was recovered by a later successful retry, so the trace has no unrecovered root cause. The recorded detail: error message "Connection timed out after 400 ms", exception type TimeoutError.',
      evidence_path: [],
      primary_failures: [],
      secondary_failures: [],
      affected_events: [],
      recovered_failures: ['ev-102'],
    },
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
    ],
    causal_graph: {
      trace_id: 'fail9999-0000-0000-0000-000000000000',
      nodes: [
        { event_id: 'ev-200', parent_id: null,     event_type: 'llm_call',  status: 'success', timestamp: ts(0), label: 'llm_call',           is_failure: false, suspicious_output: false, importance_score: 0.35, observation_level: 'full' },
        { event_id: 'ev-201', parent_id: 'ev-200', event_type: 'tool_call', status: 'failure', timestamp: ts(1), label: 'tool_call: db_query', is_failure: true,  suspicious_output: true,  importance_score: 0.98, observation_level: 'full', error_message: 'Database connection refused', exception_type: 'ConnectionRefusedError' },
        { event_id: 'ev-202', parent_id: 'ev-201', event_type: 'error',     status: 'failure', timestamp: ts(2), label: 'error',               is_failure: true,  suspicious_output: true,  importance_score: 0.97, observation_level: 'full', error_message: 'Unrecoverable: cascading failure in data pipeline', exception_type: 'RuntimeError' },
      ],
      edges: [
        { source: 'ev-200', target: 'ev-201', relation: 'parent_of' },
        { source: 'ev-201', target: 'ev-202', relation: 'parent_of' },
        { source: 'ev-201', target: 'ev-202', relation: 'influenced' },
      ],
    },
    root_cause: {
      trace_id: 'fail9999-0000-0000-0000-000000000000',
      verdict: 'root_cause',
      root_cause_event_id: 'ev-201',
      confidence: 1.0,
      category: 'tool_error',
      explanation: 'The root cause is tool_call: db_query (event_type=tool_call, status=failure); its recorded detail: error message "Database connection refused", exception type ConnectionRefusedError. The failure propagated along the evidence path tool_call: db_query → error, affecting 0 downstream event(s) and 1 secondary failure(s).',
      evidence_path: ['ev-201', 'ev-202'],
      primary_failures: ['ev-201'],
      secondary_failures: ['ev-202'],
      affected_events: [],
      recovered_failures: [],
    },
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
    // A failed run is *not* a request error: the API hands back the failed
    // trace with its diagnosis, and the failure state explains it.
    renderResults(data);
    // Keep the history in sync — the new run belongs at the top of the list.
    loadHistory(false);

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
loadHistory(false);

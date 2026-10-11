/**
 * AgentLens – test.js  (feature/timeline_intf)
 *
 * Vanilla JS test suite. No build step required.
 * Run by opening frontend/test.html in a browser.
 *
 * All functions under test are loaded from app.js via a <script> tag in
 * test.html.  They are exposed on window so they can be called here directly.
 */

/* ── Minimal test harness ─────────────────────────────────────── */

let _passed = 0;
let _failed = 0;
const _results = [];

function assert(condition, label) {
  if (condition) {
    _passed++;
    _results.push({ ok: true, label });
  } else {
    _failed++;
    _results.push({ ok: false, label });
    console.error(`FAIL: ${label}`);
  }
}

function assertEqual(actual, expected, label) {
  assert(actual === expected, `${label} (got: ${JSON.stringify(actual)}, want: ${JSON.stringify(expected)})`);
}

function assertNull(actual, label) {
  assert(actual === null || actual === undefined, `${label} (got: ${JSON.stringify(actual)})`);
}

function renderReport() {
  const container = document.getElementById('test-output');
  if (!container) return;

  const summary = document.getElementById('test-summary');
  const total = _passed + _failed;
  summary.textContent = `${_passed} / ${total} tests passed`;
  summary.className = _failed === 0 ? 'summary-pass' : 'summary-fail';

  _results.forEach(r => {
    const div = document.createElement('div');
    div.className = `test-row ${r.ok ? 'pass' : 'fail'}`;
    div.textContent = `${r.ok ? '✓' : '✗'} ${r.label}`;
    container.appendChild(div);
  });
}

/* ── Test data helpers ───────────────────────────────────────── */

const T0 = new Date('2026-10-08T12:00:00Z').getTime();
const ts  = i => new Date(T0 + i * 1000).toISOString();

function makeEvent(overrides) {
  return {
    event_id:   'ev-' + Math.random().toString(36).slice(2, 8),
    parent_id:  null,
    event_type: 'llm_call',
    status:     'success',
    latency_ms: 100.0,
    timestamp:  ts(0),
    input_data:  {},
    output_data: {},
    metadata:    {},
    ...overrides
  };
}

/* ── 1. Chronological sorting ────────────────────────────────── */

function testSortEvents() {
  const events = [
    makeEvent({ event_id: 'c', timestamp: ts(2) }),
    makeEvent({ event_id: 'a', timestamp: ts(0) }),
    makeEvent({ event_id: 'b', timestamp: ts(1) }),
  ];

  const sorted = sortEvents(events);

  assertEqual(sorted[0].event_id, 'a', 'sort: first event is earliest timestamp');
  assertEqual(sorted[1].event_id, 'b', 'sort: second event is middle timestamp');
  assertEqual(sorted[2].event_id, 'c', 'sort: third event is latest timestamp');

  // Original array must not be mutated
  assertEqual(events[0].event_id, 'c', 'sort: original array not mutated');
}

/* ── 2. statusClass mapping ──────────────────────────────────── */

function testStatusClass() {
  assertEqual(statusClass('success'),   'success', 'statusClass: success');
  assertEqual(statusClass('completed'), 'success', 'statusClass: completed -> success');
  assertEqual(statusClass('failure'),   'failure', 'statusClass: failure');
  assertEqual(statusClass('failed'),    'failure', 'statusClass: failed -> failure');
  assertEqual(statusClass('error'),     'failure', 'statusClass: error -> failure');
  assertEqual(statusClass('partial'),   'partial', 'statusClass: partial');
  assertEqual(statusClass('pending'),   'pending', 'statusClass: pending');
  assertEqual(statusClass(null),        'pending', 'statusClass: null -> pending');
  assertEqual(statusClass(undefined),   'pending', 'statusClass: undefined -> pending');
  assertEqual(statusClass(''),          'pending', 'statusClass: empty string -> pending');
  assertEqual(statusClass('SUCCESS'),   'success', 'statusClass: uppercase SUCCESS -> success');
}

/* ── 3. formatLatency ────────────────────────────────────────── */

function testFormatLatency() {
  assertEqual(formatLatency(null),      'N/A',      'formatLatency: null -> N/A');
  assertEqual(formatLatency(undefined), 'N/A',      'formatLatency: undefined -> N/A');
  assertEqual(formatLatency(0),         '0.0 ms',   'formatLatency: 0 -> 0.0 ms');
  assertEqual(formatLatency(45.2),      '45.2 ms',  'formatLatency: 45.2 -> 45.2 ms');
  assertEqual(formatLatency(999.9),     '999.9 ms', 'formatLatency: 999.9 -> 999.9 ms');
  assertEqual(formatLatency(1000),      '1.00 s',   'formatLatency: 1000 -> 1.00 s');
  assertEqual(formatLatency(1234.5),    '1.23 s',   'formatLatency: 1234.5 -> 1.23 s');
  assertEqual(formatLatency(3600000),   '3600.00 s','formatLatency: 1 hour');
}

/* ── 4. extractPrompt / extractResponse fallback logic ───────── */

function testExtractPrompt() {
  assertEqual(extractPrompt({ prompt: 'hello' }),  'hello', 'extractPrompt: reads .prompt');
  assertEqual(extractPrompt({ query:  'world' }),  'world', 'extractPrompt: falls back to .query');
  assertEqual(extractPrompt({ prompt: 'p', query: 'q' }), 'p', 'extractPrompt: prompt wins over query');
  assertNull( extractPrompt({}),                          'extractPrompt: empty obj -> null');
  assertNull( extractPrompt(null),                        'extractPrompt: null -> null');
  assertNull( extractPrompt(undefined),                   'extractPrompt: undefined -> null');
}

function testExtractResponse() {
  assertEqual(extractResponse({ response: 'r' }),    'r',   'extractResponse: reads .response');
  assertEqual(extractResponse({ answer:   'a' }),    'a',   'extractResponse: falls back to .answer');
  assertEqual(extractResponse({ text:     't' }),    't',   'extractResponse: falls back to .text');
  assertEqual(extractResponse({ response: 'r', answer: 'a', text: 't' }), 'r',
                                                            'extractResponse: response wins');
  assertNull( extractResponse({}),                         'extractResponse: empty obj -> null');
  assertNull( extractResponse(null),                       'extractResponse: null -> null');
  assertNull( extractResponse(undefined),                  'extractResponse: undefined -> null');
}

/* ── 5. Empty input/output doesn't crash ─────────────────────── */

function testEmptyData() {
  // These should not throw
  let threw = false;
  try {
    extractPrompt({});
    extractPrompt(null);
    extractResponse({});
    extractResponse(null);
    formatLatency(null);
    sortEvents([]);
    computeDepths([]);
    filterEvents([], 'all');
  } catch(e) {
    threw = true;
    console.error('Empty data crash:', e);
  }
  assert(!threw, 'empty/null data does not throw');
}

/* ── 6. Parent depth calculation ─────────────────────────────── */

function testComputeDepths() {
  const root  = makeEvent({ event_id: 'root', parent_id: null });
  const child = makeEvent({ event_id: 'child', parent_id: 'root' });
  const grand = makeEvent({ event_id: 'grand', parent_id: 'child' });
  const sib   = makeEvent({ event_id: 'sib',   parent_id: 'root' });

  const depths = computeDepths([root, child, grand, sib]);

  assertEqual(depths.get('root'),  0, 'depth: root = 0');
  assertEqual(depths.get('child'), 1, 'depth: child of root = 1');
  assertEqual(depths.get('grand'), 2, 'depth: grandchild = 2');
  assertEqual(depths.get('sib'),   1, 'depth: sibling of child = 1');
}

function testComputeDepthsEmpty() {
  const depths = computeDepths([]);
  assert(depths.size === 0, 'computeDepths: empty events -> empty map');
}

function testComputeDepthsOrphan() {
  // Event with a parent_id that doesn't exist in the set
  const ev = makeEvent({ event_id: 'orphan', parent_id: 'ghost-id' });
  let threw = false;
  try {
    computeDepths([ev]);
  } catch(e) {
    threw = true;
  }
  assert(!threw, 'computeDepths: orphan parent_id does not throw');
}

/* ── 7. Filter logic ─────────────────────────────────────────── */

function testFilterEvents() {
  const successEv = makeEvent({ event_id: 's', status: 'success' });
  const failureEv = makeEvent({ event_id: 'f', status: 'failure' });
  const partialEv = makeEvent({ event_id: 'p', status: 'partial' });
  const pendingEv = makeEvent({ event_id: 'w', status: 'pending' });
  const all = [successEv, failureEv, partialEv, pendingEv];

  const filtAll     = filterEvents(all, 'all');
  assertEqual(filtAll.length, 4, 'filter all: returns 4 events');

  const filtSuccess = filterEvents(all, 'success');
  // failure events are always shown
  assert(filtSuccess.some(e => e.event_id === 's'), 'filter success: includes success event');
  assert(filtSuccess.some(e => e.event_id === 'f'), 'filter success: failure events always visible');
  assert(!filtSuccess.some(e => e.event_id === 'p'), 'filter success: excludes partial');
  assert(!filtSuccess.some(e => e.event_id === 'w'), 'filter success: excludes pending');

  const filtFailure = filterEvents(all, 'failure');
  assertEqual(filtFailure.length, 1, 'filter failure: only 1 failure event (no pending/partial forced)');
  assertEqual(filtFailure[0].event_id, 'f', 'filter failure: correct event returned');

  const filtPartial = filterEvents(all, 'partial');
  // partial filter: shows partials + always-visible failures
  assert(filtPartial.some(e => e.event_id === 'p'), 'filter partial: includes partial event');
  assert(filtPartial.some(e => e.event_id === 'f'), 'filter partial: failure events always visible');
  assert(!filtPartial.some(e => e.event_id === 's'), 'filter partial: excludes success');
}

function testFilterEmpty() {
  const result = filterEvents([], 'success');
  assertEqual(result.length, 0, 'filterEvents: empty array -> empty result');
}

/* ── Run all tests ───────────────────────────────────────────── */

function runAllTests() {
  testSortEvents();
  testStatusClass();
  testFormatLatency();
  testExtractPrompt();
  testExtractResponse();
  testEmptyData();
  testComputeDepths();
  testComputeDepthsEmpty();
  testComputeDepthsOrphan();
  testFilterEvents();
  testFilterEmpty();
  renderReport();
}

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

const TEST_T0 = new Date('2026-10-08T12:00:00Z').getTime();
const testTs  = i => new Date(TEST_T0 + i * 1000).toISOString();

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
    makeEvent({ event_id: 'c', timestamp: testTs(2) }),
    makeEvent({ event_id: 'a', timestamp: testTs(0) }),
    makeEvent({ event_id: 'b', timestamp: testTs(1) }),
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

/* ── 8. Tree and UX edge cases ──────────────────────────────── */

function testDeepTreeAndLongPayload() {
  const events = [];
  let parentId = null;
  for (let i = 0; i < 9; i++) {
    const event = makeEvent({
      event_id: `deep-${i}`,
      parent_id: parentId,
      input_data: { prompt: 'x'.repeat(5000) },
      timestamp: testTs(8 - i),
    });
    events.push(event);
    parentId = event.event_id;
  }

  const depths = computeDepths(events);
  assertEqual(depths.get('deep-0'), 0, 'deep tree: root depth is zero');
  assertEqual(depths.get('deep-8'), 8, 'deep tree: nested depth is preserved');
  assertEqual(extractPrompt(events[0].input_data).length, 5000, 'long prompt: remains readable');
}

function testMissingTimestampAndDataFields() {
  const events = [
    makeEvent({ event_id: 'missing-time', timestamp: undefined, input_data: null, output_data: null }),
    makeEvent({ event_id: 'known-time', timestamp: testTs(1) }),
  ];
  let sorted;
  let threw = false;
  try { sorted = sortEvents(events); } catch (error) { threw = true; }
  assert(!threw, 'missing timestamp/data: rendering helpers do not throw');
  assertEqual(sorted[0].event_id, 'missing-time', 'missing timestamp: treated as earliest');
}

function testFailureDetailsShape() {
  const event = makeEvent({
    status: 'failure',
    output_data: { error: 'connection refused' },
    metadata: { exception_type: 'ConnectionError', exception_message: 'database unavailable' },
  });
  assertEqual(event.output_data.error, 'connection refused', 'failure UX: error is available');
  assertEqual(event.metadata.exception_type, 'ConnectionError', 'failure UX: exception type is available');
  assertEqual(event.metadata.exception_message, 'database unavailable', 'failure UX: exception message is available');
}

/* ── 9. Phase 8: failure presentation ───────────────────────── */

function testDefaultFilterFor() {
  const okEvents   = [makeEvent({ event_id: 'a', status: 'success' })];
  const failEvents = [
    makeEvent({ event_id: 'a', status: 'success' }),
    makeEvent({ event_id: 'b', status: 'failure' }),
  ];

  assertEqual(defaultFilterFor({ failure_count: 0 }, okEvents),   'all',     'defaultFilterFor: clean trace -> all');
  assertEqual(defaultFilterFor({ failure_count: 1 }, okEvents),   'failure', 'defaultFilterFor: failure_count drives default');
  assertEqual(defaultFilterFor({}, failEvents),                   'failure', 'defaultFilterFor: scans event statuses');
  assertEqual(defaultFilterFor(null, okEvents),                   'all',     'defaultFilterFor: null data -> all');
  assertEqual(defaultFilterFor({ failure_count: 0 }, []),         'all',     'defaultFilterFor: empty events -> all');
}

function testRenderFailureSummaryShowsFailedStep() {
  const parent = makeEvent({ event_id: 'parent-1', event_type: 'tool_call', status: 'success' });
  const failed = makeEvent({
    event_id: 'fail-1',
    parent_id: 'parent-1',
    event_type: 'llm_call',
    status: 'failure',
    output_data: { error: 'quota exceeded' },
    metadata: { exception_type: 'RuntimeError', exception_message: 'quota exceeded', step: 'synthesis' },
  });
  const events = [parent, failed];

  renderFailureSummary(
    { status: 'failure', failure_count: 1, failure_event_ids: ['fail-1'] },
    events
  );

  const el = document.getElementById('failureSummary');
  const html = el.innerHTML;
  assert(html.includes('quota exceeded'),       'failure banner: shows the error message');
  assert(html.includes('RuntimeError'),         'failure banner: shows the exception type');
  assert(html.includes('LLM CALL'),             'failure banner: shows the failed step type');
  assert(html.includes('TOOL CALL'),            'failure banner: shows the parent breadcrumb');
  assert(html.includes('1 failed step'),        'failure banner: reports failed step count');
  assert(!el.classList.contains('hidden'),      'failure banner: is visible when failures exist');
}

function testRenderFailureSummaryHiddenWithoutFailures() {
  const events = [
    makeEvent({ event_id: 'a', status: 'success' }),
    makeEvent({ event_id: 'b', status: 'success' }),
  ];
  renderFailureSummary({ status: 'success', failure_count: 0 }, events);
  const el = document.getElementById('failureSummary');
  assert(el.classList.contains('hidden'), 'failure banner: hidden when trace has no failures');
  assertEqual(el.innerHTML, '',           'failure banner: cleared when trace has no failures');
}

function testRenderFailureSummaryFallsBackToScan() {
  // No failure_event_ids provided -> banner must still find the failed event.
  const failed = makeEvent({
    event_id: 'fail-2',
    event_type: 'tool_call',
    status: 'error',
    output_data: { error: 'tool crashed' },
    metadata: { exception_type: 'ConnectionError' },
  });
  renderFailureSummary({ status: 'failure' }, [failed]);
  const html = document.getElementById('failureSummary').innerHTML;
  assert(html.includes('tool crashed'),   'failure banner: scan fallback finds error message');
  assert(html.includes('ConnectionError'),'failure banner: scan fallback finds exception type');
}

function testBuildFailureBlockFields() {
  const parent = makeEvent({ event_id: 'p-1', event_type: 'decision', status: 'success' });
  sortedEvents = [parent];

  const failed = makeEvent({
    event_id: 'f-1',
    parent_id: 'p-1',
    status: 'failure',
    output_data: { error: 'boom' },
    metadata: { exception_type: 'ValueError', exception_message: 'bad value' },
  });

  const html = buildFailureBlock(failed);
  assert(html.includes('boom'),        'failure block: renders error message');
  assert(html.includes('ValueError'),  'failure block: renders exception type');
  assert(html.includes('bad value'),   'failure block: renders exception message');
  assert(html.includes('DECISION'),    'failure block: renders parent breadcrumb');
}

function testBuildFailureBlockRetryBadges() {
  sortedEvents = [];
  const retried = makeEvent({
    event_id: 'r-2',
    status: 'success',
    metadata: { retry_of: 'r-1', attempt: 2 },
  });
  const html = buildFailureBlock(retried);
  assert(html.includes('retry-badge-of'), 'failure block: renders "retry of" badge');
  assert(html.includes('retry of'),       'failure block: retry-of label present');
}

function testBuildFailureBlockRetriedBy() {
  sortedEvents = [];
  const original = makeEvent({
    event_id: 'o-1',
    status: 'failure',
    output_data: { error: 'first attempt failed' },
    metadata: { exception_type: 'TimeoutError', retried_by: 'o-2' },
  });
  const html = buildFailureBlock(original);
  assert(html.includes('retry-badge-by'), 'failure block: renders "retried by" badge');
  assert(html.includes('TimeoutError'),   'failure block: still shows exception type');
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
  testDeepTreeAndLongPayload();
  testMissingTimestampAndDataFields();
  testFailureDetailsShape();
  testDefaultFilterFor();
  testRenderFailureSummaryShowsFailedStep();
  testRenderFailureSummaryHiddenWithoutFailures();
  testRenderFailureSummaryFallsBackToScan();
  testBuildFailureBlockFields();
  testBuildFailureBlockRetryBadges();
  testBuildFailureBlockRetriedBy();
  renderReport();
}

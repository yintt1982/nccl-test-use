import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const script = await readFile('local-report/client.js', 'utf8');
const boxes = [];
const createBox = () => {
  const nodes = { input: { value: '' }, '.local-status': {}, '.local-chart': {}, '.local-table': {}, a: {} };
  const box = { nodes, innerHTML: '', querySelector(selector) { return nodes[selector]; } };
  boxes.push(box);
  return box;
};
const panels = { chunks: { replaceChildren(child) { this.child = child; } }, timeline: { replaceChildren(child) { this.child = child; } } };
const document = {
  querySelector(selector) { return selector.includes('chunks') ? panels.chunks : selector.includes('timeline') ? panels.timeline : null; },
  createElement: createBox
};
const sample = { chunks: [{ event_id: 'step-1', source: 'sample_replay', source_event_type: 'proxy_step', rank: 0, quality: 'observed' }], timeline: [] };
const calls = [];
const timers = [];
const context = { document, location: { protocol: 'http:', search: '?run_id=replay-01' }, URLSearchParams, console,
  window: { addEventListener() {} }, setInterval: callback => { timers.push(callback); return timers.length; }, clearInterval() {},
  fetch: async url => { calls.push(url); return { ok: true, json: async () => sample }; } };
vm.runInNewContext(script, context);
await new Promise(resolve => setTimeout(resolve, 0));
const firstCount = calls.length;
sample.chunks.push({ event_id: 'step-2', source: 'sample_replay', source_event_type: 'proxy_step', rank: 1, quality: 'observed' });
for (const tick of timers) tick();
await new Promise(resolve => setTimeout(resolve, 0));
const chunks = panels.chunks.child.nodes;
if (!calls.some(url => url.includes('/replay-01/report?') && url.includes('section=chunks')) || !chunks['.local-table'].innerHTML.includes('table') ||
    !chunks['.local-status'].textContent.includes('sample') || !chunks.a.href.includes('format=csv') ||
    calls.length <= firstCount || !chunks['.local-table'].innerHTML.includes('step-2')) throw Error('UI report wiring failed');
console.log('PASS UI report query, chart, table and export wiring');

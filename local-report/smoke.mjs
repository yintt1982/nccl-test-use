import { createServer } from 'node:http';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const temp = await mkdtemp(join(tmpdir(), 'nccl-report-'));
process.env.NCCL_REPORT_DATA_DIR = temp;
const { handler } = await import('./server.mjs');

const server = createServer(handler);
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
try {
  const base = 'http://127.0.0.1:' + server.address().port;
  const data = await readFile('local-report/fixture.json', 'utf8');
  const imported = await fetch(base + '/api/runs/fixture-01/events', { method: 'POST', body: data });
  const result = await (await fetch(base + '/api/runs/fixture-01/report')).json();
  const filtered = await (await fetch(base + '/api/runs/fixture-01/report?rank=1')).json();
  const csv = await (await fetch(base + '/api/runs/fixture-01/report?rank=1&format=csv')).text();
  const duplicate = await fetch(base + '/api/runs/fixture-01/events', { method: 'POST', body: data });
  const wrong = await fetch(base + '/api/runs/fixture-01/events', { method: 'POST', body: JSON.stringify([{ schema_version: '9.0' }]) });
  const conflictData = JSON.parse(data);
  conflictData[0].rank = 7;
  const conflict = await fetch(base + '/api/runs/fixture-01/events', { method: 'POST', body: JSON.stringify(conflictData) });
  const repeated = await fetch(base + '/api/runs/fixture-02/events', { method: 'POST', body: JSON.stringify([
    { ...conflictData[0], run_id: 'fixture-02' }, { ...conflictData[0], run_id: 'fixture-02', rank: 8 }
  ]) });
  const sameTwice = await fetch(base + '/api/runs/fixture-03/events', { method: 'POST', body: JSON.stringify([
    { ...conflictData[0], run_id: 'fixture-03' }, { ...conflictData[0], run_id: 'fixture-03' }
  ]) });
  const sameTwiceReport = await (await fetch(base + '/api/runs/fixture-03/report')).json();
  const qualityEvents = [
    { schema_version: '1.0', run_id: 'quality-01', event_id: 'open', parent_event_id: null, source: 'sample_replay', source_event_type: 'collective', rank: 0, source_clock_domain: 'host_monotonic', source_timestamp_ns: 1, lifecycle: 'start' },
    { schema_version: '1.0', run_id: 'quality-01', event_id: 'bad-field', parent_event_id: null, source: 'sample_replay', source_event_type: 'proxy_step', rank: 0, source_clock_domain: 'host_monotonic', source_timestamp_ns: 2, validity: { peer: 'missing' } }
  ];
  const qualityImport = await fetch(base + '/api/runs/quality-01/events', { method: 'POST', body: JSON.stringify(qualityEvents) });
  const quality = await (await fetch(base + '/api/runs/quality-01/report')).json();
  const chunksCsv = await (await fetch(base + '/api/runs/quality-01/report?section=chunks&format=csv')).text();
  const page = await (await fetch(base + '/')).text();
  const client = await fetch(base + '/local-report/client.js');
  if (imported.status !== 200 || result.rows.length !== 3 || result.chunks.length !== 2 ||
      result.rows[0].quality !== 'observed' || result.rows[2].quality !== 'orphan' ||
      filtered.rows.length !== 1 || !csv.includes('orphan') || duplicate.status !== 200 ||
      (await duplicate.json()).accepted !== 0 || wrong.status !== 400 || conflict.status !== 409 || repeated.status !== 409 ||
      !page.includes('/local-report/client.js') || client.status !== 200 || sameTwice.status !== 200 || sameTwiceReport.rows.length !== 1 || qualityImport.status !== 200 ||
      quality.rows[0].quality !== 'unclosed' || quality.rows[1].quality !== 'missing' ||
      !chunksCsv.includes('bad-field') || chunksCsv.includes('"open"'))
    throw Error('Report contract failed');
  console.log('PASS fixture import, quality, filter and CSV');
} finally { server.close(); await rm(temp, { recursive: true, force: true }); }


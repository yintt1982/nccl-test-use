import { createServer } from 'node:http';
import { readFile, mkdir, appendFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dataDir = resolve(process.env.NCCL_REPORT_DATA_DIR || join(root, 'local-report/data'));
const port = Number(process.env.PORT || 9008);
const send = (res, status, value, type = 'application/json; charset=utf-8') => {
  res.writeHead(status, { 'content-type': type });
  res.end(type.startsWith('application/json') ? JSON.stringify(value) : value);
};
const valid = e => e && e.schema_version === '1.0' && /^[\w-]+$/.test(e.run_id || '') &&
  typeof e.event_id === 'string' && e.event_id &&
  ['sample_replay', 'abi_harness', 'nccl_profiler'].includes(e.source) &&
  ['proxy_step', 'proxy_op', 'collective', 'kernel'].includes(e.source_event_type) &&
  Number.isInteger(e.rank) && e.rank >= 0 && typeof e.source_clock_domain === 'string' &&
  Number.isFinite(e.source_timestamp_ns);
async function readEvents(runId) {
  try { return (await readFile(join(dataDir, runId + '.jsonl'), 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse); }
  catch (e) { if (e.code === 'ENOENT') return []; throw e; }
}
function project(all, params) {
  const ids = new Set(all.map(e => e.event_id));
  const rows = all.filter(e => (!params.get('rank') || String(e.rank) === params.get('rank')) &&
    (!params.get('source') || e.source === params.get('source')) &&
    (!params.get('q') || JSON.stringify(e).toLowerCase().includes(params.get('q').toLowerCase())))
    .map(e => ({ ...e, source_label: e.source === 'abi_harness' ? 'SAMPLE ABI - RUNTIME UNVERIFIED' :
      e.source === 'sample_replay' ? 'SAMPLE REPLAY' : 'NCCL PROFILER', quality: e.parent_event_id && !ids.has(e.parent_event_id) ? 'orphan' :
      e.lifecycle === 'start' && !all.some(x => x.parent_event_id === e.event_id && x.lifecycle === 'stop') ? 'unclosed' :
      e.validity && Object.values(e.validity).some(x => x === 'missing' || x === 'invalid') ? 'missing' : 'observed' }));
  return { run_id: all[0]?.run_id || null, total: all.length, rows,
    chunks: rows.filter(e => e.source_event_type === 'proxy_step'), timeline: rows };
}
export function handler(req, res) {
  (async () => {
    const url = new URL(req.url, 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/') return send(res, 200, await readFile(join(root, 'prototype-nccl/index.html'), 'utf8'), 'text/html; charset=utf-8');
    if (req.method === 'GET' && url.pathname === '/local-report/client.js') return send(res, 200, await readFile(join(root, 'local-report/client.js'), 'utf8'), 'text/javascript; charset=utf-8');
    const m = /^\/api\/runs\/([\w-]+)\/(events|report)$/.exec(url.pathname);
    if (!m) return send(res, 404, { error: 'Not found' });
    const [, runId, resource] = m;
    if (req.method === 'POST' && resource === 'events') {
      let body = '';
      for await (const part of req) { body += part; if (body.length > 2000000) return send(res, 413, { error: 'Batch too large' }); }
      let batch;
      try { batch = JSON.parse(body); if (!Array.isArray(batch) || !batch.length || batch.some(e => !valid(e) || e.run_id !== runId)) throw Error(); }
      catch { return send(res, 400, { error: 'Invalid event batch' }); }
      const old = await readEvents(runId), ids = new Map(old.map(e => [e.event_id, JSON.stringify(e)]));
      for (const event of batch) {
        const serialized = JSON.stringify(event);
        if (ids.has(event.event_id) && ids.get(event.event_id) !== serialized) return send(res, 409, { error: 'Conflicting event_id' });
        ids.set(event.event_id, serialized);
      }
      const persisted = new Set(old.map(e => e.event_id));
      const fresh = batch.filter(e => {
        if (persisted.has(e.event_id)) return false;
        persisted.add(e.event_id);
        return true;
      });
      await mkdir(dataDir, { recursive: true });
      if (fresh.length) await appendFile(join(dataDir, runId + '.jsonl'), fresh.map(JSON.stringify).join('\n') + '\n');
      return send(res, 200, { accepted: fresh.length, total: old.length + fresh.length });
    }
    if (req.method === 'GET' && resource === 'report') {
      const result = project(await readEvents(runId), url.searchParams);
      const selected = url.searchParams.get('section') === 'chunks' ? result.chunks : result.rows;
      if (url.searchParams.get('format') === 'csv') {
        const fields = ['event_id', 'source', 'source_label', 'source_event_type', 'rank', 'channel_id', 'peer', 'source_clock_domain', 'source_timestamp_ns', 'quality'];
        return send(res, 200, [fields.join(','), ...selected.map(row => fields.map(k => '"' + String(row[k] ?? '').replaceAll('"', '""') + '"').join(','))].join('\r\n'), 'text/csv; charset=utf-8');
      }
      return send(res, 200, result);
    }
    send(res, 405, { error: 'Method not allowed' });
  })().catch(e => send(res, 500, { error: e.message }));
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename))
  createServer(handler).listen(port, '127.0.0.1', function () {
    console.log('Local report: http://127.0.0.1:' + this.address().port);
  });

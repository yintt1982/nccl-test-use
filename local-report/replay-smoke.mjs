import { createServer } from 'node:http';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

const temp = await mkdtemp(join(tmpdir(), 'nccl-replay-'));
process.env.NCCL_REPORT_DATA_DIR = temp;
const { handler } = await import('./server.mjs');
const server = createServer(handler);
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
try {
  const base = 'http://127.0.0.1:' + server.address().port;
  const fixture = JSON.parse(await readFile('local-report/fixture.json', 'utf8'));
  const runId = 'replay-test';
  const batch = fixture.map(e => ({ ...e, run_id: runId }));
  for (let i = 0; i < batch.length; i++) {
    const response = await fetch(base + '/api/runs/' + runId + '/events', { method: 'POST', body: JSON.stringify([batch[i]]) });
    const report = await (await fetch(base + '/api/runs/' + runId + '/report')).json();
    if (response.status !== 200 || report.rows.length !== i + 1 || report.rows.some(e => e.source !== 'sample_replay')) throw Error('Replay batch did not refresh the report');
  }
  const csv = await (await fetch(base + '/api/runs/' + runId + '/report?format=csv')).text();
  if (!csv.includes('SAMPLE REPLAY')) throw Error('Export omitted sample provenance');
  const cliRun = 'replay-cli';
  const child = spawn(process.execPath, ['local-report/replay.mjs', cliRun, base, '5'], { stdio: 'pipe' });
  const exitCode = await new Promise(resolve => child.on('close', resolve));
  const replayed = await (await fetch(base + '/api/runs/' + cliRun + '/report')).json();
  if (exitCode !== 0 || replayed.rows.length !== fixture.length) throw Error('Replay command did not import each batch');
  console.log('PASS replay batches and SAMPLE REPLAY export');
} finally { server.close(); await rm(temp, { recursive: true, force: true }); }

import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';

const temp = await mkdtemp(join(tmpdir(), 'nccl-abi-'));
try {
  const exe = join(temp, 'profiler-host.exe');
  const output = join(temp, 'events.jsonl');
  process.env.NCCL_REPORT_DATA_DIR = temp;
  const { handler } = await import('./server.mjs');
  const server = createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const result = spawn(process.execPath, ['local-report/build-abi.mjs', temp], { stdio: 'inherit' });
  const built = await new Promise(resolve => result.on('close', resolve));
  if (built !== 0) throw Error('ABI plugin build failed');
  const ran = spawn(exe, ['--plugin', join(temp, 'profiler-plugin.dll'), '--output', output], {
    stdio: 'inherit', env: { ...process.env, NCCL_PROFILER_JSONL: output }
  });
  const exit = await new Promise(resolve => ran.on('close', resolve));
  if (exit !== 0) throw Error('ABI host failed');
  const lines = (await readFile(output, 'utf8')).trim().split(/\r?\n/);
  const events = lines.map(line => JSON.parse(line));
  const base = 'http://127.0.0.1:' + server.address().port;
  const imported = await fetch(base + '/api/runs/abi-harness/events', { method: 'POST', body: JSON.stringify(events) });
  const report = await (await fetch(base + '/api/runs/abi-harness/report')).json();
  const csv = await (await fetch(base + '/api/runs/abi-harness/report?format=csv')).text();
  if (events.length !== 6 || events[0].source !== 'abi_harness' ||
      events[0].source_event_type !== 'collective' || events[2].source_event_type !== 'proxy_op' ||
      events[4].source_event_type !== 'proxy_step' || events.filter(e => e.lifecycle === 'start').length !== 3 ||
      events.filter(e => e.lifecycle === 'stop').length !== 3 ||
      events.some(e => !e.event_id || e.source !== 'abi_harness') || imported.status !== 200 ||
      events[2].channel_id !== 2 || events[2].peer !== 1 || events[4].step !== 3 ||
      events[3].parent_event_id !== events[2].event_id ||
      report.rows.length !== 6 || report.rows.some(e => e.quality !== 'observed') ||
      !csv.includes('SAMPLE ABI - RUNTIME UNVERIFIED'))
    throw Error('ABI callbacks did not produce the expected synthetic event JSONL');
  console.log('PASS NCCL Profiler v7 ABI loading, callback lifecycle and JSONL');
  server.close();
} finally { await rm(temp, { recursive: true, force: true }); }

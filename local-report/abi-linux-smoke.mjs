import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';

if (process.platform !== 'linux') throw Error('Run this smoke test on Linux; Windows has a separate LoadLibrary harness');
const temp = await mkdtemp(join(tmpdir(), 'nccl-abi-linux-'));
process.env.NCCL_REPORT_DATA_DIR = temp;
const { handler } = await import('./server.mjs');
const server = createServer(handler);
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function run(command, args, environment = process.env) {
  const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], env: environment });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; process.stdout.write(chunk); });
  child.stderr.on('data', chunk => { output += chunk; process.stderr.write(chunk); });
  const code = await new Promise(resolve => child.on('close', resolve));
  if (code !== 0) {
    const summary = output.split(/\r?\n/).filter(line => /error:|fatal error:|failed/i.test(line)).slice(0, 3).join(' | ') || command + ' failed with exit code ' + code;
    console.error('::error title=Linux ABI harness::' + summary.replaceAll('%', '%25').replaceAll('\r', '%0D').replaceAll('\n', '%0A'));
    throw Error(command + ' failed with exit code ' + code);
  }
}
try {
  const plugin = join(temp, 'profiler-plugin.so');
  const host = join(temp, 'profiler-host');
  const sourceRoot = resolve('nccl-tests-master/src');
  await run(process.env.CXX || 'g++', ['-std=c++17', '-fPIC', '-shared', '-Ilocal-report/abi', '-I' + sourceRoot, 'local-report/abi/linux/profiler_plugin.cpp', '-o', plugin]);
  await run(process.env.CXX || 'g++', ['-std=c++17', '-Ilocal-report/abi', '-I' + sourceRoot, 'local-report/abi/linux/profiler_host.cpp', '-ldl', '-o', host]);
  const output = join(temp, 'events.jsonl');
  await run(host, ['--plugin', plugin, '--output', output], { ...process.env, NCCL_PROFILER_JSONL: output });
  const events = (await readFile(output, 'utf8')).trim().split(/\r?\n/).map(JSON.parse);
  const base = 'http://127.0.0.1:' + server.address().port;
  const imported = await fetch(base + '/api/runs/abi-harness/events', { method: 'POST', body: JSON.stringify(events) });
  const report = await (await fetch(base + '/api/runs/abi-harness/report')).json();
  const csv = await (await fetch(base + '/api/runs/abi-harness/report?format=csv')).text();
  if (events.length !== 6 || events.filter(e => e.lifecycle === 'start').length !== 3 ||
      events.filter(e => e.lifecycle === 'stop').length !== 3 || events[2].peer !== 1 ||
      events[2].channel_id !== 2 || events[2].is_send !== 1 ||
      events[2].declared_n_steps !== 4 || events[2].declared_chunk_size_bytes !== 1024 ||
      events[4].step !== 3 || imported.status !== 200 ||
      report.rows.length !== 6 || report.rows.some(e => e.quality !== 'observed') ||
      !csv.includes('SAMPLE ABI - RUNTIME UNVERIFIED')) throw Error('Linux ABI report contract failed');
  console.log('PASS Linux NCCL Profiler v7 dlopen, callbacks, JSONL import and report');
} finally {
  server.close();
  await rm(temp, { recursive: true, force: true });
}

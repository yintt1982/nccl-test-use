import { spawn } from 'node:child_process';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const temp = await mkdtemp(join(tmpdir(), 'nccl-browser-'));
const server = spawn(process.execPath, ['local-report/server.mjs'], {
  stdio: ['ignore', 'pipe', 'inherit'], env: { ...process.env, PORT: '0', NCCL_REPORT_DATA_DIR: temp }
});
let startup = '';
const base = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(Error('Local report server did not start')), 10000);
  server.stdout.on('data', chunk => {
    startup += chunk.toString();
    const match = startup.match(/http:\/\/127\.0\.0\.1:(\d+)/);
    if (match) { clearTimeout(timeout); resolve('http://127.0.0.1:' + match[1]); }
  });
  server.on('error', reject);
});
try {
  const fixture = await readFile('local-report/fixture.json', 'utf8');
  const imported = await fetch(base + '/api/runs/browser-smoke/events', { method: 'POST', body: fixture.replaceAll('fixture-01', 'browser-smoke') });
  if (!imported.ok) throw Error('Fixture import failed');
  const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  const result = spawn(chrome, ['--headless', '--no-sandbox', '--disable-gpu', '--virtual-time-budget=2500', '--dump-dom', base + '/?run_id=browser-smoke'], { stdio: ['ignore', 'pipe', 'inherit'] });
  let html = '';
  result.stdout.on('data', chunk => { html += chunk.toString(); });
  const exit = await new Promise(resolve => result.on('close', resolve));
  if (exit !== 0 || !html.includes('Local event report') || !html.includes('SAMPLE REPLAY') ||
      !html.includes('orphan-1') || !html.includes('step-1') ||
      !html.includes('synthetic sources are not hardware measurements'))
    throw Error('Browser did not render imported event rows with sample provenance: ' + JSON.stringify({
      exit, rendered: html.includes('Local event report'), replay: html.includes('SAMPLE REPLAY'),
      orphan: html.includes('orphan-1'), step: html.includes('step-1'),
      provenance: html.includes('synthetic sources are not hardware measurements'), size: html.length
    }));
  console.log('PASS browser import-to-report rendering and provenance');
} finally {
  server.kill();
  await rm(temp, { recursive: true, force: true });
}

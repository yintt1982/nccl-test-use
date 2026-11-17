import { readFile } from 'node:fs/promises';

const [runId = 'fixture-01', base = 'http://localhost:9008', delayText = '500'] = process.argv.slice(2);
const delay = Number(delayText);
if (!/^[\w-]+$/.test(runId) || !Number.isFinite(delay) || delay < 0) throw Error('Usage: node local-report/replay.mjs [run-id] [base-url] [delay-ms]');
const fixture = JSON.parse(await readFile(new URL('./fixture.json', import.meta.url), 'utf8'));
for (const [index, event] of fixture.entries()) {
  const response = await fetch(base + '/api/runs/' + runId + '/events', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify([{ ...event, run_id: runId, source: 'sample_replay' }])
  });
  if (!response.ok) throw Error('Replay batch ' + (index + 1) + ' failed: ' + response.status);
  console.log('SAMPLE REPLAY ' + (index + 1) + '/' + fixture.length);
  if (index < fixture.length - 1) await new Promise(resolve => setTimeout(resolve, delay));
}

(function () {
  if (!location.protocol.startsWith('http')) return;
  const runId = new URLSearchParams(location.search).get('run_id') || 'fixture-01';
  const fields = ['event_id', 'source', 'source_event_type', 'rank', 'channel_id', 'peer', 'source_clock_domain', 'source_timestamp_ns', 'quality'];
  const timers = [];
  const safe = value => String(value ?? '—').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  for (const section of ['chunks', 'timeline']) {
    const panel = document.querySelector('[data-report-panel=' + section + ']');
    if (!panel) continue;
    const box = document.createElement('div');
    box.className = 'report-card';
    box.innerHTML = '<h3>Local event report</h3><input type=search placeholder="Filter local events"><a download>Export CSV</a><p class=local-status></p><div class=local-chart></div><div class=local-table></div>';
    panel.replaceChildren(box);
    const render = async () => {
      const params = new URLSearchParams({ q: box.querySelector('input').value, section });
      const response = await fetch('/api/runs/' + encodeURIComponent(runId) + '/report?' + params);
      if (!response.ok) throw Error('Report unavailable');
      const rows = (await response.json())[section];
      const sources = [...new Set(rows.map(x => x.source))];
      box.querySelector('h3').textContent = 'Local event report · ' + sources.map(source =>
        source === 'abi_harness' ? 'SAMPLE ABI · RUNTIME UNVERIFIED' :
        source === 'sample_replay' ? 'SAMPLE REPLAY' : 'NCCL PROFILER').join(', ');
      box.querySelector('.local-status').textContent = rows.length + ' events · ' + (sources.join(', ') || 'none') + ' · synthetic sources are not hardware measurements';
      box.querySelector('.local-chart').innerHTML = rows.map((row, i) => '<div>' + safe(row.event_id) + ' <progress value=' + (i + 1) + ' max=' + rows.length + '></progress></div>').join('');
      box.querySelector('.local-table').innerHTML = '<table class=table><thead><tr>' + fields.map(x => '<th>' + x + '</th>').join('') + '</tr></thead><tbody>' + rows.map(row => '<tr>' + fields.map(x => '<td>' + safe(row[x]) + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
      box.querySelector('a').href = '/api/runs/' + encodeURIComponent(runId) + '/report?' + params + '&format=csv';
    };
    box.querySelector('input').oninput = () => render().catch(console.error);
    render().catch(console.error);
    timers.push(setInterval(() => {
      if (!document.hidden) render().catch(console.error);
    }, 1000));
  }
  window.addEventListener('pagehide', () => timers.forEach(clearInterval));
})();

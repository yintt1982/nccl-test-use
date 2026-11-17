# Local NCCL report path (TIN-47)

From the repository root, run node local-report/server.mjs and open http://localhost:9008.

Import the fixed fixture with PowerShell:

    Invoke-RestMethod http://localhost:9008/api/runs/fixture-01/events -Method Post -ContentType application/json -InFile local-report/fixture.json

Read http://localhost:9008/api/runs/fixture-01/report. Append ?format=csv to export filtered rows. Use rank, source, and q query parameters for both formats.

For incremental replay, run node local-report/replay.mjs replay-01 and open http://localhost:9008/?run_id=replay-01. The Chunks and Timeline report sections refresh every second while the page is visible. All replay rows and CSV exports are marked SAMPLE REPLAY.

The fixture source is sample_replay; it is not a measurement. Event files are stored under local-report/data/.

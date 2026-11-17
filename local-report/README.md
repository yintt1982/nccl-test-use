# Local NCCL report path (TIN-47)

From the repository root, run node local-report/server.mjs and open http://localhost:9008.

Import the fixed fixture with PowerShell:

    Invoke-RestMethod http://localhost:9008/api/runs/fixture-01/events -Method Post -ContentType application/json -InFile local-report/fixture.json

Read http://localhost:9008/api/runs/fixture-01/report. Append ?format=csv to export filtered rows. Use rank, source, and q query parameters for both formats.

For incremental replay, run node local-report/replay.mjs replay-01 and open http://localhost:9008/?run_id=replay-01. The Chunks and Timeline report sections refresh every second while the page is visible. All replay rows and CSV exports are marked SAMPLE REPLAY.

The fixture source is sample_replay; it is not a measurement. Event files are stored under local-report/data/.

## Profiler ABI harness (TIN-49)

On Windows with Visual Studio 2022 C++ tools installed, run node local-report/abi-smoke.mjs. The harness builds a synthetic NCCL Profiler v7 plugin, loads it with LoadLibrary, exercises init/start/state/stop/finalize callbacks, writes paired JSONL lifecycle events, and imports them through the same report API. The small local-report/abi/nccl.h only supplies NCCL result codes and the Windows pid type required by the vendored profiler interface headers. This verifies the synthetic ABI harness; it does not verify compatibility with a real NCCL runtime or GPU.

The target Linux dlopen harness is node local-report/abi-linux-smoke.mjs. It builds a synthetic .so with g++ and verifies the same callbacks and report import. It runs in CI on Ubuntu; the current Windows environment cannot execute Linux binaries.

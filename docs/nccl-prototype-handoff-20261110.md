# NCCL Tests Prototype Cross-Machine Handoff

Generated: 2026-11-10, Asia/Shanghai  
Workspace: `E:\nccl-tests`  
Purpose: continue the current NCCL Tests prototype, observability design, implementation, validation, and deployment on another machine.

## 1. Session Intent

Continue improving the existing NCCL Tests web prototype. The agreed product surfaces are:

1. Test configuration list and configuration details.
2. Logical topology for nodes, MPI processes, threads, GPUs, NCCL ranks, and split communicator groups.
3. Native nccl-tests performance reports.
4. Group concurrency report.
5. Horizontal scaling efficiency report.
6. Enhanced observability reports: Chunks, Timeline, Flows, QPairs, and Ports.
7. Every metric must show its actual source and collection state.
8. Sample or inferred values must never be presented as real measurements.

## 2. Primary Artifact

Main prototype file:

`E:\nccl-tests\prototype-nccl\index.html`

File metadata at handoff time:

- Size: `242777` bytes
- Last modified: `2026-11-07 20:26:19`
- SHA-256: `67CDB562FE747E65CB408393632A2B184A874740017C49E0C361291974D6F04E`

The prototype is a single HTML file containing CSS, JavaScript, navigation, local state, and clearly labeled demonstration data. It does not use npm or package.json.

## 3. Navigation and Configuration UI

The main sidebar only contains:

```text
Test Applications
  Collective Communication Benchmark

Analysis and Reports
  Test Report List
```

The Collective Communication Benchmark landing page is the configuration list. The Test Report List landing page is the report list.

Implemented configuration interactions include:

- Create a configuration.
- Enter an existing configuration.
- Delete an existing configuration.
- Save configuration with a generated default name.
- Back buttons name their destination, such as configuration list or report list.
- Physical setup, logical infrastructure, execution parameters, and NCCL settings.
- Logical topology for MPI process, thread, GPU, and rank relationships.
- GPU labels use the format `GPU0:  G0 C0`.
- Split topology based on `NCCL_TESTS_SPLIT`.
- Aligned configuration fields and consistent English hover help.
- Run Test opens the statistics report rather than the report list.

## 4. Report Workbench Prototype

The report page follows the accepted Variant A structure: a left-side report workbench with grouped dimensions.

Current report panels (12 total):

- `summary`
- `performance`
- `stability`
- `chunks`
- `timeline`
- `flows`
- `qpairs`
- `ports`
- `grouped`
- `scaling`
- `details`
- `iterations`

Information architecture:

```text
Summary

Base Performance
  Performance Curve
  Stability and Tail Latency
  Details
  Iterations

Enhanced Observability
  Chunks
  Timeline
  Flows
  QPairs
  Ports

Derived Analysis
  Group Concurrency
  Horizontal Scaling
```

The Summary page includes:

- Collection mode.
- Data completeness.
- Covered nodes.
- Covered ranks.
- Lost event count.
- Clock calibration status.
- Eight shortcut cards linking to report dimensions.

## 5. Prototype Data-State Rules

The UI must distinguish these states:

```text
Real collection
Sample replay
Partial
Not collected
Not applicable
Collection failed
Inferred association
```

Current prototype status:

- Chunks: sample replay for UI evaluation.
- Timeline: sample replay for UI evaluation.
- Group Concurrency: sample replay for UI evaluation.
- Horizontal Scaling: sample replay for UI evaluation.
- Flows: not collected / requires network-layer association.
- QPairs: not collected.
- Ports: not applicable or not collected depending on the run path.

Do not describe any current sample value as a real GPU test result.

## 6. UI Verification Evidence

Visual review files:

- `E:\nccl-tests\prototype-nccleport-workbench-timeline.png`
- `E:\nccl-tests\prototype-nccleport-workbench-scaling.png`

The previous session verified all 12 report navigation entries in a browser. Each navigation item had exactly one corresponding panel, no panel was blank, and Summary shortcut cards could open report panels such as Timeline.

## 7. Starting the Prototype Locally

Run in PowerShell:

```powershell
python -m http.server 8080 --directory E:
ccl-tests\prototype-nccl
```

Open:

```text
http://127.0.0.1:8080/
```

Background start:

```powershell
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList '-m','http.server',8080,'--directory','E:
ccl-tests\prototype-nccl'
```

Verification:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/
```

The local 8080 server was not running when this handoff document was generated. Start it again on the new machine.

## 8. Accepted Enhanced Observability Architecture

Target dimensions:

1. Chunks.
2. Timeline.
3. Flows.
4. QPairs.
5. Ports.

Accepted delivery phases:

```text
Phase 1: Chunks + Timeline
Phase 2: Flows + QPairs
Phase 3: Ports
Phase 4: Cross-layer correlation and root-cause analysis
```

Accepted architecture:

```text
nccl-tests
+ NCCL Profiler Plugin
+ NCCL Network Plugin
+ RDMA/QP Collector
+ NIC/HCA Port Collector
+ Local Agent
+ Correlation Service
+ Report API
```

Every reported value must declare one of these sources:

- Native nccl-tests result.
- NCCL Profiler event.
- NCCL Network Plugin event.
- RDMA/NIC hardware counter.
- Platform-derived metric.

## 9. Collection Modes and Files

Accepted user-facing collection modes:

```text
Off: base nccl-tests only
Standard: NCCL Chunks and Timeline
Network Diagnostics: QP, NIC, and Port collection in addition
```

Support both single-run collection and historical/cross-run comparison.

Recommended run directory:

```text
runs/
  run-<id>/
    manifest.json
    rank-metadata.json
    nccl-events.jsonl
    network-events.jsonl
    qp-counters.jsonl
    port-counters.jsonl
    nccl-tests.json
```

Common event fields:

```text
timestamp_ns
clock_source
clock_domain
host
pid
rank
gpu
```

Accepted correlation keys:

```text
run_id
node_id
process_id
mpi_world_rank
group_color
group_local_rank
gpu_id
collective_seq
iteration
placement
channel_id
proxy_op_id
proxy_step
peer_rank
connection_id
qp_id
port_id
```

## 10. Phase-1 Chunks and Timeline Schema

Minimum fields:

```text
run_id
collective_seq
iteration
rank
group_color
peer
channel_id
nSteps
chunkSize
isSend
event_state
start_timestamp
end_timestamp
duration
```

Keep two distinct time boundaries:

```text
NCCL Proxy Start -> Proxy Done
GPU Kernel Channel Start -> Stop
```

They are complementary and must not replace one another.

Fields that must remain explicitly unavailable until collected or correlated:

```text
Data Chunk Transfer ID
Buffer Offset
Source NIC
Destination NIC
Source QP
Destination QP
Payload Bytes
```

## 11. Deployment Model

Preferred deployment architecture:

```text
Each GPU node:
  nccl-tests
  NCCL Profiler Plugin
  Network/RDMA Collector
  NIC/HCA Port Collector
  Local Agent

Central service:
  Run Metadata
  Event Correlation
  CDF/P99
  Group Concurrency Analysis
  Horizontal Scaling Analysis
  Report API
```

The Local Agent writes locally first. Central-service outages must not silently lose data; upload can resume later.

## 12. Version Compatibility

Maintain a capability registry containing:

```text
NCCL version
Profiler API version
userTag availability
Kernel Phase availability
Proxy Step completeness
Network Plugin correlation availability
Verified event-state set
```

Rules:

- Prefer Profiler v7 sequence numbers and `userTag`.
- If `userTag` is unavailable, sequential iteration association is allowed only when labeled inferred and lower confidence.
- If Kernel Phase is unavailable, hide GPU Kernel Phase analysis.
- If Proxy Step is unavailable, show only Collective or Proxy Operation granularity.
- If Network Plugin association is unavailable, do not show precise Flow/QP mappings.
- If hardware access is unavailable, show collection failed or not applicable.

## 13. Profiler Plugin Conflict

`nccl-tests -U` currently uses:

```text
NCCL_PROFILER_PLUGIN=STATIC_PLUGIN
```

Phase 1 does not support `-U` and the custom enhanced-observability Profiler Plugin simultaneously. The UI must detect and block this conflict. It must not silently overwrite a user-selected Profiler Plugin.

## 14. Quality Gates

Initial accepted thresholds:

```text
Rank coverage                 = 100%
Collective coverage           = 100%
Event loss rate               <= 0.1%
Unclosed Chunk/Proxy events   <= 0.1%
Time-reversed events          = 0
Detailed-observation overhead <= 5%
Network-diagnostics overhead  <= 10%
```

Per-metric quality states:

```text
Complete
Partial
Not collected
Not applicable
```

If a quality gate fails, the report must say:

```text
Insufficient data quality; no diagnostic conclusion generated.
```

Overload handling must never silently drop events. Record received, written, lost, and truncated counts by event type. If disk limits are reached, stop enhanced collection while allowing the base nccl-tests run to finish.

## 15. Group Concurrency Report

Requirements:

- Based on `NCCL_TESTS_SPLIT`.
- Show Group Color, group members, and group-local ranks.
- Events must not cross groups.
- Separate per-group bandwidth from aggregate bandwidth.
- Do not invent independent G0/G1 bandwidth values without independent per-group evidence.

Golden test example:

```text
NCCL_TESTS_SPLIT=MOD 2
```

## 16. Horizontal Scaling Report

Recommended comparison points:

```text
1 node
2 nodes
4 nodes
```

Requirements:

- Align by message size.
- Reject efficiency calculation when environments are inconsistent.
- Show busbw retention.
- Show latency amplification.
- Show P99 change.
- Trace every scale point to its source task and raw event files.

## 17. Golden Acceptance Tests

1. Single node, single communicator group.
2. Multiple nodes, single communicator group.
3. Split-group concurrency.
4. Horizontal scaling across 1, 2, and 4 nodes.

Every test must be traceable to raw event files.

## 18. Verified Upstream Source Facts

Upstream source directory:

`E:\nccl-tests\nccl-tests-master`

Relevant Profiler headers:

- `nccl-tests-master/src/nccl_profiler.h`
- `nccl-tests-master/src/profiler_v5.h`
- `nccl-tests-master/src/profiler_v6.h`
- `nccl-tests-master/src/profiler_v7.h`

Verified available concepts and fields include:

```text
ncclProfileColl
ncclProfileProxyOp
ncclProfileProxyStep
ncclProfileKernelCh
ncclProfileNetPlugin
ncclProfileKernelPhase
channelId
peer
nSteps
chunkSize
isSend
rank
seqNumber
algo
proto
nChannels
Profiler v7 userTag
```

The native `-J` JSON file is for nccl-tests results and iteration data. It is not an appropriate container for all high-frequency Profiler, QP, and Port events. Use separate JSONL or Parquet event files.

## 19. Reference Documents

Read these first on the new machine:

- `E:\nccl-tests\docs\nccl-rank-topology-and-observability.md`
- `E:\nccl-tests\docs\nccl-rank-topology-and-observability.html`
- `E:\nccl-tests\docs\nccl-tests-native-report-data-analysis.md`
- `E:\nccl-tests\docs\nccl-rank-multigpu-and-split.md`
- `E:\nccl-tests\docs\nccl-rank-multigpu-and-split.html`
- `E:\nccl-tests\prototype-nccl\LOGICAL-TOPOLOGY-VERIFICATION.md`

## 20. Remote Deployment

Remote prototype URL:

```text
http://10.11.58.253:9007/
```

Deployment constraints:

- Only modify the NCCL Tests application on port 9007.
- Do not modify port 9006 or any other application, service, port, or directory.
- Back up and verify before replacement.
- Use the `nccl-tests-9007-deploy` skill.
- SSH port: 22.
- SSH user: `root`.
- The plaintext SSH password is intentionally omitted. Ask the user for it securely on the new machine if needed.

## 21. Remaining Work

1. Convert the accepted prototype into a maintainable implementation instead of continuing indefinite single-file growth.
2. Create the accepted but not-yet-written formal documents:
   - `docs/enhanced-observability-design.md`
   - `docs/enhanced-observability-schema.md`
   - `docs/enhanced-observability-acceptance.md`
3. Implement the real event model and `manifest.json`.
4. Implement capability probing.
5. Implement the Local Agent.
6. Implement a minimal Profiler Plugin collection loop.
7. Implement Chunks and Timeline file import and Report API support.
8. Replace fixed UI examples with real files or explicitly labeled replay fixtures.
9. Implement Flows, QPairs, and Ports later.
10. Repeat full browser navigation verification before deployment.

## 22. Known Risks

- `prototype-nccl/index.html` is a large single-file prototype produced through many incremental edits. It includes historical compatibility logic and repeated style layers. Make localized edits; do not rewrite the entire file without a verified migration plan.
- Some historical source text had encoding artifacts. The current browser-visible main UI is usable. Avoid whole-file encoding conversion or global string replacement.
- Current report values are not real GPU measurements.
- The final Chunks and Timeline semantics still require verification against a real Profiler event capture.
- Flow, QP, and Port mappings require evidence. Do not infer precise mappings from aggregate bandwidth or static topology.
- Local HTTP server processes do not migrate with the files and must be restarted.

## 23. Recommended Next Step

For continued UI prototyping:

1. Start the local prototype.
2. Open the Test Report List and select a sample report.
3. Review Summary, Chunks, Timeline, Flows, QPairs, Ports, Group Concurrency, and Horizontal Scaling.
4. Refine filters, empty states, quality explanations, and drill-down paths.
5. Verify every navigation entry in a browser before deployment.

For engineering implementation:

1. Turn the accepted design into formal documents.
2. Use the `implement` skill.
3. Use `tdd` for report navigation, state labels, filtering, and data import.
4. Use `code-review` for standards and specification review.

## 24. Starter Prompt for the New Machine

Copy the full `E:\nccl-tests` directory to the new machine, then send this prompt:

```text
The workspace is E:
ccl-tests.
First read docs/nccl-prototype-handoff-20261110.md.
Then inspect prototype-nccl/index.html and its report workbench.
Do not reopen already accepted architecture decisions, and do not treat sample data as real measurements.
Start the local prototype and verify Summary, Chunks, Timeline, Flows, QPairs, Ports,
Group Concurrency, and Horizontal Scaling before making changes.
Continue from my next instruction.
If deployment to 10.11.58.253:9007 is requested, use nccl-tests-9007-deploy and only modify my port-9007 application.
Do not touch any other port, service, application, or directory.
```

## 25. Handoff Probes

A new session should be able to answer:

1. What is the primary prototype file?
2. Which reports are native nccl-tests results and which require enhanced collection?
3. What is the current state of the Chunks and Timeline values?
4. Why can Flow, QP, and Port mappings not be derived precisely from static topology?
5. What conflict exists between `-U` and a custom Profiler Plugin?
6. What is permitted during remote deployment?
7. Which two enhanced reports belong to Phase 1?

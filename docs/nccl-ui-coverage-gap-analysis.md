# Current nccl-tests Prototype Coverage Gaps

Research date: 2026-10-26. Page: http://112.25.236.50:9007/. Sources: current prototype UI, NVIDIA nccl-tests upstream README, and local source at `E:/dpi/nccl-tests-master`. This is static source analysis; no GPU benchmark was run. The local `src/common.h` version macro says `2.21.1`, while its README/source tree includes features with newer NCCL requirements and has no `.git` metadata, so the local snapshot cannot be proven to be the latest upstream revision. The coverage inventory below describes the checked-out tree; verify supported binaries and dependencies on the target host and refresh the source snapshot before treating it as a latest-version audit.

## Conclusion

The prototype does **not** cover all nccl-tests functionality. It covers nine standard collective choices and most basic size/rank/timing parameters. The local source's standard benchmark set includes 11 collectives: AllReduce, AllGather, Broadcast, ReduceScatter, Reduce, AlltoAll, AlltoAllv, Scatter, Gather, SendRecv, and Hypercube. The UI is missing AlltoAllv and Hypercube. Upstream also documents two distinct test families: `comm_ops_perf` for communicator operations and GIN device API microbenchmarks. These need separate applications because their configuration and result schemas differ from collective tests.

Recommended organization: basic collective configuration, test-specific configuration, advanced execution/diagnostics, and separate Communicator Ops and GIN workspaces.

## Coverage matrix

| Area | Present in page | Missing or constrained | Recommendation |
|---|---|---|---|
| Standard collectives | 9: AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, SendRecv, AlltoAll, Gather, Scatter | AlltoAllv and Hypercube | Add both as test-specific choices. Hypercube requires a power-of-two rank count. Source: `src/Makefile` `BIN_FILES_LIST`; upstream README. |
| Data types | float, half, int, double, bfloat16 | `-d all` and other NCCL types; actual support depends on build, NCCL, CUDA, and hardware | Add “all supported types” and dynamically filter choices based on target capabilities. |
| Reduction operations | sum, prod, min, max, avg | `all` scan; source also has `mulsum`/PreMulSum capability depending on build/version | Add scan-all and supported `mulsum`; expose only for reduction collectives. This is distinct from UI `-a` timing aggregation. |
| Root selection | None | `-r <root/all>` for root collectives | Add root rank (or all roots) for supported Broadcast/Reduce/Gather/Scatter modes; validate range against ranks and adapter support. |
| Message sweep and rank layout | `-b/-e/-f/-i`, MPI process count, `-t/-g` | hostfile/nodes, MPI launcher options and binding, explicit GPU mapping/device choice and environment | Add topology and launcher configuration for reproducible multi-node runs. Keep total ranks formula visible: MPI processes × threads × GPUs/thread. |
| Execution/timing | `-w/-n/-m/-c/-N/-G/-I/-a/-C` | `-K` per-iteration skip, `-z` blocking, `-p` parallel init, `-y` default stream | Add to advanced options with dependency checks: `-K` needs `-I 1`; `-z 3` conflicts with CUDA Graph `-G`. |
| Diagnostics/reporting | `-J`, Summary/Details/Iterations sample views | `-T` timeout, `-M` memory report, `-S` timestamps, `-u` unalign, `-U` tuning | Add optional diagnostic controls and preserve command, versions, environment, device/rank mapping, and correctness status in real reports. `-J` takes a path, not a boolean. |
| Rank splitting | One MPI process group visualization | `NCCL_TESTS_SPLIT` / `NCCL_TESTS_SPLIT_MASK` can create parallel communicator groups | Add advanced rank-group configuration and show group/color. Explain that reported bandwidth is per group. |
| NCCL implementation/registration | None | `-R` registration; `-D/-V` device API; `-H` host RMA; support is collective- and version-specific | Add only after capability probing. `-D` and `-H` are mutually exclusive; both require symmetric registration `-R 2`. |
| AlltoAllv | None | Generated traffic spread or explicit traffic matrix, scaling, per-rank summary; special size semantics | Provide a dedicated panel for traffic mode, spread, matrix file/editor, scale, and summary. Do not treat it as equal-sized AlltoAll. See `src/alltoallv.cu`. |
| Communicator Ops | None | `comm_ops_perf` measures init/split/shrink/grow; separate MPI/NCCL requirements, arguments, and latency output | Add a distinct application with operation type, communicator-size sweep, resize/share/abort options, and latency report. |
| GIN device API | None | `device_api/gin` includes latency and throughput tests for put/get/signal variants; requires MPI, NCCL 2.30.7+, exactly 2 ranks and 1 GPU/rank | Add a distinct workspace with category, operation, size sweep, CTA/thread count, RSM, queue depth and GIN flags. Use its own latency, MiB/s, and message-rate report fields. |

## Suggested priorities

### First: complete common collective workflows

1. Add AlltoAllv and Hypercube, with their specific constraints.
2. Add root rank, host/node and MPI configuration, GPU mapping, and environment controls.
3. Add `-K`, `-z`, `-p`, `-y`, and timeout `-T`, including compatibility validation.
4. Support type/op “all” scans and filter choices by the installed binary's capabilities.

### Second: advanced diagnostics and implementations

1. Add `-M`, `-S`, `-u`, and `-U`.
2. Add `NCCL_TESTS_SPLIT` groups and topology visualization.
3. Add `-R/-D/-V/-H` only with version, collective support, and mutual-exclusion checks.
4. Treat the JSON output path as a path/archival setting rather than a simple on/off toggle.

### Third: separate test workspaces

1. Communicator Ops: init/split/shrink/grow tasks and millisecond latency summaries.
2. GIN: point-to-point device API latency/throughput; enforce its two-rank constraint and use its dedicated report metrics.

## Prototype boundary

Full command-line coverage does not mean displaying every control at once. Build the form from the selected test binary and detected nccl-tests/NCCL/MPI/GPU capabilities. Show only valid options, retain resolved values in the command preview, and validate dependencies before submission.

The current report is fixed illustrative data and does not change with configuration. It demonstrates report layout only; it cannot establish that a UI option is connected to a real execution backend.

## Primary source locations

- `nccl-tests-master/README.md`: CLI, rank count, split groups, Communicator Ops, and GIN.
- `nccl-tests-master/src/Makefile`: standard binary list and version-gated targets.
- `nccl-tests-master/src/common.cu`: argument parsing, compatibility checks, split groups, and device selection.
- `nccl-tests-master/src/alltoallv.cu`: traffic matrix and generated traffic options.
- `nccl-tests-master/src/comm_ops.cu`: communicator benchmark.
- `nccl-tests-master/src/device_api/gin/README.md` and `src/device_api/gin/utils/args.cc`: GIN binaries, options, and constraints.
- Upstream rolling README: https://github.com/NVIDIA/nccl-tests#usage


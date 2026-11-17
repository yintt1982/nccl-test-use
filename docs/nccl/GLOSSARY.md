# NCCL Test Observability

This context defines the terms used to configure nccl-tests runs, collect NCCL runtime data, and interpret reports.

## Language

**Run**:
A single submitted attempt to execute nccl-tests, identified independently and associated with its effective configuration and resulting evidence.
_Avoid_: Session, job (unless referring to a scheduler job)

**Run Configuration**:
The user-selected benchmark and collection settings that determine what a Run attempts to execute and observe.
_Avoid_: Command line (the command is only one representation of the configuration)

**Collection Dimension**:
A report-facing category of measurements, such as Chunks, Timeline, Flows, QPairs, or Ports; it describes what the user wants to inspect, not how that data is collected.
_Avoid_: Collector

**Collector**:
A component that observes a defined data source and produces evidence for one or more Collection Dimensions.
_Avoid_: Dimension

**Collection Capability**:
Whether the requested Collector and data source are available and compatible for a Run, including the event types that can actually be observed.
_Avoid_: Supported (without naming the source, version, and event scope)

**Configuration Snapshot**:
The immutable effective Run Configuration and relevant runtime/version metadata recorded for a Run.
_Avoid_: Current configuration (when describing an existing Run)

**Partial Report**:
A report containing valid evidence from a Run while one or more expected sources, events, or associations are missing or incomplete.
_Avoid_: Complete, zero (when data is absent)

**Live Report**:
A report view that updates while its Run is active and distinguishes provisional observations from the final result.
_Avoid_: Final report (before the Run has completed)

**Runner Agent**:
A trusted execution component associated with a GPU test node that reports node capabilities and carries out authorized Runs.
_Avoid_: Browser runner

**Toolchain Profile**:
An administrator-registered, validated combination of nccl-tests, NCCL, MPI, and required runtime dependencies available to a Runner.
_Avoid_: Arbitrary binary path

**Analysis Plan**:
A defined comparison made from multiple related Runs, such as Runs at different concurrency levels or topology sizes.
_Avoid_: Single Run

**Data Retention Policy**:
The administrator-controlled period for retaining raw collection evidence and its associated reports and run metadata.
_Avoid_: Per-user deletion setting

**Required Collection Dimension**:
A selected Collection Dimension whose loss or unavailability prevents the requested Run from being considered successful.
_Avoid_: Required report (when referring to the data source)

**Partial Run**:
A Run that produced usable benchmark or collection evidence but ended with failed, missing, or incomplete expected evidence.
_Avoid_: Successful (without qualification)

**Event Loss**:
Profiler or collector events that were generated or expected but could not be retained or delivered, with the loss count and scope recorded when known.
_Avoid_: Zero events (when collection dropped data)

**Analysis Plan Result**:
The aggregate outcome of the Runs in an Analysis Plan, including failed or missing comparison points.
_Avoid_: Successful (when the plan is incomplete)

**Analysis Baseline**:
The set of Run Configuration fields held constant across an Analysis Plan so that declared comparison variables can be interpreted.
_Avoid_: Default Run (unless it is explicitly the comparison baseline)

**Event Batch**:
An ordered, identified group of collector events delivered from a Runner to the management service and safe to retry without duplicating stored events.
_Avoid_: Unidentified upload chunk

**Report Revision**:
A monotonically increasing version of a Run's report state used to resume live updates or request a replacement snapshot.
_Avoid_: Event sequence number

**Instrumentation Condition**:
The set of collectors and profiling options active during a Run that may affect its measured performance and must be recorded when results are compared.
_Avoid_: Unqualified benchmark result

**Run Template**:
A named, editable Run Configuration that can be used to create new Runs; changes to it do not alter existing Configuration Snapshots.
_Avoid_: Existing Run Configuration (when referring to an immutable snapshot)

**Runner Connection State**:
The observed reachability of a Runner Agent, separate from the Run's execution outcome; a disconnected Runner may still be executing or spooling evidence locally.
_Avoid_: Run status (when describing agent connectivity)

**Pending Verification**:
A Run condition used when its Runner has been unreachable beyond the configured grace period and the service cannot yet confirm the execution outcome.
_Avoid_: Failed (without evidence that execution failed)

**Run Artifact**:
An immutable file produced by a Run, such as raw Profiler JSONL, native benchmark output, or Runner logs, retained as evidence separate from report summaries.
_Avoid_: Report row

**Runner Spool**:
The Runner's durable local queue of Run Artifacts and event batches awaiting confirmed receipt by the management service.
_Avoid_: Temporary in-memory buffer

**Event Schema Version**:
The explicit version of a stored event record format, distinct from the NCCL Profiler ABI and plugin version that produced it.
_Avoid_: NCCL version (when referring to the record format)

**Artifact Receipt**:
The management service's confirmation that a Run Artifact or Event Batch passed integrity checks and was durably stored.
_Avoid_: Upload complete (before durable receipt)

**Evidence Revision**:
An audited correction to a Run's interpreted outcome or report after new evidence arrives or an administrator verifies a disconnected Runner.
_Avoid_: Silent overwrite

**Metric Evidence**:
The source, collection state, scope, unit, and derivation details attached to a reported metric so its meaning and completeness can be assessed.
_Avoid_: Bare metric value

**Derived Metric**:
A value calculated from one or more evidenced inputs using a recorded formula, distinct from a directly observed measurement.
_Avoid_: Measured value

**Management Service**:
The service that authenticates users, authorizes and schedules Runs, stores Run metadata, accepts evidence, and serves reports and live updates.
_Avoid_: Runner (when referring to the control plane)

**GPU Allocation**:
The set of administrator-authorized GPUs assigned to a Run, subject to runtime visibility and occupancy checks on its Runner.
_Avoid_: GPU isolation (unless hardware or container isolation is actually enforced)

**Coordinated Start**:
A multi-Runner launch in which all target nodes report ready before receiving a shared start instruction, with actual start times recorded per node.
_Avoid_: Synchronized execution (when implying precise clock alignment)

**Cluster Scheduler**:
The trusted workload manager that allocates resources and launches a multi-node Run across its assigned nodes.
_Avoid_: Management Service (when referring to the cluster execution system)

**Native Result Point**:
A benchmark result emitted by nccl-tests after it completes a reported test configuration, such as a message-size and buffer-mode combination.
_Avoid_: Per-iteration sample (unless the benchmark explicitly emits that data)

**Runner Spool Quota**:
The administrator-defined storage limit for unconfirmed Run evidence buffered locally by a Runner.
_Avoid_: Unlimited local queue

**Profiler Compatibility Matrix**:
The administrator-published set of NCCL, Profiler ABI, plugin, and event-schema combinations verified for collection and report parsing.
_Avoid_: Assumed ABI compatibility

**Slurm Allocation**:
The nodes and resources granted by Slurm to one Run before its workload processes are launched.
_Avoid_: GPU Allocation (when referring to the complete scheduler grant)

**Preflight Result**:
The recorded outcome of validating a Run's configuration, authorization, toolchain, node resources, and required collection capabilities before workload launch.
_Avoid_: Empty report (when a Run did not reach benchmark execution)

**Collector Dependency**:
A data source or event category that the system activates to produce one or more selected Collection Dimensions.
_Avoid_: User-selected dimension (when referring to an internal acquisition requirement)

**Required Dimension**:
A selected Collection Dimension whose evidence must be available for the Run to be considered successful; selected dimensions default to required unless explicitly made optional.
_Avoid_: Optional dimension (unless explicitly configured)

**Clock Domain**:
The timebase that produced an event timestamp, such as a Runner host clock or a GPU timer, which must be identified before timestamps can be compared.
_Avoid_: Timestamp (without its source clock)

**Clock Calibration**:
An evidenced mapping between clock domains with a measured uncertainty bound, used to determine whether cross-domain time comparisons are valid.
_Avoid_: Assumed clock synchronization

**Completed Work Progress**:
A count of completed benchmark result points or Analysis Plan Runs relative to the planned total, without estimating incomplete work.
_Avoid_: Predicted percent complete

**Run Retry**:
A newly submitted Run that references an earlier attempt and has its own Run ID, Configuration Snapshot, outcome, and evidence.
_Avoid_: Reusing a terminal Run

**Scheduler Queue Deadline**:
The maximum time a Run may wait for its complete Slurm Allocation before the pending scheduler job is cancelled without launching workload ranks.
_Avoid_: Run duration limit

**Node Group**:
An administrator-managed set of Runner nodes and scheduling constraints that an authorized user may select for a Run.
_Avoid_: Arbitrary host list

**Rank Mapping**:
The resolved assignment of processes, threads, GPUs, and NCCL ranks to nodes for a Run, validated against its Slurm Allocation and Toolchain Profile.
_Avoid_: Requested topology (when referring to the resolved mapping)

**Idempotent Submission**:
A Run submission request that can be retried with the same idempotency key without creating a duplicate Run.
_Avoid_: Duplicate Run on network retry

**Profiler Plugin Package**:
An administrator-published, versioned and integrity-checked Profiler shared library plus its compatibility metadata for deployment to Runners.
_Avoid_: User-uploaded plugin binary

**NCCL Logical Communication Edge**:
A rank- and communicator-scoped peer relationship with a direction observed from NCCL runtime evidence, independent of whether packets or an external network connection were observed.
_Avoid_: Network Flow (when only Profiler peer evidence exists)

**Network Flow**:
A network-layer flow identified from observed connection or packet evidence, such as a scoped 5-tuple, rather than inferred solely from NCCL rank and peer fields.
_Avoid_: NCCL Logical Communication Edge

**RDMA Request Payload Length**:
The payload length reported for an observed RDMA operation/request, distinct from NCCL chunk size, logical collective data size, and Ethernet wire bytes.
_Avoid_: RDMA message size (without stating the observation layer)

**QP Identity**:
A Queue Pair identifier scoped by host, HCA/device, QPN, and available connection or lifecycle evidence.
_Avoid_: Global QP number

**Physical Port Counter**:
A device-level counter scoped to a host, HCA, and physical port; it is not a per-rank or per-QP measurement unless independently observed at that granularity.
_Avoid_: Per-rank port bytes (when derived by splitting an aggregate counter)

**Development Admin Grant**:
An environment-scoped, auditable administrator authorization for development and integration work that can be removed through role configuration before delivery.
_Avoid_: Global authorization bypass

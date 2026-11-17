# nccl-tests 原生采集报告数据分析

整理日期：2026-11-03。依据仓库内 `nccl-tests-master` 的 README、性能说明及 `src/common.cu`、`src/util.cu` 源码静态分析；未在 GPU/RDMA 环境执行验证。结论针对当前工作区源码快照，部署二进制需按其版本和构建重新核实。

## 1. 结论摘要

nccl-tests 已自带结构化运行报告和若干可选性能诊断数据，不应仅当作“总耗时基线”。使用 JSON 输出并打开逐迭代计时后，可以获得运行配置、rank/device 清单、逐测试点性能、逐迭代时间样本、校验/错误摘要，以及可选 NCCL tuning 字段。

这些原生数据能支撑**测试点级和迭代级性能报告**，但不提供通用的逐 Chunk/逐包事件表，也不直接提供物理 NIC port、RDMA QPair、网络 flow 的计数与映射。它们应作为报告的原生数据层，网络维度由额外的追踪、NIC/HCA 或交换机遥测补充。

## 2. 原生报告入口

| 选项/输出 | 原生能力 | 注意事项 |
|---|---|---|
| `-J <file.json>` | 输出 JSON，含元数据、配置、设备和结果 | 依据文件后缀选择格式；当前只支持 JSON。MPI 下由 world rank 0 写入。文件以独占方式创建；目标已存在时依次尝试追加 `.0`、`.1` 等后缀。 |
| stdout | 人可读的测试表、设备清单、汇总和错误文本 | 建议与 JSON 同时归档，尤其关注运行失败、内存报告等并非 JSON 字段的输出。 |
| `-I 1` | 启用逐迭代 CUDA event timing 和汇总列 | 与 `-G` CUDA Graph 不兼容；实际运行若配置冲突，源码会关闭逐迭代计时。 |
| `-K <count>` | 从逐迭代汇总统计中排除前若干样本 | 原始 JSON `times_us` 仍保留完整样本；`-K` 依赖 `-I 1`。 |
| `-S 1` | stdout 性能行添加时间戳 | 时间格式为本地墙钟秒级字符串，不是事件起止时间；JSON 测试点本身没有对应的逐事件时间戳字段。 |
| `-U 1` | 尝试报告 NCCL tuning 信息 | 要求 NCCL 2.28+；部分字段需 2.31+。该选项会设置静态 profiler plugin，因此不要默认与外部 profiler plugin 同时使用。 |
| `-M 1` | stdout 打印 GPU 内存用量分类汇总 | 当前实现以文本打印，不写入 JSON；不是 NIC/HCA 显存/队列资源报告。 |
| `-N <cycles>` | 多轮运行并逐轮打印结果 | 是重复 benchmark cycle，不是时间采样器或网络事件记录器。 |

常用相关参数还包括 `-n` 正式迭代数、`-w` warmup 数、`-m` 每轮聚合操作数、`-a` 跨 rank 时间归并、`-z` blocking 模式、`-C` CPU 时间、`-G` CUDA Graph 和 `-c` 正确性校验次数。它们影响数据解释，完整命令行也保存在 JSON 的 `args` 中。

## 3. JSON 与文本中已有的数据

### 3.1 运行级信息

JSON 顶层包括：

- 报告格式 `version`、`start_time`、`end_time`。
- 完整命令行 `args`、进程环境快照 `env`、`nccl_version`。
- `config`：线程数、每线程 GPU 数、消息扫描范围/步进、warmup/迭代/聚合/校验设置、Graph/blocking/per-iteration 计时等。
- `devices`：每个参与者的 rank、group、PID、hostname、CUDA device 编号、PCI/device 描述。
- `out_of_bounds`、`average_bus_bandwidth`、`errors` 等运行汇总。

环境快照可能含敏感变量；导入和展示时需支持脱敏。CUDA device 编号是进程可见编号，跨主机不能单独作为物理 GPU 的全局身份；应结合 hostname 和 PCI/UUID 等实际资产信息。

### 3.2 测试点结果

`results[]` 按消息大小和操作参数记录测试结果，常见字段包括：

- 测试参数：`size`、`count`、`type`、`redop`、`root`、`experiment_name`、`actual_iterations`。
- `out_of_place`、`in_place` 两种缓冲模式，各自可为 null 或包含性能结果。
- 性能字段：`time` 或 `cpu_time`、`alg_bw`、`bus_bw`、`nwrong`。
- 可选 tuning 子对象：`implementation`、`algo`、`proto`、`kernelVariant`、`syncStrategy`、`#channels`、`#warps`、`netChunkSize`。

`time`/`cpu_time` 以微秒记，带宽为 GB/s。`bus_bw` 是依 collective 和 rank 数计算的集合通信模型指标，不能解释为 NIC 端口实测吞吐。`nwrong: null` 表示没有提供错误统计，不等于零错误。

### 3.3 逐迭代时间

启用 `-I 1 -J report.json` 后，结果对象可含 `out_of_place_per_iter` / `in_place_per_iter`，其中包括：

- `times_us`：逐迭代 CUDA event 时间样本；样本基于该轮聚合操作并按聚合数归一，不能当作网络事件时间戳。
- `min_us`、`max_us`、`avg_us`、`p50_us`、`p95_us`、`p99_us`、`stdev_us`、`cv_pct`。
- `skipped_iterations`：统计汇总排除的前置样本数；原始数组不因此删除样本。
- MPI 多进程路径下可能有 `per_process_max_times_us`：各进程每轮的最大计时样本。它不是每 GPU、每 NIC、每 QP 的独立耗时。

该数据形成**迭代序列的耗时分布**，没有每个迭代的绝对起止墙钟时间，也没有每次 NCCL API、kernel、网络发送或接收事件的端点时间。不能只凭样本数组构造真实的跨主机绝对时间线。

### 3.4 额外 stdout 数据

- `-S 1`：性能报告行时间戳；逐迭代模式下格式位置/输出行为不同，应以对应版本样例解析。
- `-M 1`：初始化、用户分配、collective 等 GPU memory 分类与总计，仅文本输出。
- 设备清单、out-of-bounds 检查、平均 bus bandwidth 检查、结束状态、错误文本。

因此归档流程不应只保存 JSON；应同时捕获 stdout/stderr、启动命令和退出码。

## 4. 对 timeline / flow / port / QPair 的覆盖结论

| 目标维度 | nccl-tests 原生报告覆盖 | 可安全生成的报告 | 原生缺口 |
|---|---|---|---|
| Timeline | 有测试点平均耗时；可选迭代耗时序列；可选秒级性能行时间戳；可选 tuning 选择字段 | 每个消息大小/缓冲模式的平均指标、逐迭代延迟分布/序列、运行与测试上下文 | 无完整迭代绝对起止时间；无 NCCL API/kernel/step/chunk/packet 事件起止记录；无法凭总时长重建分段时间线 |
| Flow | 无网络 flow 表 | 可报告配置、rank/device 与测试点的逻辑上下文 | 无网络五元组、每 flow 字节/包、flow 起止时间、Rank↔flow 直接映射 |
| Port | 无 NIC/HCA 或交换机物理端口计数器 | 只能把测试结果与外部采样窗口并列展示 | 无端口身份、链路速率、收发字节/包、错误/拥塞/PFC 等端口遥测 |
| QPair | 无 RDMA QP 资源表或逐 QP 计数器 | 可把 rank/device 清单作为后续外部关联的锚点 | 无 QP 编号/状态/计数器，也无 Rank/Channel↔QP 映射 |

特别说明：`-U` 中的 `netChunkSize` 是 tuning 元数据，不是逐 Chunk 事件列表；它不包含每个块的唯一 ID、offset、peer、实际起止时间或端口/QP 归属。Profiler/tuning 的事件收集机制不应误读为完整 profiler timeline。

## 5. 原生数据到目标报告的分层

推荐将报告明确分为两层：

1. **nccl-tests 原生报告**：保留 JSON 原文和 stdout/stderr；规范化运行、配置、rank/device、测试点指标、迭代样本、校验、tuning 和内存摘要。原生字段与单位保持可追溯。
2. **外部关联遥测**：以同一 `run_id` 和主机为组织键，额外采集 NCCL/CUDA 事件、NIC/HCA QP 统计、物理端口/交换机计数器或网络 flow。每种记录保存数据源、工具/驱动版本、时间窗、原始记录引用和关联置信度。

只有原始数据源明确提供映射时，才将 rank 关联到 QP 或将 QP 关联到 flow。仅按同一运行时间窗重叠时，应标记为相关性/推断，不应显示为精确归属。端口计数器是共享端口的窗口聚合数据，也不能自动拆分到某个 collective 或 rank。

## 6. 推荐采集配置与解析规则

用于建立 nccl-tests 原生基线，可按目标版本支持情况启用：

```text
-J <run>.json -I 1 -K <summary-skip-count> -S 1
```

`-K` 是从 `-I` 汇总统计中排除的前置计时样本数，不等同于 `-w` warmup 次数；JSON 原始迭代数组保留完整样本。

是否启用 `-U 1` 应单独决定：它有版本要求，并可能影响 `NCCL_PROFILER_PLUGIN` 的配置。`-M 1` 可加入诊断运行，但需从 stdout 单独提取。开始前确认 `-I` 与 `-G` 的兼容关系，并保存最终实际启动参数、环境和退出码。

解析侧要求：

- 以 JSON `version` 和运行时 nccl-tests/NCCL 版本选择解析规则，不假设字段永久固定。
- 保留原始 JSON 与原始日志；未知字段不丢弃，null 不改写成零。
- 将时间、带宽、大小的单位规范化存储，同时保留原值和原始单位。
- 以 run、消息大小、collective 参数和 in-place/out-of-place 模式组成测试点身份；正确处理同一次运行中多测试点及多 cycle。
- 保存设备关联维度至少包括 hostname、rank、group、PID、CUDA 可见设备编号和 PCI/device 信息；有条件时补采设备 UUID，但不要从本地序号臆造 UUID。
- `env` 可能含凭据或内部配置，展示与共享前脱敏。
- `actual_iterations` 优先于配置里的 `iterations` 描述实际结果数量；`nwrong=null`、tuning `N/A` 和缺失的可选对象都要按“不可用/未提供”表达。

## 7. 实施顺序与验收

1. 对目标部署二进制执行 `--help`，核对版本、编译选项和可用参数；不能仅依赖仓库快照推断线上能力。
2. 单机运行一次小规模 collective，启用 JSON 和 per-iteration timing，核对字段、单位、样本数、rank/device 和文件重名行为。
3. 双机 MPI 运行相同测试，核对 JSON writer、`devices` rank 语义、`per_process_max_times_us` 顺序及跨主机日志采集。
4. 对比无 `-I`、有 `-I`、有 `-U` 的结果和耗时，记录额外采集开销及插件冲突。
5. 将原生数据解析为报告，保留原始证据并验证 null、缺字段、多结果模式、失败退出和敏感环境变量处理。
6. 再接入外部 timeline、flow、port、QPair 来源，并分别验证时间同步、对象映射和采集扰动。

原生报告验收标准：每个测试结果能回溯到原始 JSON/文本；关键性能数据单位正确；迭代样本数量与实际迭代设置/结果一致；rank/device 能定位到主机资源；报告没有把迭代耗时、`bus_bw` 或 `netChunkSize` 误称为逐块事件、端口计数或 QP/flow 数据。

## 8. 本地依据

- [`README.md`](../nccl-tests-master/README.md)：参数、JSON 输出、逐迭代计时、时间戳、内存和 tuning 选项。
- [`doc/PERFORMANCE.md`](../nccl-tests-master/doc/PERFORMANCE.md)：time、algorithm bandwidth、bus bandwidth 的语义和计算模型。
- [`src/common.cu`](../nccl-tests-master/src/common.cu)：命令行解析、计时循环、跨 rank 归并、校验阶段、内存报告开关和兼容性检查。
- [`src/util.cu`](../nccl-tests-master/src/util.cu)：JSON 版本/元数据、设备清单、测试结果字段、迭代数组、tuning 字段和运行汇总。
- [`docs/nccl-report-and-timing-research.md`](nccl-report-and-timing-research.md)：已有计时及 JSON 字段研究。
- [`docs/nccl-rank-topology-and-observability.md`](nccl-rank-topology-and-observability.md)：Rank/device 语义和外部追踪维度边界。

## 9. 报告样例与维度说明

### 9.1 双机多 Rank Collective stdout 样例

NVIDIA `nccl-tests` 上游 issue #389 于 2026-07-22 发布了一次双机 `all_reduce_perf` 实测日志：两台 8xB200 节点、ConnectX-7 NDR 400 Gb/s InfiniBand、共 16 ranks；样例命令扫描 512 MiB 至 16 GiB，factor 2。日志报告的二进制版本是 `nccl-tests 2.19.6`，因此这是近期公开的**旧版 stdout 形态样例**，不是当前 checkout 所有新选项的示范。

该实测日志的一条 1 GiB 测试点可整理为：

| 缓冲模式 | time (us) | alg_bw (GB/s) | bus_bw (GB/s) | nwrong |
|---|---:|---:|---:|---:|
| out-of-place | 11694.5 | 91.82 | 172.16 | 0 |
| in-place | 11691.4 | 91.84 | 172.20 | 0 |

输出开头还列出各个 Rank、所属主机、PID、CUDA device 和 GPU 型号；结果表按消息大小逐项给出两种缓冲模式的性能指标。该 issue 是用户提交的实测，不代表 NVIDIA 对相同硬件的性能保证。

这份样例至少包含两类维度：

- **运行/配置维度**：nccl-tests/NCCL 版本、线程和 GPU 数、消息大小范围、迭代设置。
- **参与者/资源维度**：Rank、Group、PID、hostname、进程可见 CUDA device、PCI 地址和 GPU 型号。
- **测试点性能维度**：单条消息大小、元素数、数据类型、归约操作、root，以及 out-of-place/in-place 的 time、algbw、busbw、校验错误数。
- **运行质量维度**：out-of-bounds 校验、平均 bus bandwidth 摘要。

其中结果表是跨参与者归并后的测试点结果，不是每个 Rank 独立的网络流量表。Rank/device 清单可帮助定位参与 GPU，但没有 NIC port、QP、flow 字段。上游 issue #389 页面提供完整命令、Rank 0–15 清单和每个消息大小的结果行。

### 9.2 单机 P2P `sendrecv_perf` 样例

上游 issue #331（2025-07-14）展示了单机 2 GPU `sendrecv_perf` 报告和约 71–73 GB/s 的 P2P 带宽结果。该样例同时附有 NCCL INFO 与 `nvidia-smi topo -m` 输出：前者包括 Rank/communicator/channel 路径、NCCL 发现的 `mlx5_*` IB 设备和 P2P chunk-size 配置；后者显示 GPU 与 NIC 的静态 PCIe/NVLink 拓扑关系。

这组材料覆盖 **GPU↔GPU P2P 性能、参与者/设备、NCCL 逻辑 Channel/传输选择、静态 GPU-NIC 邻接关系**。NCCL Channel 里的 Rank 对和 `P2P Chunksize` 是逻辑执行/配置信息，不等于网络五元组或实际 Chunk 时间线；`nvidia-smi topo -m` 是静态拓扑，也不是端口流量计数。样例没有逐 QP 计数、端口收发字节或 flow 映射。

### 9.3 `-I` JSON 迭代计时结构示意

本工作区没有找到真实运行生成的 nccl-tests JSON 样本文件。下面是按当前源码字段结构整理的**示意片段**，数值仅作格式说明，不代表实测：

```json
{
  "version": 4,
  "start_time": "2026-11-03 10:00:00",
  "nccl_version": 23102,
  "config": {
    "iterations": 4,
    "aggregated_iterations": 1,
    "per_iter_timing": true,
    "per_iter_skip": 1
  },
  "devices": [
    { "rank": 0, "group": 0, "hostname": "node1", "device": 0, "device_info": "NVIDIA GPU" },
    { "rank": 1, "group": 0, "hostname": "node2", "device": 0, "device_info": "NVIDIA GPU" }
  ],
  "results": [
    {
      "size": 1048576,
      "count": 262144,
      "type": "float",
      "redop": "sum",
      "out_of_place": { "time": 120.0, "alg_bw": 8.74, "bus_bw": 13.11, "nwrong": 0 },
      "out_of_place_per_iter": {
        "skipped_iterations": 1,
        "times_us": [121.0, 119.0, 120.0, 120.0],
        "min_us": 119.0,
        "max_us": 120.0,
        "avg_us": 119.67,
        "p50_us": 120.0,
        "p95_us": 120.0,
        "p99_us": 120.0
      }
    }
  ]
}
```

这个 JSON 结构组合的是 **运行级、Rank/device 和测试点/迭代级性能维度**。`times_us` 是相对的逐迭代耗时样本，不是带绝对时间戳的事件；它不能直接画成 NCCL API、Chunk、报文或 NIC 端口的 start/end 时间线。实际 JSON 可能还包含其他配置、in-place 结果、tuning、错误和汇总字段；字段以目标二进制版本为准。

### 9.4 NCCL GIN Device API 测试输出

当前仓库还包含 `device_api/gin` 微基准。README 说明这些测试面向 NCCL GIN 的 `put`、`get`、`signal` 及组合操作：吞吐类输出 message size、message count、MiB/s 和 MPPS；延迟类输出 size、iteration count 和单向延迟（us），并可显示 RSM、CTA/thread 等解析后的配置。该路径需要 MPI、NCCL 2.30.7+，且固定为两个 Rank、每 Rank 一张 GPU。

这是 **GIN 操作 API 性能维度**，比 collective 测试多了 message rate 和 GIN 操作类型，但仍不是通用 QP 遥测：输出没有物理 port、QP ID/state/counters 或 flow 五元组。依据：[`device_api/gin/README.md`](../nccl-tests-master/src/device_api/gin/README.md)。

### 9.5 样例维度速查

| 样例 | 主要数据维度 | 能回答的问题 | 不能回答的问题 |
|---|---|---|---|
| `all_reduce_perf` stdout/JSON | 测试点性能、消息大小、缓冲模式、Rank/device 上下文 | 某配置/消息大小下 collective 的平均性能和正确性如何 | 哪个端口/QP/flow 承载了多少字节、某个 chunk 的网络起止时刻 |
| `-I 1` JSON | 逐迭代延迟序列与分位统计 | 同一测试点的迭代耗时是否抖动、尾延迟如何 | 迭代内多段传输、绝对跨机事件时间线 |
| `-U 1` tuning 扩展字段 | 算法、协议、kernel variant、channels、chunk size 等选择元数据 | NCCL 选择/报告了哪些调优参数 | 每个 chunk 的身份/偏移/实际发送完成时间 |
| `sendrecv_perf` | Rank 对的 GPU P2P 带宽/延迟 | 点对点测试性能如何 | QP/port/flow 的直接归属 |
| `device_api/gin` 微基准 | GIN put/get/signal 的延迟、带宽或消息率 | GIN Device API 操作性能如何 | RDMA QP、NIC port、网络 flow 的遥测和映射 |

### 9.6 样例来源和解释边界

- [NVIDIA/nccl-tests issue #389：8xB200 nccl test performance validation](https://github.com/NVIDIA/nccl-tests/issues/389)：双机 16-rank stdout、设备清单和性能表。属于用户提交的实测日志，不是官方承诺的标准数值。
- [NVIDIA/nccl-tests issue #331：A100 intra-node P2P bandwidth](https://github.com/NVIDIA/nccl-tests/issues/331)：单机 sendrecv 性能与 Rank/device/NCCL 初始化日志样例。
- [NVIDIA/nccl-tests issue #400：alltoallv bus bandwidth factor discussion](https://github.com/NVIDIA/nccl-tests/issues/400)：2026-10-14 的字段解释讨论，提醒 `busbw` 受 collective/流量模式公式影响；issue 中的推导是提交者观点，不应视为已合并的规范定义。
- [NVIDIA nccl-tests README](https://github.com/NVIDIA/nccl-tests/blob/master/README.md) 与 [`doc/PERFORMANCE.md`](../nccl-tests-master/doc/PERFORMANCE.md)：当前参数和性能指标定义。

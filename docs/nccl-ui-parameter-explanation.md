# 当前配置页参数与 nccl-tests 源码对照说明

研究日期：2026-10-26  
页面：`http://112.25.236.50:9007/`，原型页标题 `NCCL-tests / Collective Benchmarks Prototype`。  
依据：当前页面可访问状态与本地 `nccl-tests-master` 的 README、`src/common.cu`、各 collective adapter；静态对照，没有执行页面上的真实 GPU 测试。页面明确标注 `PROTOTYPE / 非真实运行数据`，点击 Run Trial 只显示固定示例报告。

## 1. 页面当前表达的运行配置

当前配置页展示这些区域：

1. **基准测试配置**：集合通信测试、数据类型、归约操作、Rank 列表、`-b/-e/-f/-i` 消息大小扫描。
2. **GPU 与 Rank 布局**：MPI 进程数、`-t` 每进程线程数、`-g` 每线程 GPU 数、MPI 启动器和 rank/GPU 示意图。
3. **执行选项**：`-w/-n/-m/-c/-N/-G/-I/-J/-a/-C`。
4. **命令预览**：展示由配置合成的命令字符串。
5. **统计报告页**：Summary / Details / Iterations 三个示例维度，示例值不跟随当前配置变化。

页面默认示例命令为：

```bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -t 1 -g 4 -d float -o sum -w 5 -n 20 -m 1 -c 1 -N 1 -G 0 -I 1 -a 1 -C 0 -J results.json
```

这条命令本身没有 `mpirun -np`，所以单进程启动；`-g 4` 代表该进程的一个测试线程使用 4 张 GPU，即 4 个 NCCL ranks。当前页面里的 `MPI 进程数=1` 应由外层启动命令表达；若配置为 2，预览应类似 `mpirun -np 2 ./build/all_reduce_perf ...`。总 NCCL ranks 公式为：

```text
MPI 进程数 × 每进程线程数(-t) × 每线程 GPU 数(-g)
```

`MPI 启动器` 是网管层选择的执行器，不是 nccl-tests 的命令行参数；二进制还必须以 MPI 支持方式构建，才能多进程运行。[README 的 rank 定义](../nccl-tests-master/README.md#L35) [命令行参数解析](../nccl-tests-master/src/common.cu#L1198)

## 2. 基准测试配置

### 集合通信测试

下拉框当前有 9 个测试：AllReduce、AllGather、ReduceScatter、Broadcast、Reduce、SendRecv、AlltoAll、Gather、Scatter。这些名称对应 9 种 nccl-tests 可执行程序/测试 adapter，通常程序文件名是 `all_reduce_perf`、`all_gather_perf` 等。选择测试类型决定 collective 语义与可用参数，不只是报告名称。[常规测试程序列表](../nccl-tests-master/src/Makefile#L44)

需要标清当前页面覆盖边界：本地源码还包含 **AlltoAllv**、**Hypercube** 等测试，但当前下拉框没有它们；`comm_ops_perf` 是 communicator 管理操作测试，不是这里这组常规 collective benchmark。AlltoAllv 还需流量矩阵/spread 配置，不能只作为一个同质下拉项无说明地加入。[AlltoAllv adapter](../nccl-tests-master/src/alltoallv.cu#L8) [Hypercube adapter](../nccl-tests-master/src/hypercube.cu#L60) [构建目标](../nccl-tests-master/src/Makefile#L44)

适配器语义差别会影响后续统计字段：AllGather 每 rank 输入一块、输出拼接的全量结果；ReduceScatter 输入全量向量、输出规约后的 rank 分块；Broadcast/Reduce 有 root；AlltoAll 是每 peer 等长数据块；SendRecv 在 rank 环上收发。相同 `-b` 并不必然表示每个 rank 收发相同总字节数，报告的 `size`、API `count` 和带宽分子也可能不同。[Collective 布局研究](nccl-validation-layout-research.md) [Adapter 走读](nccl-adapter-deep-dive.md)

### 数据类型（`-d`）

当前选项：float、half、int、double、bfloat16。nccl-tests 的 `-d` 选定 collective 的数据类型；它会改变每元素字节数，因而相同 `-b` 对应的元素数量不同。当前主线也支持更多类型（如 int8/uint8/int32/uint32/int64/uint64，以及在构建和 NCCL 版本满足条件时的 BF16/FP8），但 `all` 才是命令行的全类型扫描语义；页面目前看起来只展示单一类型选择。[类型表](../nccl-tests-master/src/common.cu#L35) [版本门槛](../nccl-tests-master/src/common.cu#L1172)

建议类型值显示为 NCCL 实际 token，例如 `int32`，而不要只写含糊的 `int`；同时按本机 binary/NCCL build 能力过滤 BF16/FP8，不能仅凭静态列表断言目标机器可运行。对不同 collective/op，某些类型和操作组合也可能不受当前 NCCL 版本或硬件支持。

### 归约操作（`-o`）

下拉框当前为求和、乘积、最小值、最大值、平均值，对应 `sum/prod/min/max/avg`。此选项**只对具有 reduction 语义的测试有效**，例如 AllReduce、Reduce、ReduceScatter；AllGather、Broadcast、Gather、Scatter、AlltoAll、SendRecv 并不使用所选 reduction op。此类测试的 adapter 会以 `none` 或固定操作路径运行，不应让 UI 看起来更改了它们的归约行为。[README 参数说明](../nccl-tests-master/README.md#L72) [AllGather 不使用归约 op](../nccl-tests-master/src/all_gather.cu#L130) [AllReduce 遍历 op](../nccl-tests-master/src/all_reduce.cu#L545)

`avg` 受 NCCL 版本支持约束；源码版本低于 2.10 时不会启用 ncclAvg。页面还未暴露 `mulsum`（PreMulSum）及 `all`（扫描操作）的选择。归约 op 的“平均值”与执行区 `-a` 的统计平均不是同一件事：前者是 GPU collective 的数值运算；后者是测试计时值的聚合策略。[运算版本门槛](../nccl-tests-master/src/common.cu#L56) [初始化版本检测](../nccl-tests-master/src/common.cu#L1173)

### Rank 列表

截图上方的 `R0–R3` 是所选 collective 参与 ranks 的示意；测试的参与者数量真正由 MPI 进程数、`-t` 和 `-g` 决定。除非页面已经配置并应用 rank split，否则这个列表不应被解释成可独立任意增删的 rank 集合。

对于 root 操作（如 Broadcast、Reduce、Gather、Scatter），还要有 root rank 配置；当前页面没有 root 字段。源码 `-r root/all` 支持指定 root 或依次测试所有 root，但 README 简述主要提到 Broadcast/Reduce，实际 adapter 的 root 支持范围需要按每个测试确认。建议选择需要 root 的测试时才显示该字段，并提供合法范围校验。[root 参数文档](../nccl-tests-master/README.md#L73) [Broadcast root 扫描](../nccl-tests-master/src/broadcast.cu#L113)

### 消息大小扫描：`-b/-e/-f/-i`

| 页面参数 | nccl-tests 参数 | 含义与限制 |
|---|---|---|
| 最小消息大小 | `-b` / `--minbytes` | 扫描起点，单位字节；默认是 32 MiB。页面示例 `8` 表示 8 字节。 |
| 最大消息大小 | `-e` / `--maxbytes` | 扫描终点，单位字节；`128M` 表示 128 MiB，不是 128 MB 十进制。必须大于等于最小值。 |
| 倍增系数 | `-f` / `--stepfactor` | 下一测试规模按倍数增长；页面示例 2 表示每一步翻倍。 |
| 固定步长 | `-i` / `--stepbytes` | 下一测试规模增加固定字节数。 |

`-f` 和 `-i` 是两种扫描方式，应该二选一；页面提示正确。建议把输入解析成明确字节数并显示预览点列表，明确 `M/G` 使用二进制 MiB/GiB。消息大小是 adapter 的统一输入尺寸；实际 API count 会按数据类型、rank 数和 adapter 对齐规则重新计算，特别是 AllGather/ReduceScatter/AlltoAll。最终 JSON `size` 取 send/expected buffer 字节数中的较大值，`count` 是 adapter 的 paramCount，两者不要混成一个字段。[README 尺寸参数](../nccl-tests-master/README.md#L66) [尺寸换算](../nccl-tests-master/src/common.cu#L928) [报告字段](../nccl-tests-master/src/util.cu#L501)

## 3. GPU 与 Rank 布局

### MPI 进程数

页面提供的“MPI 进程数”是启动器参数，例如 `mpirun -np 2`；它没有对应 nccl-tests 内部的 `-p`。`-p` 是 parallel_init（并行初始化 NCCL），不是进程数。单机多 GPU 常用一个 MPI 进程配多个 `-g` GPU；多机/多进程时由 MPI launcher 创建进程。[README 说明 MPI 管理进程数](../nccl-tests-master/README.md#L37) [参数表](../nccl-tests-master/src/common.cu#L1198)

### 每进程线程数（`-t`）

`-t` 是每个 MPI 进程的 host 线程数。多线程时每线程管理 `-g` 张 GPU，所以总 ranks 是 `np × t × g`。页面的 GPU/Ranks 示意目前以 `R0/G0` 方式展示设备到 rank 对应关系；多进程时还应能区分 hostname、MPI proc、localRank、线程号、CUDA device id 与 NCCL rank。这些编号不是同义词。[rank 映射研究](nccl-mpi-rank-mapping.md) [README 参数](../nccl-tests-master/README.md#L63)

### 每线程 GPU 数（`-g`）

`-g` 是每个 nccl-tests host thread 使用的 GPU 数；一个 CUDA device 对应一个 NCCL rank。它不是全局 GPU 数，也不是每个 MPI 进程 GPU 总数（多线程时每进程 GPU 数为 `t×g`）。设备默认按本机 MPI process 的 localRank 分配连续 ID；`NCCL_TESTS_DEVICE` 可覆写起始设备编号。[设备分配](../nccl-tests-master/src/common.cu#L1657)

### MPI 启动器

`mpirun` 与 `mpiexec` 是执行命令前缀的选择，不是 nccl-tests CLI flag。配置 1 个 MPI 进程时也可以不加启动器，直接运行 binary；多进程时 binary 需要 MPI build。还需记录 MPI 实现、hostfile/节点列表和绑定策略（当前页面未提供），否则“进程数”本身不能完整复现实验。[README 构建及 MPI 使用](../nccl-tests-master/README.md#L24) [多进程示例](../nccl-tests-master/README.md#L47)

### 图示上的“MPI 进程组 0”

默认无 `NCCL_TESTS_SPLIT` 时，所有进程进入同一个 communicator，标成一个组是合理的。设置 `NCCL_TESTS_SPLIT`/`NCCL_TESTS_SPLIT_MASK` 后，会依据 MPI world proc rank 计算 color，同 color 进程形成不同 communicator；此时一个进程组图示可能需要展示多个颜色分组。split 下性能报告的带宽是每个组的口径，不应把各并行组数值直接当成单组 busbw。[README split](../nccl-tests-master/README.md#L110) [split 实现](../nccl-tests-master/src/common.cu#L1587)

## 4. 执行选项

### 预热迭代次数（`-w`）

正式计时前运行的 collective 次数，不计入正式计时。用于让首次通信连接、路径初始化等先发生。页面默认 5；nccl-tests 默认 1。[README](../nccl-tests-master/README.md#L77) [warmup 路径](../nccl-tests-master/src/common.cu#L720)

### 计时迭代次数（`-n`）

正式外层计时迭代数，默认 20。通常最终时间是迭代总耗时除以 `-n × -m` 后的每次操作平均耗时；它不是报告行数，也不是校验次数。[计时循环](../nccl-tests-master/src/common.cu#L753) [时间归一](../nccl-tests-master/src/common.cu#L795)

### 每次计时聚合操作数（`-m`）

每个外层计时迭代中成组发起的 collective 个数，默认 1。时间再除以 `-m`，因此报告的 latency 仍按每个 collective 归一。提高该值可能减少每次迭代中的 host 开销影响，但不能将配置数当作单次操作的延迟。[README](../nccl-tests-master/README.md#L79) [聚合执行](../nccl-tests-master/src/common.cu#L757)

### 正确性校验次数（`-c`）

校验循环次数，不是“抽查计时迭代百分比”。每个校验循环会重新初始化数据、另发起一轮 collective，再对结果与 expected 比较；这段校验不混入性能计时。默认 1；0 表示不做 datacheck，此时不能把 `nwrong=0` 展示成通过，JSON 应体现未校验/null 状态。[README](../nccl-tests-master/README.md#L89) [校验循环](../nccl-tests-master/src/common.cu#L828) [nwrong 输出](../nccl-tests-master/src/util.cu#L612)

### 运行轮次（`-N`）

重复完整的消息扫描并逐轮输出，不是计时迭代数。默认 1；0 表示无限重复，因此生产网管应对 0 做显著提示和任务停止控制。[README](../nccl-tests-master/README.md#L80) [外层 cycle](../nccl-tests-master/src/common.cu#L952)

### CUDA Graph 启动次数（`-G`）

启用 CUDA Graph capture/replay；0 表示关闭，正值表示回放次数。代码会将总测量时间按 graph launch 次数归一。它与 `-I 1` per-iteration timing 不兼容；与 blocking mode `-z 3` 也不兼容（当前页面没暴露 `-z`）。必须按编译使用的 NCCL/CUDA 版本能力判断；不支持时源码会忽略/关闭该选项。[README](../nccl-tests-master/README.md#L95) [Graph 计时](../nccl-tests-master/src/common.cu#L774) [互斥检查](../nccl-tests-master/src/common.cu#L1500)

### 逐次迭代计时（`-I`）

页面开关对应 `-I 1/0`，记录每次迭代 CUDA event 时间，并在报告中输出 min/max/avg/percentile/stdev 等统计及原始样本。它不是只给一个整体平均时间。与 `-G > 0` 不兼容。当前配置页没有 `-K`，所以不能从页面丢弃 warmup-like 的前置样本；`-K` 控制的是 per-iteration summary 排除前几个计时样本，且不删原始 JSON `times_us`。[README](../nccl-tests-master/README.md#L82) [per-iteration 输出](../nccl-tests-master/src/util.cu#L656) [兼容性](../nccl-tests-master/src/common.cu#L1496)

### 写入 JSON 报告（`-J`）

这是页面的布尔开关，但实际 CLI `-J` 接收**文件路径**而不是 0/1。开启时命令预览需生成文件名，例如 `-J results.json`；关闭时不应出现 `-J 0`。JSON 保存完整命令参数、进程环境、NCCL 版本、配置、设备、每个测试点结果及可选 per-iteration/tuning 数据。目标已存在时源码会尝试创建 `.0`、`.1` 后缀文件，不覆盖旧报告。[README](../nccl-tests-master/README.md#L102) [JSON 创建](../nccl-tests-master/src/util.cu#L364)

### MPI 统计方式（`-a`）

选项 0/1/2/3 分别是 rank 0、平均、最小、最大计时。这里的实现要准确解释：它在 host 侧先把一个 MPI 进程内部多个 nccl-tests 线程的计时做对应归并；`-a 1/2/3` 再通过 MPI 对 **MPI_COMM_WORLD 各进程的计时值**做平均/最小/最大。它不是对每张 GPU/NCCL rank 的原始延迟做一个通用的 rank 归约。`-a 0` 不执行 MPI 汇总，以 MPI rank 0 的报告值为准。该选项只有 MPI-enabled build 才有跨进程归并意义。[README 选项](../nccl-tests-master/README.md#L81) [线程与 MPI 归并实现](../nccl-tests-master/src/common.cu#L324)

另外，`-a` 的“平均”与 AllReduce 测试运算 `-o sum/avg` 无关。若用了 communicator split，时间统计仍有自己的参与进程归并口径，而 busbw 的理论系数使用当前测试 communicator rank 数；展示时应把 group 与计时归并策略同时保留。[split rank 映射](nccl-mpi-rank-mapping.md) [带宽调用 rank 数](../nccl-tests-master/src/common.cu#L823)

### 报告 CPU 时间（`-C`）

开关对应 `-C 1/0`。关闭时报告程序计时得到的操作延迟；开启时报告 CPU wall-clock 计时口径（微秒）。这会改变指标含义，不能与标准 latency 字段混为一谈，也要与 `time`/`cpu_time` 字段名对应。[README](../nccl-tests-master/README.md#L96) [输出字段选择](../nccl-tests-master/src/util.cu#L612)

## 5. 页面当前没有提供但容易被误解的参数

当前命令预览没有这些选项。它们不是页面当前配置项，下面仅说明源码中相关语义，避免误把已有 UI 控件当成完整复现配置：

- `-r/--root`：root 选择；页面当前无 root 控件，但 Broadcast/Reduce/Gather/Scatter 等需按 adapter 确认 root 语义。
- `-p/--parallel_init`：是否并行执行 NCCL communicator 初始化，不是 MPI 进程数。
- `-z/--blocking`：collective 等待/barrier 模式，会影响计时方式。
- `-K/--per_iter_skip`：逐迭代统计时忽略开头若干样本，只影响 summary，不删原始 JSON 样本。
- `-D/-V`：NCCL device API 实现及其 CTA 数；只有部分 collective 支持，且 `-D>0` 需要 `-R 2` symmetric registration。
- `-H`：host RMA 实现/context 数；仅部分 collective 支持，与 `-D` 互斥且需要 `-R 2`。
- `-R`：内存注册模式，和 Device API/Host RMA 能力相关。
- `-U`：报告 NCCL tuning 信息，需要 NCCL 2.28+；部分字段需要更高版本及 profiler 支持。
- `NCCL_TESTS_SPLIT[_MASK]`：将参与 ranks 分成并行 communicator group；它是环境变量，不是普通 CLI 参数。
- AlltoAllv 专属环境变量：矩阵文件、scale、spread、rank summary。页面没有流量矩阵编辑器，也没有 AlltoAllv 测试类型。

以上选项的约束见 [README 完整参数表](../nccl-tests-master/README.md#L87) 与源码参数检查逻辑。[互斥及版本检查](../nccl-tests-master/src/common.cu#L1475)

## 6. 如何读页面右侧的命令预览和统计报告

命令预览应视为“配置翻译结果”，不是实际任务记录。当前页面的 `Run Trial` 会打开固定示例报告，页面已说明报告数据不会随配置改变，因此示例表里的 8 ranks、示例消息大小和 busbw 数值不能反过来解释当前配置（默认配置实际是 4 ranks）。

真实运行报告中常见字段：

- `size`：该测试点公共报告尺寸；在不同 collective 上不一定等于单 rank 发送字节数。
- `count`：adapter 的 API count/测试参数元素数，具体含义随 collective 变化。
- `time` 或 `cpu_time`：微秒；分别对应 latency/CPU 计时开关。
- `alg_bw`、`bus_bw`：测试按操作公式计算的十进制 GB/s；busbw 是推导值，不是网卡计数器实测。
- `nwrong`：校验发现的错误元素数；null/N/A 表示没有错误报告，不等同 0。
- `out_of_place` 与 `in_place`：两种缓冲区模式各自的性能/校验结果；不是两种消息大小。
- `*_per_iter`：`-I 1` 开启后的单次迭代分布与原始时间数组。

页面的 Summary / Details / Iterations 三层可分别呈现汇总、逐消息规模和逐迭代信息；是否有逐 rank/逐进程详情，取决于原始报告真正包含什么。尤其 `-a` 的聚合结果不应标成每 GPU 的测量值。[JSON schema 研究](nccl-report-and-timing-research.md) [性能字段实现](../nccl-tests-master/src/util.cu#L594)

## 7. 本页配置对照覆盖状态

已逐项对照页面当前可访问配置区的所有控件：集合通信测试、数据类型、归约操作、rank 列表、`-b/-e/-f/-i`、MPI 进程数、`-t/-g`、MPI 启动器、`-w/-n/-m/-c/-N/-G/-I/-J/-a/-C`、命令预览与统计报告页。结论均来自页面当前展示状态及本地 nccl-tests 源码；本页原型尚未连接执行后端，因此没有真实任务运行、硬件能力探测或参数联动校验。

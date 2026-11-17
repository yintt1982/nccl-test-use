# nccl-tests：非标准与变长 Collective 源码走读

研究日期：2026-10-26。源码基线：`nccl-tests-master`（此前与上游快照核对）；本次为静态阅读，未在 GPU 上编译或运行。本文承接 [正确性校验与 Collective 数据布局](nccl-validation-layout-research.md) 和 [算子适配层、指标与验证](nccl-collectives-metrics-research.md)。源码链接均指仓库相对路径及行号。

## 1. 本轮结论

这些测试的名字看起来都是 collective，但实现层次并不一样：AlltoAll 在新 NCCL 上可调用 `ncclAlltoAll`、旧版本用成组 Send/Recv，也有 RMA/device kernel 分支；Gather/Scatter 也有版本兼容实现；AlltoAllv 和 Hypercube 则由 nccl-tests 自己用多对 Send/Recv 组成。界面和报告模型应记录实际测试类型/实现模式，不能仅凭 collective 名称推断底层调用。

## 2. 操作与数据布局

用 `N` 表示 communicator ranks，`C` 表示 adapter 的 `paramcount`（元素数），`w` 为每元素字节数。公共框架将用户输入的字节规模换算为元素数，再由 adapter 返回 API 参数、send/receive 缓冲需求及 in-place 偏移。[公共尺寸换算](../nccl-tests-master/src/common.cu#L928)

| 测试 | 缓冲布局及语义 | 默认/兼容执行方式 | 原地与校验 |
|---|---|---|---|
| AlltoAll | 每 rank 的 send/recv 均为 `N*C` 元素；send 按目标 rank 连续分块，recv 按源 rank 放入对应分块。`C` 是每个 peer 的 count；总数据按 `N*C` 计。[count 与初始化](../nccl-tests-master/src/alltoall.cu#L17) | 支持的 NCCL 版本调用 `ncclAlltoAll`；否则对每个 rank 配对 `ncclSend`/`ncclRecv`；另有 HOST RMA 与 device API 实现。[dispatch](../nccl-tests-master/src/alltoall.cu#L468) | 适配器执行 in-place 路径，但源码明确说不支持 in-place 并关闭该路径错误报告；不能把这一行的 `#wrong=0` 作为原地正确性证据。[标记](../nccl-tests-master/src/alltoall.cu#L25) |
| AlltoAllv | 每个 rank 给每个 peer 发不同长度。send buffer 按目的 rank 顺序串接；recv/expected 按来源 rank 顺序串接。buffer 按全 rank 最大 send/recv 总量分配。[布局](../nccl-tests-master/src/alltoallv.cu#L398) [初始化](../nccl-tests-master/src/alltoallv.cu#L406) | 对每个 peer 计算该方向的 send/recv count，再用一个 `ncclGroupStart/End` 批量发起非零长度 Send/Recv。[执行](../nccl-tests-master/src/alltoallv.cu#L515) | in-place 不受支持且关闭错误报告。[标记](../nccl-tests-master/src/alltoallv.cu#L462) |
| Gather | 每 rank 输入 `C`；root 接收 `N*C`，其他 rank 也由 adapter 计算 expected，但实际结果只在 root 汇聚。[布局](../nccl-tests-master/src/gather.cu#L10) | NCCL 版本够新时调用 `ncclGather`；较老支持版本用 root 收取各 rank 的 Send/Recv；特定分支可走 HOST RMA。[执行](../nccl-tests-master/src/gather.cu#L89) | in-place 输入放入 root 接收缓冲区对应 rank 槽位；偏移由公共框架提供，非 root 的 gather receive buffer 不作为最终报告语义。[offset/初始化](../nccl-tests-master/src/gather.cu#L10) |
| Scatter | root 输入 `N*C`，每 rank 接收 `C`；expected 是当前 rank 的块。[布局](../nccl-tests-master/src/scatter.cu#L10) | 新版本调用 `ncclScatter`；兼容路径由 root 对每个 rank Send、所有 rank 从 root Recv；另有 HOST RMA 分支。[执行](../nccl-tests-master/src/scatter.cu#L86) | in-place root 从接收 buffer 开始位置提供完整输入；接收偏移记录在 adapter 元数据中。[初始化/offset](../nccl-tests-master/src/scatter.cu#L10) |
| SendRecv | 每 rank 发送 `C` 给 `(rank+1)%N`，并从 `(rank-1+N)%N` 接收 `C`；expected 对应前一 rank 的输入。[布局与 expected](../nccl-tests-master/src/sendrecv.cu#L10) | 成组 `ncclSend` + `ncclRecv`；HOST RMA 实现也保留同样的环形 peer 关系。[执行](../nccl-tests-master/src/sendrecv.cu#L94) | in-place 不受支持且关闭错误报告；NCCL communicator 配置提示每 rank 有两个 P2P peer。[标记/config](../nccl-tests-master/src/sendrecv.cu#L33) |
| HyperCube | 每 rank 起始持有 `C`，最终接收区为 `N*C`，实现的是 AllGather 数据布局。[布局](../nccl-tests-master/src/hypercube.cu#L12) | 每轮 mask 倍增，peer 为 `rank XOR mask`，每轮交换已收集的 `mask*C` 元素；这是测试代码里的算法，不是 `ncclAllGather` 调用。[循环](../nccl-tests-master/src/hypercube.cu#L48) | 支持原地布局；runTest 在 rank 数不是 2 的幂时跳过测试，因此报告需区分 skip 与 pass。[原地初始化/跳过](../nccl-tests-master/src/hypercube.cu#L21) [rank 检查](../nccl-tests-master/src/hypercube.cu#L104) |

## 3. AlltoAllv 的流量配置和报告

AlltoAllv 有两种流量模式：[源码头部说明](../nccl-tests-master/src/alltoallv.cu#L20)

- **自动生成模式**（默认）：每 rank 对所有 peer 的总发送量由测试尺寸控制；`NCCL_TESTS_ALLTOALLV_SPREAD` 在 0 到 1 之间调节均匀分配和按 rank 环形距离加权分配。peer 的字节数向下按 16 字节对齐。[环境变量解析](../nccl-tests-master/src/alltoallv.cu#L152) [分配计算](../nccl-tests-master/src/alltoallv.cu#L361)
- **矩阵文件模式**：`NCCL_TESTS_ALLTOALLV_MATRIX_FILE` 给出发送矩阵，单元格 `[i][j]` 表示 rank i 发往 rank j 的字节数；`NCCL_TESTS_ALLTOALLV_MATRIX_SCALE` 缩放矩阵。源码文档要求矩阵为方阵，维度不小于 ranks；`-b` 与 `-e` 相等，且 `-e` 至少容纳任一 rank 的最大 send/recv 总量。[矩阵说明](../nccl-tests-master/src/alltoallv.cu#L50) [约束](../nccl-tests-master/src/alltoallv.cu#L76)

默认标准行指标按所有 rank 中最大的 send/recv 字节量计算 algbw，再乘 `(N-1)/N` 得 busbw；异构流量下文档建议用 `-a 3` 取 ranks 最大时间，避免平均时间被低负载 rank 拉低。`NCCL_TESTS_ALLTOALLV_PRINT_SUMMARY=1` 会额外输出逐 rank send/recv bytes 与带宽。[指标](../nccl-tests-master/src/alltoallv.cu#L467) [行级带宽](../nccl-tests-master/src/alltoallv.cu#L498) [输出说明](../nccl-tests-master/src/alltoallv.cu#L83)

因此 UI/报告应保留流量模式、spread 或矩阵标识、每 rank 的 send/recv bytes、计时聚合方式和时间。只显示一个“消息大小”会掩盖不均衡模式；也不能把矩阵模式的 `paramcount=0` 解读为零字节，实际 peer count 来自矩阵。[矩阵与 paramcount](../nccl-tests-master/src/alltoallv.cu#L398)

## 4. 指标口径

本地 adapter 的 `getBw` 公式如下；速率使用十进制 GB/s（除以 `1e9`）。

| 测试 | algbw 分子 | busbw/algbw 系数 |
|---|---:|---:|
| AlltoAll | `N*C*w` | `(N−1)/N` |
| AlltoAllv | `max(max_send_bytes, max_recv_bytes)` | `(N−1)/N` |
| Gather/Scatter | `N*C*w` | `(N−1)/N` |
| SendRecv | `C*w` | `1` |
| HyperCube | `(N−1)*C*w` | `1` |

依据：[AlltoAll](../nccl-tests-master/src/alltoall.cu#L47)、[Gather](../nccl-tests-master/src/gather.cu#L38)、[Scatter](../nccl-tests-master/src/scatter.cu#L34)、[SendRecv](../nccl-tests-master/src/sendrecv.cu#L38)、[Hypercube](../nccl-tests-master/src/hypercube.cu#L40)、[AlltoAllv](../nccl-tests-master/src/alltoallv.cu#L498)。这些 busbw 仍是测试端按公式推导的 collective 带宽口径，不是 NIC/交换机硬件计数器读数。

## 5. 对管理界面/报告数据模型的直接影响

在进入 Keysight 界面分析前，可以先把 nccl-tests 一侧的最小结果实体定为：

1. **Run 配置**：二进制/测试名、MPI ranks、threads、每线程 GPU 数、数据类型、op/root、尺寸扫描、warmup/迭代/校验次数、NCCL/device implementation、环境变量。
2. **测试点配置**：adapter 的 `paramcount`、发送/接收缓冲字节数、in-place/out-of-place 标志；AlltoAllv 还要记录 traffic mode 和 matrix/spread 设置。
3. **测试点结果**：时间及其跨 rank 聚合方法、algbw、busbw、wrong elements、skip/错误状态；若可用，附每 rank 的实际 send/recv 字节和耗时。
4. **证据来源**：原始 stdout/JSON、运行命令、主机/GPU/NCCL 信息。解析层应保留原始值和单位，避免把派生 busbw 标成硬件实测值。

这只是依据源码字段与行为形成的数据建模建议，不代表已经开始 Keysight 对照或原型工作。

## 6. 仍待覆盖与实测

本轮补齐了先前列出的 AlltoAll、AlltoAllv、Gather、Scatter、SendRecv、Hypercube 静态阅读。接下来应继续核对 JSON 输出 schema、计时选项/跨 rank 聚合语义、MPI 多机映射、环境/版本配置的采集字段；之后整理一份“nccl-tests 能力与证据覆盖清单”。当前没有 GPU 实测，因此尚需在真实 GPU/NCCL/MPI 环境验证缓冲区布局、各 adapter 的校验行为、AlltoAllv 不均衡计时和 device/RMA 分支。

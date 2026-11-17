# nccl-tests 正确性校验与 Collective 数据布局

研究日期：2026-10-26。源码基线：`E:/dpi/nccl-tests-master`，与已核验的上游 master 快照一致。静态源码阅读，未运行 GPU 测试。本文承接 [AllReduce 执行走读](nccl-allreduce-walkthrough.md)。

## 1. 校验不是“拿一组固定随机数比对”

`verifiable/` 为每个元素按 datatype、reduction op、rank 数、rank 编号、seed 和元素下标生成输入；对应生成逻辑可复现，因此期望结果可以独立构造。一般 reduction 的 `genOutput()` 按 rank 顺序把每个 rank 的 `genInput()` 经过规约函数折叠起来。[输入/输出通用生成](../nccl-tests-master/verifiable/verifiable.cu#L752)

生成 kernel `prepareInput2` 在每个 GPU 上并行写输入；`ncclVerifiablePrepareInput` 按操作选择 Sum/Min/Max/Product/Avg 等生成器。`prepareExpected2` 用 `genOutput()` 并行生成期望结果，API `ncclVerifiablePrepareExpected` 做操作分发。[输入生成 kernel](../nccl-tests-master/verifiable/verifiable.cu#L1005) [输入 API](../nccl-tests-master/verifiable/verifiable.cu#L1060) [expected 生成](../nccl-tests-master/verifiable/verifiable.cu#L1089)

例如 AllReduce Sum：每个 rank 对同一元素索引生成 rank 专属输入，expected 对所有 rank 的同位置输入做 Sum。AllReduce 适配器把输入初始化与 expected 构造放在每张 GPU 上，然后计时结束后重新初始化并另发起一次 collective 做比对。[AllReduce 初始化](../nccl-tests-master/src/all_reduce.cu#L42) [校验调用顺序](../nccl-tests-master/src/common.cu#L831)

## 2. 如何判定一个元素错误

`CheckData()` 对每个本机 GPU 取得输出缓冲区及 expected，调用 `ncclVerifiableVerify()`；返回的错误元素数先按本机 GPU 累加，再经框架的进程/线程汇总，形成 `#wrong` / JSON `nwrong`。[CheckData](../nccl-tests-master/src/common.cu#L449) [错误归并与报告](../nccl-tests-master/src/common.cu#L876)

当 expected 已预先生成时，验证 kernel 对结果和 expected 做逐元素比较并累计超过 tolerance 的元素数量。浮点 AllReduce Average 设 tolerance=2 ULP，以容纳平均值预乘/浮点运算带来的尾数差异；代码注释说明其原因。不能笼统宣称所有浮点规约都是零容差，也不能把 2 ULP 推广到所有操作。[prepared comparison](../nccl-tests-master/verifiable/verifiable.cu#L1189) [容差选择](../nccl-tests-master/verifiable/verifiable.cu#L1306)

数据生成不是为了模拟业务分布，而是为多种 datatype、reduction op 和 rank 数构造可预测且有针对性的结果。整数 reduction 的通用输入由 index/rank/seed 的 hash 构造；浮点 Sum 有专门输入/输出构造逻辑，确保结果可预测。[整数输入](../nccl-tests-master/verifiable/verifiable.cu#L755) [浮点 Sum](../nccl-tests-master/verifiable/verifiable.cu#L836)

## 3. 对 AllReduce 校验链的具体读法

AllReduce 的 `AllReduceInitData()`：

1. 按当前 datatype 把 send/expected 字节数换成元素数。
2. 得到全局 communicator rank 数 `nProcs*nThreads*nGpus`。
3. 对每张 GPU，计算其 rank 编号并清零 recv buffer。
4. 根据 out-of-place/in-place 决定在哪里生成输入。
5. 调 `InitData()` 生成该 rank 输入，再调 `InitDataReduce()` 为该 rank 生成规约 expected。
6. 同步设备，保证准备完成。

计时完成后，`BenchTime()` 再做一轮初始化与 collective，然后 `CheckData()` 比较每张 GPU 输出。检查迭代不计入 benchmark time；`-c` 是检查循环数，不是计时迭代的抽样比例。[AllReduce adapter](../nccl-tests-master/src/all_reduce.cu#L42) [检查阶段](../nccl-tests-master/src/common.cu#L831)

## 4. 三种 Collective 的数据布局

用 `N` 表示 communicator rank 数，用 `E` 表示初步按用户消息大小和 datatype 得到的元素数。AllGather 和 ReduceScatter 适配器计算 `base=(E/N) & ~(16/eltSize - 1)`；这是按 rank 切分后再向下对齐的元素数。故在某些组合下，实际有效元素数会小于未对齐的理想切分，不能只用 `E/N` 假设。[AllGather 布局](../nccl-tests-master/src/all_gather.cu#L10) [ReduceScatter 布局](../nccl-tests-master/src/reduce_scatter.cu#L10)

| 操作 | 每 rank 发送 | 每 rank 接收 | 本地语义 | `algbw` 的有效字节分子 | `busbw` 因子 |
|---|---:|---:|---|---:|---:|
| AllReduce | E | E | 所有 rank 对应元素规约，结果复制到所有 rank | E × typeSize | 2(N−1)/N |
| AllGather | base | base×N | 收集各 rank 的等长输入块，按 rank 拼到每个接收缓冲区 | base×N×typeSize | (N−1)/N |
| ReduceScatter | base×N | base | 对输入向量分块规约，每个 rank 只收一个结果块 | base×N×typeSize | (N−1)/N |
| Broadcast | E | E | root 输入传播到所有 rank | E × typeSize | 1 |

公式与代码：[AllGather 带宽](../nccl-tests-master/src/all_gather.cu#L38)、[ReduceScatter 带宽](../nccl-tests-master/src/reduce_scatter.cu#L37)、[Broadcast 带宽](../nccl-tests-master/src/broadcast.cu#L34)、[通用理论说明](../nccl-tests-master/doc/PERFORMANCE.md#L65)。`algbw` 的公式分子和报告的 `size` 字段不能混为一谈：通用报告 `size` 取 `max(sendBytes, expectedBytes)`，`count` 来自适配器的 `paramCount`。[行前缀生成](../nccl-tests-master/src/common.cu#L932)

### AllGather

每 rank 提供 base 个元素，接收缓冲区容量为 base×N。in-place 时，该 rank 的输入先放在接收缓冲区按 rank 对应的分块位置；expected 则按 rank 顺序把每个 rank 的独立输入拼接起来。NCCL API count 是每个 rank 提供的 count，而不是总接收元素数。[初始化与偏移](../nccl-tests-master/src/all_gather.cu#L19) [调用参数](../nccl-tests-master/src/all_gather.cu#L102)

### ReduceScatter

每 rank 提供 base×N 个输入元素，接收 base 个规约后元素。expected 由总 expected buffer 初始化后，按本 rank 所负责的块偏移生成对应规约结果。in-place 输出偏移也按 rank 块位置设置。[初始化与块偏移](../nccl-tests-master/src/reduce_scatter.cu#L19) [调用参数](../nccl-tests-master/src/reduce_scatter.cu#L45)

### Broadcast

每 rank 的发送/接收数量都是 E，但只有 root 的输入是待广播数据；其他 rank 输入不构成结果来源。所有 rank 的 expected 都从相同 seed 生成同一预期值。busbw 换算因子固定为 1。[初始化](../nccl-tests-master/src/broadcast.cu#L18) [公式与调用](../nccl-tests-master/src/broadcast.cu#L34)

## 5. 看报告时要区分的量

- **用户请求的 min/max bytes**：命令行扫描范围。
- **适配器的参数 count**：传给 NCCL API 的元素数，或定义当前测试点的基础元素数。
- **sendBytes/expectedBytes**：每个 rank 本地缓冲区需求。
- **报告 size**：公共报告取 max(sendBytes, expectedBytes) 后的字节数。
- **algbw 分子**：各操作适配器定义的有效逻辑数据量；AllGather/ReduceScatter 会用 N 倍 count。
- **busbw**：algbw 再乘操作相关理论换算因子。
- **nwrong**：计时后独立校验中未通过容差比较的元素计数。

对 AllGather/ReduceScatter，size、count 和带宽分子明显不一样。界面如果只存一个 `message_size` 数字，就会丢掉 API count 与用户数据规模的区别。

## 6. 当前掌握与未覆盖

目前已通过源码跟通 AllReduce 主链，理解校验机制，并研究 AllGather、ReduceScatter、Broadcast 以及其他六类 adapter；追加的 [adapter 走读](nccl-adapter-deep-dive.md) 与 [计时和 JSON 报告研究](nccl-report-and-timing-research.md) 覆盖了变长流量、rank 时间聚合、per-iteration timing 和输出 schema。仍未在 GPU/NCCL/MPI 环境实测，也未完整验证多机 rank 映射和不同 device/RMA 路径。下一步应整理 nccl-tests 功能/参数/结果证据覆盖清单，并标注静态源码结论与待硬件验证项；满足用户要求的源码理解阶段后，再进入 Keysight 录屏对照。


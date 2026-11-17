# 从一条命令理解 nccl-tests 的 AllReduce 主流程

研究日期：2026-10-26。源码基线：`E:/dpi/nccl-tests-master`，与上游 master `afd59abab774a6a4b492eb344bd0c845a0b7bb4d` 的 79 个文件逐一校验一致。本笔记是静态源码走读，不是 GPU 实测报告。源码行号相对该目录。

## 1. 示例命令与假设

```bash
./build/all_reduce_perf -b 8 -e 8 -g 2 -d float -o sum -w 1 -n 2 -c 1 -J allreduce.json
```

假设这是非 MPI 构建，单机至少有两张可见 GPU，且没有通过 `NCCL_TESTS_DEVICE` 改变起始设备。该命令使用两张 GPU、每卡一个 NCCL rank，对 8 字节的 float 数组做 Sum AllReduce。它只跑一个数据规模；每种缓冲区模式做 1 次预热、2 次计时迭代，并在计时之后独立做 1 次正确性检查。框架分别测 out-of-place 与 in-place。

本文不填造硬件实测的性能数值。公式示例只用符号或明确标注的假设时间。

## 2. 参数进入程序

`main()` 用 getopt 解析参数：`-b/-e` 进入 `minBytes/maxBytes`，`-g` 设置每线程 GPU 数，`-d/-o` 解析成 NCCL datatype/reduction op，`-w/-n/-c` 分别设置 warmup、计时迭代、检查迭代；`-J` 选择 JSON 输出。默认还有 `nThreads=1`、step factor 关闭、datacheck=1、average=1。[参数解析](../nccl-tests-master/src/common.cu#L1247)

因 `minBytes=maxBytes=8`，`TimeTest()` 的循环只测 8 字节一个点。`AllReduceRunTest()` 将数据类型和规约操作设为单值并调用 `TimeTest()`。[AllReduce 测试入口](../nccl-tests-master/src/all_reduce.cu#L545) [公共大小扫描](../nccl-tests-master/src/common.cu#L942)

## 3. GPU、进程与 rank

非 MPI 时只有一个进程；默认一个线程；`-g 2` 表示每线程使用两张 GPU，因此 communicator 有两个 NCCL rank。默认可见设备从 GPU 0 开始时，GPU 0、1 分别对应 rank 0、1。代码中 rank 计算为 `((proc*nThreads + thread)*nGpus + i)`。[GPU 选择](../nccl-tests-master/src/common.cu#L1660) [rank 公式](../nccl-tests-master/src/common.cu#L568)

`initComms()` 为每张 GPU 建立对应 communicator；单线程管理多 GPU 时，发起 collective 的位置用 `ncclGroupStart/End` 包围。MPI 多进程场景才额外使用 MPI 广播 unique ID、协调进程；被测 AllReduce 调用仍通过 NCCL。[communicator 初始化](../nccl-tests-master/src/common.cu#L200) [MPI ID 广播和分组](../nccl-tests-master/src/common.cu#L1570)

## 4. 从 8 字节到 NCCL 调用

float 每元素 4 字节，所以每 rank 的元素数 `count=8/4=2`。AllReduce 的发送和接收元素数都等于 count，in-place 偏移为 0。[AllReduce 数据量规则](../nccl-tests-master/src/all_reduce.cu#L34) `setupArgs()` 将元素数乘元素大小，得到发送和预期输出缓冲区的字节数。[字节数换算](../nccl-tests-master/src/common.cu#L928)

out-of-place 时从独立 send buffer 读、写入 recv buffer。in-place 时输入和输出都使用 recv buffer。默认 `deviceImpl=0` 时，适配器调用 `ncclAllReduce(send, recv, count, type, op, comm, stream)`；显式启用其他实现时会走不同路径。[调用分发](../nccl-tests-master/src/common.cu#L554) [AllReduce 实现](../nccl-tests-master/src/all_reduce.cu#L500)

两个 rank 的每个位置都得到两张 GPU 输入值的和，且输出相同。报告中的 `size=8` 是每 rank 的逻辑输入消息大小，不是所有 rank 传输字节数之和。[NCCL 操作语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)

## 5. 计时与正确性检查

每种缓冲区模式先初始化（开启校验时），执行 warmup，然后同步。接着每个外层迭代发起一次 collective；本例 agg_iters 默认 1，故有两次计时迭代。框架等待 stream 完成，再计算每次 operation 的平均时间。默认不启用逐次 CUDA event 统计。[预热与计时](../nccl-tests-master/src/common.cu#L720)

计时结束后，框架重新初始化数据，再执行单次 collective 并比较实际结果和预期结果。`-c 1` 是这项独立正确性检查的重复数，不是对两次计时迭代逐一检查。输入和 expected 根据数据类型、规约操作、rank 数和 seed 构造；错误结果累计为错误元素数。[输入初始化](../nccl-tests-master/src/all_reduce.cu#L42) [计时后验证](../nccl-tests-master/src/common.cu#L831) [verifiable 接口](../nccl-tests-master/verifiable/verifiable.h#L37)

## 6. time、algbw、busbw

AllReduce 的逻辑消息大小 S=8 bytes。若平均操作时间为 t 秒：

```text
algbw = S / t / 1e9                         (GB/s)
busbw = algbw × 2 × (nRanks - 1) / nRanks   (GB/s)
```

本例 nRanks=2，修正因子为 1，所以 busbw 等于 algbw。仅作公式演算：若假设 t=1 微秒，则 algbw=0.008 GB/s。该值是算例，不是本机实测。若 rank 数为 8，修正因子为 1.75。busbw 是按 collective 通信模型修正的算法带宽，不是 NIC 计数器测得的吞吐。[AllReduce 带宽实现](../nccl-tests-master/src/all_reduce.cu#L59) [AllReduce 公式](../nccl-tests-master/doc/PERFORMANCE.md#L33)

**单位差异：** 该源码实现把秒乘 `1e6` 输出 time，JSON 字段是 `time` 或 `cpu_time`，值的单位为微秒；迭代数据字段也带 `_us`。仓库性能文档第 3 行写毫秒，与实现不一致。解析或展示该版本数据时应按源码实现与 JSON 字段约定处理，并显式标注微秒。[时间换算](../nccl-tests-master/src/common.cu#L886) [结果字段](../nccl-tests-master/src/util.cu#L594) [迭代字段](../nccl-tests-master/src/util.cu#L668)

文本结果前缀含 `size/count/type/redop/root`，主体含 `time/algbw/busbw/#wrong`。AllReduce 的 root 不适用，但通用输出仍打印默认 root=0。JSON 的每条记录分别含 `out_of_place`、`in_place` 两种结果对象，内有 time、alg_bw、bus_bw 和 nwrong；顶层记录版本、argv、环境变量、NCCL 版本、设备信息等复现元数据。[结果前缀与字段](../nccl-tests-master/src/util.cu#L501) [JSON 元数据](../nccl-tests-master/src/util.cu#L364)

## 7. 容易误读的地方

- 一个 MPI 进程可以管理多个 NCCL rank；本例是一个进程、两张 GPU、两个 rank。
- size 是每 rank 的逻辑消息大小；float 的 count 是元素个数，本例为 2。
- busbw 是推导指标，不是 NIC 实测流量。
- `-n 2` 是计时迭代数；`-c 1` 是计时后校验重复数。
- AllReduce 会分别报告 out-of-place 和 in-place；不是所有 collective 都支持 in-place。
- 本版本的 time 实现单位为微秒，不能照搬性能文档开头的毫秒描述。
- 显存不足时程序可能下调 maxBytes，需看实际输出和 JSON，而不只看请求参数。[内存约束](../nccl-tests-master/src/common.cu#L1628)

## 8. 后续阅读

下一步沿 `AllReduceInitData → ncclVerifiablePrepareInput/PrepareExpected → ncclVerifiableVerify` 阅读输入模式与 expected 的构造；再比较 AllGather、ReduceScatter、Broadcast 的 count、缓冲区布局和 bus bandwidth 因子。随后追踪 `-I 1` 的逐迭代 JSON，以及 CUDA Graph / blocking 参数对测量口径的影响。当前阶段仍专注理解 nccl-tests，不进入 Keysight 界面原型。

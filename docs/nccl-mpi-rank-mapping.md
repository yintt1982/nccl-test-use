# nccl-tests：MPI rank、GPU 映射与 split communicator

研究日期：2026-10-26。静态源码研究，不代表多机实测。依据 `src/common.cu` 与 README。

## 1. 四种编号不能混为一谈

- `proc`：MPI_COMM_WORLD 中的进程 rank，由启动器分配。
- `localRank`：按 hostname 对比，在本机 MPI 进程中的序号；代码统计 MPI 全局序列中本进程之前、hostname hash 相同的进程数。
- CUDA device ID：默认以 `localRank * nThreads * nGpus` 为起始号连续分配；`NCCL_TESTS_DEVICE` 可覆盖起始编号。
- NCCL rank：在本次 NCCL communicator 中的 rank。未 split 时，进程段起点为 `proc * nThreads * nGpus`；线程和 GPU 在该段内继续排列。[host/local rank 与 split](../nccl-tests-master/src/common.cu#L1570) [GPU 选择与 communicator init](../nccl-tests-master/src/common.cu#L1647) [firstRank](../nccl-tests-master/src/common.cu#L1018)

总 rank 数为 MPI 进程数 × 每进程线程数 × 每线程 GPU 数。MPI 进程数由 `mpirun -np` 管理，不是 nccl-tests `-g`。[README](../nccl-tests-master/README.md#L37)

## 2. 示例映射

假设单机两个 MPI 进程、每进程 `-g 2 -t 1`，默认无 split，且没有 `NCCL_TESTS_DEVICE` 覆盖：

| MPI proc | localRank | CUDA devices | NCCL ranks |
|---:|---:|---|---|
| 0 | 0 | 0, 1 | 0, 1 |
| 1 | 1 | 2, 3 | 2, 3 |

多线程时每进程仍按 thread 再 GPU 排列：`proc * threads * gpus + thread * gpus + gpuIndex`。设备选择也为每个进程的本地 GPU 区间连续编号。[rank 公式](../nccl-tests-master/src/common.cu#L1018) [逐 GPU 初始化](../nccl-tests-master/src/common.cu#L1657)

## 3. split 如何改变 communicator

设置 `NCCL_TESTS_SPLIT` 后，程序由全局 MPI proc 计算 color，再通过 `MPI_Comm_split` 将同 color 的进程组成子 communicator。`NCCL_TESTS_SPLIT_MASK` 等价于按掩码 AND。程序在子 communicator 中计算 `ncclProcs/ncclProc`，然后以子组进程数 × threads × GPUs 创建 NCCL communicator；GPU 本地选择和设备报告仍使用原 MPI proc/localRank 体系。[split 计算与 MPI split](../nccl-tests-master/src/common.cu#L1587) [子组 NCCL 规模](../nccl-tests-master/src/common.cu#L1613) [communicator 创建](../nccl-tests-master/src/common.cu#L1639)

README 举例：8 GPU/节点情况下，`AND 0x7` 可让同一节点编号对应的 rank 跨节点成组，`OR 0x7` 则将节点内 GPU 分到组；实际 color 取决于 MPI proc 的全局排列，因此 launcher 的 rank 排布会影响分组。带宽按每组报告，要估算所有并行组总吞吐量需乘组数。[split 例子及口径](../nccl-tests-master/README.md#L110)

## 4. 性能归并的一个边界

`-a` 的 0/1/2/3 分别选择 rank 0、跨 ranks 平均、最小、最大时间。实现的 MPI 时间归并使用 `MPI_COMM_WORLD` 做 Allreduce，并在平均模式除以全局 MPI 进程数 × 线程数；GPU rank 时间先在本地线程/进程范围内聚合。因此启用 split 时，各子 communicator 测试时间仍在 MPI_COMM_WORLD 归并。要解释并发 split 的单组负载性能时，应同时保留 color/group 信息以及 `-a` 口径，不能假定报告时间只在同一个 NCCL 子组内取最大值。[归并实现](../nccl-tests-master/src/common.cu#L324) [README 参数](../nccl-tests-master/README.md#L81)

这部分来自静态阅读；不同 MPI launcher 的 proc 排列及其实际对网络/NIC 的影响尚需实机确认。

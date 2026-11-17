# NCCL/RDMA 统计报告研究（KAI DC Builder 风格）

日期：2026-11-13  
范围：为 nccl-tests 原型设计可落地的统计报告字段和图表；参考 Keysight KAI Data Center Builder 的报告/可视化定位、NVIDIA NCCL 官方文档与本仓库源码。

## 一手来源

- Keysight AI Data Center Builder 产品页（产品定位与网络/AI 数据中心测试）：<https://www.keysight.com/us/en/products/network-test/keysight-ai-data-center-builder.html>（页面对自动化测试、拓扑/流量可视化等描述可能随站点更新）。
- Keysight 数据表（可下载 PDF，报告与测试结果展示的产品资料）：<https://www.keysight.com/us/en/assets/3124-1163/data-sheets/AI-Data-Center-Builder.pdf>
- NCCL 环境变量（含 profiler plugin、调试/日志相关配置）：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
- NCCL-tests 性能定义（官方仓库）：<https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>
- NCCL 源码 profiler v7 ABI（事件描述字段）：<https://github.com/NVIDIA/nccl/blob/master/src/profiler_v7.h>
- 本地镜像：`nccl-tests-master/doc/PERFORMANCE.md`、`nccl-tests-master/src/profiler_v7.h`、`nccl-tests-master/src/util.cu`。

Keysight 资料适合作为“测试工作台 + 总览 KPI + 拓扑/结果下钻”的信息架构参考；它不能证明 NCCL 运行时已采集某个 RDMA 计数器。每个图表都应显示采集来源、时间范围和缺失状态。

## 建议的报告页面与图表

### 1. 摘要页（KAI 风格总览）

- KPI 卡：总样本数、成功/失败、消息大小范围、rank 数、节点数、P50/P99 延迟、峰值 algbw、峰值 busbw。
- “健康状态”只由可验证规则生成：例如 `#wrong > 0`、超时、缺失 rank、校验失败；不要把低带宽直接标成网络故障。
- 配置摘要：测试二进制/版本、NCCL 版本、CUDA/GPU、节点、网卡、拓扑来源、环境变量快照、采集时间（UTC）。

### 2. 消息大小扫描

- 折线图：x=message bytes（对数轴），y 可切换 `time_ms`、`algbw_GBps`、`busbw_GBps`；按 collective（AllReduce/AllGather 等）和数据类型分组。
- 同图显示 warmup 与 measured 样本的标记；展示 min/avg/max，必要时 P50/P99 带状区间。
- NCCL-tests 定义：`algbw = S/t`。`busbw` 是按 collective 和 rank 数修正后的可比硬件带宽，不能与 algbw 混用。AllReduce 修正为 `algbw*2*(n-1)/n`；ReduceScatter/AllGather 为 `algbw*(n-1)/n`；Broadcast/Reduce 为 `algbw`。详情见 PERFORMANCE.md。

### 3. 延迟与稳定性

- 箱线图或分位数带：每个 size/collective 展示 P50、P90、P99、min/max；样本数不足时标注 `n`，不要伪造分位数。
- CV（变异系数）和异常数可作为表格列；仓库输出格式还包含 `#wrong`、`i_min`、`i_max`、`i_p99`、`i_cv%`（见 `src/util.cu`）。
- 时间单位固定为 ms；带宽固定为 GB/s，并在导出中写明十进制/二进制换算口径。

### 4. Rank/节点/链路下钻

- 热力图：行=rank 或 GPU，列=size/collective，颜色=busbw、P99 或错误率；缺失样本使用灰色并保留原因。
- 箱线/条形图：按节点、rank、peer、channel 聚合的持续时间和字节数。
- 拓扑图：节点、GPU、NIC、rank 映射与 peer 关系；边上只显示实际采集的字节/事件计数，推断值必须标注“derived”。

### 5. 运行时事件时间线（若启用 NCCL profiler plugin）

- 甘特/时间线：collective API、collective kernel、P2P、proxy、network plugin 事件；支持按 rank、stream、channel 筛选。
- 事件详情可显示 operation/function、count、datatype、root/peer、seqNumber、nChannels、algo、proto、kernelVariant、userTag、开始/结束时间。
- 适合计算事件持续时间、重叠率、按 channel 的 busy time；不要把 API duration 当成 GPU kernel 或 RDMA wire duration。

## 可直接实现的字段（有来源约束）

| 字段 | 来源/计算 | 报告口径 |
|---|---|---|
| `collective`, `datatype`, `count`, `root`, `peer` | NCCL profiler v7 `collApi`/`p2pApi` | 原始字符串/整数，缺失即 null |
| `rank`, `commId`, `nNodes`, `nranks`, `commName` | profiler `init` 参数及事件 | 作为筛选和配置摘要 |
| `seqNumber`, `nChannels`, `algo`, `proto`, `kernelVariant` | profiler v7 `coll` | 仅在该事件类型支持时显示 |
| `time_ms` | nccl-tests measured timing 或 profiler start/stop 时间戳 | 明确测量层级（API/kernel） |
| `algbw_GBps` | `S/t` | S、t、单位均写入导出 |
| `busbw_GBps` | PERFORMANCE.md collective 公式 | 必须同时保存 `nranks` 和 collective |
| `#wrong`, `i_min`, `i_max`, `i_p99`, `i_cv%` | nccl-tests 输出列/`util.cu` | 显示样本数和聚合窗口 |
| `graphCaptured`, `groupDepth`, `userTag` | profiler v7 描述 | 作为上下文标签，不自行解释语义 |
| `proxyOp`（peer、channel、steps、chunk、send/recv） | profiler v7 | 事件级调试视图，不能直接等同 RDMA 包 |

## 不可直接推断的字段

- **真实 RDMA 链路吞吐、端口利用率、重传、ECN、PFC、拥塞根因**：NCCL algbw/busbw 或 profiler 事件不包含 NIC 端口计数器；需要 ethtool、厂商 NIC telemetry、RoCE counters 或交换机遥测，并关联时间戳。
- **物理拓扑/线缆/交换机端口映射**：rank、peer、channel 只表示 NCCL 逻辑关系；除非另有 inventory/拓扑证据，不得绘制为物理链路。
- **算法/协议未出现的原因**：事件字段为空可能是版本、激活 mask 或插件能力差异；不能断言“未使用 ring/tree”或“未启用 RDMA”。
- **GPU SM/内存占用、功耗、温度、PCIe/NVLink 带宽**：需要 CUPTI/NVML/Nsight 等独立采集。
- **因果结论（某 NIC/交换机导致慢）**：报告最多标记时间相关性和异常窗口，根因需要对齐 NIC/交换机/GPU 证据。
- **百分位和稳定性**：只有平均时间或单次样本不能推导 P99、CV；必须保留原始重复样本。
- **跨版本或跨硬件“理论峰值百分比”**：除非报告包含明确的链路额定带宽、协议开销和测量层级，否则不计算“利用率”。

## 实施约束与验收建议

1. 每个数值带 `source`（nccl-tests、NCCL profiler、NIC telemetry、derived）和 `scope`（API/kernel/run）。
2. 图表支持按 collective、size、rank、节点、stream、channel 过滤，并能导出原始行。
3. 缺失、不可用和未采集分开表示；`0` 不能代替 null。
4. 报告页脚显示 NCCL/CUDA/驱动、测试命令、环境变量、时区和采集时间，保证可复现。
5. busbw 公式测试应覆盖不同 rank 数与 AllReduce/AllGather/ReduceScatter/Broadcast/Reduce，避免把修正后的值当作原始 algbw。

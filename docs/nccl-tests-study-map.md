# nccl-tests 源码研究地图与覆盖状态

更新日期：2026-11-17。当前研究覆盖 nccl-tests/NCCL 执行原理，并形成前端运行与观测报告的端到端方案。本文件用于导航已有研究结论、标记研究边界。

本轮入口：[测试驱动与计时边界](nccl-tests-code-learning-guide.md) → [NCCL 核心执行链](nccl-core-architecture-research-notes.md)。前者以本地 nccl-tests 为准，后者单独固定 NVIDIA/nccl 提交；两个仓库的版本不能混用。

产品架构方案：[前端运行与可观测报告端到端方案](nccl/nccl-tests-frontend-runner-end-to-end-plan.md)；架构决策：[NCCL Run 与观测证据 ADR](adr/0001-nccl-runner-and-evidence-architecture.md)。该方案仍是待实现设计，当前前端原型没有真实 Runner/API。

## 研究层次

| 层次 | 要回答的问题 | 当前资料 | 状态 |
|---|---|---|---|
| 1. 仓库与程序结构 | 有哪些测试程序、入口和公共框架？ | [整体架构](nccl-tests-architecture.md) | 已建立 |
| 2. 单个测试的执行主链 | 命令行怎样走到 AllReduce NCCL 调用？ | [AllReduce 执行走读](nccl-allreduce-walkthrough.md) | 已跟读静态源码 |
| 3. Collective 语义与 buffer 布局 | 每 rank 输入/输出什么、count 和消息字节如何变化？ | [校验与布局](nccl-validation-layout-research.md)、[adapter 深入走读](nccl-adapter-deep-dive.md) | 11 类常规测试已覆盖源码层 |
| 4. 正确性校验 | 输入/expected 如何生成，错误数和容差是什么意思？ | [校验与布局](nccl-validation-layout-research.md)、[指标与验证](nccl-collectives-metrics-research.md) | 已覆盖主要机制；未硬件实测 |
| 5. 性能指标 | time/algbw/busbw 的公式、单位和口径是什么？ | [指标与验证](nccl-collectives-metrics-research.md)、[adapter 深入走读](nccl-adapter-deep-dive.md) | 已覆盖常规口径；busbw 是派生指标 |
| 6. 计时与运行参数 | warmup、迭代、聚合、blocking、graph、rank 聚合如何影响结果？ | [计时与 JSON 报告](nccl-report-and-timing-research.md)、[MPI rank 映射](nccl-mpi-rank-mapping.md) | 已覆盖静态语义 |
| 7. 结果报告 | stdout/JSON 有什么字段，如何保留配置和证据？ | [计时与 JSON 报告](nccl-report-and-timing-research.md) | 已覆盖当前 schema |
| 8. NCCL 内部执行 | NCCL 如何选算法/协议、走 GPU/网络传输？ | [核心执行链](nccl-core-architecture-research-notes.md) | 已静态核验固定提交的 API、调度、Ring、NET Proxy 主线；特殊后端与实机选择尚未全面验证 |
| 9. 实机环境行为 | 编译/运行、多机 rank 映射、不同 GPU/NIC 下结果如何？ | 需 GPU、CUDA、NCCL、MPI 环境 | 未验证 |
| 10. Keysight 网管体验 | Keysight 录屏里的页面、流程、信息架构如何映射？ | 用户提供的两份录屏 | 按用户要求暂缓，等 nccl-tests 阶段完成 |

## 11 类测试覆盖

| 测试 | 语义/源码路径 | 数据校验关注点 | 现阶段结论 |
|---|---|---|---|
| AllReduce | 全 rank 同索引规约后复制给所有 rank | expected 独立规约；Average 有特定 ULP 容差 | 已深入主链 |
| AllGather | 各 rank 输入块按 rank 拼接并广播到全体 | count 是单 rank chunk；接收长度乘 rank 数 | 已研究 |
| ReduceScatter | 规约后按块分发 | API 输入与带宽分子区分 | 已研究 |
| Broadcast | root 输入复制到各 rank | root 与非 root 输入角色不同 | 已研究 |
| Reduce | 所有 rank 输入，结果只在 root | root / 非 root expected 语义 | 结构与指标已覆盖，建议补 root 校验细读 |
| AlltoAll | 每个 rank 向每个 peer 发等长块 | in-place 路径不支持且不报告错误 | 已研究 |
| AlltoAllv | peer 间不同长度，支持生成流量或矩阵 | in-place 不支持；最大工作量和最长 rank 时间口径 | 已研究 |
| Gather | 所有 rank 的块汇总到 root | root receive buffer 含全 rank 块 | 已研究 |
| Scatter | root 大数组分块发给各 rank | 各 rank expected 是自己的分片 | 已研究 |
| SendRecv | rank 环：发给 next、从 previous 收 | in-place 不支持且不报告错误 | 已研究 |
| HyperCube | 自定义 Send/Recv 倍增交换，完成 AllGather 布局 | rank 数非 2 的幂时跳过 | 已研究 |

源码覆盖指理解 adapter 的静态结构，不等同所有版本分支、所有 datatype/op、所有配置组合都已运行验证。

## 需要完成的研究闭环

1. **命令行参数矩阵**：将 README 所有参数按资源拓扑、消息扫描、校验、计时、实现后端、报告分类；注明组合限制和各自改变的测量语义。
2. **MPI 多机映射**：已完成静态跟读，见 [rank 映射笔记](nccl-mpi-rank-mapping.md)；仍需不同 launcher 的多机实测。
3. **Reduce 校验细节与异常行为**：补 root-only 输出的错误检查解释；区分跳过、不支持、校验关闭和通过。
4. **NCCL 库源码层（按需）**：如果管理界面要呈现 tuning/算法选择，结合 `-U` profiler 字段与对应 NCCL 代码/文档，追踪字段的来源、可用版本和缺失原因。
5. **GPU 实验计划**：至少覆盖单机双 GPU、多 GPU，AllReduce 与 AllGather/ReduceScatter，校验开关，JSON/per-iter，AlltoAllv 不均衡矩阵；记录硬件、驱动、CUDA/NCCL、命令、原始日志和 JSON。没有这些环境时只标静态结论，不虚构性能值。
6. **阶段完成判定**：源码研究笔记、参数/输出字段表、错误/skip 状态解释和实测缺口整理完成后，再开始观看 Keysight 录屏并提取页面及流程需求。

## 源码基线说明

研究基于本地 `nccl-tests-master`。上游核验记录在 [源码核验 JSON](nccl-source-verification.json)；结论仍应与实际部署使用的 tag/commit 再核对。到目前为止没有声称已编译或运行 GPU benchmark。


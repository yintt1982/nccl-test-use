# NCCL 可观测性报告规格：以 Profiler Proxy 采集为第一阶段

## Problem Statement

当前原型已经有 Chunks、Timeline、Flows、QPairs、Ports 等独立报告页面，但真实数据的采集路径和各指标的来源边界尚未转化为可执行方案。用户希望先交付能够通过 NCCL Profiler 插件获取的数据，再逐步接入 RDMA/QP、NIC 端口和交换机数据。只运行 nccl-tests 的原生 benchmark 输出，不能产生逐 Proxy Step、GPU Channel、QP 或 NIC 端口指标；把样例或推导数据当作实测会误导分析。

本规格必须回答：怎样让 NCCL 加载 Profiler Plugin、插件要实现哪些回调、启用哪些事件类型、每类事件包含什么可用字段、如何与报告中的指标对应、如何处理不完整或无法关联的数据，以及哪些指标明确留给后续采集阶段。

## Solution

先交付 **Profiler Proxy 阶段**：在 NCCL Profiler Plugin 的事件回调中接收并持久化 Collective、ProxyOp、ProxyStep、GPU Kernel Channel 事件；用事件句柄和 NCCL 提供的父子对象关系关联事件；保留原始事件与时钟域；将结果导入 Chunks（界面上标注为 Proxy-step 视图）和 Timeline 两个独立报告页面。此阶段依赖 NCCL 实际加载的 Profiler ABI 和事件回调，不假设 nccl-tests 自带 profiler 已经完成了这些采集。

报告仍采用 Keysight KAI 风格：每个统计维度单独一页，图表在上、完整宽表格在下。Flows、QPairs、Ports 页面可以先保留，但其无真实来源的指标显示 `SAMPLE REPLAY`、未采集、不适用或采集失败；不得显示成真实零值。

### 采集方式与插件职责

1. **启用插件**：为 NCCL 准备与实际运行库 ABI 版本兼容的 Profiler Plugin，并通过 `NCCL_PROFILER_PLUGIN` 指向插件名/库。运行前记录 NCCL 版本、Profiler ABI、插件版本、启用配置和运行 rank。启动后检查 NCCL 日志或插件初始化记录，证明每个 rank 都成功加载插件；未加载时该次报告不得声称采集了 Profiler 事件。
2. **实现插件生命周期回调**：Profiler 插件提供 `init`、`startEvent`、`stopEvent`、`recordEventState`、`finalize`。`init` 记录 communicator/run 上下文并返回所需 event activation mask；`startEvent` 按事件类型读取描述符、分配事件句柄并记录开始时间；`recordEventState` 记录 Proxy 状态转换和增量参数；`stopEvent` 结束事件并写入可关闭记录；`finalize` 刷盘并写明丢失/未闭合事件数。回调中的耗时工作应尽量进入有界缓冲区，避免在 NCCL 热路径同步写文件。
3. **启用事件类别**：激活 Collective/P2P 上下文、ProxyOp、ProxyStep、Kernel Channel；若安装版本支持且确实需要，再启用 Kernel Phase、NetPlugin、CE 事件。激活掩码必须按 ABI 头文件和运行库版本编译/验证，不能假定不同 NCCL 版本的位值相同。记录请求的掩码和实际收到的事件类型。
4. **记录 ProxyOp 描述符**：保存 PID、channelId、peer、nSteps、chunkSize、isSend、原始事件类型和 communicator/rank 上下文。将 nSteps 命名为“声明/计划步数”，不能直接当作实际捕获到的步数。
5. **记录 ProxyStep 状态**：ProxyStep 描述符提供 step 序号；`recordEventState` 可能提供发送/接收等待、GPU wait、传输等状态转换。按 ABI 对状态参数的定义保存原始 state、参数、事件时间戳及有效性，不把状态名臆测成 RDMA request、网络完成或 GPU 消费完成。
6. **记录 Kernel Channel**：保存 channelId、GPU timer (`pTimer`)、所属 rank 和可证实的 parent 对象。其时间戳属于 GPU 时钟域；与主机回调时间比较前必须有可验证的时钟转换和误差界限。Kernel Channel 描述符本身没有独立 peer，不能复制 ProxyOp.peer 伪装成 GPU kernel 的目的端。
7. **写入事件流**：Profiler 回调把事件写入每个进程独立的有界队列，由后台写入 append-only JSONL（或等价格式）。事件记录包含 run_id、host、pid、rank、comm_id/comm_key、ABI、source_event_type、event_handle_id、parent_handle_id（可用时）、start/end 或 state timestamp、clock_source/domain、字段有效性、采集状态。事件明细与 nccl-tests 的 `-J` benchmark 结果分文件保存，并由 manifest 引用。
8. **配置冲突处理**：nccl-tests 的 `-U` 路径会选择其内置 `STATIC_PLUGIN` Profiler。若该路径与自定义 Profiler Plugin 冲突，启动预检必须明确阻止或要求使用兼容路径；不得静默覆盖用户指定的 plugin，也不能宣称两种插件同时工作。
9. **迭代关联**：v7 描述符中有 `seqNumber`、`userTag` 等字段，但有字段不表示测试驱动已经为每次 iteration 赋 tag。精确关联优先使用经验证的 userTag 或采集侧显式上下文；否则只保留 communicator 内序号或标注 inferred。nccl-tests 的 measured iteration 可能聚合 `agg_iters` 个 collective，原生 `times_us` 不得拆分伪造为逐 iteration/ProxyStep 时长。

### Proxy 阶段可以生成的指标

| 报告维度/指标 | Profiler 来源与生成方式 | 结果口径和限制 |
|---|---|---|
| Collective 清单 | Collective start/stop 描述符；func/count/datatype/root、seqNumber、algo、proto、nChannels、kernelVariant、userTag（按版本/事件类型取可用字段） | 原样保存字段；缺失字段为 null 并给原因；不把 communicator 序号自动解释成测试 iteration |
| Proxy 操作数/方向 | ProxyOp start/stop 及外层 rank/communicator | 可按 rank、peer、channel、send/recv、collective 统计事件数；这是软件代理活动，不是网络包数 |
| 声明 Proxy 步数 | ProxyOp.nSteps | 表示描述符报告的步数；与捕获到的 ProxyStep 数分开显示 |
| 观测 Proxy Step 数 | 被成功接收并归属到 ProxyOp 的 ProxyStep 描述符/状态 | 只有明确 ABI 语义下可以计数；漏事件、重复事件和孤儿事件需在质量摘要中报告 |
| Proxy 状态次数/占比 | `recordEventState` 的已知状态转换 | 可按明确状态统计数量和驻留时间；仅在对应 start/stop 成对且时钟域一致时计算 duration |
| 声明 Chunk Size | ProxyOp.chunkSize | 显示为 `declared_chunk_size_bytes`；不等于尾块实际长度、Payload Bytes、RDMA Message Size 或 packet size |
| 逻辑通信边 | ProxyOp.peer + isSend + rank + communicator | 表示 NCCL 逻辑 peer 方向；不等于跨主机实际匹配成功的发送/接收传输 |
| Channel 分布 | ProxyOp.channelId、Kernel Channel.channelId、Collective.nChannels | 按各事件原始作用域显示；channel 需限定 communicator/run，不能视为全局 ID |
| Proxy 时间线 | ProxyOp start/stop 和 ProxyStep 状态时间戳 | 仅表示 Profiler 定义的 host/proxy 事件边界；不得命名为 wire latency 或远端 GPU 完成时间 |
| GPU Kernel Channel 时间线 | KernelCh start/stop 或 ABI 给定的 pTimer 事件 | GPU 时间域的 channel 活动；转换到 host 时间前要校准并展示 uncertainty |
| Proxy/Kernel 重叠度 | 在同一 run 有可靠时钟映射、事件边界和覆盖率时，由时间区间计算 | 派生指标；显示输入事件范围、样本数、时钟误差；数据不满足条件则不可用 |
| Data Size | nccl-tests 参数或 Collective count × datatype size | 标注为每 rank 输入/输出或 collective 逻辑数据量；不是网络实际发送字节 |
| Payload Bytes / RDMA Message Size / Packets | Proxy 描述符不能独立提供可信值 | Proxy 阶段显示未采集；不得用 Data Size、chunkSize 或 nSteps 计算代替 |
| Flow / QP / NIC / Port / PFC / ECN / CNP / CRC | 需要 Network Plugin、RDMA/QP 连接和硬件遥测 | 本阶段不采集；显示后续阶段、样例或缺失状态，不能由 Proxy 字段推算 |

### 分阶段交付

1. **阶段一：Profiler Proxy**——完成插件加载/能力探测、Collective/ProxyOp/ProxyStep/Kernel Channel 事件采集、manifest/JSONL、Chunks Proxy-step 视图、Timeline、质量与来源标记。
2. **阶段二：Network、Flow、QP**——接入 NCCL NetPlugin/IB profiling 与连接生命周期；按实际可得的 WR、QP、device、opcode、length 和成功/失败状态形成有来源的 Flow/QP 视图。需要时再补连接两端映射，不能预设一次 ProxyStep 对应一个 WR 或 QP。
3. **阶段三：Port/NIC**——接入 NIC/HCA/交换机硬件计数器，提供端口范围的 bytes/packets、PFC、ECN/CNP、CRC 等设备实际支持的数据，保留 counter scope 和采样窗口。
4. **阶段四：跨层关联**——基于 rank/comm、parent handle、连接代次、QP/端口映射和时钟校准关联层间事件；输出相关关系和置信状态，不仅凭时间相关性断言因果根因。

## User Stories

1. 作为 NCCL 基准测试使用者，我希望先用 Profiler Plugin 采集 Proxy 相关事件，以便在部署 RDMA 和 NIC 采集器之前获得第一批细粒度报告。
2. 作为运行管理员，我希望能确认每个 rank 成功加载了指定版本的 Profiler Plugin，以便区分真实事件缺失与插件根本没有运行。
3. 作为插件实现者，我希望依据 init 返回的事件激活掩码选择所需事件，以便控制采集开销并识别未启用的事件类别。
4. 作为插件实现者，我希望通过 startEvent、recordEventState、stopEvent 和 finalize 完整处理事件生命周期，以便保留开始、状态变化、结束及未闭合情况。
5. 作为分析者，我希望查看 Collective 到 ProxyOp 再到 ProxyStep 的可验证父子关系，以便定位某个 collective 对应的代理活动。
6. 作为分析者，我希望单独查看 ProxyOp 的 rank、peer、channel、方向、nSteps 和 chunkSize，以便理解代理层声明的工作结构。
7. 作为分析者，我希望看到 ProxyStep 的实际观测数与 ProxyOp 声明的 nSteps 并列显示，以便识别部分采集、漏事件或未完成工作。
8. 作为分析者，我希望按 Proxy 状态转换查看等待和活动时间，以便辨别代理线程在哪些状态停留；同时知道这些状态不自动代表网络完成。
9. 作为分析者，我希望查看 Kernel Channel 事件并与 Proxy 时间轴对照，以便观察 CPU proxy 与 GPU channel 活动之间的时间关系。
10. 作为分析者，我希望主机时间和 GPU pTimer 保留各自时钟域，以便避免未经校准就比较不同域时间戳。
11. 作为分析者，我希望按 run、host、rank、communicator、collective 序号、peer、channel 和事件类型筛选，以便下钻到具体通信活动。
12. 作为分析者，我希望 Data Size、chunkSize、ProxyStep 数和 Payload Bytes 使用不同字段名及来源说明，以便不把逻辑数据量或配置大小误认为实际传输字节。
13. 作为分析者，我希望报告明确指出 Proxy 阶段不能直接得到 RDMA message size、包数和端口字节，以便知道何时需要启用后续采集阶段。
14. 作为分析者，我希望 Flows、QPairs、Ports 页面在采集未就绪时显示未采集或不适用状态，以便保留完整报告导航而不编造数据。
15. 作为分析者，我希望样例数据页面始终标记 `SAMPLE REPLAY`，以便不会把设计演示当成实测。
16. 作为分析者，我希望每个指标行记录来源、事件 ID、scope 和采集状态，以便从图表追溯到原始 Profiler 事件。
17. 作为运行管理员，我希望查看请求的事件掩码、实际出现的事件类型和缺失类型，以便发现 NCCL 版本或插件能力不匹配。
18. 作为运行管理员，我希望采集结果包含每个 rank 的事件收到、写入、丢失、重复、孤儿和未闭合数量，以便判断报告完整度。
19. 作为运行管理员，我希望 Profiler 回调使用有界缓冲和异步落盘，以便降低详细采集对 NCCL 热路径的影响。
20. 作为 nccl-tests 使用者，我希望 `-U` 与自定义 Profiler Plugin 的冲突在启动前明确提示，以便避免静默加载错误插件。
21. 作为分析者，我希望 `userTag`、collective `seqNumber` 与测试 iteration 的区别清楚显示，以便不把 communicator 序号错当 iteration。
22. 作为分析者，我希望原生 `times_us` 与 Proxy、Kernel 事件时长分开显示，以便基准轮次时延不被拆解成不存在的逐块时延。
23. 作为分析者，我希望 rank/peer 映射限定在 communicator 内，以便不同 communicator 中重复的 rank 编号不会被错误关联。
24. 作为分析者，我希望孤儿事件、重复状态、缺少 stop 的事件和时间倒退都能在报告中显式呈现，以便判断数据是否足够用于结论。
25. 作为分析者，我希望阶段二按 WR/QP/连接证据生成 Flows 和 QPairs，以便不把 ProxyStep 和网络请求假设为一对一。
26. 作为分析者，我希望阶段三的 NIC 计数保留端口级作用域，以便不会把物理端口聚合值伪装成 rank 或 QP 值。
27. 作为分析者，我希望跨层关联显示直接关联、推断关联或不可关联状态，以便评估报告结论的证据强度。
28. 作为报告使用者，我希望每个维度页面都采用图表加完整宽表格布局，以便先观察趋势，再核对原始行。
29. 作为报告使用者，我希望图表过滤条件与表格行、汇总统计和导出内容一致，以便报表结果可复核。
30. 作为项目维护者，我希望事件文件与 nccl-tests `-J` benchmark 文件分开保存并由 manifest 关联，以便高频 Profiler 事件不会改变原生结果格式。

## Implementation Decisions

- Profiler Plugin 是 NCCL 的 profiler 回调插件，不是替代 NCCL 网络传输的 Network Plugin。第一阶段只接入 Profiler ABI，不更换传输实现。
- 插件通过环境配置加载；加载能力、ABI 版本和日志证据纳入 run manifest。插件必须匹配运行时 NCCL Profiler ABI。
- 插件实现生命周期回调 `init`、`startEvent`、`recordEventState`、`stopEvent`、`finalize`。事件描述符和状态参数以目标 ABI 头文件为准，版本差异由能力表表示。
- `init` 返回 event mask。目标事件包括 Collective/P2P 上下文、ProxyOp、ProxyStep、Kernel Channel；其他事件按目标 ABI 能力和产品需求选择，不要求所有版本一律具备。
- 回调仅复制必要的字段和原始 state 参数到有界缓冲。后台线程异步写 JSONL；每个事件保留本地唯一 event ID、parent ID（若 ABI 提供）、源事件类型、原始时间和 clock domain。
- 在真实 GPU smoke test 之前，为每个目标 ABI 提供无 GPU ABI 测试宿主：按 versioned header 动态加载插件并驱动 `init/startEvent/recordEventState/stopEvent/finalize`，用合成描述符验证字段、parent 关联、异常状态、队列/异步落盘和清理。宿主输出只标记 `SAMPLE REPLAY`/合成 ABI 事件，不能证明真实 NCCL 已加载插件或实际产生事件。
- 实际关联优先使用 NCCL event handle/parentObj 层级。没有可证明父关系时保留 orphan 或 inferred，禁止按“最近时间事件”静默补链。
- ProxyOp 直接字段保存为 `pid`、`channel_id`、`peer`、`declared_n_steps`、`declared_chunk_size_bytes`、`is_send`。ProxyStep step 索引与 event-state 变化分开记录。
- `recordEventState` 的每一条状态按 enum 原始值、ABI 版本、状态参数、有效字段和 timestamp 保存。只有状态边界完整、语义已核实、时钟可比时才派生等待时间。
- Kernel Channel 保存 channel_id 和 GPU timer；它没有独立 peer。与 host 事件的换算必须提供校准来源及不确定度，否则不生成联合持续时间或重叠率。
- `seqNumber`、`userTag`、测试 iteration、aggregation index 是不同字段。若测试程序没有设置可验证 tag，则 iteration 关联为空或 inferred。
- 事件数据写入独立于 nccl-tests `-J` 结果的 JSONL/等价事件文件。manifest 保存 run、rank、NCCL/ABI/plugin 版本、激活掩码、收集能力、时钟、测试命令和事件质量计数。
- Chunks 页面在第一阶段称为 Proxy-step view；`nSteps`、实际 ProxyStep 数、传输请求数分别展示。不得用 `ceil(Data Size/chunkSize)` 生成实际块数。
- Timeline 分开表示 Collective API、Proxy、Kernel Channel 等测量边界。Profiler callback 的 start/stop 不自动命名为 GPU 执行、RDMA wire 或远端 GPU 完成时间。
- 每个指标具有来源、scope 和状态：真实采集、派生、样例回放、未采集、不适用、推断关联、采集失败。缺失值使用 null 并说明原因；0 仅表示来源明确测得为零。
- 保留独立维度页面和 KAI 风格布局：页面顶部图表、下方完整宽表格；支持筛选和导出原始行。
- `-U` 与自定义 Profiler Plugin 的不兼容情形需要预检、说明和清晰失败信息，不静默改写插件选择。
- 阶段二再接 Network/IB plugin 与 WR/QP/connection source；阶段三再接 NIC/HCA/交换机计数；阶段四做跨层关联。端口聚合计数不得拆分成 per-rank/per-QP 实测值。
- 本规格不包含 `executor unavailable` 外部执行工作流错误的修复。

## Testing Decisions

- 采用已确认的最高层测试边界：导入带版本的 Profiler JSONL fixture，经报告查询边界验证 Chunks/Timeline 的用户可见行、图表汇总和来源状态。
- 无 GPU 时增加 ABI 测试宿主验收：动态加载目标 plugin，按对应 ABI 驱动完整回调生命周期和合成事件场景；其结果与 fixture 报告测试、真实 NCCL/GPU smoke test 分开记录，不能相互替代。
- 黄金 fixture 包含一条 Collective、其 ProxyOp、多个 ProxyStep/状态转换、Kernel Channel 事件、缺失可选字段，以及孤儿/未闭合事件；验证实际关联和质量信息。
- 覆盖 ABI/event mask 能力：请求与实际事件类型分别可见；缺失类型不得被报告成“零事件”或插件成功采集。
- 覆盖 `declared_n_steps` 与 `observed_proxy_steps` 分离、chunkSize 不产生 payload/RDMA/packet bytes、Proxy 状态不伪装成网络完成。
- 覆盖主机与 GPU 时钟域：没有校准时不生成跨域 duration/overlap；有校准时报告不确定度。
- 覆盖 userTag/seqNumber/iteration 语义分离，以及 `agg_iters` 下不把 benchmark timing 拆成细粒度事件样本。
- 覆盖样例始终显示 `SAMPLE REPLAY`；空值包含原因；真实零值只允许来自明确测得为零的数据。
- 覆盖过滤器、图表统计、表格行和导出原始行对同一 fixture 一致；覆盖重复、乱序、截断、孤儿、未闭合事件的可见质量结果。
- 覆盖无 NetPlugin/QP/硬件遥测输入时，Flows/QPairs/Ports 不生成真实指标；端口/PID 聚合不得呈现为单 rank 或单 QP 数据。
- 好的测试验证导入和报告对使用者的外部行为，不绑定内部函数调用或 UI 私有实现。当前原型是单文件 HTML，仓库没有对应的自动化报告测试框架；工程实现应建立一个报告导入/查询集成边界，用真实结构的 fixture 验证端到端报告行为。

## Out of Scope

- 阶段一实现 RDMA WR、QP 生命周期、Network Plugin 传输事件、NIC/HCA 端口与交换机计数器。
- 从 Proxy 字段推导真实 RDMA message size、packet size、wire bytes、Flow Size、per-QP/per-port bytes、PFC/ECN/CNP/CRC 或拥塞根因。
- 声称 ProxyStep 与 Keysight 私有 Chunk、Buffer Offset 或 Flow 语义完全等价。
- 用静态拓扑、`chunkSize`、Data Size、rank/peer 或 NCCL algbw/busbw 推断物理路径与实际网络吞吐。
- 修改 nccl-tests 原生 benchmark 算法或将 `times_us` 改造成细粒度事件时间。
- 修复独立的外部 executor 可用性错误。
- 远程部署或修改任何外部服务。

## Further Notes

- 已检查的本地 nccl-tests Profiler v7 描述符含 ProxyOp 的 pid/channelId/peer/nSteps/chunkSize/isSend，ProxyStep 的 step，Kernel Channel 的 channelId/pTimer，Collective 的 seqNumber/userTag 等字段。ABI 存在这些字段，不代表 nccl-tests 当前内置回调完整采集并持久化了它们。
- 本地 nccl-tests 内置 profiler 当前 activation mask 包含 Coll/P2P/ProxyOp/CE，但不包含完整 ProxyStep/Kernel Channel；`recordEventState_v5/v6/v7` 当前为空实现。它只保存代表性 Proxy chunkSize。实现阶段一需要补齐或提供独立可加载 Profiler Plugin，验证回调事件确实到达。
- `-U` 会选择 `STATIC_PLUGIN`；与自定义插件的加载路径存在冲突，需要在实现中以目标构建和 NCCL 版本实测确认。
- 本地 `common.cu` measured loop 可在一次测量迭代聚合多个 collective；原生 `times_us` 是 benchmark 边界计时，不能分解为 rank/step 时间。
- `chunkSize` 是 Proxy 描述符的配置/工作参数，不等于每步有效 payload、RDMA 消息长度或网络 packet size。
- 术语表为通用网络分析术语，未定义 NCCL Profiler 术语；仓库快照未找到相关 ADR。本规格沿用研究文档对 ProxyOp、ProxyStep、Kernel Channel、rank、QP、port 的限定语义。
- 测试边界已由用户确认：版本化 Profiler JSONL fixture → 报告查询边界 → Chunks/Timeline 表格、图表、来源状态，覆盖孤儿/缺失事件和 SAMPLE REPLAY。
- 发布到 issue tracker 仍需先配置正确的 NCCL 项目/team 和 `ready-for-agent` 标签。当前连接的 Linear workspace 只有 TIN 团队，且未发现该标签；按 `to-spec` skill，应运行 `/setup-matt-pocock-skills` 配置 tracker 与 triage 词汇后再发布。

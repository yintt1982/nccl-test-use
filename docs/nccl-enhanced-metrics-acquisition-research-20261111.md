# NCCL-tests 增强指标采集研究：Chunks / Timeline / Flows / QPairs / Ports

研究日期：2026-11-11（Asia/Shanghai）。范围：官方接口、源码与可实施采集方案；未连接 GPU/RDMA 服务器，未进行真实 NCCL 或硬件计数验证。本报告不修改 UI、不部署服务。

## 1. 结论与证据边界

**可以实现这五类报表，但不能承诺只运行原生 nccl-tests 就获得截图全部字段，也不能承诺与截图工具的私有指标语义一致。** 可先实现 NCCL Collective → ProxyOp → ProxyStep 与 GPU Channel 时间轴；精确 NIC/QP/网络字节需要网络事件、连接元数据及硬件采集；模拟器的组件名和未公开指标定义应保持不可用。[L1][L2][S1][S2][S3]

以下三种证据等级贯穿本文：

- **已验证源码/API**：在本文固定提交或本地文件中存在相应字段、调用或计数实现；不表示部署环境已支持。
- **设计建议**：本平台定义的关联键、统计口径、采集流程；不能冒充 NVIDIA 或截图厂商定义。
- **待实机验证**：运行库 ABI、编译开关、NIC/固件/驱动、事件完整率、计数作用域、时间误差及开销。

关键判断：

1. 本地 Profiler v5/v6/v7 有 channel、peer、nSteps、chunkSize、isSend、seqNumber；v7 有 userTag。字段存在不等于 nccl-tests 已为每个 iteration 设置 tag。本地源码搜索只在 v7 头文件发现 userTag，不能据此宣称 iteration 已精确关联。[L2][L3]
2. 固定上游 NCCL 提交的内置 IB 网络事件已有 device、wr_id、opcode、qpNum、length，值得优先复用；但它受 **NET_PROFILER 编译开关**控制，本文提交默认值为 0。不能只设置环境变量就承诺得到 QP 事件。[S2][S3][S4]
3. 该提交的 NetPlugin parentObj 非空时可关联 ProxyStep；v6+ 允许其为空。此链条只证明本地层级，不自动提供跨节点发送/接收一一配对、远端 QPN、物理 NIC 或应用缓冲区偏移。[S1][S2][S5]
4. Linux 支持 QP 绑定计数器；auto pid/type 是多个 QP 的分类聚合。本文 Linux 提交还存在可绑定 QP 的 optional RDMA packets/bytes、CE/CNP 流计数能力，但需要对应驱动、固件、硬件、绑定方式和权限。不能把所有 Linux RDMA 环境一概称为支持，也不能一概称为不支持 per-QP bytes。[S8][S9][S10]
5. Port、vport、priority、QP counter set 是不同作用域。ethtool 端口聚合数不能按 rank/QP 平分后标为真实 QP 指标。[S8][S11]
6. CQE/网络请求完成、GPU 可见、GPU 消费完成、Collective 完成必须分开。CQE 时间不能直接命名为远端 GPU 消费完成。[S6][S7]

## 2. 研究版本、资料与既有架构

本地目录实际为 D:/codeprojects/Vibecoding/nccl-tests；用户上下文里的 D:/代码工程 路径在本次环境中不存在。仓库副本未提供可用 Git 元数据，因此不把目录名 master 或本地元信息作为上游版本证明。本地 profiler_v7.h SHA-256：96CCC69492850FEC143549F17177155F8A3A6FC8437137B642446ACA731E9048。[L2]

| 对象 | 本次核验基线 | 可据此声称什么 |
|---|---|---|
| 本地 nccl-tests 副本 | src/profiler_v5.h / v6.h / v7.h、nccl_profiler.h、common.cu | 文件实际包含的接口与 -U 行为；不证明实际加载的 NCCL ABI |
| NVIDIA/nccl | ad2d192c1f279aefcaf340f67e07880b65a1116b，提交时间 2026-11-09；version.mk 为 2.32.6 | 本文固定源码能力；**不声称此版本已安装或为最新正式发行版** [S0] |
| Linux | 067c5c24cef23d913b67eaaf71ae67e7169a6620，提交时间 2026-11-10 | mlx5 RDMA counter / flow-counter 实现 [S9][S10] |
| iproute2 | 1d64d3416d60bc8161dff3aa6c3befd9edbb2b62，提交时间 2026-11-10 | rdma statistic / resource 命令语义 [S8][S12] |
| rdma-core | d04695bbd52ae49f7fbe0cdbfc8abd0328c908d3，提交时间 2026-11-11 | verbs 文档语义 [S7] |
| mlx5 ethtool 文档 | Linux 官方文档在线版，访问日期 2026-11-11 | 命名与作用域；部署机实际输出仍需探测 [S11] |

沿用已接受架构与阶段，不重新选择 UI：nccl-tests + Profiler + Network instrumentation + RDMA/QP Collector + Port Collector + Local Agent + Correlation Service + Report API。阶段仍为 Chunks/Timeline → Flows/QPairs → Ports → 跨层关联。阶段一可以提前做只读端口能力探测，但不改变交付顺序。[L1]

原生结果仍独立保留：本地 common.cu L728 有 warmup，L759–764 的 measured iteration 内还可包含 agg_iters 个 collective；util.cu L680 起写 times_us / per_process_max_times_us。因此原生 times_us 不是rank级chunk时间轴，也不是Chunk/Flow完成时间CDF原始样本。只有明确同一测量边界和聚合规则时才做对照，不能把一个原生iteration计时拆成虚构chunk时间。[L3][L4]

## 3. 截图字段的解释边界

截图中的 generic.1.nic.0 和 Emulated Tx Component 表明至少该字段描述模拟组件；不能把它自动映射为真实机器的 mlx5_0、PCI BDF 或 netdev。真实采集模式该列应为 Not applicable，另用明确命名的 Source NIC / Destination NIC 展示实机设备；如果导入模拟器文件，保存原始字符串并标 simulation/replay。以上是基于用户截图的解释与本平台设计，不是对未知厂商实现的认证。

截图 Buffer Offset 出现 0/1 且无单位，不能确定是字节偏移、buffer 槽位或逻辑索引；Chunk Count、Flow、Completion Time 的统计定义也未给出。**不得用同名列暗示语义已对齐。** 需要截图工具的 schema/导出文件/官方文档才能确认等价。截图 QP Bytes Tx=68,125,696 与 Chunk Payload=67,108,864 的差额不能用于推出通用协议开销、帧长度或重传数量。

以下矩阵的“可实现”是本平台明确口径下的能力，不是截图值的复刻。状态建议：Real collection、Sample replay、Partial、Not collected、Not applicable、Collection failed、Inferred association；缺失值为 null + reason，不以 0 替代。[L1]

## 4. Chunks / Timeline 字段矩阵

| 截图字段 | 采集来源 / 建议口径 | 能力与限制 |
|---|---|---|
| Data Size | nccl-tests size + datatype/count；Profiler collective count/datatype | 可验证输入量；AllGather/ReduceScatter 等需标明 per-rank input/output 口径，不等于 wire bytes [L3][S1] |
| Collective | coll.func，algo，proto | 直接字段；算法/协议一并保留 [S1][L2] |
| Iteration | 测试驱动侧 phase、iteration、placement、graph replay、aggregation index；可用 userProfilerTag | 需要驱动插桩；seqNumber 是 communicator 内 collective 序号，不是天然 iteration [L2][S1][S13] |
| Source Rank / Destination Rank | proxyOp.rank + peer + isSend，并记录 communicator/group | 表示该通信边的端点，不是 AllReduce 原始数据贡献者；不能跨 communicator 复用裸 rank [S1] |
| Data Chunk Transfer ID | 平台生成 run/comm/rank/op/channel/step/event sequence 的 ID | Profiler 没有截图式跨节点传输 ID；本地事件 ID 与双端 transfer ID 分开，未配对时禁止伪造 [L2][S1] |
| Buffer Offset | 若插桩能追踪注册区间与实际 SGE，定义 registration_offset_bytes；另行追踪 application_buffer_offset_bytes | collective sendBuff/recvBuff 加网络 data 指针不足以跨 staging、FIFO、分片证明应用偏移；截图单位未知，第一阶段不可用 [S1][S6][S7] |
| Chunk Count | 建议分别显示 observed_proxy_steps、declared_nSteps、实际 transfer_count | nSteps 是 proxy 工作的 steps，不保证等于截图 chunk 数；不可直接 ceil(dataSize/chunkSize) 或照搬 nSteps [S1] |
| Channel ID | proxyOp.channelId 或 kernelCh.channelId | 直接字段；必须附 comm/op，不能当全局 channel [S1][L2] |
| Emulated Tx Component | 模拟器导入字段 | 真实 GPU/NIC 模式 Not applicable；不能编造 generic.* 标签 |
| Transmit Type | 分列 direction(send/recv)、transport(IB/RoCE/socket/本地)、verbs opcode、algo/proto | 标准 Profiler isSend 不是完整截图 transmit type；opcode 仅在已验证有效的网络事件路径显示 [S1][S2][S3] |
| Source NIC / Destination NIC | 网络 device index → properties → PCI/GUID/port；远端依连接握手 | 精确双端映射需要连接记录；逻辑/虚拟 NIC 可含多个物理设备，静态拓扑只能作为候选 [S5][S6] |
| Chunk Size | proxyOp.chunkSize | 可显示 declared_chunk_size_bytes；尾块及协议实际传输量需另采，不能当每一块 payload [S1][L2] |
| Payload Bytes | ProxyStep transSize；发送侧 net IB length；明确观察层 | 本地 v5 已有 transSize；net.cc 对实际发送/接收大小赋值。可作为 transport_step_bytes，不自动等于应用净有效字节或含头 wire bytes [L2][S14] |
| Start(ms) | 事件原始时间经同一域换算，减明确 run/iteration origin | v5-v7 主机回调时间与 GPU pTimer 分开；跨域必须校准；默认不把回调发出时间当 GPU 执行开始 [S1][S15] |
| Completion Time CDF | 对明确事件边界的 duration 或 completion_offset 分别统计 | 平台派生；须显示样本单位、N、过滤项、未闭合数、单位及时间不确定度，见第8节 |
| Rank 时间轴连接 | 本地父子层级；双端 matched transfer 才画精确边 | peer 只支持逻辑通信边；无共享 transfer key 时画 inferred/dashed，不能声称实际每块发送接收对应 [S1][S3][S5] |

**阶段一建议把 Chunks 标为 Proxy-step view。** 用户仍可在 Chunks 页面查看，但 tooltip 明确“观察对象为 NCCL proxy step，未证明等价于截图模拟器 chunk”。单机走 NVLink/共享内存时可以有 GPU 时间轴，却没有同一套 IB QP/Port；相应网络指标显示 Not applicable，而不是 0。[S1][L1]

补充 KernelCh 字段边界：本地 v7 的 kernelCh 描述符只有 channelId 和 pTimer，加上外层 rank/parentObj；没有独立 peer。GPU Channel 行不能直接复制 ProxyOp 的 peer 并表示已测的单一目的端。只有额外关联证据证明对应关系时才显示关联端点，否则该字段为未采集/不适用，并保留原始事件类型。[L2]

## 5. Flows 字段矩阵与网络插件方案

Flows 中重复的 Data Size、Collective、Iteration、Source/Destination Rank、Transfer ID、Buffer Offset、Chunk Count、Channel ID、Emulated Tx Component、Transmit Type、Source/Destination NIC、Chunk Size、Payload Bytes、Start(ms)，均继承第4节的来源和限制；不会因为切到 Flows 视图而获得额外事实。新增字段如下。

| 字段 | 建议实现 | 能力与限制 |
|---|---|---|
| Destination QP ID | 建连阶段 remoteQpNum + 远端设备/port/QP generation | IB NetPlugin v1 只携带本地 qpNum；RC/UC 的 dest_qp_num 可由拥有 QP 的进程查询/连接插桩导出；不能从 RC CQE 的 src_qp 获得，src_qp 仅对 UD 有效 [S2][S5][S7] |
| Flow Size | 定义为选定 flow 下去重的 transport request/WR 数据总量 | 必须明确 flow 是一次逻辑 transfer、一次 request、一个方向的 QP 生命周期、还是固定窗口；WR、QP、step 多对多，不能直接把 Data Size 当 Flow Size [S3][S6] |
| Flow Metric Completion Time CDF | 使用同一 flow 定义和完成边界生成分布 | 全寿命 QP 在测试中复用，通常不是天然的一次流完成；建议先做 request/transfer duration CDF |

### 5.1 优先复用内置 IB profiling

已验证固定 NCCL 源码：

- net_ib_v1.h 的 ncclProfileQp 描述包含 device、wr_id、opcode、qpNum、length。[S2]
- p2p.cc L214-231 在发送 WR 提交附近启动事件，length 来自 SGE；L848/L919 的完成处理停止相应事件，另有清理路径停止事件。因此 stop 需要附 outcome/close_reason，不能把所有停止都视为成功。[S3]
- p2p.cc L517-519 的 prepostReceiveWorkRequests 路径直接跳过该接收启动段；L523-538 的接收事件仅显式赋 device/wr_id/qpNum。没有证据保证接收 opcode/length 完整，必须按方向与路径设置 validity bitmap。[S3]
- makefiles/common.mk L24 默认 NET_PROFILER=0，L226-228 才根据非0值添加 NCCL_ENABLE_NET_PROFILING。部署包可能没有这些事件。[S4]
- README L334-340：v6+ 可以仅启用 NetPlugin；parentObj 若非空才是 ProxyStep；为空不能擅自回接最近 ProxyOp。建议主动开启 ProxyOp+ProxyStep+NetPlugin，并记录 orphan 数。[S1]

因此最小工程路线是：自建 Profiler 解析标准与 IB 扩展事件 → 在已验证构建上测试 → 缺失连接字段再补小范围网络插桩。**无需先假定必须完整重写 NCCL Network Plugin。** 若最终选外部 Network Plugin，需实现相匹配的 versioned ABI，并验证 GPU buffer registration、flush、进度、错误处理等，不能仅为埋点替换传输逻辑。[S6]

### 5.2 必须补充的连接事件

设计建议新增 connection_open/qp_state/connection_close：run_id、host_boot_id、pid_start、comm_key、connection_generation、local_device、physical_members、PCI/GUID、local_port、local_qpn、remote_host、remote_qpn、remote_gid、transport、QP type、created_at、destroyed_at。保存 parent step 与 request/WR event 的关系表；一个 request 可分散到多个 QP，一个 QP 可服务多个 request，不做一对一强制假设。[S3][S5][S6]

Buffer offset 需分开记录 application_buffer、registered_memory_region、staging_buffer 三种基准；只在同一已知区间做指针差。地址转成进程内 buffer_id + offset 后持久化。跨主机两端 wr_id 不保证相同，且可复用；不能用 wr_id 或 qpn 单独作为全局键。[S3][S7]

## 6. QPairs 字段矩阵：软件活动与硬件计数分开

| 截图字段 | 来源 / 口径 | 边界 |
|---|---|---|
| Source QP ID / Destination QP ID | QP 创建/连接握手、ibv_query_qp | QPN 加 host/device/generation 才唯一；远端字段可能不可得，显示 null [S5][S7] |
| Source NIC / Destination NIC | QP → HCA/port → PCI/netdev；远端连接记录 | 需要实测映射，不能把 logical NCCL device index 直接命名物理 NIC [S6][S12] |
| Messages Sent / Received | 平台定义 successful transport requests 或 posted/completed WRs，分别计数 | 不等于以太网帧，也不等于 NCCL collective 次数；RDMA Write 与 SEND 接收机制不同，不能泛化 receive CQE 计消息 [S3][S7] |
| Messages Failed | post 失败、completion failure、aborted request 分列，去重归并 | CQ error 可使多个请求失效；仅计 CQE status 非成功会漏 unsignaled/未完成请求。错误 CQE 仅部分字段有效 [S7] |
| Bytes Tx/Rx | 优先明确标为软件 request_bytes；可选单QP硬件 RDMA byte counters 单独列 | 软件数据长度不含全套 wire 开销；硬件计数需确认可选counter开启、绑定成员为单QP、方向与字节边界 [S3][S8][S9][S10] |
| Frames Tx/Rx | 已验证单QP硬件 packet counter 或明确的抓包层 | 可选 rdma_tx/rx_packets 在本文 Linux 源码存在；不是所有 NIC 都支持，packet 与截图 frame 的边界未证实 [S9][S10] |
| Frames Delta | 定义 delta=同窗同单位 tx_packets-rx_packets | 平台派生；仅在配对端点、同步窗口和相同过滤范围下比较。非零不直接等于丢包，可能含重传、控制包、窗口截断、统计差异 |
| SeqError（字幕提及） | packet_seq_err、out_of_sequence、implied_nak_seq_err 等原始名称 | 不是同义列；分别暴露作用域、是否可按QP绑定和驱动定义，不汇总成未定义 SeqError [S9] |
| 重传（字幕提及） | duplicate_request、local_ack_timeout_err、roce_adp_retrans、roce_adp_retrans_to 等候选 | 超时/重复请求/重传事件是不同计数，不能相加称“重传帧总数”；能力及硬件语义待验证 [S9] |

### 6.1 rdma statistic 的正确使用

官方手册规定 auto 模式把同类别对象绑定同一个 counter set：auto pid、auto type、auto pid,type 都可能聚合多个 QP。手动 bind lqpn 且不指定 cntn 会申请新 counter；指定已有 cntn 可共享。输出必须保存 counter_id、实际 member_qpns、scope、binding_start/end，不可仅凭 rdma statistic qp 命令名称标单QP。[S8]

本文 Linux mlx5 counters.c L38-109 包括请求、序列、重试等 q-counter 家族；L155-160 包括 optional rdma_tx/rx_packets/bytes；L459起、L589-654 包括 per-QP optional counter 的读取与绑定。fs.c 使用 QP 相关匹配安装流计数规则。因此“Linux 绝不可能按 QP 取得 bytes/packets”已过于绝对；准确描述是“存在已验证源码能力，实机是否具备取决于完整软硬件栈和安装/绑定成功”。[S9][S10]

不要把默认 /sys/class/infiniband/.../ports/.../hw_counters 或 rdma statistic show link 输出自动解释为某个 QP。端口拥塞计数也不因旁边有一列 QPN 就变成 per-QP。可选计数设置可能替换当前启用列表，采集器应读取原配置、只在隔离测试端口变更并恢复；资源不足/无权限须保留具体错误。[S8][S9]

## 7. Ports 字段矩阵

| 截图字段 | Linux/mlx5 候选来源 | 范围与注意事项 |
|---|---|---|
| Device / Port | RDMA device、port、netdev、PCI BDF/GUID 清单 | 记录 PF/VF/representor、logical/vport/physical 的关系，不能混写 [S6][S11][S12] |
| Ranks | 本次 run 的 rank→实际连接→port 关联 | 平台派生；静态就近拓扑不证明实际使用；一个rank可用多port，一个port可被其他进程共享 [S5][S6] |
| Speed | ethtool link speed、NCCL properties.speed(Mbps)、RDMA port 属性 | 记录采样时刻与 link state，区别配置/协商速率，不等于测得吞吐 [S6] |
| Frames Tx/Rx | tx_packets_phy / rx_packets_phy | 物理端口；rx_packets_phy 不包括因FCS/帧长等丢弃的包，不能等同所有上线路帧 [S11] |
| Bytes Tx/Rx | tx_bytes_phy / rx_bytes_phy | 文档明确 rx_bytes_phy 包括 Ethernet header/FCS；不要默认所有统计器有相同边界，且不可归因到单run [S11] |
| PFC Rx | rx_prio[p]_pause，另存 pause_duration | 每优先级0–7；只有相应PFC启用时可用。次数与时长单位不同，global pause不是同一指标 [S11] |
| ECN-CE Rx | np_ecn_marked_roce_packets 或 optional cc_rx_ce_pkts | 以设备实际暴露与定义为准；**rx_prio[p]_marked 表示设备因主机侧拥塞给包打ECN标记，不能当收到CE包数** [S9][S10][S11] |
| CNP Tx | np_cnp_sent 或 optional cc_tx_cnp_pkts | 区分 notification-point 与匹配流计数，保存原始counter名称、作用域和能力 [S9][S10] |
| CNP Rx | rp_cnp_handled / rp_cnp_ignored，或 optional cc_rx_cnp_pkts | handled/ignored 不自动等于全部收到的CNP；除非证明互斥完备，不盲目求和替代 Rx [S9][S10] |
| CRC Errors | rx_crc_errors_phy | 外部端口收包 FCS 错误；不是 PCIe CRC，也不是完整RDMA数据校验失败总数 [S11] |
| RoCEv2 QPs Configured | QP 生命周期清单；以RoCEv2/GID类型/连接证据筛选 | 这是资源/状态数，不是通用 ethtool 硬件计数；明确 count_at_time 或 created_during_run [S7][S12] |
| RoCEv2 QPs Ready | 平台明确为已观测 RTS 的目标数据QP数，并附配置状态 | RTS 是本地QP状态，不能保证对端可达/链路健康；排除 flush/control QP，不能混为“业务已成功” [S5][S7] |
| RoCEv2 Data Frames Tx | 优先经验证的协议/操作码硬件匹配或抓包 | tx_vport_rdma_unicast_packets 计 RoCE/UD/RC 等RDMA流量，不能直接命名“仅RoCEv2数据帧”；rdma_tx_packets 也须确认过滤边界，必要时显示 RDMA packets Tx [S10][S11] |

端口统计采用前后快照与周期样本：delta=value_end-value_start，遇重置、设备重启、计数宽度回绕不直接给负增量；保留原值、读开始/结束时间、counter bit width（若已知）、reset_generation。共享端口下标 port_total/shared，不声称所有增量均来自当前 NCCL。即使隔离运行也先做空闲背景基线。以上为设计建议。

## 8. 时间边界、同步、CDF 与数据质量

### 8.1 四类时间不能混为同一个完成

| 边界 | 建议列名 | 可证明的内容 |
|---|---|---|
| Profiler ProxyOp/ProxyStep start→stop/state | proxy_duration / step_state_duration | NCCL主机proxy观察的活动边界 [S1] |
| KernelCh/KernelPhase pTimer start→stop | gpu_channel_duration / gpu_phase_duration | GPU记录的时间域；不能直接减主机时钟 [L2][S15] |
| Network request post→test(done) 或 QP event stop | network_request_duration / observed_qp_event_duration | 请求/WR完成处理相关，受轮询和清理路径影响 [S3][S6] |
| 接收可见/消费与 collective执行完成 | gpu_visibility / gpu_consumption / collective_completion | 需要各自显式事件；iflush用于GPU可见，irecvConsumed另表达设备消费，不能由CQE直接推导 [S6][S7] |

v5-v7 插件在回调时读取的主机时间应名 callback_timestamp；KernelCh 使用原始 pTimer。本文上游新增 v8：可在 init 选择 CLOCK_MONOTONIC/REALTIME，并启用 gpuClockSync；描述符/状态含 tsCpu 与 tsCpuUncertainty，GPU映射的不确定度是校准锚点界限，不覆盖锚点间漂移。默认不开启，CE事件还有独立限制。该能力值得加入 capability registry，但不能强行把本地 v7 二进制当 v8 解析，也不能称其已替代跨节点同步。[S15]

跨主机设计：每节点保留 monotonic 原始时间、host_boot_id、时钟域；有 PTP/PHC 则记录系统钟/PHC 的校准与误差，无PTP则提供校准探针的估计偏差/漂移。Linux PTP 提供硬件时钟访问和同步接口，但“安装了PTP”不等于“所有事件已精确对齐”。每条跨节点边带误差范围 ε；当先后差不足以超过双方误差界时，不下因果先后结论。[S16]

### 8.2 CDF 定义（平台设计）

- duration_ms=(end-start)/1e6，与 completion_offset_ms=(end-origin)/1e6 分成两个指标；截图只写 Completion Time，无法判定是哪一个。
- 观测单位必须是一个已闭合 proxy step / request / flow；固定 collective、data size、phase、placement、方向、算法/协议后才汇总；不把相同事件在 Chunks 与 Flows 重复计入同一CDF。
- 经验CDF定义 F(t)=完成且有效样本中 duration≤t 的数量 / 有效完成样本数。图例同时给 total_started、completed、failed、unclosed、filtered、sampled、lost。该CDF明确是“完成样本条件分布”。
- 未闭合不能当0，也不能静默删除后声称代表全体；如需完成概率/尾延迟推断，另行采用明确的删失分析并说明假设。高丢失时不自动做此推断。
- 保留原始ns和显示ms；图表竖线不能证明样本量。跨主机端到端 duration 只有双端关联与时钟误差可接受时才输出，否则仅显示本地主机同域duration。

### 8.3 质量门槛

沿用已有验收目标：rank覆盖100%、collective覆盖100%、event loss≤0.1%、未闭合≤0.1%、time-reversed=0、Standard开销≤5%、Network Diagnostics开销≤10%。它们是项目验收目标，**不是本次已测结果**。[L1]

新增建议：按 event type/direction/path 报告有效字段率、parent关联率、双端配对率、counter绑定成员数、计数采样间隔/读取耗时、clock uncertainty、采样比例；任何不满足字段级完整性的值设 Partial。能绘制本地时间轴不代表能下跨层根因结论。失败时保留原始记录，显示 Insufficient data quality; no diagnostic conclusion generated。[L1]

## 9. 最小 PoC（待执行，不是已有结果）

### 9.1 目标与准备

两个 Linux GPU 节点，每节点一个 GPU/一个rank，先固定单个消息大小、单一collective、单一算法/协议作为对照；确认实际使用IB/RoCE路径。另做单机本地路径和4rank的 MOD 2 分组，检查N/A与组边界。记录 NCCL runtime/构建版本、Profiler ABI、插件SHA、CUDA/driver、kernel、rdma-core/iproute2、NIC固件、PCI/port、MTU/GID/traffic class。不要用仓库头文件版本代替动态加载库版本。[L1][S4][S6]

先做只读发现（命令选项依部署版 --help 确认）：

```bash
uname -r
rdma -V
rdma link show
rdma -j resource show qp
rdma statistic mode supported link mlx5_0/1
rdma statistic qp show link mlx5_0/1
ethtool -i <netdev>
ethtool <netdev>
ethtool -S <netdev>
./build/all_reduce_perf --help
```

这些命令是采集建议；若 JSON/子命令不支持，记录版本差异后使用该版本实际提供的方式，不把命令失败改写成0。[S8][S12]

### 9.2 PoC A：Profiler 与单方向QP事件

1. 为选定固定NCCL源码构建测试库，明确 NET_PROFILER=1；保留构建日志和实际运行库路径。先验证普通运行正确，再加载插件。[S4]
2. 插件导出实际支持的 versioned symbol，按ABI独立适配；激活 Coll/ProxyOp/ProxyStep/KernelCh/NetPlugin；在支持时额外KernelPhase。回调仅复制有效字段到有界队列，后台写JSONL；禁用未知netPlugin id/version解析。[S1][S2][S15]
3. 使用 libnccl-profiler-observe.so，NCCL_PROFILER_PLUGIN=observe；按实际MPI运行环境传播变量和库路径。NCCL_DEBUG=INFO 留存选中网络和插件加载证据。先不开 -U。[S1][L3]
4. 初始参数建议 fixed-size -b 64M -e 64M -n 20 -w 5 -g 1；若本地 --help 支持，显式 -G 0、-m 1、-U 0，禁用graph/aggregation以便首轮配对。MPI两节点各1进程；不要未检查就复制参数到不同版本。
5. 第一轮仅要求原始事件落盘及正确层级，不要求所有截图列都有值。检验发送QP事件长度、接收方向缺失、prepost路径、stop清理语义。
6. 新增驱动事件phase=warmup/measured/validation、iteration、placement、aggregation_index、graph_execution_id。只在实际支持 userProfilerTag 的调用路径赋tag并建立映射表；图重放多次不能靠捕获阶段一次tag区分所有执行。无tag时标 inferred 并限制实验条件。[L2][L3][S13]

建议启动骨架（需替换host、路径，变量转发按MPI实现配置）：

```bash
export NCCL_PROFILER_PLUGIN=observe
export NCCL_DEBUG=INFO
mpirun -np 2 --host <hostA>:1,<hostB>:1 \
  ./build/all_reduce_perf -b 64M -e 64M -n 20 -w 5 -g 1 -G 0 -m 1 -U 0
```

**-U冲突继续保留。** 本地 common.cu L1519-1525 会将用户插件选择覆盖为 STATIC_PLUGIN。虽然本文上游README支持最多两个Profiler插件，但当前驱动覆盖行为没有因此自动消失；不要宣称已可共存。后续单独修正驱动、核验多插件ABI与开销后才能放开。[L3][S1]

### 9.3 PoC B：连接与per-QP计数

1. 在建立目标数据QP后、业务流量开始前建立采集同步点，导出本地/远端QPN、port、connection generation；进程退出前再保留同步点以读取尾快照。独立collector不能拿别的进程裸指针调用 ibv_query_qp。[S5][S7]
2. 先测手动单QP新counter绑定：rdma statistic qp bind link mlx5_0/1 lqpn <QPN>；保存返回counter、成员和绑定时间，验证只有目标QP。另做双QP同pid反例，证明 auto pid 的聚合不能冒充单QP。[S8]
3. optional counters 先查询支持/当前启用集，再按运行版CLI开启需要的 rdma_tx/rx_packets/bytes、CE/CNP；确认绑定时也启用可选counter。不同版本手动bind可选语法可能不同，不提供未经验证的通用命令。只开端口optional而未绑定QP，不可称per-QP。[S8][S9]
4. 设计两个QP的对照负载：仅A有流量→仅B有流量→二者并发；观察单QP计数隔离、sum与port范围差异、reset与销毁情况。失败就保留软件request_bytes，硬件字段Not collected/Collection failed。
5. 对比未采集、Profiler、Profiler+QP的多轮基线；检查未改变算法/协议/吞吐路径、验证错误数、P50/P99和采集丢失率。只在实机达到目标后承诺开销。

### 9.4 PoC C：Port 与负例

采集空闲、运行前、周期、运行后 ethtool/RDMA计数快照。检查PFC优先级、CE/CNP来源、共享流量标记。做单机本地路径、两组MOD2、两个并行作业共享端口，以及clock未同步/事件丢失/QP销毁等负例；保证UI能准确显示N/A、Partial和未配对，不显示虚假0。

## 10. 数据关联、存储与实施阶段

### 10.1 建议数据契约

每个指标存 value、unit、scope、source_kind、raw_counter_or_event、source_version、collection_status、validity_reason、window_start/end、clock_domain、uncertainty_ns、sampling_ratio。每个关联存 association_kind=exact/inferred/unmatched、evidence_ids、confidence_reason。模板和截图回放必须带 sample=true。[L1]

键设计：

```text
run_key       = run_id
process_key   = run_id + host_boot_id + pid + process_start
comm_key      = run_id + communicator_identity + communicator_generation
rank_key      = comm_key + group_local_rank  (另存 mpi_world_rank/group_color)
event_key     = process_key + monotonic_event_sequence
step_key      = rank_key + collective_seq + channel_id + proxy_op_id + proxy_step
qp_key        = host_boot_id + HCA_identity + port + qpn + create_generation
counter_key   = host_boot_id + HCA_identity + port + counter_id + binding_generation
connection_key= explicit_connection_id + connection_generation
```

此处是设计建议；commId应保存原值但仍核验跨rank一致性与生命周期，裸parentObj指针只在进程内即时转换为稳定ID，不能直接做跨进程关联。QP/WR/request复用、重连、虚拟NIC、多rail、聚合接收和分组并发都须保留多对多边表。[S1][S3][S5][S6]

沿用 runs/run-id/{manifest.json,rank-metadata.json,nccl-events.jsonl,network-events.jsonl,qp-counters.jsonl,port-counters.jsonl,nccl-tests.json}；建议补 connections.jsonl、clock-calibration.jsonl。先本地写盘，异步上传；CDF/聚合由服务计算，保留可回溯原始event IDs。高频事件不塞入原生 -J 结果文件。[L1]

### 10.2 实施交付

| 阶段 | 可评审交付 | 完成门槛 |
|---|---|---|
| 0 能力探测 | manifest、版本/ABI/NET_PROFILER/字段能力表、采集状态 | 能解释每一列为何可用/不可用；未提供数据不显示0 |
| 1 Chunks + Timeline | Profiler插件、phase/iteration标记、raw events导入、proxy与GPU分轨、同域CDF | 明确粒度与时间边界；覆盖/丢失/开销达项目目标 |
| 2 Flows + QPairs | 内置IB事件验证、连接扩展、request/WR/QP关系、软件计数与可用硬件单QP计数 | 双QP隔离对照、双端关联证据、无聚合伪装per-QP |
| 3 Ports | 作用域完整的counter adapter、快照增量、PFC/CE/CNP/CRC与资源状态 | 命名与真实counter一致、共享端口透明、reset和缺失正确处理 |
| 4 跨层关联 | 经校准的rank连线、flow→QP→port追溯、带证据的瓶颈提示 | 时钟误差、关联完整率和控制实验足以支持结论 |

尚待真实GPU/RDMA环境回答：目标NCCL版本/构建ABI；是否启用NET_PROFILER；实际IB插件事件路径和parent完整率；目标mlx5硬件/固件是否支持单QP optional counters；端口/priority计数语义与背景流量；精确iteration/graph映射；时钟误差和采集开销。任何一个未知项都不能由截图或静态拓扑补成“已测”。

## 11. 可点击证据索引

本报告仅使用官方源码/文档和本地项目资料；网页工具本次未返回可引用内容，官方内容通过HTTPS直接读取。以下固定提交链接可复查；动态官方文档注明访问日期。

- [L1] [已接受的交接与架构](nccl-prototype-handoff-20261110.md)，重点第5、8–14、18、21节。
- [L2] 本地 [profiler_v5.h](../nccl-tests-master/src/profiler_v5.h)、[profiler_v6.h](../nccl-tests-master/src/profiler_v6.h)、[profiler_v7.h](../nccl-tests-master/src/profiler_v7.h)、[nccl_profiler.h](../nccl-tests-master/src/nccl_profiler.h)。
- [L3] 本地 [common.cu](../nccl-tests-master/src/common.cu)，-U覆盖见L1519–1525；本次对src搜索userTag只在v7头文件命中。
- [L4] 本地 [util.cu](../nccl-tests-master/src/util.cu)，原生JSON timing输出见L680起。
- [S0] [NCCL固定提交版本文件](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/makefiles/version.mk)。
- [S1] [NVIDIA Profiler插件官方README](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/plugins/profiler/README.md)，NetPlugin parent规则见L334–340；还包含事件、状态、时序和插件加载说明。
- [S2] [NCCL Net IB profiler v1描述符](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/include/plugin/profiler/net_ib_v1.h)。
- [S3] [NCCL IB P2P实际埋点](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net_ib/p2p.cc#L214)，发送L214–231，接收L517–538，完成L848/L919，清理L699/L710。
- [S4] [NCCL NET_PROFILER编译选项](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/makefiles/common.mk#L24)，开关转换见L226–228。
- [S5] [NCCL IB连接与远端QPN](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net_ib/connect.cc#L486)，另见L845/L1339/L1417。
- [S6] [NCCL Network插件v11 ABI](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/include/plugin/net/net_v11.h)，getProperties、isend/irecv、test、iflush、irecvConsumed、vProps。本文检查v11，不声称它是唯一或最新ABI。
- [S7] rdma-core官方 [ibv_poll_cq](https://github.com/linux-rdma/rdma-core/blob/d04695bbd52ae49f7fbe0cdbfc8abd0328c908d3/libibverbs/man/ibv_poll_cq.3)、[ibv_query_qp](https://github.com/linux-rdma/rdma-core/blob/d04695bbd52ae49f7fbe0cdbfc8abd0328c908d3/libibverbs/man/ibv_query_qp.3)、[ibv_post_send](https://github.com/linux-rdma/rdma-core/blob/d04695bbd52ae49f7fbe0cdbfc8abd0328c908d3/libibverbs/man/ibv_post_send.3)。
- [S8] [iproute2 rdma-statistic官方手册](https://github.com/iproute2/iproute2/blob/1d64d3416d60bc8161dff3aa6c3befd9edbb2b62/man/man8/rdma-statistic.8)，auto分类、手动bind、optional-counters、示例。
- [S9] [Linux mlx5 RDMA counters实现](https://github.com/torvalds/linux/blob/067c5c24cef23d913b67eaaf71ae67e7169a6620/drivers/infiniband/hw/mlx5/counters.c#L38)，可选packet/byte L155–160，per-QP读取L459起，绑定L623起。
- [S10] [Linux mlx5 RDMA flow counters实现](https://github.com/torvalds/linux/blob/067c5c24cef23d913b67eaaf71ae67e7169a6620/drivers/infiniband/hw/mlx5/fs.c#L1050)，QP匹配及opcounter规则。
- [S11] [Linux官方 mlx5 ethtool counters说明](https://docs.kernel.org/networking/device_drivers/ethernet/mellanox/mlx5/counters.html)，访问2026-11-11；重点vport/physical/priority三类表。
- [S12] [iproute2 rdma-resource官方手册](https://github.com/iproute2/iproute2/blob/1d64d3416d60bc8161dff3aa6c3befd9edbb2b62/man/man8/rdma-resource.8)。
- [S13] [NCCL调用配置userProfilerTag](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/nccl.h.in#L211)，默认0；应用tag需保留最高位为0；字段存在不保证所有既有调用路径已设置。
- [S14] [NCCL网络proxy实际transfer大小与状态](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net.cc#L1448)，接收大小L1660、RecvGPUWait L1735。
- [S15] [NCCL Profiler v8时间戳与校准契约](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/include/plugin/profiler/profiler_v8.h#L13)，CPU/GPU边界、uncertainty与init配置。
- [S16] [Linux PTP硬件时钟基础设施官方文档](https://github.com/torvalds/linux/blob/067c5c24cef23d913b67eaaf71ae67e7169a6620/Documentation/driver-api/ptp.rst)。

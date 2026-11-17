# NCCL 核心架构学习笔记：从一次 AllReduce 到 GPU 与网络

研究日期：2026-11-16。本文静态阅读 NVIDIA 官方源码，不宣称已在 GPU/RDMA 环境执行验证。

## 1. 版本基线与阅读边界

固定源码基线为 NVIDIA/nccl 提交 [`ad2d192c1f279aefcaf340f67e07880b65a1116b`](https://github.com/NVIDIA/nccl/tree/ad2d192c1f279aefcaf340f67e07880b65a1116b)。该提交的 [makefiles/version.mk](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/makefiles/version.mk) 标识 2.32.6；这只是被检查源码的版本字段，不表示当前机器安装了此版本，也不表示它是正式发行版。源码通过 GitHub raw/API 在线读取，临时副本不属于当前工程的完整 NCCL checkout。

本提交的模块结构与许多旧版 NCCL 博客不同：入队模块位于 `src/enqueue/enqueue.cc` 及 task_prep/task_sched 子目录，tuning 位于 `src/tuning`。不能照搬旧版 `src/enqueue.cc`、`src/graph/tuning.cc` 的行号和函数链。本提交还包含 legacy/general/symmetric/CE 等不同执行路径，本文先抓共同主线，并用可直接读懂的 legacy Ring 算法展示数据推进；不把它说成所有调用实际走的唯一实现。

## 2. 应先分开的五件事

| 层次 | 回答的问题 | 示例 |
|---|---|---|
| Collective 语义 | 用户希望得到什么结果？ | AllReduce：所有 rank 参与归约，每个 rank 得到完整结果 |
| Algorithm | 各 rank 以什么通信结构完成结果？ | Ring、Tree、NVLS、CollNet 等，实际支持受 collective/硬件约束 |
| Protocol | 工作如何切片、缓冲和同步？ | SIMPLE、LL、LL128；不是 TCP/RDMA 协议的同义词 |
| Execution | 用哪些 GPU/CPU 工作单元执行？ | CUDA kernel、CTA/channel、Proxy progress |
| Transport | 某个 peer 连接实际怎么访问/搬运数据？ | P2P、SHM、NET、CollNet；NET 可再落到 IB/RoCE 或 socket 后端 |

算法和协议名称的枚举转换可查 [collectives.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/collectives.cc#L90)；传输列表及选择可查 [transport.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport.cc#L15)。例如“Ring + SIMPLE + NET”是在不同层次作出的选择，不是三种相互替代的通信方式。

## 3. 初始化阶段：communicator 不只是 rank 编号

用户创建 communicator 时传入参与 rank 数、当前 rank 与共享 unique ID。源码中的 `ncclCommInitRank` 进入 `ncclCommInitRankDev`，再安排初始化工作。初始化过程需要 bootstrap 交换参与者信息、建立本地与跨节点的认识、识别系统拓扑、计算候选通信图并准备连接。[init.cc：公开入口](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/init.cc#L2967)

在这个固定提交中，可按以下函数认识初始化职责，而不是一开始阅读整个数千行文件：

1. `bootstrapInit`、`bootstrapAllGather`：交换各 rank 所需的控制信息。bootstrap 控制通信与实际大数据传输不是同一个概念。
2. `ncclTopoGetSystem`、`ncclTopoComputePaths`：识别拓扑并计算路径。
3. `ncclTopoCompute`：为 Ring、Tree 及可用的其他算法准备图。
4. `ncclTopoPreset`、`ncclTopoPostset`：将各节点/各 rank 信息组织为最终通道结构。
5. transport setup/connect：为需要的 peer 连接准备资源；本提交也有 runtime connection 路径，因此不能说所有数据连接都必然在初始化时一次性完成。

这些调用可直接见 [init.cc 的拓扑准备区段](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/init.cc#L1446) 和 [bootstrap 初始化区段](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/init.cc#L2221)。

communicator 因而是后续每次 collective 使用的完整运行上下文，包括 rank 与设备信息、拓扑、channel/peer 连接、调优上下文、任务规划与错误状态。MPI 常被 nccl-tests 用来启动进程和交换 NCCL unique ID，但不能据此把每次 NCCL 数据传输解释为 MPI 负责搬运。

## 4. API 阶段：调用不是“立即发一个网络消息”

`ncclAllReduce` 接受 send/recv buffer、count、datatype、reduction op、communicator 和 CUDA stream。源码先在 `ncclAllReduceConfigImpl` 中构造 `ncclInfo`，随后调用 `ncclEnqueueCheck`。[collectives.cc：AllReduce](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/collectives.cc#L192)

`ncclEnqueueCheck` 做 communicator 状态检查、`ncclCommEnsureReady`、参数检查，调用 `taskAppend` 将本次请求加入任务体系，然后通过内部 group end 处理后续工作。若外面有显式 group，内层结束不会等价于最外层组提交；因此多个 NCCL 调用可以先聚合再准备和启动。[enqueue.cc：入口](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/enqueue/enqueue.cc#L3570)

这里应分清三个时刻：API 在主机上返回、CUDA stream 上的通信工作开始、GPU/通信工作真正完成。API 返回一般不能被当成全部 GPU 数据已经可消费；应用应依据 stream 顺序和同步要求判断完成。[官方 CUDA Stream 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)

## 5. 准备与调度：把逻辑任务转为可执行工作

固定提交中的关键入口是 `ncclPrepareTasks`、`ncclTasksRegAndEnqueue` 和 group 模块的 launch 流程：准备任务、根据实际需要建立连接、注册资源、安排工作计划，再启动对应执行路径。[enqueue.cc：任务准备](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/enqueue/enqueue.cc#L432)；[group.cc：准备与入队](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/group.cc#L825)

学习时可将 `ncclTaskColl` 理解为还在主机端规划的 collective 任务；`ncclKernelPlan` 是一次启动所需的计划容器。任务与计划可能聚合，不能预设“一次 API 调用 = 一个 CUDA kernel = 一个 ProxyOp = 一个 RDMA WR”。源码中 group launch 还区分 CE、RMA 与普通 kernel 分支。[group.cc：launch](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/group.cc#L431)

### 算法和协议如何选择

`ncclGetAlgoInfo` 根据 datatype、collective、rank 数、逻辑数据量、支持的 CollNet/NVLS 能力、资源与用户约束等构造 tuning input，再调用 `ncclTuningCompute`。结果包含 algorithm、protocol、warps、channel 限制及估计时间。环境变量或 tuner plugin 也可能约束/改变选择。[enqueue.cc：ncclGetAlgoInfo](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/enqueue/enqueue.cc#L2220)；[tuning.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/tuning/tuning.cc)

所以“消息小一定 Tree、消息大一定 Ring”只能作为粗略直觉，不能成为源码事实。具体选择取决于版本、拓扑、操作、硬件支持、注册状态、配置和调优结果；报告应记录实际 algo/proto，而不是由 size 猜测。

## 6. GPU 执行：用 Ring AllReduce 读懂数据推进

最适合第一遍学习的源码是 legacy [device/all_reduce.h 的 runRing](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/device/all_reduce.h#L14)。它取得当前 channel 的工作区间、chunkCount 和 ring prev/next，再在循环中执行：

1. `directSend`：先发送一块本地输入。
2. `directRecvReduceDirectSend`：接收上一 rank 的部分结果，与本地输入归约，再继续发送。
3. `directRecvReduceCopyDirectSend`：完成该块最后一次归约，把最终结果写入输出并继续传递；这处把 ReduceScatter 的末步与 AllGather 的首步融合。
4. `directRecvCopyDirectSend`：接收已经完成归约的结果块，复制到输出并向下一 rank 传播。
5. `directRecv`：接收最后一个结果块。

Ring 的每一轮通常可理解为 ReduceScatter 的 N−1 个通信阶段，加上 AllGather 的 N−1 个阶段；代码中可以融合中间步骤，而且大数组会分成多个 chunk/loop，多个 channel 也会并行推进。因此不能拿 `2(N−1)` 直接当整次调用观测到的 ProxyStep 数或网络 WR 数。

4 个 rank 的教学例子：每个 rank 有 A/B/C/D 四个片段。ReduceScatter 结束后，每个 rank 持有某一个片段的四方归约结果；AllGather 再传播这些最终片段，每个 rank 得到完整 A/B/C/D。每个片段的归属顺序由 ring 排列和索引决定，不必强记 rank0 永远负责 A。

本提交还有 [general/all_reduce.cuh](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/device/general/all_reduce.cuh#L14)，可看到新的 FlowProcessor/GIN/LSA 执行机制。它仍能按同样的“发送—归约发送—归约复制发送—复制传播—接收”理解 Ring 数据推进，但不同后端的具体同步和进度责任不同。不能把 legacy primitives 的所有实现细节直接套到新路径。

### Channel、Chunk、Slice、Step 的作用

- Channel 是并行通信工作组织单元，绑定相应逻辑图/peer 工作与设备执行资源；不是 NIC 队列/QP 的同义词。
- Chunk 是在某种算法/协议下分配或循环处理的数据分片；tail 会调整长度，不能假设每块都与配置的 chunkSize 一样大。
- Slice 是更细的流水线处理单位。协议会影响 chunkSteps/sliceSteps 以及缓冲/同步方式。
- Step 是内部缓冲和进度推进的计数单位；其定义属于具体实现层。算法阶段、ProxyStep、NET request 与 packet 不天然一一对应。

主机端计算 stepSize、chunkSteps、sliceSteps、chunkSize 的代码见 [enqueue.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/enqueue/enqueue.cc#L2334)，GPU primitives 的缓冲/同步实现见 [prims_simple.h](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/device/prims_simple.h)。SIMPLE/LL/LL128 不能理解成固定 MTU 或 Ethernet packet 格式。

## 7. CPU Proxy 与 NET：谁推进跨节点传输

在需要 Proxy 的 NET 路径中，主机计划会保存 Proxy 工作并启动 Proxy 进度。`ncclProxySaveOp`、`ncclProxyStart` 与 `ncclProxyProgress` 分别帮助理解工作提交、唤醒和推进。[enqueue.cc：Proxy 提交](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/enqueue/enqueue.cc#L1579)；[proxy.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/proxy.cc#L801)

Proxy progress 遍历活动工作并调用连接所属 transport 的 `proxyProgress`。对普通 NET 路径，关键是 `sendProxyProgress` 和 `recvProxyProgress`：

- 发送端跟踪 posted/transmitted/done 等进度，等待待发送数据可用，然后调用 `ncclNet->isend`；随后以 `test` 查询请求完成并更新缓冲区/步骤进度。
- 接收端通过 `irecv` 投递接收，使用 `test` 获取完成与实际长度；根据路径需要执行 `iflush` 或可见性处理，再通知 GPU/缓冲进度。

源码可见 [transport/net.cc：sendProxyProgress](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net.cc#L1327) 与 [recvProxyProgress](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net.cc#L1496)。

重要含义：CPU Proxy 通常负责网络请求控制/进度，不代表所有 GPU payload 都经 CPU memcpy；GPUDirect 路径可让 NIC 直接访问已注册的 GPU buffer。反过来，也不能因为存在 GDR 就断言没有 CPU Proxy。新 GPU 发起通信后端、特殊算法和局部传输又可能有不同路径，本节明确限定到已引用 NET Proxy 实现。

还需分清：NET request 完成、接收数据对 GPU 可见、GPU 已消费、collective 全部结束。这些不是同一个边界，`test` 完成时间不能直接命名为“远端 GPU 业务计算结束”。

## 8. 与当前 Proxy Profiler 方案的连接

Profiler Plugin 是观察插件；Network Plugin 是传输接口扩展；NCCL Proxy 是内部进度工作。这三者都叫 plugin/proxy 附近的名字，但职责不同。

| 观察对象 | 能直接帮助解释什么 | 不能仅靠它解释什么 |
|---|---|---|
| Collective API/执行事件 | 调用与 NCCL 执行上下文、实际算法/协议（字段可用时） | 某端口实际 wire bytes |
| ProxyOp | 某 rank/channel/peer 的方向和代理工作参数 | 一次 API 对应多少 Ethernet packets |
| ProxyStep/状态 | 内部代理步骤、等待/进度边界 | 跨节点双端一一匹配、真实 WR/QP 映射 |
| Kernel Channel | GPU channel 活动与 timer | 描述符未提供的独立 peer/QP |
| NetPlugin/WR/QP | 进一步补充网络请求、长度、连接身份 | 无额外硬件数据时的精确端口拥塞计数 |

原始 buffer size、chunkSize、logical size、transport length、wire bytes 应分列保存。此处引用的 NET 发送路径中存在实际 `size`，接收 `test` 返回实际长度，说明传输长度应从那个观测层采集，而不是用 chunkSize 代替。[net.cc：isend 与实际 size](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/transport/net.cc#L1440)

## 9. 最有效的源码阅读顺序

1. 读 `collectives.cc` 的 `ncclAllReduce`：确认 API 只描述逻辑请求。
2. 读 `ncclEnqueueCheck`：看检查、taskAppend、group 关系，不先深入所有特殊分支。
3. 读 group 的 prepare/reg/enqueue/launch 主线：理解任务何时变为执行计划。
4. 读 `ncclGetAlgoInfo` 与 tuning input/result：理解算法协议是选择结果，而非固定标签。
5. 读 legacy `runRing` 的五类 primitives：用 4 rank、4 分片画出 ReduceScatter/AllGather。
6. 读 `sendProxyProgress`/`recvProxyProgress`：把 GPU buffer 进度与异步 NET 请求连接起来。
7. 最后回到 init/topology、primitives 同步、特殊执行后端：此时各结构为何存在会更清楚。

本文没有进行动态调用栈、CUDA trace、RDMA 请求跟踪或开销实测。固定提交源码存在某能力不表示本地 nccl-tests 编译版本或实际设备启用了它；下一步实机学习应锁定 NCCL 构建版本，以日志/Profiler 验证实际 algo/proto/transport，再对照相应分支。

# nccl-tests：算子适配层、指标与正确性验证

研究日期：2026-10-26。范围：本地 `E:/dpi/nccl-tests-master`，静态阅读；本笔记不声称该目录为最新上游版本，也未执行 GPU 测试。源码位置均相对该目录，供主研究文档整合。

## 1. 结构与职责

nccl-tests 公共框架并不为每种 collective 复制一套测试驱动，而是通过两层 C 风格函数指针接口接入不同算子（`src/common.h:115–148`）：

- `testEngine` 是测试程序级接口：`getBuffSize` 决定缓冲区需求，`runTest` 选择数据类型/归约操作/root 等测试组合；可选 `initCommConfig` 定制 communicator 配置；按 NCCL 编译版本可提供 `getDevCommRequirements` 声明 device API 资源需求。
- `testColl` 是单种通信操作的接口：`getCollByteCount` 把统一测试尺寸换算为 sendcount、recvcount、API paramcount 和 in-place 偏移；`initData` 构造输入与预期输出；`runColl` 发起真正通信；`getBw` 按算子语义计算带宽。
- `common.cu` 负责统一调度、计时、调用上述回调及检查数据；例如 `getCollByteCount` 位于 933 行，`getBw` 位于 824 行，检查流程位于 828–887 行。
- `verifiable/` 是可复现测试数据生成与结果验证模块，不是另一个通信后端；其 API 定义位于 `verifiable/verifiable.h:8–57`。

以 AllReduce 为例：`ncclTestEngine.runTest → AllReduceRunTest → args->collTest=&allReduceTest → TimeTest → 公共执行框架 → AllReduceRunColl → ncclAllReduce`。AllReduceRunTest 遍历指定/全部 type 与 op（`src/all_reduce.cu:532–589`），默认实现 `deviceImpl=0` 直接调用 NCCL（500–508 行）。

## 2. 测试算子覆盖

本地 Makefile 的常规程序列表包括 11 项：all_reduce、all_gather、broadcast、reduce_scatter、reduce、alltoall、alltoallv、scatter、gather、sendrecv、hypercube（`src/Makefile:44`）。以下是功能语义：

| 测试 | 做什么 | 源码依据 |
|---|---|---|
| AllReduce | 所有 rank 的数组逐元素归约，完整结果送达所有 rank | `src/all_reduce.cu:42–64,500–537` |
| ReduceScatter | 逐元素归约后，每个 rank 获得完整结果的一段 | `src/reduce_scatter.cu:10–50` |
| AllGather | 每个 rank 提供一段数据，每个 rank 收到全部段拼接 | `src/all_gather.cu:10–44,102–114` |
| Broadcast | root 的同一份数组分发到所有 rank | `src/broadcast.cu:18–39,74–96` |
| Reduce | 归约结果只在 root 有意义 | `src/reduce.cu:18–49` |
| AlltoAll | 每个 rank 给每个 peer 各发一个等长数据块 | `src/alltoall.cu:17–52,468–489` |
| AlltoAllv | peer 间可不同长度；支持自动生成分布或显式流量矩阵 | `src/alltoallv.cu:20–81` |
| Gather | 所有 rank 的各自片段聚集到 root | `src/gather.cu:89–115` |
| Scatter | root 的完整数组切片分发给各 rank | `src/scatter.cu:86–114` |
| SendRecv | 每 rank 发给 rank+1、从 rank-1 接收，构成环 | `src/sendrecv.cu:94–108` |
| Hypercube | 用异或配对、倍增块大小的 Send/Recv 序列实现 AllGather 模式 | `src/hypercube.cu:48–68` |

不能将这些文件一律解释为“薄封装 NCCL 单个函数”。例如 AlltoAll 可以回退为成组 Send/Recv（`src/alltoall.cu:468–489`）；本地 AllReduce 另含基于 device API 的 LSA/Multimem CUDA kernel（`src/all_reduce.cu:7–21,500–529`）；AllGather 包含按 NCCL 版本启用的 host RMA 路径（`src/all_gather.cu:46–114`）。高级分支必须随实际 NCCL 编译版本和硬件能力核实。

## 3. 指标原理：先统一消息量定义

设 N 为 communicator 的 rank 数，w 为数据类型字节数，t 为每次操作耗时（秒），C 为传给 `getBw` 的 count。带宽单位使用十进制 GB/s，即除以 1e9。下面是本地实现公式：

| 算子 | algbw | busbw/algbw | 依据 |
|---|---|---|---|
| AllReduce | Cw/(1e9·t) | 2(N−1)/N | `src/all_reduce.cu:59–64` |
| AllGather | CNw/(1e9·t) | (N−1)/N | `src/all_gather.cu:38–43` |
| ReduceScatter | CNw/(1e9·t) | (N−1)/N | `src/reduce_scatter.cu:37–42` |
| Broadcast | Cw/(1e9·t) | 1 | `src/broadcast.cu:34–39` |
| Reduce | Cw/(1e9·t) | 1 | `src/reduce.cu:36–39` |
| AlltoAll | CNw/(1e9·t) | (N−1)/N | `src/alltoall.cu:47–52` |
| Gather/Scatter | CNw/(1e9·t) | (N−1)/N | `src/gather.cu:38–43`; `src/scatter.cu:34–39` |
| SendRecv | Cw/(1e9·t) | 1 | `src/sendrecv.cu:38–43` |
| Hypercube | C(N−1)w/(1e9·t) | 1 | `src/hypercube.cu:40–45` |

AlltoAllv 不能套用等长 AlltoAll 的简单解释：源码按所有 rank 的最大 send/recv 字节计算带宽，并建议不均衡负载使用 `-a 3` 最大时间。流量生成、矩阵模式和逐 rank 汇总详见 [AlltoAllv 深入笔记](nccl-adapter-deep-dive.md)。

`count` 并非所有算子都是“完整数组元素数”。例如 AllGather 输入 C、输出 CN；ReduceScatter 输入 CN、输出 C。其统一测试尺寸还会除以 N 并对齐，见 `src/all_gather.cu:10–17` 和 `src/reduce_scatter.cu:10–17`。

`busbw` 是在 algbw 上施加通信量归一化系数得到的推导指标，不是交换机端口计数器或某张 NIC 的直接吞吐读数（证据：各 GetBw 实现；数学说明见 `doc/PERFORMANCE.md` 的 Bus bandwidth 和 Summary）。官方在线同名文档也解释该系数用于 collective 的硬件带宽比较；本笔记公式以本地代码为准。

另有一个容易误读之处：本地 `doc/PERFORMANCE.md` 开头写 time 单位 ms，但实际 `common.cu:886` 将秒乘 1e6，输出变量名为 timeUsec。报告解析应以实际输出表头和实现为准，而不是沿用该文档的 ms 说法。

## 4. 正确性验证调用链

以 AllReduce 为例（`src/all_reduce.cu:42–55`）：

1. 为每个 GPU/rank 建立输入，in-place 时使用接收缓冲区作为输入。
2. `InitData → ncclVerifiablePrepareInput`：根据 type、op、seed、rank 数、当前 rank 生成数据（`src/common.cu:288–290`）。
3. `InitDataReduce → ncclVerifiablePrepareExpected`：独立生成预期归约结果（`src/common.cu:283–285`）。
4. 性能计时结束后，按 datacheck 次数重新初始化，执行一个检查用通信并等待完成（`src/common.cu:824–876`）。因此不能说性能计时的每一轮都被逐元素验证。
5. `CheckData → CheckDelta → ncclVerifiableVerify`：比较实际输出与 expected，汇总本线程各 GPU 的不匹配元素数，再跨线程/进程求和（`src/common.cu:277–280,449–492,878–887`）。

`verifiable.h:8–10` 明确说明数据经过构造，使浮点归约具有可预测输出。这不是用随机真实训练数据运行一个完整 CPU reference model。验证代码根据类型和 op 设置容差；默认 tolerance=0，浮点 Avg 允许 tolerance=2，比较实现以相应位宽整数表示的差为基础（`verifiable/verifiable.cu:1188–1228,1306–1339`）。因此不应概括成“任意浮点误差都可接受”或“一律逐位完全相同”。

## 5. 容易形成错误理解的边界

- SendRecv 是环形 peer 配对，不能直接描述成任意两卡点对点拓扑测试（`src/sendrecv.cu:100–108`）。
- in-place 列存在不等于所有算子支持正确的原地语义。本地 AlltoAll 和 SendRecv 显式标注不支持，且设置 `reportErrors=0`（`src/alltoall.cu:42–43`; `src/sendrecv.cu:33–34`）。
- `#wrong=0` 只对启用的检查、对应测试数据和执行条件有意义；关闭检查或禁用报告不能据此宣称正确性得到验证（`src/common.cu:831–887`）。
- 相同 `size` 字段在不同算子中需先解释 send/recv/API count，再比较带宽，不能假定发送、接收缓冲区总量完全相同（`src/common.h:117–120`）。
- 通信算子 API 与底层 ring/tree/传输路径是不同层次；默认 AllReduce 适配器在此只调用 `ncclAllReduce`，算法选择须继续追踪 NCCL 本体（`src/all_reduce.cu:505–508`）。
- Hypercube 实现用 `rank ^ mask` 作为对端，并未在上述循环做非 2 次幂 rank 修正；适用 rank 条件需进一步验证，不能把它等同通用 NCCL AllGather（`src/hypercube.cu:61–67`）。

## 6. 后续深入顺序

已完成默认 AllReduce 主链、AllGather/ReduceScatter 数据布局、验证机制、各 adapter 的基本语义，以及计时/JSON 报告结构；详见本目录关联笔记。尚需整理多机 rank 映射与参数覆盖矩阵，并在可用 GPU 环境实测。Device API/RMA 和 NCCL 内部算法/传输仍作为独立深入层次，不能仅凭 nccl-tests adapter 推导其具体算法选择。


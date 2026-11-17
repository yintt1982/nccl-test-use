# Keysight 截图 NCCL Config 与 NVIDIA nccl-tests 对照

核对日期：2026-10-27。

## 结论及证据边界

截图24项全部可在本次下载的 NVIDIA 官方 NCCL 环境变量文档（页面标题 NCCL 2.32.3）中找到，均不是 Keysight 专有变量。NVIDIA nccl-tests 是 NCCL 的测试程序，链接 NCCL 库；这些配置通过运行环境交给 NCCL 库处理，不是 nccl-tests 的24个命令行选项，也不等同于 ncclConfig_t 的字段。

不能笼统承诺24项在任意版本、硬件上全部有效：23项是当前文档仍描述的常规配置，1项 NCCL_CHECKS_DISABLE 自2.2.12已弃用。当前源码是否继续读取这一旧参数尚未完成在线确认，不应宣称设置后有效，也不应仅凭 deprecated 断言已删除。实际支持范围取决于运行时加载的 NCCL 版本、网络后端及硬件条件。

本次 NVIDIA 官方在线文档抓取成功，快照为 ../docs/nccl-env-upstream.html。GitHub 页面/API/raw请求超时，未能对最新master做实时核验。架构判断同时参考本地 nccl-tests-master/src/Makefile（NCCL_HOME include/lib，LIBRARIES += nccl）及 src/common.cu（ncclCommInitRank/Config、ncclGetVersion）。本地目录有扩展代码，不能当作本次在线核对的纯净最新上游。已有 nccl-source-verification.json 记录曾对上游 afd59abab774a6a4b492eb344bd0c845a0b7bb4d（2026-10-05）核对79个文件，但不代表今天最新master。

## 逐项核对

“支持”指官方NCCL文档提供此环境变量，可由采用相应NCCL版本的nccl-tests使用；不代表本机已做GPU实测。

| 参数 | 截图值 | 判断及注意事项 |
|---|---|---|
| NCCL_P2P_DISABLE | 1 | 支持，关闭GPU间直接P2P传输。 |
| NCCL_P2P_LEVEL | 0 | 支持，旧整数0对应LOC；建议使用LOC。已关闭P2P时此项冗余。变量始于2.3.4。 |
| NCCL_SHM_DISABLE | 1 | 支持，关闭共享内存传输。 |
| NCCL_DEBUG | WARN | 支持，WARN主要输出错误警告，不是详细初始化日志。 |
| NCCL_DEBUG_FILE | /tmp/nccl-debug.log | 支持，始于2.2.12；多进程建议使用/tmp/nccl-debug.%h.%p.log，防止覆盖。 |
| NCCL_DEBUG_SUBSYS | INIT | 支持，始于2.3.4；需INFO级别才能查看所需初始化信息。 |
| NCCL_TOPO_DUMP_FILE | /tmp/nccl-topo.xml | 支持，始于2.6；输出NCCL探测到的拓扑，各节点路径须可写。 |
| NCCL_BUFFSIZE | 4194304 | 支持，单位字节，为4 MiB通信缓冲区，不是测试消息大小。 |
| NCCL_CROSS_NIC | 0 | 支持，控制ring/tree跨NIC选择，单NIC场景无效果。 |
| NCCL_CHECKS_DISABLE | 0 | 已弃用：始于2.0.5，2.2.12起deprecated；不作为现代NCCL有效调优项保证。 |
| NCCL_NET_PLUGIN | none | 支持，始于2.11，none为合法值；使用内置网络实现。 |
| NCCL_NET | IB | 支持，始于2.10；Linux内置IB指IB verbs网络实现，可服务IB/RoCE，并非仅限原生InfiniBand。 |
| NCCL_IB_DISABLE | 0 | 支持，允许IB/RoCE后端；0不保证后端一定可用。 |
| NCCL_IB_GID_INDEX | 3 | 支持，RoCE GID索引；3是否对应RoCEv2必须看实际设备GID表，不是固定含义。 |
| NCCL_IB_HCA | mlx5 | 支持，匹配名称以mlx5开头的所有HCA及其端口，不是特指一块卡。 |
| NCCL_NET_GDR_LEVEL | PHB | 支持，限制GPU Direct RDMA使用的拓扑距离；还须硬件和驱动支持。旧名称NCCL_IB_GDR_LEVEL于2.4.0改为本名。 |
| NCCL_NET_GDR_READ | 1 | 支持，允许发送时从GPU内存直接RDMA读取；受GDR能力及拓扑条件限制。 |
| NCCL_IB_TIMEOUT | 18 | 支持，超时约4.096微秒乘2的18次方，即1.074秒，不是18秒。当前文档默认20，截图不是所有版本默认值。 |
| NCCL_IB_RETRY_CNT | 7 | 支持，IB重试次数。 |
| NCCL_IB_SL | 0 | 支持，InfiniBand Service Level。 |
| NCCL_IB_TC | 0 | 支持，traffic class；实际QoS效果还取决于网络配置。 |
| NCCL_IB_SPLIT_DATA_ON_QPS | 1 | 支持，始于2.18；多个QP时拆分消息。需结合NCCL_IB_QPS_PER_CONNECTION；单QP无法体现多QP分流。 |
| NCCL_ALGO | Ring | 支持，始于2.5，限定集合通信算法。 |
| NCCL_PROTO | Simple | 支持，始于2.5，限定NCCL通信协议。 |

## 使用与验证

不需要为上述普通环境变量逐项修改nccl-tests源码。启动测试前设置环境变量即可；多机MPI运行必须确保每个rank均收到变量，每个节点均加载预期版本NCCL。不要只核对编译时NCCL_HOME，而忽略运行时动态库搜索路径。

示意命令（Linux；实际HCA/GID须按机器确定）：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
NCCL_NET_PLUGIN=none \
NCCL_NET=IB \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
```

该示意命令不是截图的完整复现，也不保证单机两GPU一定走网络。需要复现截图时应逐项注入已核实参数并检查日志中的网络选择、HCA和GDR状态。NCCL_CHECKS_DISABLE 与 nccl-tests 的 -c 结果正确性检查不是同一个开关。NCCL_CHECK_POINTERS 是另一个现代检查变量，但不是旧变量同名替代或简单取反。

截图并非NCCL环境变量全集；官方还包括NCCL_SOCKET_IFNAME、NCCL_SOCKET_FAMILY、NCCL_IB_QPS_PER_CONNECTION、NCCL_IB_ROCE_VERSION等。前端可提供常用模板加自定义键值输入；版本校验、MPI环境传递、运行时NCCL版本记录比硬编码24项白名单更有价值。

## 官方来源

- NVIDIA NCCL环境变量文档（本次成功获取）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- nccl-tests仓库：https://github.com/NVIDIA/nccl-tests
- nccl-tests构建文件：https://github.com/NVIDIA/nccl-tests/blob/master/src/Makefile
- nccl-tests初始化源码：https://github.com/NVIDIA/nccl-tests/blob/master/src/common.cu
- CHECKS_DISABLE：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-checks-disable
- QP拆分：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-split-data-on-qps

GitHub链接是供复核的官方位置，不代表本次已成功在线读取。

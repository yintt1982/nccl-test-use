# nccl-tests 运行复现与诊断配置细化

研究日期：2026-10-26  
页面：`http://112.25.236.50:9007/`  
依据：页面当前配置、`E:/dpi/nccl-tests-master/README.md`、`src/common.cu`、`src/util.cu`。以下为源码静态分析，未执行 GPU 实测。

## 1. 为什么当前“MPI 进程数 + 每进程 GPU 数”还不够

页面当前可以表达：

```text
MPI 进程数 × 每进程线程数(-t) × 每线程 GPU 数(-g) = 总 NCCL rank 数
```

但多机测试还需要回答四个问题：

1. MPI 进程分布在哪些节点？
2. 每个 MPI 进程被分配了哪些 GPU？
3. CUDA device id 与 NCCL rank、MPI rank、local rank 如何对应？
4. 是否把 MPI world 拆成了多个 NCCL communicator group？

这些信息会改变通信拓扑和带宽口径，所以应该成为任务配置和报告的一部分。

## 2. 节点 / hostfile 配置

### 2.1 页面应增加的字段

建议在“GPU 与 Rank 布局”下增加“节点与 MPI 启动”区域：

| 字段 | 作用 | 示例 |
|---|---|---|
| 节点列表 | 明确允许 MPI 使用的主机 | `gpu01,gpu02` |
| Hostfile | 使用 MPI 的主机文件及槽位 | `gpu01 slots=4`、`gpu02 slots=4` |
| 总 MPI 进程数 | 对应 `mpirun -np` / `mpiexec -n` | `8` |
| 每节点进程数 | 对应 launcher 的 ppr 或 machinefile 约束 | `4` |
| 每进程线程数 | `-t` | `1` |
| 每线程 GPU 数 | `-g` | `1` |
| 进程绑定 | CPU/NUMA 绑定策略 | `bind-to none`、`slot`、` numa` |
| 额外 MPI 参数 | 不同 MPI 实现的专用参数 | Open MPI、MPICH、Intel MPI 分开保存 |
| nccl-tests binary | 实际运行的路径及构建类型 | `all_reduce_perf_mpi` |

不要把“节点列表”和“hostfile”当成两个必须同时填写的条件。它们是两种启动来源：简单任务可以用节点列表生成 launcher 参数；复杂集群可以直接引用已审计的 hostfile。

### 2.2 命令预览应呈现完整启动命令

单节点时可以是：

```bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

多节点时，页面不能只显示 binary 后面的参数，而应显示完整命令，例如：

```bash
mpirun -np 8 \
  --host gpu01:4,gpu02:4 \
  --map-by ppr:4:node --bind-to none \
  ./build/all_reduce_perf_mpi -b 8 -e 8G -f 2 -g 1
```

`--host`、`--map-by`、`--bind-to` 是 MPI 实现相关的示意参数；实际页面应根据用户选择的 Open MPI、MPICH 或其他 launcher 生成对应语法，不应把某个 MPI 实现的参数硬编码成通用字段。

### 2.3 必须做的校验

提交前至少检查：

- `MPI 进程数 × 每进程 GPU 数` 是否与节点 GPU 总量匹配；
- `-t × -g` 是否超过每个 MPI 进程所在节点可用 GPU 数；
- binary 是否以 `MPI=1` 构建，多进程时不能使用单进程 binary；
- hostfile 中的节点是否可达、槽位是否足够；
- 每个节点的 CUDA/NCCL/MPI 版本和 GPU 型号是否满足一致性要求；
- `CUDA_VISIBLE_DEVICES`、launcher 的 GPU 绑定参数和 `NCCL_TESTS_DEVICE` 是否互相覆盖。

## 3. GPU 映射配置

### 3.1 当前 nccl-tests 的默认映射

源码在每个 MPI 进程中计算 `localRank`，默认给该进程分配连续 GPU：

```text
默认起始 GPU = localRank × -t × -g
本进程第 i 个测试 GPU = 默认起始 GPU + i
```

例如每节点 4 张 GPU、每个进程 1 个线程、每线程 2 张 GPU：

| local rank | 分配的 CUDA device |
|---:|---|
| 0 | 0, 1 |
| 1 | 2, 3 |
|
如果通过 `NCCL_TESTS_DEVICE=4` 设置起点，则进程从 device 4 开始按连续编号分配。这个环境变量是“起始 device”语义，不是任意 GPU 列表映射。

### 3.2 页面应区分的编号

报告里建议同时展示：

| 编号 | 含义 |
|---|---|
| MPI world rank | MPI_COMM_WORLD 中的进程编号 |
| local rank | 当前节点内的 MPI 进程序号 |
| NCCL group/color | split 后所属 communicator 组 |
| NCCL rank | 当前 communicator 内的 rank |
| thread id | nccl-tests 进程内 host 线程编号 |
| CUDA device | 实际 CUDA device id |
| hostname | 节点名 |
| PCI bus id | GPU 的 PCI 地址 |

源码的设备报告已经输出 `Rank`、`Group`、`Pid`、`hostname`、`device`、PCI 地址和 GPU 型号，并写入 JSON 的 `devices` 数组。网管应直接把这些字段作为拓扑表，而不是只画 `R0/G0`。

### 3.3 任意映射怎么表达

nccl-tests 本身的默认逻辑主要是连续 device 分配。要表达非连续映射，例如：

```text
MPI rank 0 -> GPU 0
MPI rank 1 -> GPU 3
MPI rank 2 -> GPU 1
MPI rank 3 -> GPU 2
```

网管需要在 launcher 层使用节点级 `CUDA_VISIBLE_DEVICES`、MPI local-rank 环境变量或包装脚本完成。页面应把这类内容保存为“GPU 映射策略/启动模板”，不要伪装成 `-g` 能完成的功能。

## 4. `NCCL_TESTS_SPLIT` rank 分组

### 4.1 它做什么

`NCCL_TESTS_SPLIT` 不是普通 nccl-tests CLI 参数，而是环境变量。程序先读取 MPI world rank，再根据表达式计算 `color`，然后调用 `MPI_Comm_split`：

```text
MPI world rank -> color/group -> group 内 communicator -> NCCL ranks
```

同一 color 的 MPI 进程进入同一个 communicator。每个 MPI 进程中的 `-t × -g` 个 GPU 会跟随这个进程加入该组。

可用形式包括：

```text
NCCL_TESTS_SPLIT="AND 0x7"   # 或 &0x7
NCCL_TESTS_SPLIT="OR 0x7"    # 或 |0x7
NCCL_TESTS_SPLIT="MOD 8"      # 或 %8
NCCL_TESTS_SPLIT="DIV 8"      # 或 /8
NCCL_TESTS_SPLIT_MASK="7"     # 等价于 AND 0x7
```

### 4.2 8 个 MPI 进程的例子

对于 MPI world rank `0..7`：

- `MOD 2`：分成两个组，偶数 rank 一组、奇数 rank 一组；
- `MOD 8`：形成 8 个组，每组一个 MPI 进程；
- `DIV 4`：rank 0–3 为一组，rank 4–7 为另一组；
- `AND 0x7`：在 8 个进程的范围内，效果相当于按低 3 位形成颜色。

网管要在提交前根据 MPI 进程数计算预览：

```text
world rank | color | group size | group-local rank | assigned GPUs
```

### 4.3 报告口径的变化

split 后，测试的 communicator rank 数不再等于 MPI world size。带宽、计时和正确性都属于各自 group。README 明确说明报告的 bandwidth 是“每组”带宽；如果要估计所有并发组的总吞吐，需要在组规模和测试目的相同的前提下再汇总。

因此报告中至少要有：

- world size 与 group size；
- group/color；
- group-local rank；
- 每组的 algbw/busbw；
- 并行组数量；
- 是否展示“每组值”还是“所有组估算总和”。

不能把 split 后多个组的 busbw 直接平均后标成单个 NCCL communicator 的性能。

## 5. `-M/-S/-u/-U` 诊断选项

### `-M`：内存使用报告

`-M 1` 打开内存报告。源码在测试结束时报告至少三类 GPU 内存：

- `Initialization`：NCCL communicator 初始化消耗；
- `User-Allocated`：测试 send/recv/expected 等用户缓冲区；
- `Collective`：collective 执行期间的额外内存。

适合用于判断：消息规模变大时是否接近显存上限、初始化开销是否异常、不同注册/实现模式是否改变显存占用。它不是 GPU 实时监控曲线，也不能替代 NVML 的功耗、温度和实时显存采样。

UI 建议：增加“内存报告”开关；报告页增加按节点、GPU、阶段的内存表，并把测试时的 maxBytes 自动降级记录下来。源码在内存不足时可能降低最大测试尺寸，报告必须显示实际运行的 `maxBytes`。

### `-S`：报告时间戳

`-S 1` 在性能报告行中增加格式类似 `%Y-%m-%d %H:%M:%S` 的时间戳。它适合：

- 对齐 NCCL 日志、系统日志、交换机/网卡监控；
- 排查某一轮测试是否与系统事件同时发生；
- 在长时间、多轮 `-N` 运行中定位异常轮次。

它是“报告行时间戳”，不是每个 CUDA event 的高精度时间戳，也不等于完整 trace。页面应显示测试时区和任务开始/结束时间，避免只保存一段无时区的字符串。

### `-u`：故意制造 buffer 非对齐

`-u N` 让 send/recv buffer 从第 `N` 个元素开始偏移，源码按数据类型元素大小换算为字节偏移。这是鲁棒性和边界场景测试：

- 检验非理想对齐下是否仍能正确运行；
- 观察对齐变化对性能的影响；
- 暴露自定义 kernel、注册内存或特殊实现对对齐的隐含假设。

它不属于普通性能基线，建议放在“压力/诊断场景”，默认关闭。报告必须把 `unalign` 值作为配置维度，否则不同对齐条件的结果容易被错误地放在同一趋势图中。

### `-U`：NCCL tuning 信息

`-U 1` 请求报告 NCCL 实际选择的调优信息。源码可在结果中增加：

- implementation；
- algorithm；
- protocol；
- kernel variant；
- synchronization strategy；
- channel 数；
- warp 数；
- network chunk size。

它主要用于解释“为什么这次带宽变了”，不是直接指定算法和协议。需要注意版本门槛：NCCL 2.28+ 才支持 tuning 报告，完整的对称 collective 标识和 kernel variant 信息需要更高版本；开启时源码还可能把 `NCCL_PROFILER_PLUGIN` 调整为 `STATIC_PLUGIN`。

UI 建议：把它做成“记录 NCCL 实现选择”开关。报告页提供按消息大小展开的 tuning 列，并支持比较两次任务的 algorithm/protocol/kernel 差异。

## 6. `-R/-D/-V/-H` 为什么要动态显示

这些选项改变的是 NCCL buffer/collective 的底层实现，不是所有测试都适用，也不是所有安装环境都支持。

| 参数 | 作用 | 适用范围/限制 |
|---|---|---|
| `-R 0/1/2` | 不注册、local registration、symmetric registration | NCCL 版本有门槛；`-D` 和 `-H` 都要求 `-R 2` |
| `-D N` | 选择 device API collective 实现 | 当前源码说明并非每个 collective 都有，主要是 AllReduce、AlltoAll 等；要求 NCCL 2.28+ 和 `-R 2` |
| `-V N` | Device API 使用的 CTA 数 | 只在 device implementation 生效；必须为合法正数，并受 binary 校验 |
| `-H N` | 使用 Host RMA，配置 RMA context 数 | 支持 SendRecv、AlltoAll、AllGather、Broadcast、Gather、Scatter；要求 `-R 2`，与 `-D` 互斥；多 context 需要更高 NCCL 版本 |

### 6.1 动态显示的判断条件

页面加载或选择测试时，后端应综合四类能力：

1. **构建能力**：binary 是否 MPI build，是否编译进对应实现；
2. **NCCL 版本**：通过 `ncclGetVersion` 或启动探测获得实际运行库版本；
3. **测试类型**：当前 adapter 是否实现 Device API 或 Host RMA 路径；
4. **硬件与内存能力**：GPU 架构、P2P、对称内存注册等是否满足要求。

不能只根据页面下拉框静态决定“支持/不支持”。例如 AllReduce 可以显示 `-D`，AllGather 可能只显示 `-H`，SendRecv 可能显示 Host RMA；如果目标节点 NCCL 版本过低，则这些控件应隐藏或置灰并说明原因。

### 6.2 依赖和互斥校验

最少实现以下规则：

```text
-D > 0  => -R = 2
-H > 0  => -R = 2
-D > 0  与 -H > 0  互斥
-H > 1  => NCCL >= 2.31（按当前源码要求）
-D/-V  => 当前测试必须声明支持 device implementation
-H     => 当前测试必须声明支持 host RMA
```

命令预览不应只在最后拼接参数，而应显示一条能力说明，例如：

```text
已启用 Device API implementation=1，原因：AllReduce 支持，NCCL 版本满足，symmetric registration 已开启。
```

或者：

```text
-H 不可用：当前选择的 Reduce adapter 没有 Host RMA 实现。
```

## 7. 推荐的页面组织

### 基础配置

保留当前页面中的 collective、数据类型、op、消息范围、`-t/-g`、warmup/iters/check 等常用字段。

### 拓扑与复现

新增：节点列表/hostfile、MPI launcher 参数、每节点进程数、CPU/NUMA 绑定、GPU 映射策略、`CUDA_VISIBLE_DEVICES`、`NCCL_TESTS_DEVICE`、NCCL 环境变量。

### 高级执行与诊断

新增：`-M/-S/-u/-U/-T/-K/-z/-p/-y`，并显示依赖和冲突状态。

### 实现模式

仅在能力探测后显示：`-R/-D/-V/-H`。把“支持原因、版本门槛、互斥条件、实际展开命令”同时呈现出来。

### 报告

保存完整复现包：

```text
任务配置 JSON
完整 launcher 命令
nccl-tests binary 与版本
NCCL/CUDA/MPI 版本
节点、hostname、PID
MPI rank/local rank/NCCL group/rank
CUDA device、PCI bus、GPU 型号
环境变量
诊断开关
每个消息规模的结果和 tuning 信息
```

这样报告才可以回答“结果是多少”和“这个结果是在什么拓扑、什么实现、什么诊断条件下得到的”。

## 关键源码位置

- `src/common.cu`：命令行解析、`-R/-D/-V/-H` 依赖与互斥、split、默认 GPU 分配、版本检查。
- `src/util.cu`：JSON 初始化、设备/rank 报告、`-S` 输出、`-U` tuning 字段、内存报告调用。
- `README.md`：MPI rank 公式、split 语法、参数说明和支持范围。

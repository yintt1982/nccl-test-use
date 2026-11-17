# nccl-tests GPU、MPI、线程与 Rank 逻辑拓扑

> 综合说明：见 [资源拓扑、Rank 编号与追踪采集能力](nccl-rank-topology-and-observability.md)，包含编号对照、分组图、五类采集维度和真实采集验证方案。

本文按仓库中的 nccl-tests 实现说明节点、MPI 进程、进程内线程、每线程 GPU，以及 MPI/NCCL Rank 的关系。图中示例假设每个 GPU 只分配给一个测试参与者，且设备按本机 Rank 顺序连续分配。

## 层次关系

```text
MPI 启动器（mpirun / mpiexec）
└── MPI_COMM_WORLD：所有节点上的 MPI 进程
    ├── 节点 gpu01
    │   ├── MPI 全局 Rank 0（本机 localRank 0）
    │   │   ├── 线程 0（-t）
    │   │   │   ├── GPU（-g）→ NCCL Rank
    │   │   │   └── GPU（-g）→ NCCL Rank
    │   │   └── 线程 1（-t）
    │   │       ├── GPU（-g）→ NCCL Rank
    │   │       └── GPU（-g）→ NCCL Rank
    │   └── MPI 全局 Rank 1（本机 localRank 1）
    │       ├── 线程 0 → 2 GPU → 2 NCCL Rank
    │       └── 线程 1 → 2 GPU → 2 NCCL Rank
    └── 节点 gpu02
        ├── MPI 全局 Rank 2（本机 localRank 0）
        │   ├── 线程 0 → 2 GPU → 2 NCCL Rank
        │   └── 线程 1 → 2 GPU → 2 NCCL Rank
        └── MPI 全局 Rank 3（本机 localRank 1）
            ├── 线程 0 → 2 GPU → 2 NCCL Rank
            └── 线程 1 → 2 GPU → 2 NCCL Rank
```

## 具体示例：一个通信组

启动参数示例：2 个节点，每节点 2 个 MPI 进程；每个进程 `-t 2 -g 2`。不设置 `NCCL_TESTS_SPLIT`，因此所有参与者属于一个 communicator。

```text
                          MPI_COMM_WORLD
                  4 个 MPI 进程：world rank 0..3
                                 │
                  NCCL_TESTS_SPLIT 未设置
                 所有进程进入同一个 MPI 子通信器
                                 │
                    一个 NCCL communicator
                       16 个 NCCL Rank
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
 节点 gpu01                                           节点 gpu02
 ├─ MPI world rank 0                                  ├─ MPI world rank 2
 │  localRank 0                                       │  localRank 0
 │  ├─ thread 0: GPU 0,1 → NCCL Rank 0,1             │  ├─ thread 0: GPU 0,1 → NCCL Rank 8,9
 │  └─ thread 1: GPU 2,3 → NCCL Rank 2,3             │  └─ thread 1: GPU 2,3 → NCCL Rank 10,11
 └─ MPI world rank 1                                  └─ MPI world rank 3
    localRank 1                                          localRank 1
    ├─ thread 0: GPU 4,5 → NCCL Rank 4,5                ├─ thread 0: GPU 4,5 → NCCL Rank 12,13
    └─ thread 1: GPU 6,7 → NCCL Rank 6,7                └─ thread 1: GPU 6,7 → NCCL Rank 14,15
```

这个表中的 NCCL Rank 顺序用于解释拓扑关系；不要将它与 MPI world rank 混用。实际 NCCL communicator 的成员/设备顺序由 nccl-tests 和 NCCL 初始化流程建立，分析日志时应以程序输出的 Rank 映射为准。

## `NCCL_TESTS_SPLIT`：按 MPI Rank 先分组

split 表达式先为每个 **MPI 全局 Rank** 计算 color，再通过 `MPI_Comm_split` 创建 MPI 子通信器；每个子通信器分别创建自己的 NCCL communicator。线程不单独计算 color，同一 MPI 进程的所有线程和它管理的 GPU 都属于该进程的组。

沿用上例，设置 `NCCL_TESTS_SPLIT="MOD 2"`：

```text
MPI_COMM_WORLD：world rank 0, 1, 2, 3
├── color 0：world rank 0、2 → 各进程 4 个 GPU → 组内 NCCL Rank 0..7
└── color 1：world rank 1、3 → 各进程 4 个 GPU → 组内 NCCL Rank 0..7
```

每个 communicator 都有自己的 NCCL Rank 编号空间，所以不同组内都可以存在 NCCL Rank 0。不同组测得的带宽是各组带宽；总带宽需按组分别统计后相加。

## 数量关系

设：

- `N`：MPI 进程总数（`mpirun -np N`），即 MPI world rank 数量。
- `T`：每个 MPI 进程的线程数（`-t T`）。
- `G`：每个线程使用的 GPU 数（`-g G`）。

在每个进程都使用相同参数且每个 GPU 独占的前提下：

```text
每 MPI 进程的 GPU 数 = T × G
整次运行的 GPU 参与者数 = N × T × G
无 split 时的 NCCL communicator rank 数 = N × T × G
```

使用 split 时，若某组包含 `N_group` 个 MPI 进程：

```text
该组的 GPU / NCCL Rank 数 = N_group × T × G
```

节点数不直接决定 Rank 数；节点数与每节点进程分布决定各进程在哪台机器以及本机设备编号如何分配。

## 编号与设备映射

| 名称 | 范围与用途 |
|---|---|
| MPI world rank (`proc`) | 整个 MPI 作业范围内唯一；split color 使用它计算。 |
| 本机 local rank (`localRank`) | 在当前主机上按 MPI world rank 顺序计数；用于默认选择 GPU 起点。 |
| MPI 子通信器 rank (`ncclProc`) | split 后在对应子通信器内重新编号；nccl-tests 用它选择该组的 NCCL ID 广播根进程。 |
| NCCL Rank | 某一个 NCCL communicator 内的 GPU 参与者编号；每个线程可管理多个 GPU/Ranks。 |
| thread index | MPI 进程内编号，范围 `0..T-1`；不是 MPI Rank，也不是独立节点进程。 |
| CUDA device index | 节点上的 GPU 设备编号；默认映射可被 `NCCL_TESTS_DEVICE` 覆盖。 |

上游源码的默认 GPU 起点按下式计算：

```text
默认设备起点 = localRank × T × G
本进程所用 GPU = 默认设备起点 ... 默认设备起点 + (T × G - 1)
```

这是本机的 CUDA 设备编号，不是 NCCL Rank。它假设进程按可预测顺序映射到节点、每节点可见 GPU 排列一致，并且 GPU 分配不重叠。使用 hostfile、调度器绑定、容器 GPU 可见性或 `NCCL_TESTS_DEVICE` 时，需要检查实际设备映射，不能仅凭公式推定物理 GPU。

## 对配置界面的对应

| 配置字段 | nccl-tests / MPI 含义 |
|---|---|
| 节点列表 | MPI 作业可用的主机名。 |
| MPI 进程总数 | MPI world 的总进程数；每个进程获得一个 MPI world rank。 |
| 每节点 MPI 进程上限 | launcher 的节点进程分配约束，不等同于每节点 GPU 数。 |
| 每进程线程数 | `-t`，每个 MPI 进程创建的测试线程数。 |
| 每线程 GPU 数 | `-g`，每个测试线程负责的 GPU 数。 |
| NCCL_TESTS_DEVICE | 覆盖 nccl-tests 默认的 GPU 设备起点；需确保进程之间不冲突。 |
| NCCL_TESTS_SPLIT | 按 MPI world rank 计算 color 并创建多个独立通信组。 |

## 源码依据

- 上游 README：`nccl-tests-master/README.md` 的 Arguments 和 Running multiple operations in parallel 小节，说明 `-t`、`-g` 及 split 表达式。
- 上游实现：`nccl-tests-master/src/common.cu` 的 `run()`，计算 `proc`、`localRank`、split color、MPI 子通信器及默认 GPU 设备起点。
- 上游实现：同文件构造 `threads` 参数的部分，为每个 thread 传入 `ncclProc`、线程索引、`nGpus` 以及对应 GPU 数组切片。


## 当前截图：3 个 MPI 进程的 MOD 2 分组

参数：3 节点，每节点 1 个进程，`-t 4 -g 2`。color 按 MPI world rank 取模。

```text
MPI_COMM_WORLD：3 个进程，共 24 个 GPU 参与者
├── G0：world rank 0、2 → 独立 NCCL communicator，16 个 Rank
│   ├── gpu01 / world rank 0 / 组内 MPI rank 0 / localRank 0
│   │   ├── T0 → GPU 0,1 → NCCL Rank 0,1
│   │   ├── T1 → GPU 2,3 → NCCL Rank 2,3
│   │   ├── T2 → GPU 4,5 → NCCL Rank 4,5
│   │   └── T3 → GPU 6,7 → NCCL Rank 6,7
│   └── gpu03 / world rank 2 / 组内 MPI rank 1 / localRank 0
│       ├── T0 → GPU 0,1 → NCCL Rank 8,9
│       ├── T1 → GPU 2,3 → NCCL Rank 10,11
│       ├── T2 → GPU 4,5 → NCCL Rank 12,13
│       └── T3 → GPU 6,7 → NCCL Rank 14,15
└── G1：world rank 1 → 独立 NCCL communicator，8 个 Rank
    └── gpu02 / world rank 1 / 组内 MPI rank 0 / localRank 0
        ├── T0 → GPU 0,1 → NCCL Rank 0,1
        ├── T1 → GPU 2,3 → NCCL Rank 2,3
        ├── T2 → GPU 4,5 → NCCL Rank 4,5
        └── T3 → GPU 6,7 → NCCL Rank 6,7
```

G0 跨节点通信，G1 仅节点内通信。分组不改变设备分配；同一 MPI 进程的线程和 GPU 不会被拆散到不同组。不同组的 NCCL Rank 编号独立，因此同时出现 Rank 0 是正常的。

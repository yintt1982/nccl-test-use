# nccl-tests：计时语义与 JSON 报告结构

研究日期：2026-10-26。依据本地 `nccl-tests-master` 源码静态阅读，未进行 GPU 运行验证。

## 1. 计时不是单一开关

`-w` warmup 次数不计入正式时间；`-n` 是外层正式迭代数；`-m` 是每个外层迭代聚合发起的 collective 数。源码在正式循环中每个外层迭代发起 `agg_iters` 次操作，并将总秒数除以 `iters * agg_iters`，所以常规 time 是每次 collective 的平均延迟，而非一批 `-m` 操作的总时间。[README 选项](../nccl-tests-master/README.md#L77) [计时循环](../nccl-tests-master/src/common.cu#L720) [平均计算](../nccl-tests-master/src/common.cu#L795)

计时路径还受 `-z` blocking 模式、`-G` CUDA Graph、`-C` CPU time 影响。`-z 3` 在每个外层迭代同步并 barrier，报告时间排除 barrier；`-G` 捕获/回放 graph，时间再按 graph launches 归一；`-C` 报告 CPU 侧时间。使用 `-I` per-iteration timing 时，CUDA events 记录每次聚合组的设备时间，并除以 `-m`，得到每个 collective 的样本；其汇总统计可排除前 `-K` 个样本，但原始 JSON 样本保留完整。[blocking/graph 计算](../nccl-tests-master/src/common.cu#L750) [event 除聚合数](../nccl-tests-master/src/common.cu#L666) [选项说明](../nccl-tests-master/README.md#L82)

`-a` 控制跨 rank 计时归并：0 取 rank 0，1 取平均，2 取最小，3 取最大。它控制的是 benchmark duration 的跨 rank 归并，不是 collective reduction op；两者语义独立。[README](../nccl-tests-master/README.md#L81) [归并实现](../nccl-tests-master/src/common.cu#L321)

校验通过 `-c` 控制。源码在 benchmark 结束后单独重新初始化数据，再执行校验用 collective 和比较；校验本身不混入性能时间。[校验阶段](../nccl-tests-master/src/common.cu#L828)

## 2. JSON 顶层对象

`-J file.json` 根据 `.json` 后缀启用 JSON。MPI 场景只让 MPI rank 0 创建报告文件；以独占创建方式打开，若目标已存在会追加 `.0`、`.1` 等编号，而不是覆盖。顶层记录 `version`、`start_time`、完整 `args`、完整 `env` 与 `nccl_version`，结束时写 `end_time`。[初始化和元信息](../nccl-tests-master/src/util.cu#L364) [结束时间](../nccl-tests-master/src/util.cu#L434)

主体大致包含：

- `config`：线程数、每线程 GPU 数、字节扫描范围/步进、warmup/正式迭代/聚合迭代/验证次数、CUDA Graph、blocking、per-iteration timing 等。[config](../nccl-tests-master/src/util.cu#L740)
- `devices`：rank、group、pid、hostname、CUDA device、PCI ID、设备描述。[rank 字段](../nccl-tests-master/src/util.cu#L487) [设备采集](../nccl-tests-master/src/util.cu#L765)
- `results`：逐测试点记录。[结果序列开启/结束](../nccl-tests-master/src/util.cu#L887)
- `out_of_bounds`、`average_bus_bandwidth`、`errors`：整次运行的越界错误、平均 bus bandwidth 检查状态、错误列表。[全局结果](../nccl-tests-master/src/util.cu#L897)

每条 `results[]` 记录含 `size`、`count`、`type`、`redop`、`root`；然后包含 `out_of_place` 和 `in_place` 结果对象（某些模式可将 out-of-place 置为 null）。性能字段是 `time` 或 `cpu_time`、`alg_bw`、`bus_bw`、`nwrong`；若错误报告被 adapter 禁用，`nwrong` 写成 JSON null。[测试点字段](../nccl-tests-master/src/util.cu#L501) [性能对象](../nccl-tests-master/src/util.cu#L576)

启用 per-iteration timing 时，结果对象另带 `out_of_place_per_iter` / `in_place_per_iter`，包括 skipped iteration 数、min/max/avg/p50/p95/p99/stdev（微秒）、cv 百分比及原始 `times_us`；MPI 情况下还可能包含 `per_process_max_times_us`。[per-iter JSON](../nccl-tests-master/src/util.cu#L656)

若 `-U` tuning 启用，结果行还可能带 tuning 子对象：implementation、algo、proto、kernelVariant、syncStrategy、channels、warps、netChunkSize；版本/collector 未提供时会写 `N/A`。[tuning 输出](../nccl-tests-master/src/util.cu#L516)

## 3. 报告解析与展示注意事项

1. `size` 是公共报告使用的 `max(sendBytes, expectedBytes)`，`count` 是 adapter `paramcount`。不同 collective 的二者含义不同；AlltoAllv 矩阵模式中 `count=0` 仍可能有非零流量。[公共 preamble](../nccl-tests-master/src/common.cu#L928) [输出字段](../nccl-tests-master/src/util.cu#L501)
2. `time` 数值单位为微秒；JSON per-iteration 字段显式标 `_us`。`cpu_time` 仍以 microseconds 输出，只是测量对象变成 CPU 端计时。[输出](../nccl-tests-master/src/common.cu#L886) [JSON key](../nccl-tests-master/src/util.cu#L612)
3. `nwrong=null` 表示没有提供/启用错误报告，不表示零错误。文本表中的 `N/A` 对应 JSON null。[输出逻辑](../nccl-tests-master/src/util.cu#L604)
4. `actual_iterations` 是结果终止记录写入的实际迭代数，不能无条件用 config 的 `iterations` 替代，尤其某些执行模式可能改变实际执行量。[结果终止符](../nccl-tests-master/src/util.cu#L516)
5. tuning 信息是可选且依赖 NCCL 版本、NCCL profiler/plugin 信息与 `-U` 设置。没有 tuning 子对象或值为 N/A，不足以断言 NCCL 没有选择算法。[README 版本要求](../nccl-tests-master/README.md#L106)
6. JSON `env` 是执行进程环境快照，适合复现，但也可能含环境中其他变量；导入或展示时要支持敏感字段过滤策略。该建议来自其完整采集行为，不是 nccl-tests 自带脱敏承诺。[环境写入](../nccl-tests-master/src/util.cu#L419)

## 4. 对后续管理网管的报告层建议

解析层以 JSON 原始对象和 stdout 原文为证据来源；规范化层保留原始字段，同时补充明确单位、时间归并策略、测试实现/adapter、每 rank 资源与状态；展示层区分延迟、带宽、校验状态与 per-iteration 分布。一次运行的配置必须和结果点关联保存，特别是 `-a/-m/-n/-w/-c/-z/-G/-C/-I/-K/-D/-H/-R/-U` 与 AlltoAllv 流量环境变量。

这仍是 nccl-tests 侧报告和数据契约研究，没有开始 Keysight 录屏对照或 UI 原型。

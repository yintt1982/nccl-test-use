# KAI 配置与结果界面迁移到 nccl-tests 的评估

日期：2026-10-25

## 结论

可以迁移 KAI 的交互骨架和信息布局，但不能直接迁移 KAI 的全部指标和网络仪表页面。适合 nccl-tests 的方案是：保留 KAI 的“测试任务 -> 配置 -> 运行 -> 结果摘要 -> 详情”流程，以及暗色、分区、拓扑可视化的表达方式；结果页只呈现 nccl-tests 实际输出或 JSON report 能提供的字段。

## 观察来源

- 报告录屏：`D:\工具开发\同行\KAI截图和录屏\record\KAI_仪表介绍_完整版.mp4`
- 配置录屏：`D:\工具开发\同行\KAI截图和录屏\record\kai_intro_06_nccltests.mp4`
- 关键帧归档：`E:\dpi\prototype-video-frames\contact.jpg`
- 指标字段对照：[nccl-kai-field-comparison.md](./nccl-kai-field-comparison.md)

## 配置界面观察

配置录屏采用暗色工作区，顶部有产品/工程导航，中间是测试拓扑和设备连接示意，侧边或右侧是参数表单。配置动作大致按以下顺序组织：

1. 选择测试任务或测试类型。
2. 选择参与测试的主机、GPU、rank 或网络端点。
3. 查看拓扑/连接关系，确认设备映射。
4. 设置 collective、数据规模、迭代和运行参数。
5. 保存配置并启动测试。

这套顺序适合 nccl-tests，因为 nccl-tests 的核心配置本身就是测试程序、GPU/rank 映射、collective、数据规模、warmup、iterations、datatype 和 NCCL 环境变量。KAI 的“物理仪表端口、RoCE、PFC、ECN、故障注入”部分不应迁移到 NCCL 配置页，除非后续产品明确接入额外的网络控制器。

## 结果界面观察

报告录屏表现出三个层次：

- 运行/任务层：测试名称、状态、时间和任务入口。
- 摘要层：按数据规模展示主要性能结果，适合快速判断趋势。
- 详情层：展开配置、拓扑、单次结果和更细的运行信息。

这个层次适合 nccl-tests。NCCL 的摘要表可展示 `size`、`count`、`type`、`redop`、`root`、`time`、`algbw`、`busbw`、`#wrong`；JSON report 还可以展示运行参数、设备信息、迭代统计、校验和 tuning 信息。

## 可迁移与排除范围

| KAI 设计 | nccl-tests 迁移判断 |
|---|---|
| 暗色控制台、顶部导航、任务工作区 | 适合迁移 |
| 配置表单分组 | 适合迁移 |
| GPU/rank 拓扑示意 | 适合迁移，但只显示 NCCL 能确认的设备和 rank 映射 |
| 测试任务列表和运行状态 | 适合迁移 |
| 按消息规模的结果表和趋势图 | 适合迁移 |
| 配置与结果联动查看 | 适合迁移 |
| 物理机箱、端口、交换机链路编辑 | 不迁移 |
| PFC、ECN、CNP、CRC、重传、QP、flow、FCT | 不呈现，nccl-tests 没有对应数据源 |
| KAI 理想带宽 `Ideal (%)` | 不呈现，除非另行定义并实现 NCCL 参考带宽模型 |
| KAI 故障注入与网络仿真控制 | 不迁移到 NCCL 原型 |

## 建议的 nccl-tests 原型结构

### 配置页

- 左侧：测试任务列表和运行状态。
- 中央：GPU/rank 拓扑图，显示 host、rank、GPU、device id。
- 右侧：配置表单。
  - Collective
  - 数据类型和 reduction op
  - GPU 数量
  - 最小/最大消息大小
  - 步进因子或步进字节数
  - warmup iterations
  - timed iterations
  - validation
  - in-place/out-of-place
  - NCCL algorithm/protocol/environment variables
- 底部：生成的命令行预览和启动按钮。

### 结果页

- 顶部：任务名、运行状态、开始/结束时间、主机和 GPU 数量。
- 第一行摘要：最大 `algbw`、最大 `busbw`、最小 `time`、`#wrong` 总数。
- 主区域：`size` 对 `time/algbw/busbw` 的曲线和表格。
- 详情标签页：
  - Performance：逐数据规模结果
  - Iterations：min/max/avg/P50/P95/P99、标准差、CV
  - Configuration：本次实际运行配置
  - Devices：rank、host、GPU、device info
  - Validation：out-of-bounds、错误数和测试状态

## 迁移判断

迁移是可行的，且 KAI 的布局适合作为 nccl-tests 的交互参考。迁移的核心是复用工作流和视觉组织方式，不复用 KAI 的数据模型边界。NCCL 原型应保持“集合通信性能测试工具”的范围，所有没有 nccl-tests 数据来源的 KAI 网络诊断字段都从界面和统一报告中删除。

下一步原型验证应重点观察三个问题：配置表单是否能覆盖 nccl-tests 常用参数；拓扑图是否能帮助用户确认 GPU/rank 映射；结果页是否能让用户从摘要快速定位到某个消息规模的 `time/algbw/busbw/#wrong`。

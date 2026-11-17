# nccl-tests 是否支持模型 Workload：源码分析

核查日期：2026-10-27。
核查对象：本地 `E:/dpi/nccl-tests-master` 源码快照。目录无 `.git`，本次不能独立确认其对应的上游提交，也不将本地新增参数一概认定为所有上游版本均有。在线检索未返回可引用内容，直接请求 GitHub 原始文件连接超时；以下结论以可逐行复查的本地代码为依据。

## 结论

如果 workload 指“选择 ResNet、AlexNet 等模型，执行其训练或推理”，则不支持。用户所写 alexret 暂按 AlexNet 理解。
如果 workload 指“配置集合通信类型、数据量、数据类型、并发 GPU/rank、迭代次数”，则支持。通信微基准不等于模型基准。

## 源码证据链

| 检查点 | 位置 | 发现 |
|---|---|---|
| 项目定位 | [README.md](../nccl-tests-master/README.md)，第 3 行 | 测试 NCCL 操作的性能与正确性 |
| 参数入口 | [common.cu](../nccl-tests-master/src/common.cu)，1198–1234 行 | longopts 有 minbytes/maxbytes/step/iters/ngpus/datatype/op 等；没有 model/workload/模型权重/数据集输入 |
| 测试循环 | common.cu，955–959 行 | 按消息大小遍历，分别调用原位和非原位 BenchTime |
| 数据生成 | [all_reduce.cu](../nccl-tests-master/src/all_reduce.cu)，42–54 行；common.cu，283–289 行 | InitData/InitDataReduce 调用可验证输入与预期值生成器，生成测试缓冲区 |
| 通信调用 | all_reduce.cu，500–507 行 | 默认实现直接调用 ncclAllReduce |
| 算子测试入口 | all_reduce.cu，545–575 行 | 循环数据类型与归约操作；没有模型构建、前向传播、反向传播和优化器流程 |

在 src 中搜索 resnet、alexnet、alexret、pytorch、tensorflow、onnx、checkpoint、dataset，未发现对应实现。结论不仅来自关键词缺失，也来自参数解析、输入生成、调度、实际操作的完整调用链。

## 已支持的配置与限制

- 基准程序选择通信操作，例如 all_reduce_perf。
- `-b/-e/-i/-f` 控制消息大小及扫描；`-d` 控制通信数据类型；`-o` 控制归约运算，例如 sum，不是选择神经网络算子。
- `-n/-w/-m` 控制计时、预热和聚合迭代；`-g/-t` 配合 MPI 进程数控制 rank 规模。精确定义见 README 参数表。
- `-G` 捕获并重放当前基准操作的 CUDA Graph，不是导入模型计算图或训练 trace。
- `-J` 是结果 JSON 输出，不是 workload JSON 输入。
- `NCCL_TESTS_SPLIT` 将 ranks 分组并行执行相同操作，不会自动建立模型张量并行、流水线并行或计算依赖。

重要例外：本地 [alltoallv.cu](../nccl-tests-master/src/alltoallv.cu) 第 50–81 行描述了 `NCCL_TESTS_ALLTOALLV_MATRIX_FILE` 与 `NCCL_TESTS_ALLTOALLV_MATRIX_SCALE`，可以从文件设置 rank-to-rank 流量矩阵。实际解析入口见 153–173 行。矩阵元素以字节表示，按 16 字节对齐，维度至少覆盖 ranks；矩阵模式要求 `-b == -e`，缓冲区必须覆盖最大总发送/接收量。该能力属于通信模式配置，不是模型 workload 加载。

## 若要按模型进行测试

1. **真实模型基准**：另接模型训练/推理执行后端，由后端构造指定模型、输入、精度、batch size 和分布式策略；NCCL 作为通信库，nccl-tests 保留用于链路基线。后端应输出迭代时间、样本吞吐、显存等真实模型指标。
2. **模型通信近似**：先采集指定模型、框架、并行策略和 batch 配置的通信操作、消息大小与次数，再用 nccl-tests 测量对应通信点。必须标注“通信近似”，不能把一次模型参数总量的 AllReduce 当作完整训练，也不能把通信时间相加直接称为训练耗时。
3. **模型通信/计算回放**：若需要时间序列、跨 rank 依赖、计算通信重叠等，应增加 trace 回放和调度后端。只增加模型下拉框无法完成此能力。

建议界面把“通信基准配置”和“模型 workload”分成不同测试类型。ResNet/AlexNet 选择项只有接入实际模型执行或已验证回放后端后才具有对应含义。若仅保存 nccl-tests 参数组合，应命名为通信配置模板。

本次仅新增分析文档，未修改源码、界面或部署服务。
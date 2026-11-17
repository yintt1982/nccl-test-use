# nccl-tests Workload 配置能力核对

核对日期：2026-10-27。用户要求：先确认支持，如支持再增加选择配置。

## 结论

NVIDIA/nccl-tests 可配置集合通信微基准负载，但不原生提供 Keysight KAI Workload Emulation 意义上的模型工作负载库、训练计算/通信序列回放、模型分区/并行训练调度。没有发现 --workload 或模型/trace输入接口。故本次不添加 Workload Emulation 入口，不将参数模板伪装成原生模型仿真。

## 已在线核对的证据

- https://raw.githubusercontent.com/NVIDIA/nccl-tests/master/README.md
  Arguments 列出消息范围/步长、迭代、数据类型、规约、线程/GPU数等；Running multiple operations in parallel 描述分组并行执行相同操作。-J 为结果文件输出，非workload JSON输入。-G为基准操作CUDA graph捕获重放，非任意训练trace回放。
- https://raw.githubusercontent.com/NVIDIA/nccl-tests/master/src/common.cu
  longopts及getopt_long（本次页面1108–1147行）无workload输入选项；测试循环（879–886行）按消息大小扫描，调用BenchTime执行原位/非原位测试。
- https://keysight.github.io/kai-dc-builder/solution/intro/
  明确将 Collective Benchmarks 与 Workload Emulation 分为两个应用。后者回放计算与通信步骤序列，分析模型分区对JCT的影响。
- https://www.keysight.com/nl/en/about/newsroom/news-releases/2025/0401-pr25-050-keysight-introduces-ai-data-center-builder-to-validate-and-optimize-network-architecture-and-host-design.html
  描述GPT/Llama库及DP/FSDP/3D并行模式。

## 可选扩展（本次未实施）

1. 基准负载模板：在前端保存/选择 nccl-tests 支持的参数组合，如小消息扫描、大消息扫描、固定消息测试。属于平台封装，不是上游workload选项。不要给模板冠以已验证的模型语义。
2. 多阶段通信测试：代理顺序调用多个测试程序并分别归档。需额外编排，不能直接把多个进程运行时间之和解释为训练JCT。
3. 模型工作负载回放：需要额外后端、计算/通信依赖与调度模型、工作负载文件规范和结果验证，不能仅用下拉框接入nccl-tests完成。

本次仅新增核对记录，原型界面未改动。

?????Keysight?????????????workload-kai-intro.txt?????????????GitHub?????????nccl-tests?????README.md?src/common.cu????????????master?

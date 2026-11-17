# NCCL Run 与观测证据架构

状态：Accepted（2026-11-17）

采用“浏览器 → OIDC 管理服务 → Slurm → Linux Runner Agent → nccl-tests/Profiler”的控制链。原生 benchmark 输出、NCCL Profiler 事件、网络/QP 事件和 NIC/交换机计数作为有版本、来源和作用域的独立证据保存；由管理服务生成可修订的报告并通过 SSE 发布 Report Revision。选择此边界是为了让前端不执行任意 shell、让高频采集不阻塞 NCCL 回调，并避免把逻辑 chunk/peer 指标冒充物理网络测量。

管理元数据存 PostgreSQL，原始 Run Artifact 存文件/对象存储；Runner 以幂等批次上传，服务端持久化确认后才清理本地 spool。首发使用单管理服务实例和备份恢复。多节点通过受限 Slurm allocation 与登记的 `srun`/MPI/PMIx 组合执行；无 Slurm 环境首发只支持单节点。

开发与联调阶段通过受控身份策略授予开发账号管理员权限，以便调整测试配置和采集设置；所有操作仍经过服务端 RBAC 并审计。运行操作员/只读权限从实现初期即建模和测试。交付前通过角色与资源授权配置切换权限，不使用全局鉴权绕过，也不依赖最后修改业务代码。

采集按 Proxy-first 分阶段：首发细粒度报告使用 ABI 兼容的 NCCL Profiler Plugin；Flows/QPairs 需要 Network/IB、连接/WR/QP 证据；Ports 需要设备侧计数器。所有跨层映射必须保存关联证据和时钟不确定度；不能由 Proxy 字段推算 request、packet、wire 或端口字节。

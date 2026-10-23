# 冗余代码清理与接口迁移

本次清理统一重复实现，并让原来没有效果的配置给出明确反馈。`FEDAVG`、`FEDAVG_BASE`、`FEDAVG_DP`、`FEDPROX` 继续保留各自语义。

## 已清理的实现

- 删除 `blades/adversaries/spear_adversary.py` 中可执行代码之后的历史注释版本。
- 将 SPEAR 与 SPEAR-SL 相同的 z 搜索提取到私有 `_spear_search.py`。各自的 DnC 模拟方法和随机数调用顺序保持不变。
- 删除 SPEAR-SL2 未被使用的标准 ALIE 分位数计算，以及未使用的导入、局部变量和私有状态。
- 删除未接入训练链的私有 `blades/_internal/worker_group.py` 和 `core/execution/reduce_result_dict_fn.py`。
- ActorPool 共用 `blades.utils.actor_pool.ActorPool`，执行核心保留 `_ActorPool` 别名。亲和任务正确占用执行器并在释放后继续调度，排队任务仍按提交顺序返回结果。旧训练器注册表转发到统一的算法注册表。
- 张量工具复用 Ray/PyTorch 实现和共同的裁剪计算，保留公开函数名以及训练所用裁剪函数的返回约定。

## ClientConfig

内置客户端的数据集、模型和损失由训练器与 Task 管理。旧 `ClientConfig.dataset()`、`model()` 设置此前没有效果，现在会提示迁移并拒绝不支持的设置；字典配置和构建阶段也执行校验。历史默认配置仍可回读，自定义客户端仍可消费自己的扩展字段。

```python
from blades.algorithms.fedavg import FedavgConfig

config = (
    FedavgConfig()
    .data(dataset_config={"type": "MNIST", "train_batch_size": 32})
    .training(global_model="mlp")
    .client(client_config={"lr": 0.1, "momentum": 0.9})
)
```

内置客户端目前使用 SGD。旧 `optimizer("sgd", opt_kargs={"lr": ..., "momentum": ...})` 会转接到现有 `training()` 配置；其他优化器、额外优化器参数和调度器会明确报错。需要这些功能时，覆写自定义 Task 的 `init_optimizer()`；自定义损失覆写 `Task.loss()`。推荐直接使用 `ClientConfig.training(lr=..., momentum=...)`。

直接给 `opt_kargs` 赋值不会更改客户端使用的 `lr/momentum`。当非默认旧参数与实际训练值冲突时，构建客户端会给出可见警告；使用上述配置方法应用这些设置。

## 兼容与弃用

- `Signguard(max_tau=..., linkage=...)` 仍接受旧参数，但显式传入会发出弃用警告；这两个参数此前没有参与聚合。聚合计算保持原样。
- CLI 的 `--trace` 源自 TF2 eager tracing，当前 PyTorch 训练器不支持。参数保留并给出可见警告，帮助文本也标明弃用。
- `run_blades_experiments(algo=...)` 保留为兼容参数并发出弃用警告；实际算法始终由 `experiments[name]["run"]` 指定，内部调用不再透传多余参数。
- `blades.trainers.registry.TRAINERS` 与 `blades.algorithms.registry.ALGORITHMS` 共用一张表。`FEDAVG` 指向带攻击版本；基础训练器请用 `FEDAVG_BASE`。
- `blades.session`、`blades.utils.queue.Queue` 保留旧行为并标记弃用。当前客户端使用 `blades.core.execution.session.get_session(client_id)`；一般队列使用 `ray.util.queue.Queue`。旧 Queue 的 CUDA IPC 协议不同，不直接替换其实现。
- 无仓库内引用的公开张量工具保留兼容入口；私有设备检查由 PyTorch 的参数转换函数承担。

许可证、原始迁入清单以及各攻击变体保留。历史实验配置的适用范围仍见 [NEW_EXAMPLES_CHECK.md](NEW_EXAMPLES_CHECK.md)。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests blades/aggregators/tests blades/datasets/partitioners/tests blades/datasets/tests/test_clientdataset.py blades/models/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m examples.smoke_train
.\.venv\Scripts\python.exe -m ruff check blades --select F401,F811,F821,F841 --no-cache
```

冗余清理阶段的上述回归测试共 180 项通过；后续 [FedAvg 实现统一](FEDAVG_UNIFICATION.md) 后扩展至 188 项通过，Ruff 指定检查全部通过。测试输出仅含既有的 wandb/Sentry 弃用警告和 Dirichlet 张量构造警告。

专项回归覆盖固定种子的 SPEAR 数值与随机数状态、旧接口兼容、无效客户端配置、裁剪的原位行为，以及亲和执行器排队时的有序与无序结果消费。训练验证使用合成数据运行两轮 CPU 实验；未覆盖 GPU 或多节点训练。

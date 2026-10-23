# FedAvg 实现统一

FedAvg 的实际实现集中于 `blades/trainers/fedavg`。基础版、攻击版、DP 版本共用 `FedavgTrainer` 的初始化、训练循环、评估分发与 checkpoint 入口。

```text
blades/trainers/fedavg/
  fedavg.py         基础配置与唯一训练引擎
  callbacks.py      联邦轮次回调
  adversarial.py    攻击配置与 Fedavg 薄子类
  fedavg_dp.py      DP 配置与 FedavgDP 薄子类
```

`blades/algorithms/fedavg/fedavg.py` 和 `fedavg_dp.py` 仅转发类定义。通过新旧路径导入得到同一个类对象，继承关系保持：

```text
Trainer
└── FedavgTrainer
    ├── Fedavg
    │   └── FedavgDP
    └── FedProxTrainer
```

## 算法入口

| YAML run | 类 | 作用 |
| --- | --- | --- |
| FEDAVG_BASE | FedavgTrainer | 基础联邦训练 |
| FEDAVG | Fedavg | 联邦训练与攻击回调 |
| FEDAVG_DP | FedavgDP | 攻击配置与 DP 客户端 |
| FEDPROX | FedProxTrainer | 基础训练与 proximal 客户端回调 |

注册表直接从统一实现加载各类，原有四个名字和 YAML 用法保留。`get_default_config()` 将 `algo_class` 绑定到调用它的类，用户子类也能构建自身类型。

推荐导入：

```python
from blades.trainers.fedavg import (
    FedavgTrainer, FedavgTrainerConfig,
    Fedavg, FedavgConfig,
    FedavgDP, FedavgDPConfig,
)
```

旧 `blades.algorithms.fedavg` 包级导入、`fedavg` / `fedavg_dp` 模块导入以及基础训练器和回调的原路径均继续可用。

## 回调与训练顺序

每轮仍执行：同步模型 → 客户端训练 → `on_local_round_end` 回调 → 提取更新 → 聚合并更新全局模型。

`FedavgTrainerConfig._get_callback_specs()` 复制普通回调配置列表；攻击配置在新的临时列表中追加一个攻击配置。统一的 `build_callbacks()` 校验冻结状态并创建回调实例。重复构建同一配置不会修改 `callbacks_config`，也不会累积攻击回调。已有自定义回调的顺序保持，自动安装的攻击回调位于其后。

攻击模块在构建攻击配置时延迟导入 `Adversary`，回调类型引用使用 `TYPE_CHECKING`，基础引擎无需在模块顶层加载攻击包。

## 评估兼容

权重同步、数据加载与评估分发只有一份实现，通过 `_compile_eval_results()` 选择汇总策略：

- 基础版及 FedProx 继续调用 `server.task.compile_eval_results()`，保留自定义 Task 的汇总逻辑。
- 攻击版及 DP 继续输出旧版 `ce_loss`、`acc_top_1`，按客户端样本数加权，保留原有自定义 Task 场景下的分类汇总行为。
- 默认 Classifier 和攻击版共用 `tasks.classifier.compile_classification_results()`，消除重复的指标计算。

本次迁移保留 DP 隐私公式、`train_bs` 配置约定及 FedProx 的原有计算。checkpoint 的既有空实现集中保留在基础类。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests blades/aggregators/tests blades/datasets/partitioners/tests blades/datasets/tests/test_clientdataset.py blades/models/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m examples.smoke_train
.\.venv\Scripts\python.exe -m examples.smoke_spectral_spear
.\.venv\Scripts\python.exe -m ruff check blades --select F401,F811,F821,F841 --no-cache
```

专项测试覆盖导入别名和注册表身份、全新解释器的导入顺序、重复回调构建、基础版与零攻击版本的模型更新一致性、攻击在聚合前执行、自定义评估策略、DP 和 FedProx 扩展。训练验证使用合成数据与 CPU。

验证结果：188 项回归测试通过（包括 8 项新增统一回归），Ruff 指定检查、Python 语法解析与 `git diff --check` 通过。普通攻击和 Spectral SPEAR 示例分别完成两轮 CPU 训练。基础版与零攻击版本的实际两轮参数逐项完全一致；训练循环 `training_step()` 的 AST 与原实现相同。未执行 GPU 或多节点训练。

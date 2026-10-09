# 合并说明

源目录是 `../blades/blades` 与 `../fedlib-main/fedlib-main`。两个目录保持原样；迁入文件的源版本校验值见 `source-manifest.json`。

## 代码组织

统一分发名为 `BladesALL`，Python 导入名为 `blades`。所有业务模块和实验配置中的包路径都指向本项目，不创建 `fedlib` 兼容包，也不使用安装目录中的源码。

| 原功能 | 合并位置 |
| --- | --- |
| Blades 攻击，包括 SPEAR / SPEAR-SL | `blades/adversaries` |
| 防御、数据、模型、训练器 | `blades/aggregators`、`blades/datasets`、`blades/models`、`blades/trainers` |
| 基础客户端 | `blades/clients/base_client.py` 中的 `BaseClient` |
| 支持攻击的客户端 | `blades/clients/client.py` 中的 `Client`，继承本地 `BaseClient` |
| 基础回调和梯度裁剪回调 | 合并到 `blades/clients/callbacks.py` |
| 两个项目的实验入口 | `blades/train.py` |
| 原框架示例 | `examples/core`、`blades/config_examples` |

`FEDAVG` 保留攻击版语义，`FEDAVG_BASE` 指向基础训练器，另有 `FEDAVG_DP` 和 `FEDPROX`。修正了 FedProx 默认配置类型、Python 配置文件的训练类解析和命令行 Ray 资源参数传递。许可证及归属信息均保留。

## 验证范围

`tests` 下的新测试验证本地模块解析、禁止外部框架导入、客户端继承与构造、攻击/防御交互、动态注册、回调及 CLI。`examples/smoke_train.py` 用合成数据执行两轮带攻击和防御的 CPU 训练。详细运行结果见 `VERIFICATION.md`。

本次保持现有攻击和聚合算法的数学实现。尚未验证完整数据集实验、GPU、多节点训练或每个历史测试。源代码中的 checkpoint 保存函数仍是空实现；FedProx 原有惩罚项使用了不带梯度的保存张量，其数学效果需要单独审查。上述问题不属于包合并修复范围。

原 Blades 根目录的 `test.py` 是 AGNews 手动诊断脚本，引用了提供的框架源码中不存在的 `datasets.agnews` 模块，因此未将它接入新项目测试。原 `new_examples` 中涉及 AGNews 的 YAML 作为历史配置保留，运行它们前仍需实现相应数据集和文本模型。

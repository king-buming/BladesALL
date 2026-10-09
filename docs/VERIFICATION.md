# 验证记录

日期：2026-09-04。

## 环境

Windows，Python 3.12.13，PyTorch 2.7.1+cpu，torchvision 0.22.1+cpu，torchaudio 2.7.1，Ray 2.39.0。项目 `.venv` 中已完成可编辑安装，可直接使用 `.venv\Scripts\python.exe`。

验证环境创建时使用了 `--system-site-packages` 复用本机 `D:\miniconda3\envs\lerobot2` 中的通用依赖，项目所需的 Ray、PyTorch 等版本另装在本项目 `.venv` 内。未修改该 Conda 环境。迁移到其他电脑时，请按 README 创建新的环境并安装本项目。

## 结果

- 合并回归、CLI 及现有聚合器/数据划分/模型测试：**27 passed**。5 条警告来自原数据划分代码的 tensor 构造方式和 wandb/Sentry 弃用提示，没有测试失败。
- `python -m examples.smoke_train`：完成两轮本地 CPU 联邦训练，1 个恶意客户端，SignFlip 攻击，Median 聚合；模型参数发生更新。两轮评估 loss 为 `0.8482406`、`0.8457373`。
- 本地模块、字符串类路径、客户端与回调组合、算法注册表验证通过。测试主动拒绝任何外部框架包导入。
- `python -m blades --help` 通过；Python 实验文件能够解析实际训练类。
- Poetry wheel 构建及 `pip install -e . --no-deps --no-build-isolation` 成功。wheel 包含全部主要模块及 25 个 YAML 配置，依赖元数据无外部框架包。
- 所有迁入 Python 源码语法解析成功，静态内部导入路径均存在。
- 两个源包的 149 个文件路径全部保留（同名路径做语义合并）；源目录清单中 222 个文件的 SHA-256 校验值未变。
- 独立代码审查未发现尚未处理的合并问题。

复现测试命令（项目根目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests blades/aggregators/tests blades/datasets/partitioners/tests blades/datasets/tests/test_clientdataset.py blades/models/tests -q
.\.venv\Scripts\python.exe -m examples.smoke_train
```

验证没有覆盖全量历史实验、真实数据下载、GPU、多节点训练及 Sphinx HTML 构建。源项目遗留问题和示例适用范围见 [MIGRATION.md](MIGRATION.md)。

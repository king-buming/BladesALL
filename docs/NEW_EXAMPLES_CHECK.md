# new_examples 使用检查

日期：2026-09-04。检查对象为 `blades/new_examples` 中全部 8 个 YAML。

启动方式保持原框架的 `train.py file <yaml>` 形式；可从根目录使用等价的 `python -m blades file <yaml>`。8 个 YAML 解析出的参数与所提供的原 Blades 项目一致，本次检查没有调整实验参数。

| 文件 | 网格实验数 | 组件检查结果 |
| --- | ---: | --- |
| fedsgd_cnn_fashion_mnist.yaml | 7 | 模型、数据集、攻击和聚合器实现存在，配置解析通过 |
| fedsgd_cnn_fashion_mnist_noniid.yaml | 21 | 同上，含三个 Dirichlet alpha 值 |
| fedsgd_mlp_mnist.yaml | 7 | 模型、数据集、攻击和聚合器实现存在，配置解析通过 |
| fedsgd_mlp_mnist_noniid.yaml | 2 | 不能直接运行：活动配置使用 MCA、FedCut，所提供源码没有这两个聚合器 |
| fedsgd_resnet_cifar10.yaml | 1 | 组件和配置解析通过；100 个客户端中 30 个运行 SPEAR 攻击 |
| fedsgd_resnet_cifar10_noiid.yaml | 1 | 组件和配置解析通过；文件名保留原来的 noiid 拼写 |
| fedsgd_textcnn_agnews.yaml | 1 | 不能直接运行：没有 AGNews 数据集实现，global_model 的 texttnn 字典也不受模型目录支持 |
| fedsgd_textrnn_agnews.yaml | 1 | 不能直接运行：没有 AGNews 数据集和 textrnn 模型，model_config 参数未被模型构造使用 |

共 41 组实验。上述缺失模块在用户提供的 `fedlib-main/fedlib-main` 源码中也不存在；没有猜测其算法实现或擅自替换实验方案。

## GPU、攻击与运行轮数

- 全部 YAML 设置了正数的 `num_gpus_for_driver` 和 `num_gpus_per_worker`（0.3 或 0.5），且 `num_remote_workers=0`。即便没有远程 worker，本地 worker 仍按配置使用 CUDA。
- 当前 `.venv` 的 PyTorch 是 CPU 版本，不能直接执行这些 GPU 配置；使用支持 CUDA 的环境，或把 YAML 的 `config.num_gpus_for_driver` 和 `config.num_gpus_per_worker` 都改成 `0`。仅传 `--ray-num-gpus 0` 不会改写试验中的这两个字段。
- 除 CIFAR10 IID 配置之外，其他 7 个 YAML 的 `num_malicious_clients` 都是 `grid_search: [0]`；指定了攻击类也不会生成恶意客户端。
- 默认 2000 轮；前述实验数来自当前活动的 grid_search，不包含注释中的候选算法。
- 数据集加载会尝试下载 MNIST/FashionMNIST/CIFAR10 到用户目录中的 `~/fldata`，需要相应网络和存储条件。

## 本次修复与验证范围

修复了 `blades/train.py` 的两个入口问题：明确按 UTF-8（兼容 BOM）读取 YAML；使用 Ray 2.39 的 `checkpoint.path` 并处理没有 checkpoint 的情况。新增两个回归测试，已复现原失败并验证修复。未改动 8 个 YAML 的参数。

已验证全部 YAML 加载、全部网格配置类校验、攻击构造、聚合器构造；针对已有 7 个聚合器执行真实张量运算，针对 FashionCNN/MLP/ResNet 执行图像张量前向与反向。机器可读结果见 `new_examples-audit.json`。

这不是完整真实数据训练验证：未下载这些数据集，未运行 2000 轮网格实验，也未测试 GPU 训练。原训练器的 checkpoint 保存仍是空实现，因此建议先使用 `--no-checkpoint-at-end`，不要把现有 checkpoint 当成已验证可恢复的模型。

完成环境/资源配置后，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m blades file .\blades\new_examples\fedsgd_mlp_mnist.yaml --no-checkpoint-at-end
```

也可沿用原来的目录与命令形式（先在使用的 Python 环境可编辑安装项目）：

```powershell
cd blades
python train.py file .\new_examples\fedsgd_mlp_mnist.yaml --no-checkpoint-at-end
```

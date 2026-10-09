# SPEAR 的谱方向扩展与审稿问题处理

日期：2026-09-11。新增实现称为 **SPEAR-Spectral 实验变体**。它在原有 DnC 代理验收框架上增加了实际参与向量构造的 SVD 几何。旧 `spear_adversary.py` 和旧实验配置保留，以便复现与对照。本文描述已经实现的机制、可检验的数学性质和仍需补齐的研究证据；不把旧论文表格归给新版。

## 当前代码和 Adaptive-DnC 的关系

核对文件为 `iotjpaper/review.docx`、`BladesALL/blades/adversaries/spear_adversary.py` 的生效代码，以及 [NDSS 2021 原文](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf) §VII-B、式 (8)、Algorithm 1。式 (6) 是 Min-Max，不能用作 Adaptive-DnC 的定位。

原文已经以统一的均值加缩放扰动构造恶意上传，通过恶意客户端在 DnC 中的存活情况选择系数，并明确不知道服务器实际抽到的坐标。本地旧 SPEAR 则固定沿正标准差方向，在 [0,5] 做 10 次二分，每次用 5 组新坐标模拟检测。两者在攻击框架上高度重合，方向符号、搜索预算和实现细节存在差别。

旧代码中的 SVD 只计算 DnC 投影评分，没有构造谱约束扰动；也没有跨轮 ramp-up 状态。因此不能用旧代码支持“首次自适应规避 DnC”“首次谱方向优化”或“已实现动态 ramp-up”。固定随机子空间也不足以恢复二分所需的单调性，因为恶意候选本身会改变混合均值和主奇异向量。

| 比较维度 | 旧 SPEAR | 新 SPEAR-Spectral |
| --- | --- | --- |
| 候选方向 | 固定逐坐标标准差 σ | 多随机子空间 SVD 得到风险矩阵，再解椭球内的逆均值方向 |
| SVD 用途 | 模拟防御评分 | 构造方向，以及重新计算混合总体的防御评分 |
| 优化对象 | 搜索更大的存活系数 z | 在经验存活约束下，选择最差代理组上的聚合逆均值位移最大的候选 |
| 搜索方法 | 对随机判据二分 | 完整遍历事先固定的 z 和谱惩罚有限网格 |
| 随机性 | 每候选重采样，消耗全局 RNG | 独立的拟合、搜索、事后检验随机流；所有候选共享搜索组 |
| 失败返回 | 未检验的标准 ALIE 系数 | 明确记录无可行候选，返回良性均值作为回退 |
| 已知信息 | 全部良性更新及身份 | 相同的全知设定；尚未实现 compromised-only |

## 实际优化问题

设本轮可见良性更新为 B∈R^(b×d)，均值 μ，样本标准差 σ。项目上传的是 `本地参数 − 全局参数`，服务器沿聚合结果更新参数，因此 −μ 是与正常更新相反的方向。该方向是损害的局部代理，不能直接等同于训练损失梯度。

对 K 个独立抽样的坐标集合 Ω_l，中心化良性子矩阵并做 reduced SVD：

\[
B_{:,\Omega_l}-\bar B_{\Omega_l}=U_l S_l V_l^\top.
\]

保留前 r 个数值非零右奇异向量；r 受真实秩限制。将每个 v_lj 按原坐标嵌回 d 维，并赋权 s_lj/(s_l1√K)，将这些列组成 A。全部相同的良性更新没有有效谱列。不同子空间重叠的坐标贡献会相加。

定义

\[
H_\lambda=I+\lambda AA^\top,\quad \lambda\ge 0,\qquad
\rho=z\|\sigma\|_2.
\]

方向子问题为

\[
\max_\delta -\mu^\top\delta
\quad\text{s.t.}\quad \delta^\top H_\lambda\delta\le\rho^2.
\]

对非零 μ 与 ρ，解为

\[
\delta^*=-\frac{\rho H_\lambda^{-1}\mu}
{\sqrt{\mu^\top H_\lambda^{-1}\mu}},\qquad a=\mu+\delta^*.
\]

这是真正改变候选方向的谱惩罚：沿抽样主方向的能量受到额外约束，而未抽到的坐标保留单位范数约束。**不是硬投影到严格零空间，也没有保证 DnC 分数必然下降。** 新候选会旋转混合总体的主方向，故仍必须重新中心化并重新做混合 SVD 检验。

该闭式解来自标准椭球线性优化：令 y=H^(1/2)δ，用 Cauchy–Schwarz 得最优值 ρ√(μᵀH⁻¹μ)。H≽I 还给出 ||δ||₂≤ρ。μ=0 或 ρ=0 时返回零扰动。λ=0 恢复普通逆均值方向，可直接用于移除谱惩罚的消融。这里的标准数学工具及 Woodbury 恒等式不能单独作为原创定理。

实现仅在坐标并集上保存 A，利用

\[
H_\lambda^{-1}\mu=\mu-\lambda A(I+\lambda A^\top A)^{-1}A^\top\mu
\]

求解至多 (Kr)×(Kr) 的系统，不建立 d×d 协方差。SVD 失败、非有限输入或求解退化会显式报错，不用随机向量假装谱检验成功。

## 搜索、验收与诊断

1. 按实际本轮结果顺序放回恶意候选的 f 个相同副本；使用本轮 f，不使用全局配置人数。保留 NumPy 默认排序，与仓库 DnC 的同分规则对齐；数值近似同分仍可能受硬件/库影响。
2. 所有候选共享 E 个搜索组，每组包含 T 次 DnC 坐标抽样。每次检测都重新计算候选污染后的中心和首右奇异向量，按平方投影保留 n−floor(c f) 行，最后求 T 次保留集合的交集。
3. 基准总体保留同样的良性行和恶意位置，将恶意位置替换成 μ。它是同规模的“均值替换控制”，不是无恶意训练的准确率基线。
4. 对每个组 e 计算 D_e(a)=−μ_hatᵀ[DnC_e(B,a)-DnC_e(B,μ)]。默认要求每个搜索组中全部恶意客户端存活；`min_survival` 指“全部 f 个存活的组数比例”，不是平均恶意客户端保留比例。选择最大 min_e D_e(a) 的可行候选；平分时优先更高组存活率，再选较小扰动。
5. 网格全部评估，不早停、不二分。多组 λ 的 z=0 重复点只算一次。返回的是该有限列表的经验最优候选，没有连续域全局最优或总体概率保证。
6. 最终向量只在独立事后组上检验一次。记录 `audit.all_survival_rate`、每组保留恶意数和 `damage_by_bank`；**这些结果不参与重新选候选**。它不是服务器真实结果。k≥d 时各抽样覆盖全部坐标，不再提供实质坐标子集的多样性。
7. 没有候选满足阈值时，`status=no_feasible_candidate`、`selected=null`，提交 μ 并记录其事后结果。均值回退也可能被 DnC 剔除，不声称成功。

拟合、搜索和事后组使用三个局部 CPU `torch.Generator`，种子由配置种子和轮号确定，不消耗训练/真实服务器的全局 torch 或 NumPy 随机流。`last_diagnostics` 包含配置、轮号、种子、全部候选记录、选择结果、事后结果、人数、行位置和 SVD 次数。保存这些记录才能复核实际运行参数。

## 使用方式

入口：`blades/adversaries/spear_spectral_adversary.py` 的 `SPEARSpectralAdversary`。
数学核心：`blades/adversaries/spectral_spear.py`。

在现有实验的 `adversary_config` 中使用：

```yaml
adversary_config:
  type: blades.adversaries.spear_spectral_adversary.SPEARSpectralAdversary
  z_values: [0.0, 0.5, 1.0, 2.0, 4.0]
  spectral_penalties: [0.0, 1.0, 10.0]
  spectral_rank: 2
  fit_subspaces: 3
  dnc_sub_dim: 10000
  dnc_num_iters: 5
  search_banks: 2
  audit_banks: 4
  dnc_filter_frac: 1.0
  min_survival: 1.0
  seed: 2026
```

攻击代理参数不会自动修改真实服务器配置。配置中的 `server_config.aggregator` 必须明确选为 DnC 并记录其参数；实验可有意设置失配。数据下载和完整训练未由这个片段触发。

从 BladesALL 根目录运行离线 CPU 冒烟训练：

```powershell
.\.venv\Scripts\python.exe -m examples.smoke_spectral_spear
```

核心和接入测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_spectral_spear.py tests/test_spectral_spear_round.py -q
```

## 本次实际验证

本机环境为 Python 3.12.13、PyTorch 2.7.1+cpu；CUDA 不可用。2026-09-11 实际执行：

- 新增核心与接入测试 **30 项通过**。覆盖闭式解与稠密数值解、约束半径、重叠坐标嵌入、非单调反例、固定坐标下与本地 DnC 一致性、独立随机流、常数/单客户端输入、回退状态及真实训练回调。
- `python -m pytest tests -q`：项目顶层 `tests` 目录 **124 项通过**，13.52 秒；仅有既存 wandb/Sentry 弃用警告。这不是对所有历史模块测试目录的全量验证。
- `python -m examples.smoke_spectral_spear`：离线 CPU 两轮实际训练成功。两轮状态均为 `proxy_feasible`，事后两组各保留 1 个恶意上传；loss 为 0.8073618293、0.8030748487。这里是生成的二维二分类数据，训练与评估共用玩具样本，只验证接入；不作为泛化、攻击效果或论文准确率数据。

独立复核发现并修复了常数 float32 更新中心化导致伪谱方向，以及大谱惩罚下 Woodbury 消减误差导致超出约束半径的问题。新增回归测试先复现失败，再验证修复。良性谱拟合使用锚定中心化；低秩求解与二次型累积使用 float64，返回时检查上传精度下的约束。代理 DnC 保持本地防御的中心化方式以便对照。

## 计算成本与适用边界

记 Q 为去重后的网格候选数，J 为事后组数，k=min(d,sub_dim)，s≤Kr 为有效谱列数，u≤min(d,Kk) 为坐标并集大小。SVD 次数为 K+(Q+1)ET+2JT，其中基准总体也需检查。默认 Q=13，共 **183 次 SVD/轮**；旧 SPEAR 是 50 次。新版增加了检验预算，不能把任何效果增加全部归给方向机制，也不能声称比旧版更快。

谱拟合约 O(K b k min(b,k))，方向求解每个 λ 约 O(u s²+s³+d)，主要存储 O(us+s²)。有限搜索及事后检查约 O(((Q+1)E+2J)T[nk min(n,k)+n log n])，另有总体构造/聚合 O((QE+J)nd)、抽样 O((K+ET+JT)d) 和临时更新存储 O(nd)。库与设备会影响真实耗时，SVD 次数不是墙钟加速比。

目前只实现全良性更新可见的研究设定；没有在客户端本地估计隐藏良性统计，也没有借用真实服务器随机坐标。SPEAR-SL 仍是另一条独立路径；本变体没有自动执行 SignLoader，追加符号变换后必须重新检验最终向量。

## 怎么处理这次审稿意见

可以承认旧 SPEAR 属于已有的 DnC 感知标量搜索家族，再清楚说明本次修订增加了什么。SVD 构造方向、椭球问题、聚合方向目标和独立事后检验提供了可检查的算法差别，但不足以单凭实现宣布“首次”或“显著优于 Adaptive-DnC”。

本次定向检索还发现 [Good Gradients Poison Your Model，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38328) 已讨论方向优化与边界自适应；其摘要并不能排除与本方案的细节重合。[AutoAdapt，NDSS 2024](https://www.ndss-symposium.org/wp-content/uploads/2024-1366-paper.pdf) 也是需要逐式比较的多约束攻击近邻。尚未完成覆盖这些近邻全文的创新性审查。因此这里只称“相对现有本地代码和 DnC 固定方向构造的实质扩展”，不作全领域首创结论。

重投前至少补齐：

- 按原文说明复现 Adaptive-DnC；旧 SPEAR、新版、λ=0、固定方向加相同有限搜索分别比较。固定方向应覆盖 ±σ 与 −μ，明确符号约定。
- 同候选评估次数、同总 SVD 预算和同墙钟预算分别报告，分离搜索预算与方向优化的贡献。新版选择多个 λ 会扩大候选数量，不能忽略这一点。
- 报告真实服务器保留率、代理和事后保留率差、模型准确率/损失、每轮攻击耗时；在相同数据分割和种子下配对多次运行并报告均值与标准差。
- 检验子维度、有效秩、谱惩罚、恶意比例、非 IID 程度和代理/服务器参数失配。区分全知设定与仅受控客户端信息；后者需要新增信息隔离实现。
- 若组合 SignLoader，比较变换前后最终向量的谱验收与攻击效果，不能复用变换前的通过判据。

英文回复可先采用以下事实定位，再补真实实验数据：

> We agree that the previous implementation belongs to the established DnC-aware scalar-search family and did not sufficiently demonstrate a distinct attack construction. In this revision, we implement a separate experimental variant that constructs perturbation directions using benign spectral geometry. Specifically, it solves a linear anti-update objective under a positive-definite spectral ellipsoid, then evaluates a finite set of candidates using the aggregate displacement after DnC filtering. Candidate selection uses fixed local coordinate banks, while a separate bank is used only for post-selection diagnostics. These changes define an algorithmic extension; comparative effectiveness and its dependence on the information and computational budgets require separate experiments.

这段不能改成“我们已证明优于所有攻击”或加入未经运行的结果。原稿的旧数据继续对应旧实现；新版实验应单独标识。

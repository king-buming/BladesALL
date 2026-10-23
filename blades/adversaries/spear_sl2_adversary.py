import torch
from blades.constants import CLIENT_UPDATE, CLIENT_ID
from blades.trainers import Trainer as Algorithm
from blades.trainers import Trainer
from .adversary import Adversary


class SPEARSL2Adversary(Adversary):
    r"""SPEAR-SL2 (SignLoader 2.0): 幅值保留的符号共识对齐攻击。

    针对 SPEAR-SL 中 SignLoader 削弱攻击效果的四个根因：
    1. 顺序缺陷: z 搜索在未修改向量上验证 DnC，符号修补后谱特征已变；
    2. 幅值侵蚀: 多余符号被置零/贴近零常数，攻击能量被直接抹掉；
    3. 零位错配: 良性零是结构性的（死单元），SLv1 的零放在任意位置；
    4. 范数不恢复: 符号计数对正缩放不变，范数本可免费恢复到裁剪预算上限。

    SL2 的设计：
    - 每维符号取良性客户端在该维的多数符号（共识模板），
      幅值保留 ALIE 结构 g[j] = s[j] * |mu[j] + z * sigma[j]|；
    - 需要的零放置在良性"最常为零"的维度（结构零），幅度代价趋近于零；
    - 残差计数修复用取反（保幅值），且只翻转 |mu[j]*g[j]| 最小的维度
      （对余弦相似度损失最小）；
    - 对齐后范数裁剪到不超过良性中位数范数 M（与 SignGuard /
      ClippedClipping 的服务端裁剪等价，方向不变，符号计数不变）；
    - z 的二分搜索在【对齐 + 裁剪后的最终向量】上模拟 DnC（联合验证）。
    """

    def on_trainer_init(self, trainer: Trainer):
        super().on_trainer_init(trainer)

        self.num_clients = trainer.config.num_clients
        self.num_byzantine = len(self.clients)

        # DnC 参数（模拟防御者的参数）
        self.dnc_sub_dim = 10000  # 子维度
        self.dnc_num_iters = 5    # 迭代次数
        self.dnc_filter_frac = 1.0  # 过滤比例
        self.dnc_sim_repeats = 2  # 每个候选 z 的独立模拟次数（鲁棒于子采样随机性）

    # ------------------------------------------------------------------
    # 良性统计：平均符号计数 + 每维共识模板 + 每维零频率
    # ------------------------------------------------------------------
    def _benign_sign_stats(self, benign_updates: torch.Tensor):
        """
        返回:
            (avg_p, avg_z, avg_n): 平均符号计数（SignGuard 的聚类特征）
            template: 每维多数符号 +1/-1/0（共识模板）
            zero_freq: 每维良性客户端为零的频率（结构零定位）
        """
        num_benign, d = benign_updates.shape
        pos_mask = benign_updates > 0
        neg_mask = benign_updates < 0
        zero_mask = benign_updates == 0

        # 平均符号计数（与 AverageSign 一致）
        avg_p = int(pos_mask.sum(dim=1).float().mean().round().item())
        avg_z = int(zero_mask.sum(dim=1).float().mean().round().item())
        avg_n = int(neg_mask.sum(dim=1).float().mean().round().item())
        while avg_p + avg_z + avg_n < d:
            avg_p += 1
        while avg_p + avg_z + avg_n > d:
            if avg_n > 0:
                avg_n -= 1
            elif avg_p > 0:
                avg_p -= 1
            else:
                avg_z -= 1

        # 每维多数符号（共识模板）: +1 / -1 / 0
        pos_count = pos_mask.sum(dim=0)
        neg_count = neg_mask.sum(dim=0)
        zero_count = zero_mask.sum(dim=0)
        template = torch.zeros(d, dtype=torch.int8, device=benign_updates.device)
        template[pos_count > neg_count] = 1
        template[neg_count > pos_count] = -1
        # 多数良性为零的维度（死单元）→ 模板为零
        template[zero_count > torch.maximum(pos_count, neg_count)] = 0

        zero_freq = zero_count.float() / num_benign
        return (avg_p, avg_z, avg_n), template, zero_freq

    # ------------------------------------------------------------------
    # 幅值保留的符号对齐
    # ------------------------------------------------------------------
    def _align_signs(self, g, benign_updates, template, zero_freq, avg_counts):
        """
        对齐 g 的符号统计到良性平均 (avg_p, avg_z, avg_n)，
        全程只用取反（保幅值）与结构零放置（近零幅度代价）。
        """
        mu = benign_updates.mean(dim=0)
        avg_p, avg_z, avg_n = avg_counts
        g = g.clone()

        # 1. 结构零放置：零放在良性最常为零的维度（平局取 |g| 小的维度）
        need_zero = avg_z - int((g == 0).sum().item())
        if need_zero > 0:
            g_abs = g.abs()
            # 评分: 零频率降序，|g| 升序；已是零的维度排除（哨兵 -3.0）
            score = zero_freq * 2.0 - g_abs / (g_abs.max() + 1e-12)
            score = torch.where(g == 0, torch.full_like(score, -3.0), score)
            zero_dims = torch.argsort(score, descending=True)[:need_zero]
            g[zero_dims] = 0.0

        # 2. 共识对齐：模板非零的维度强制取多数符号（取反保幅值）
        flip = (torch.sign(g) != template) & (template != 0) & (g != 0)
        g[flip] = -g[flip]

        # 3. 残差计数修复：只翻转 |mu[j]*g[j]| 最小的维度（余弦损失最小）。
        #    由于 cur_p + cur_n + cur_z = d = avg_p + avg_z + avg_n 且
        #    cur_z == avg_z，p/n 缺口互补，单次修复即可闭合。
        cur_p = int((g > 0).sum().item())
        cur_n = int((g < 0).sum().item())
        if cur_p > avg_p:
            excess = cur_p - avg_p
            pos_dims = torch.where(g > 0)[0]
            cost = mu[pos_dims].abs() * g[pos_dims].abs()
            flip_dims = pos_dims[torch.argsort(cost)[:excess]]
            g[flip_dims] = -g[flip_dims]
        elif cur_n > avg_n:
            excess = cur_n - avg_n
            neg_dims = torch.where(g < 0)[0]
            cost = mu[neg_dims].abs() * g[neg_dims].abs()
            flip_dims = neg_dims[torch.argsort(cost)[:excess]]
            g[flip_dims] = -g[flip_dims]

        return g

    def _build_candidate(self, z: float, benign_updates: torch.Tensor, stats) -> torch.Tensor:
        """构造【最终上传形态】的恶意更新：符号对齐 + 范数裁剪。"""
        mu = benign_updates.mean(dim=0)
        sigma = benign_updates.std(dim=0)
        avg_counts, template, zero_freq = stats

        g = mu + sigma * z
        g = self._align_signs(g, benign_updates, template, zero_freq, avg_counts)

        # 范数裁剪：不超过良性中位数范数 M。SignGuard / ClippedClipping 会把
        # 超过 M 的更新裁剪到 M（方向不变），预先裁剪等价，且保证 DnC 模拟的
        # 正是防御者实际看到的向量。符号计数对正缩放不变，裁剪不破坏符号伪装。
        # 注意：不做向上缩放——放大低范数向量（纯信号、无噪声结构）反而使其
        # 在谱空间成为离群点。
        M = benign_updates.norm(dim=1).median()
        g_norm = torch.norm(g)
        if g_norm > M:
            g = g * (M / g_norm)
        return g

    # ------------------------------------------------------------------
    # DnC 模拟（与 SPEAR 相同，修复了 randperm 的 device 问题）
    # ------------------------------------------------------------------
    def simulate_dnc_filtering(self, all_updates: torch.Tensor, benign_mask: torch.Tensor) -> torch.Tensor:
        """
        模拟 DnC 防御，返回每轮被认为"可信"的客户端索引
        """
        device = all_updates.device
        n_clients, d = all_updates.shape
        num_keep = n_clients - int(self.dnc_filter_frac * self.num_byzantine)

        all_benign_ids = []

        for iter_idx in range(self.dnc_num_iters):
            # 1. 随机选子维度
            indices = torch.randperm(d, device=device)[:self.dnc_sub_dim]
            sub_updates = all_updates[:, indices]

            # 2. 中心化
            mu = sub_updates.mean(dim=0)
            centered = sub_updates - mu

            # 3. SVD 找主方向
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                v = vh[0, :]  # 第一右奇异向量
            except Exception:
                v = torch.randn(self.dnc_sub_dim, device=device)
                v = v / v.norm()

            # 4. 计算投影距离
            projections = torch.matmul(centered, v)  # (n_clients,)
            s_values = projections ** 2

            # 5. 排序，保留距离最小的 num_keep 个
            sorted_indices = torch.argsort(s_values)
            benign_ids = sorted_indices[:num_keep]
            all_benign_ids.append(benign_ids)

        # 6. 取交集（DnC 的核心）
        intersection = set(all_benign_ids[0].cpu().numpy())
        for ids in all_benign_ids[1:]:
            intersection &= set(ids.cpu().numpy())

        return torch.tensor(list(intersection), device=device)

    # ------------------------------------------------------------------
    # 联合谱验证的 z 搜索：在最终上传向量上模拟 DnC
    # ------------------------------------------------------------------
    def find_adaptive_z_max(self, benign_updates: torch.Tensor, stats) -> float:
        """
        找到最优的 z_max：候选恶意更新先做符号对齐 + 范数裁剪，
        再用最终形态模拟 DnC，保证上传向量本身通过谱约束。
        DnC 的子维度采样是随机的，每个候选 z 用 dnc_sim_repeats 次独立
        模拟的最差结果判定，使选出的 z 对子采样随机性鲁棒。
        """
        n_benign = benign_updates.shape[0]
        n_total = self.num_clients

        z_low, z_high = 0.0, 5.0
        best_z = 0.0  # 保守回退：对齐后的近良性向量

        for _ in range(10):
            z_mid = (z_low + z_high) / 2

            # 构造最终形态的候选恶意更新（对齐 + 裁剪）
            candidate = self._build_candidate(z_mid, benign_updates, stats)

            all_updates = torch.cat([
                benign_updates,
                candidate.unsqueeze(0).repeat(self.num_byzantine, 1)
            ], dim=0)

            malicious_indices = set(range(n_benign, n_total))

            # 多次独立模拟取最差结果（子采样随机性鲁棒）
            num_malicious_passed = self.num_byzantine
            for _ in range(self.dnc_sim_repeats):
                trusted_indices = self.simulate_dnc_filtering(all_updates, None)
                trusted_set = set(trusted_indices.cpu().numpy())
                num_malicious_passed = min(
                    num_malicious_passed,
                    len(malicious_indices & trusted_set),
                )
                if num_malicious_passed == 0:
                    break

            if num_malicious_passed == self.num_byzantine:
                # 全部通过，z 可以更大（更攻击性）
                best_z = z_mid
                z_low = z_mid
            elif num_malicious_passed == 0:
                # 全部没通过，z 需要更小
                z_high = z_mid
            else:
                # 部分通过，调整策略
                if num_malicious_passed > self.num_byzantine / 2:
                    z_low = z_mid
                else:
                    z_high = z_mid

        return best_z

    def on_local_round_end(self, trainer: Algorithm):
        # 1. 获取良性更新与其符号统计
        benign_updates = self.get_benign_updates(trainer)
        stats = self._benign_sign_stats(benign_updates)

        # 2. 联合谱验证的 z 搜索（在对齐+缩放后的最终向量上模拟 DnC）
        adaptive_z = self.find_adaptive_z_max(benign_updates, stats)

        # 3. 生成最终攻击更新（与搜索时验证的向量完全一致）
        update = self._build_candidate(adaptive_z, benign_updates, stats)

        # 4. 应用到所有恶意客户端
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update.clone()

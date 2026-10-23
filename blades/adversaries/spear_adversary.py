import torch
from blades.constants import CLIENT_UPDATE, CLIENT_ID
from blades.trainers import Trainer as Algorithm
from blades.trainers import Trainer
from .adversary import Adversary
from ._spear_search import _AdaptiveZSearchMixin

class SPEARAdversary(_AdaptiveZSearchMixin, Adversary):
    def on_trainer_init(self, trainer: Trainer):
        super().on_trainer_init(trainer)

        self.num_clients = trainer.config.num_clients
        self.num_byzantine = len(self.clients)

        # DnC 参数（模拟防御者的参数）
        self.dnc_sub_dim = 10000  # 子维度
        self.dnc_num_iters = 5    # 迭代次数
        self.dnc_filter_frac = 1.0  # 过滤比例

        # 计算标准 ALIE 的 z_max（作为参考）
        s = torch.floor_divide(self.num_clients, 2) + 1 - self.num_byzantine
        cdf_value = (self.num_clients - self.num_byzantine - s) / (
            self.num_clients - self.num_byzantine
        )
        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
        # self.z_max_standard = dist.icdf(cdf_value)
        self.z_max_standard = dist.icdf(torch.tensor(cdf_value))

    def simulate_dnc_filtering(self, all_updates: torch.Tensor, benign_mask: torch.Tensor) -> torch.Tensor:
        """
        模拟 DnC 防御，返回每轮被认为"可信"的客户端索引
        目标：找到 DnC 的过滤边界
        """

        device = all_updates.device
        n_clients, d = all_updates.shape
        num_keep = n_clients - int(self.dnc_filter_frac * self.num_byzantine)

        # 记录每轮的可信索引
        all_benign_ids = []

        for iter_idx in range(self.dnc_num_iters):
            # 1. 随机选子维度
            # indices = torch.randperm(d)[:self.dnc_sub_dim]
            indices = torch.randperm(d, device=device)[:self.dnc_sub_dim]
            sub_updates = all_updates[:, indices]

            # 2. 中心化
            mu = sub_updates.mean(dim=0)
            centered = sub_updates - mu

            # 3. SVD 找主方向
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                v = vh[0, :]  # 第一右奇异向量
            except:
                # SVD 失败，用随机方向
                # v = torch.randn(self.dnc_sub_dim, device=all_updates.device)
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

        # return torch.tensor(list(intersection), device=all_updates.device)
        return torch.tensor(list(intersection), device=device)

    def on_local_round_end(self, trainer: Algorithm):
        # 1. 获取良性更新
        benign_updates = self.get_benign_updates(trainer)

        # 2. 计算标准统计量
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)

        # 3. 【关键】自适应调整 z_max 以绕过 DnC
        adaptive_z = self.find_adaptive_z_max(benign_updates)

        # 4. 生成攻击更新
        update = mean + std * adaptive_z

        # 5. 应用到所有恶意客户端
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update.clone()

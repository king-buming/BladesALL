"""Shared legacy SPEAR search; simulator implementations stay on each attack."""

import torch


class _AdaptiveZSearchMixin:
    """Search through the concrete attack's DnC simulator."""

    def find_adaptive_z_max(self, benign_updates: torch.Tensor) -> float:
        """
        找到最优的 z_max，使得恶意更新刚好通过 DnC 检测
        策略：二分搜索，找到第 (m-f) 位的边界
        """
        n_benign = benign_updates.shape[0]

        # 计算良性更新的统计量
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)

        # DnC 保留 n_benign 个更新；每次检查是否所有恶意更新均通过。

        n_total = self.num_clients
        # 二分搜索 z_max
        z_low, z_high = 0.0, 5.0  # 搜索范围
        best_z = self.z_max_standard.item()

        for _ in range(10):  # 10次迭代
            z_mid = (z_low + z_high) / 2

            # 构造候选恶意更新
            malicious_candidate = mean + std * z_mid

            # 构造完整的更新矩阵（良性 + 恶意）
            all_updates = torch.cat([
                benign_updates,
                malicious_candidate.unsqueeze(0).repeat(self.num_byzantine, 1)
            ], dim=0)

            # 模拟 DnC
            trusted_indices = self.simulate_dnc_filtering(all_updates, None)

            # 检查恶意客户端是否被包含
            malicious_indices = set(range(n_benign, n_total))
            trusted_set = set(trusted_indices.cpu().numpy())

            # 计算有多少恶意客户端通过了检测
            num_malicious_passed = len(malicious_indices & trusted_set)

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

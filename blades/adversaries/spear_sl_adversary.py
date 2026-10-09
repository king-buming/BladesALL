from typing import List
import torch
import random
from blades.constants import CLIENT_UPDATE, CLIENT_ID
from blades.trainers import Trainer as Algorithm
from blades.trainers import Trainer
from blades.aggregators import Signguard
from .adversary import Adversary

class SPEARSLAdversary(Adversary):
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
        self.z_max_standard = dist.icdf(cdf_value)

    def simulate_dnc_filtering(self, all_updates: torch.Tensor, benign_mask: torch.Tensor) -> torch.Tensor:
        """
        模拟 DnC 防御，返回每轮被认为"可信"的客户端索引
        目标：找到 DnC 的过滤边界
        """
        n_clients, d = all_updates.shape
        num_keep = n_clients - int(self.dnc_filter_frac * self.num_byzantine)
        
        # 记录每轮的可信索引
        all_benign_ids = []
        
        for iter_idx in range(self.dnc_num_iters):
            # 1. 随机选子维度
            indices = torch.randperm(d)[:self.dnc_sub_dim]
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
                v = torch.randn(self.dnc_sub_dim, device=all_updates.device)
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
        
        return torch.tensor(list(intersection), device=all_updates.device)

    def find_adaptive_z_max(self, benign_updates: torch.Tensor) -> float:
        """
        找到最优的 z_max，使得恶意更新刚好通过 DnC 检测
        策略：二分搜索，找到第 (m-f) 位的边界
        """
        n_benign = benign_updates.shape[0]
        d = benign_updates.shape[1]
        
        # 计算良性更新的统计量
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)
        
        # 创建候选的恶意更新（使用不同的 z）
        # 我们需要让恶意更新在 DnC 排序中，刚好排在第 (n_benign + self.num_byzantine - self.num_byzantine) = n_benign 位？
        # 不对，应该是：总客户端 n = n_benign + num_byzantine
        # DnC 保留 n - num_byzantine = n_benign 个
        # 所以恶意更新需要排在第 n_benign 位或更前
        
        n_total = self.num_clients
        target_rank = n_total - self.num_byzantine  # 刚好不被过滤的排名
        
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

    def _get_average_sign_stats(self, benign_updates):
        """
        计算每个良性客户端的正号、负号、零的比例，然后求平均
        """
        d = benign_updates.shape[1]
        total_p = 0
        total_z = 0
        total_n = 0
        num_benign = benign_updates.shape[0]
        
        for update in benign_updates:
            p = torch.sum(update > 0).item()
            z = torch.sum(update == 0).item()
            n = torch.sum(update < 0).item()
            total_p += p
            total_z += z
            total_n += n
        
        avg_p = int(total_p / num_benign)
        avg_z = int(total_z / num_benign)
        avg_n = int(total_n / num_benign)
        
        # 确保总和正确
        while avg_p + avg_z + avg_n != d:
            if avg_p + avg_z + avg_n < d:
                avg_p += 1
            else:
                if avg_p > 0:
                    avg_p -= 1
                elif avg_n > 0:
                    avg_n -= 1
                else:
                    avg_z -= 1
        
        return avg_p, avg_z, avg_n

    def _adjust_signs_comprehensive(self, g, ref_p, ref_z, ref_n):
        """
        综合调整正、零、负符号的梯度
        """
        g_final = g.clone()
        d = g_final.shape[0]
        
        # 获取当前的符号统计
        curr_p = torch.sum(g_final > 0).item()
        curr_z = torch.sum(g_final == 0).item()
        curr_n = torch.sum(g_final < 0).item()
        
        # 计算符号差距
        diff_p = curr_p - ref_p
        diff_z = curr_z - ref_z
        diff_n = curr_n - ref_n
        
        # 获取最小的正值和最大的负值（用于后续替换）
        positive_values = g_final[g_final > 0]
        negative_values = g_final[g_final < 0]
        
        min_positive = positive_values.min().item() if len(positive_values) > 0 else 0.01
        max_negative = negative_values.max().item() if len(negative_values) > 0 else -0.01
        
        # 第一步：处理正号过多的情况
        if diff_p > 0:
            # 首先看零是否需要增加
            if diff_z < 0:
                num_to_zero = min(diff_p, -diff_z)
                positive_indices = torch.where(g_final > 0)[0]
                sorted_indices = positive_indices[torch.argsort(g_final[positive_indices])]
                g_final[sorted_indices[:num_to_zero]] = 0.0
                diff_p -= num_to_zero
                diff_z += num_to_zero
            
            # 如果还有多余的正号，改为负号
            if diff_p > 0 and diff_n < 0:
                num_to_negative = min(diff_p, -diff_n)
                positive_indices = torch.where(g_final > 0)[0]
                sorted_indices = positive_indices[torch.argsort(g_final[positive_indices])]
                g_final[sorted_indices[:num_to_negative]] = max_negative
                diff_p -= num_to_negative
                diff_n += num_to_negative
        
        # 第二步：处理负号过多的情况
        if diff_n > 0:
            # 首先看零是否需要增加
            if diff_z < 0:
                num_to_zero = min(diff_n, -diff_z)
                negative_indices = torch.where(g_final < 0)[0]
                sorted_indices = negative_indices[torch.argsort(-g_final[negative_indices])]
                g_final[sorted_indices[:num_to_zero]] = 0.0
                diff_n -= num_to_zero
                diff_z += num_to_zero
            
            # 如果还有多余的负号，改为正号
            if diff_n > 0 and diff_p < 0:
                num_to_positive = min(diff_n, -diff_p)
                negative_indices = torch.where(g_final < 0)[0]
                sorted_indices = negative_indices[torch.argsort(-g_final[negative_indices])]
                g_final[sorted_indices[:num_to_positive]] = min_positive
                diff_n -= num_to_positive
                diff_p += num_to_positive
        
        # 第三步：处理零过多的情况
        if diff_z > 0:
            # 首先看正号是否需要增加
            if diff_p < 0:
                num_to_positive = min(diff_z, -diff_p)
                zero_indices = torch.where(g_final == 0)[0]
                # 随机选择零值位置修改为正号
                if len(zero_indices) > 0:
                    perm = torch.randperm(len(zero_indices))
                    selected_indices = zero_indices[perm[:num_to_positive]]
                    g_final[selected_indices] = min_positive
                    diff_z -= num_to_positive
                    diff_p += num_to_positive
            
            # 然后看负号是否需要增加
            if diff_z > 0 and diff_n < 0:
                num_to_negative = min(diff_z, -diff_n)
                zero_indices = torch.where(g_final == 0)[0]
                if len(zero_indices) > 0:
                    perm = torch.randperm(len(zero_indices))
                    selected_indices = zero_indices[perm[:num_to_negative]]
                    g_final[selected_indices] = max_negative
                    diff_z -= num_to_negative
                    diff_n += num_to_negative
        
        return g_final

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
        
        # 4.1 新的符号检测模块：基于参考良性客户端符号比例进行修改
        # 计算参考良性客户端符号比例（平均）
        ref_p, ref_z, ref_n = self._get_average_sign_stats(benign_updates)
        
        # 综合调整梯度符号
        update = self._adjust_signs_comprehensive(update, ref_p, ref_z, ref_n)
        
        # 5. 应用到所有恶意客户端
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update.clone()
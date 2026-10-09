from typing import List
import torch
import random
from blades.constants import CLIENT_UPDATE, CLIENT_ID
from blades.trainers import Trainer as Algorithm
from blades.trainers import Trainer
from blades.aggregators import Signguard
from .adversary import Adversary

class SPEARAdversary(Adversary):
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
        
# from typing import List
# import torch
# import random
# from blades.constants import CLIENT_UPDATE, CLIENT_ID
# from blades.trainers import Trainer as Algorithm
# from blades.trainers import Trainer
# from blades.aggregators import Signguard
# from .adversary import Adversary

# class SPEARAdversary(Adversary):
#     def on_trainer_init(self, trainer: Trainer):
#         super().on_trainer_init(trainer)
        
#         self.num_clients = trainer.config.num_clients
#         self.num_byzantine = len(self.clients)
        
#         # DnC 参数（模拟防御者的参数）
#         self.dnc_sub_dim = 10000  # 子维度
#         self.dnc_num_iters = 5    # 迭代次数
#         self.dnc_filter_frac = 1.0  # 过滤比例
        
#         # 计算标准 ALIE 的 z_max（作为参考）
#         s = torch.floor_divide(self.num_clients, 2) + 1 - self.num_byzantine
#         cdf_value = (self.num_clients - self.num_byzantine - s) / (
#             self.num_clients - self.num_byzantine
#         )
#         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
#         self.z_max_standard = dist.icdf(cdf_value)

#     def simulate_dnc_filtering(self, all_updates: torch.Tensor, benign_mask: torch.Tensor) -> torch.Tensor:
#         """
#         模拟 DnC 防御，返回每轮被认为"可信"的客户端索引
#         目标：找到 DnC 的过滤边界
#         """
#         n_clients, d = all_updates.shape
#         num_keep = n_clients - int(self.dnc_filter_frac * self.num_byzantine)
        
#         # 记录每轮的可信索引
#         all_benign_ids = []
        
#         for iter_idx in range(self.dnc_num_iters):
#             # 1. 随机选子维度
#             indices = torch.randperm(d)[:self.dnc_sub_dim]
#             sub_updates = all_updates[:, indices]
            
#             # 2. 中心化
#             mu = sub_updates.mean(dim=0)
#             centered = sub_updates - mu
            
#             # 3. SVD 找主方向
#             try:
#                 _, _, vh = torch.linalg.svd(centered, full_matrices=False)
#                 v = vh[0, :]  # 第一右奇异向量
#             except:
#                 # SVD 失败，用随机方向
#                 v = torch.randn(self.dnc_sub_dim, device=all_updates.device)
#                 v = v / v.norm()
            
#             # 4. 计算投影距离
#             projections = torch.matmul(centered, v)  # (n_clients,)
#             s_values = projections ** 2
            
#             # 5. 排序，保留距离最小的 num_keep 个
#             sorted_indices = torch.argsort(s_values)
#             benign_ids = sorted_indices[:num_keep]
#             all_benign_ids.append(benign_ids)
        
#         # 6. 取交集（DnC 的核心）
#         intersection = set(all_benign_ids[0].cpu().numpy())
#         for ids in all_benign_ids[1:]:
#             intersection &= set(ids.cpu().numpy())
        
#         return torch.tensor(list(intersection), device=all_updates.device)

#     def find_adaptive_z_max(self, benign_updates: torch.Tensor) -> float:
#         """
#         找到最优的 z_max，使得恶意更新刚好通过 DnC 检测
#         策略：二分搜索，找到第 (m-f) 位的边界
#         """
#         n_benign = benign_updates.shape[0]
#         d = benign_updates.shape[1]
        
#         # 计算良性更新的统计量
#         mean = benign_updates.mean(dim=0)
#         std = benign_updates.std(dim=0)
        
#         # 创建候选的恶意更新（使用不同的 z）
#         # 我们需要让恶意更新在 DnC 排序中，刚好排在第 (n_benign + self.num_byzantine - self.num_byzantine) = n_benign 位？
#         # 不对，应该是：总客户端 n = n_benign + num_byzantine
#         # DnC 保留 n - num_byzantine = n_benign 个
#         # 所以恶意更新需要排在第 n_benign 位或更前
        
#         n_total = self.num_clients
#         target_rank = n_total - self.num_byzantine  # 刚好不被过滤的排名
        
#         # 二分搜索 z_max
#         z_low, z_high = 0.0, 5.0  # 搜索范围
#         best_z = self.z_max_standard.item()
        
#         for _ in range(10):  # 10次迭代
#             z_mid = (z_low + z_high) / 2
            
#             # 构造候选恶意更新
#             malicious_candidate = mean + std * z_mid
            
#             # 构造完整的更新矩阵（良性 + 恶意）
#             all_updates = torch.cat([
#                 benign_updates,
#                 malicious_candidate.unsqueeze(0).repeat(self.num_byzantine, 1)
#             ], dim=0)
            
#             # 模拟 DnC
#             trusted_indices = self.simulate_dnc_filtering(all_updates, None)
            
#             # 检查恶意客户端是否被包含
#             malicious_indices = set(range(n_benign, n_total))
#             trusted_set = set(trusted_indices.cpu().numpy())
            
#             # 计算有多少恶意客户端通过了检测
#             num_malicious_passed = len(malicious_indices & trusted_set)
            
#             if num_malicious_passed == self.num_byzantine:
#                 # 全部通过，z 可以更大（更攻击性）
#                 best_z = z_mid
#                 z_low = z_mid
#             elif num_malicious_passed == 0:
#                 # 全部没通过，z 需要更小
#                 z_high = z_mid
#             else:
#                 # 部分通过，调整策略
#                 if num_malicious_passed > self.num_byzantine / 2:
#                     z_low = z_mid
#                 else:
#                     z_high = z_mid
        
#         return best_z

#     def _get_average_sign_stats(self, benign_updates):
#         """
#         计算每个良性客户端的正号、负号、零的比例，然后求平均
#         """
#         d = benign_updates.shape[1]
#         total_p = 0
#         total_z = 0
#         total_n = 0
#         num_benign = benign_updates.shape[0]
        
#         for update in benign_updates:
#             p = torch.sum(update > 0).item()
#             z = torch.sum(update == 0).item()
#             n = torch.sum(update < 0).item()
#             total_p += p
#             total_z += z
#             total_n += n
        
#         avg_p = int(total_p / num_benign)
#         avg_z = int(total_z / num_benign)
#         avg_n = int(total_n / num_benign)
        
#         # 确保总和正确
#         while avg_p + avg_z + avg_n != d:
#             if avg_p + avg_z + avg_n < d:
#                 avg_p += 1
#             else:
#                 if avg_p > 0:
#                     avg_p -= 1
#                 elif avg_n > 0:
#                     avg_n -= 1
#                 else:
#                     avg_z -= 1
        
#         return avg_p, avg_z, avg_n

#     def _adjust_signs_comprehensive(self, g, ref_p, ref_z, ref_n):
#         """
#         综合调整正、零、负符号的梯度
#         """
#         g_final = g.clone()
#         d = g_final.shape[0]
        
#         # 获取当前的符号统计
#         curr_p = torch.sum(g_final > 0).item()
#         curr_z = torch.sum(g_final == 0).item()
#         curr_n = torch.sum(g_final < 0).item()
        
#         # 计算符号差距
#         diff_p = curr_p - ref_p
#         diff_z = curr_z - ref_z
#         diff_n = curr_n - ref_n
        
#         # 获取最小的正值和最大的负值（用于后续替换）
#         positive_values = g_final[g_final > 0]
#         negative_values = g_final[g_final < 0]
        
#         min_positive = positive_values.min().item() if len(positive_values) > 0 else 0.01
#         max_negative = negative_values.max().item() if len(negative_values) > 0 else -0.01
        
#         # 第一步：处理正号过多的情况
#         if diff_p > 0:
#             # 首先看零是否需要增加
#             if diff_z < 0:
#                 num_to_zero = min(diff_p, -diff_z)
#                 positive_indices = torch.where(g_final > 0)[0]
#                 sorted_indices = positive_indices[torch.argsort(g_final[positive_indices])]
#                 g_final[sorted_indices[:num_to_zero]] = 0.0
#                 diff_p -= num_to_zero
#                 diff_z += num_to_zero
            
#             # 如果还有多余的正号，改为负号
#             if diff_p > 0 and diff_n < 0:
#                 num_to_negative = min(diff_p, -diff_n)
#                 positive_indices = torch.where(g_final > 0)[0]
#                 sorted_indices = positive_indices[torch.argsort(g_final[positive_indices])]
#                 g_final[sorted_indices[:num_to_negative]] = max_negative
#                 diff_p -= num_to_negative
#                 diff_n += num_to_negative
        
#         # 第二步：处理负号过多的情况
#         if diff_n > 0:
#             # 首先看零是否需要增加
#             if diff_z < 0:
#                 num_to_zero = min(diff_n, -diff_z)
#                 negative_indices = torch.where(g_final < 0)[0]
#                 sorted_indices = negative_indices[torch.argsort(-g_final[negative_indices])]
#                 g_final[sorted_indices[:num_to_zero]] = 0.0
#                 diff_n -= num_to_zero
#                 diff_z += num_to_zero
            
#             # 如果还有多余的负号，改为正号
#             if diff_n > 0 and diff_p < 0:
#                 num_to_positive = min(diff_n, -diff_p)
#                 negative_indices = torch.where(g_final < 0)[0]
#                 sorted_indices = negative_indices[torch.argsort(-g_final[negative_indices])]
#                 g_final[sorted_indices[:num_to_positive]] = min_positive
#                 diff_n -= num_to_positive
#                 diff_p += num_to_positive
        
#         # 第三步：处理零过多的情况
#         if diff_z > 0:
#             # 首先看正号是否需要增加
#             if diff_p < 0:
#                 num_to_positive = min(diff_z, -diff_p)
#                 zero_indices = torch.where(g_final == 0)[0]
#                 # 随机选择零值位置修改为正号
#                 if len(zero_indices) > 0:
#                     perm = torch.randperm(len(zero_indices))
#                     selected_indices = zero_indices[perm[:num_to_positive]]
#                     g_final[selected_indices] = min_positive
#                     diff_z -= num_to_positive
#                     diff_p += num_to_positive
            
#             # 然后看负号是否需要增加
#             if diff_z > 0 and diff_n < 0:
#                 num_to_negative = min(diff_z, -diff_n)
#                 zero_indices = torch.where(g_final == 0)[0]
#                 if len(zero_indices) > 0:
#                     perm = torch.randperm(len(zero_indices))
#                     selected_indices = zero_indices[perm[:num_to_negative]]
#                     g_final[selected_indices] = max_negative
#                     diff_z -= num_to_negative
#                     diff_n += num_to_negative
        
#         return g_final

#     def on_local_round_end(self, trainer: Algorithm):
#         # 1. 获取良性更新
#         benign_updates = self.get_benign_updates(trainer)
        
#         # 2. 计算标准统计量
#         mean = benign_updates.mean(dim=0)
#         std = benign_updates.std(dim=0)
        
#         # 3. 【关键】自适应调整 z_max 以绕过 DnC
#         adaptive_z = self.find_adaptive_z_max(benign_updates)
        
#         # 4. 生成攻击更新
#         update = mean + std * adaptive_z
        
#         # 4.1 新的符号检测模块：基于参考良性客户端符号比例进行修改
#         # 计算参考良性客户端符号比例（平均）
#         ref_p, ref_z, ref_n = self._get_average_sign_stats(benign_updates)
        
#         # 综合调整梯度符号
#         update = self._adjust_signs_comprehensive(update, ref_p, ref_z, ref_n)
        
#         # 5. 应用到所有恶意客户端
#         for result in trainer.local_results:
#             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 result[CLIENT_UPDATE] = update.clone()










# # import random

# # import torch

# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard

# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)
        
# #         # 计算良性客户端数量
# #         num_benign = self.num_clients - num_byzantine
# #         # 自由度设置为：良性客户端总数 - 1
# #         self.df = num_benign - 1
        
# #         # 原有的cdf_value计算逻辑保持不变
# #         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
# #         cdf_value = (self.num_clients - num_byzantine - s) / (
# #             self.num_clients - num_byzantine
# #         )
        
# #         # 关键修改：将正态分布替换为t分布（StudentT），传入自由度df
# #         dist = torch.distributions.studentT.StudentT(df=self.df, loc=torch.tensor(0.0), scale=torch.tensor(1.0))
# #         self.z_max = dist.icdf(cdf_value)

# #         self.negative_indices = None

# #     def on_local_round_end(self, trainer: Algorithm):
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)
# #         std = benign_updates.std(dim=0)
# #         update = mean + std * self.z_max
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = update


# # import random
# # import torch
# # import scipy.stats as stats
# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from typing import Dict, Tuple
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
# #         cdf_value = (self.num_clients - num_byzantine - s) / (
# #             self.num_clients - num_byzantine
# #         )
# #         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
# #         self.z_max = dist.icdf(cdf_value)

# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         # 遍历模型所有可训练参数，记录每个参数的切片范围
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()  # 参数元素总数
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def _get_orthogonal_vector(self, v: torch.Tensor, device: torch.device) -> torch.Tensor:
# #         """
# #         生成与输入向量v垂直的单位向量
# #         :param v: 输入向量（一维Tensor）
# #         :param device: 设备（cpu/cuda）
# #         :return: 与v垂直的单位向量（维度与v一致）
# #         """
# #         if v.ndim != 1:
# #             raise ValueError("输入向量必须是一维Tensor")
        
# #         # 情况1：v是零向量 → 直接返回随机单位向量
# #         if torch.allclose(v, torch.zeros_like(v)):
# #             random_vec = torch.randn_like(v, device=device)
# #             return random_vec / torch.norm(random_vec)
        
# #         # 情况2：非零向量 → 构造正交向量
# #         # 1. 随机生成初始向量
# #         random_vec = torch.randn_like(v, device=device)
# #         # 2. 计算random_vec在v上的投影：proj = (random_vec · v / ||v||²) * v
# #         dot_product = torch.dot(random_vec, v)
# #         v_norm_sq = torch.dot(v, v)
# #         projection = (dot_product / v_norm_sq) * v
# #         # 3. 减去投影得到正交向量
# #         orthogonal_vec = random_vec - projection
# #         # 4. 归一化（避免零向量）
# #         orthogonal_norm = torch.norm(orthogonal_vec)
# #         if orthogonal_norm < 1e-8:  # 极端情况：随机向量与v共线 → 重新生成
# #             return self._get_orthogonal_vector(v, device)
        
# #         return orthogonal_vec / orthogonal_norm

# #     def on_local_round_end(self, trainer: Algorithm):
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)  # 展平的均值向量
# #         std = benign_updates.std(dim=0)    # 展平的标准差向量
# #         device = mean.device  # 获取设备（保证张量在同一设备）

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 2. 保留SignGuard的负向索引逻辑（原逻辑不变，可选启用）
# #         if isinstance(trainer.server.aggregator, Signguard):
# #             if self.negative_indices is None:
# #                 num_elements = len(std)
# #                 num_negate = num_elements // 2
# #                 self.negative_indices = random.sample(range(num_negate), num_negate)
# #             std[self.negative_indices] *= -1

# #         # 3. 初始化最终更新向量（默认所有层用mean）
# #         final_update = mean.clone()

# #         # 4. 仅对linear层（最后一层全连接层）生成垂直向量攻击
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             # 匹配linear层的所有参数（linear.weight / linear.bias）
# #             if param_name.startswith(self.target_fc_prefix):
# #                 # 提取当前层的良性梯度均值
# #                 mean_layer = mean[param_slice]
# #                 # 生成与mean_layer垂直的单位向量
# #                 ortho_vec = self._get_orthogonal_vector(mean_layer, device)
                
# #                 # 可选：缩放垂直向量（匹配原梯度的幅值，增强攻击效果）
# #                 # 方式1：用当前层梯度的标准差缩放
# #                 std_layer = std[param_slice]
# #                 ortho_vec = ortho_vec * std_layer.mean()  # 用标准差均值缩放
# #                 # 方式2：用原均值向量的幅值缩放（等价强度）
# #                 # ortho_vec = ortho_vec * torch.norm(mean_layer)
                
# #                 # 替换目标层的更新为垂直向量
# #                 final_update[param_slice] = ortho_vec

# #         # 5. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update


# # #84 83 84
# # import torch
# # from blades.trainers import Trainer as Algorithm
# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from .adversary import Adversary
# # from typing import Dict, Tuple, List


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Algorithm):
# #         super().on_trainer_init(trainer)

# #         self.num_clients = trainer.config.num_clients
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（根据实际模型调整）
# #         self.target_fc_prefix = "linear"

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def _scaling_attack_on_layer(self, mean: torch.Tensor, benign_updates: torch.Tensor,
# #                                   layer_slices: List[slice], threshold: float,
# #                                   gamma_init: float = 10.0, omega: float = 0.001) -> torch.Tensor:
# #         """
# #         对指定层（最后一层）进行缩放攻击，使用与ScaleSign相同的步长递减搜索策略。
# #         参数：
# #             mean: 良性更新的均值向量（整个模型展平）
# #             benign_updates: 良性更新集合 (m, d)
# #             layer_slices: 最后一层参数对应的切片列表（可能有多个参数，如weight和bias）
# #             threshold: 距离约束阈值（良性更新之间的最大距离）
# #             gamma_init: 初始缩放因子
# #             omega: 步长终止阈值
# #         返回：
# #             满足阈值约束的整体更新向量
# #         """
# #         # 提取最后一层所有参数的均值部分，并拼接成一维向量
# #         mean_layer_parts = [mean[slc] for slc in layer_slices]
# #         mean_layer = torch.cat(mean_layer_parts)

# #         # 根据符号确定正负梯度索引
# #         theta1 = torch.where(mean_layer > 0)[0]
# #         theta2 = torch.where(mean_layer < 0)[0]

# #         # 如果没有正或负梯度，则无法进行缩放攻击，直接返回原均值
# #         if len(theta1) == 0 or len(theta2) == 0:
# #             return mean.clone()

# #         gamma = gamma_init
# #         step = (gamma - 1) / 2  # 初始步长

# #         # 当前最佳更新（默认均值）
# #         best_update = mean.clone()

# #         # 循环搜索，直到步长小于等于omega
# #         while step > omega:
# #             # 构造缩放向量phi_layer
# #             phi_layer = torch.ones_like(mean_layer)
# #             phi_layer[theta1] = gamma
# #             phi_layer[theta2] = 1.0 / gamma

# #             # 生成临时更新：其他层保持mean，最后一层缩放
# #             temp_update = mean.clone()
# #             start = 0
# #             for slc in layer_slices:
# #                 length = slc.stop - slc.start
# #                 temp_update[slc] = mean_layer[start:start+length] * phi_layer[start:start+length]
# #                 start += length

# #             # 计算与所有良性更新的最大距离
# #             dists = torch.cdist(temp_update.unsqueeze(0), benign_updates, p=2).squeeze(0)
# #             max_dist = dists.max().item()

# #             if max_dist <= threshold:
# #                 # 满足约束时增大gamma以增强攻击
# #                 gamma += step / 2
# #                 best_update = temp_update.clone()  # 记录当前满足条件的更新
# #             else:
# #                 # 不满足约束时减小gamma
# #                 gamma -= step / 2

# #             step = step / 2  # 步长减半

# #         # 返回最后一次满足条件的更新（若从未满足，则best_update为mean）
# #         return best_update

# #     def on_local_round_end(self, trainer: Algorithm):
# #         # 获取良性客户端的更新（形状：[num_benign, dim]）
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)

# #         # 首次运行时解析模型参数层信息
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 收集最后一层所有参数的切片
# #         layer_slices = []
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             if param_name.startswith(self.target_fc_prefix):
# #                 layer_slices.append(param_slice)

# #         # 如果没有找到目标层，则回退到使用均值（即不攻击）
# #         if not layer_slices:
# #             final_update = mean.clone()
# #         else:
# #             # 计算良性更新之间的最大距离作为阈值
# #             threshold = torch.cdist(benign_updates, benign_updates, p=2).max().item()
# #             # 应用缩放攻击生成满足阈值约束的恶意更新
# #             final_update = self._scaling_attack_on_layer(mean, benign_updates, layer_slices, threshold)

# #         # 将最终恶意更新赋给所有恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update


# # # #50%
# # import random
# # import torch
# # import scipy.stats as stats  # 导入scipy统计库
# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from typing import Dict, Tuple  # 核心缺失的导入
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
# #         cdf_value = (self.num_clients - num_byzantine - s) / (
# #             self.num_clients - num_byzantine
# #         )
# #         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
# #         self.z_max = dist.icdf(cdf_value)

# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         # 遍历模型所有可训练参数，记录每个参数的切片范围
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()  # 参数元素总数
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def on_local_round_end(self, trainer: Algorithm):
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)  # 展平的均值向量
# #         std = benign_updates.std(dim=0)    # 展平的标准差向量

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 2. 保留SignGuard的负向索引逻辑（原逻辑不变）
# #         if isinstance(trainer.server.aggregator, Signguard):
# #             if self.negative_indices is None:
# #                 num_elements = len(std)
# #                 num_negate = num_elements // 2
# #                 self.negative_indices = random.sample(range(num_negate), num_negate)
# #             std[self.negative_indices] *= -1

# #         # 3. 初始化最终更新向量（默认所有层用mean）
# #         final_update = mean.clone()

# #         # 4. 仅对linear层（最后一层全连接层）应用ALIE攻击公式
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             # 匹配linear层的所有参数（linear.weight / linear.bias）
# #             if param_name.startswith(self.target_fc_prefix):
# #                 # final_update[param_slice] = mean[param_slice] + std[param_slice] * self.z_max
# #                 # final_update[param_slice] = mean[param_slice] + std[param_slice] * 1
# #                 # final_update[param_slice] = -mean[param_slice] - std[param_slice] * 1
# #                 final_update[param_slice] = -mean[param_slice] * 3

# #         # 5. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update




# # import random
# # import torch
# # import scipy.stats as stats  # 导入scipy统计库
# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from typing import Dict, Tuple, List  # 补充必要导入
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
# #         cdf_value = (self.num_clients - num_byzantine - s) / (
# #             self.num_clients - num_byzantine
# #         )
# #         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
# #         self.z_max = dist.icdf(cdf_value)

# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         # 遍历模型所有可训练参数，记录每个参数的切片范围
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()  # 参数元素总数
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info
    
# #     def _get_median_l2_update(self, benign_updates: torch.Tensor) -> torch.Tensor:
# #         """
# #         计算良性梯度中L2范数为中位数的更新向量
# #         Args:
# #             benign_updates: 良性客户端的更新张量，形状为 [num_benign, num_params]
# #         Returns:
# #             median_l2_update: L2范数中位数对应的更新向量，形状为 [num_params]
# #         """
# #         # 计算每个良性客户端更新的L2范数
# #         l2_norms = torch.norm(benign_updates, p=2, dim=1)
# #         # 找到L2范数的中位数
# #         median_norm = torch.median(l2_norms)
# #         # 找到最接近中位数的更新向量（若有多个，取第一个）
# #         median_idx = torch.argmin(torch.abs(l2_norms - median_norm))
# #         median_l2_update = benign_updates[median_idx].clone()
        
# #         return median_l2_update

# #     def on_local_round_end(self, trainer: Algorithm):
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)  # 展平的均值向量
# #         std = benign_updates.std(dim=0)    # 展平的标准差向量

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)
        
# #         # 2. 计算良性梯度中L2范数为中位数的更新向量
# #         median_l2_update = self._get_median_l2_update(benign_updates)

# #         # 4. 初始化最终更新向量（使用L2范数中位数的更新值）
# #         final_update = median_l2_update.clone()

# #         # 5. 仅对linear层（最后一层全连接层）应用ALIE攻击公式
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             # 匹配linear层的所有参数（linear.weight / linear.bias）
# #             if param_name.startswith(self.target_fc_prefix):
# #                 final_update[param_slice] = -mean[param_slice] 

# #         # 6. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update



# # import random
# # import torch
# # from typing import Dict, Tuple

# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         # 初始化参数
# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"
# #         # MinMax二分查找参数
# #         self.threshold_diff = 1e-4  # 二分查找精度
# #         self.high_init = 5.0         # 二分查找上界
# #         self.low_init = 0.0          # 二分查找下界

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def _get_linear_layer_slice(self) -> slice:
# #         """合并所有linear层参数的切片范围，返回整体slice"""
# #         linear_slices = []
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             if param_name.startswith(self.target_fc_prefix):
# #                 linear_slices.append(param_slice)
        
# #         if not linear_slices:
# #             raise ValueError(f"未找到前缀为 {self.target_fc_prefix} 的层参数")
        
# #         # 合并linear层所有参数的切片范围
# #         start = min(s.start for s in linear_slices)
# #         end = max(s.stop for s in linear_slices)
# #         return slice(start, end)

# #     def _get_median_l2_update(self, benign_updates: torch.Tensor) -> torch.Tensor:
# #         """计算良性梯度中L2范数为中位数的更新向量"""
# #         # 计算每个良性客户端更新的L2范数
# #         l2_norms = torch.norm(benign_updates, p=2, dim=1)
# #         # 找到L2范数的中位数
# #         median_norm = torch.median(l2_norms)
# #         # 找到最接近中位数的更新向量（若有多个，取第一个）
# #         median_idx = torch.argmin(torch.abs(l2_norms - median_norm))
# #         median_l2_update = benign_updates[median_idx].clone()
        
# #         return median_l2_update

# #     def _attack_by_binary_search(self, benign_updates: torch.Tensor, median_l2_update: torch.Tensor) -> Tuple[torch.Tensor, float]:
# #         """
# #         严格按你的逻辑实现MinMax二分查找：
# #         1. mal_update = median_l2_update（非全连接层） + mean - mid*std（全连接层）
# #         2. 用完整mal_update计算loss来更新mid
# #         """
# #         # 1. 基础计算
# #         mean_grads = benign_updates.mean(dim=0)
# #         deviation = benign_updates.std(dim=0)
# #         threshold = torch.cdist(benign_updates, benign_updates, p=2).max()
# #         linear_slice = self._get_linear_layer_slice()

# #         # 2. 二分查找最优mid值
# #         low = self.low_init
# #         high = self.high_init
# #         mid = (low + high) / 2
# #         best_mid = mid  # 记录最优mid值

# #         while abs(high - low) > self.threshold_diff:
# #             mid = (low + high) / 2
# #             # 构造当前mid对应的mal_update（核心：非全连接层用median，全连接层用扰动值）
# #             mal_update = median_l2_update.clone()
# #             mal_update[linear_slice] = mean_grads[linear_slice] - mid * deviation[linear_slice]
            
# #             # 用完整的mal_update计算loss（和良性更新的最大L2距离）
# #             loss = torch.cdist(mal_update.unsqueeze(0), benign_updates, p=2).max()
            
# #             # 调整二分查找边界
# #             if loss < threshold:
# #                 low = mid
# #                 best_mid = mid  # 更新最优mid
# #             else:
# #                 high = mid

# #         # 3. 计算最终的全连接层扰动值（用最优mid）
# #         final_perturbed_linear = mean_grads[linear_slice] - best_mid * deviation[linear_slice]
# #         return final_perturbed_linear, best_mid

# #     def on_local_round_end(self, trainer: Algorithm):
# #         self.trainer = trainer
# #         benign_updates = self.get_benign_updates(trainer)

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 2. 步骤1：计算良性客户端L2范数中位数的更新向量（基准）
# #         median_l2_update = self._get_median_l2_update(benign_updates)

# #         # 3. 步骤2：执行MinMax二分查找（严格按你的逻辑构造mal_update）
# #         perturbed_linear_grad, mid = self._attack_by_binary_search(benign_updates, median_l2_update)

# #         # 4. 步骤3：构造最终恶意更新（和二分查找中的mal_update完全一致）
# #         final_update = median_l2_update.clone()
# #         linear_slice = self._get_linear_layer_slice()
# #         final_update[linear_slice] = perturbed_linear_grad

# #         # 5. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update


# # import random
# # import torch
# # from typing import Dict, Tuple

# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         # 初始化参数
# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"
# #         # MinMax二分查找参数（复用MinMaxAdversary的核心配置）
# #         self.threshold_diff = 0.01  # 二分查找精度
# #         self.high_init = 5.0         # 二分查找上界
# #         self.low_init = 0.0          # 二分查找下界

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def _get_linear_layer_slice(self) -> slice:
# #         """合并所有linear层参数的切片范围，返回整体slice"""
# #         linear_slices = []
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             if param_name.startswith(self.target_fc_prefix):
# #                 linear_slices.append(param_slice)
        
# #         if not linear_slices:
# #             raise ValueError(f"未找到前缀为 {self.target_fc_prefix} 的层参数")
        
# #         # 合并linear层所有参数的切片范围
# #         start = min(s.start for s in linear_slices)
# #         end = max(s.stop for s in linear_slices)
# #         return slice(start, end)

# #     def _compute_linear_layer_mid(self, benign_updates: torch.Tensor, linear_slice: slice) -> float:
# #         """仅基于linear层参数执行MinMax二分查找，计算最优mid值（替代原z_max）"""
# #         # 提取良性更新中仅linear层的参数
# #         benign_linear = benign_updates[:, linear_slice]
# #         # 计算linear层参数的均值和标准差
# #         mean_linear = benign_linear.mean(dim=0)
# #         std_linear = benign_linear.std(dim=0)
# #         # 计算benign客户端linear层参数的最大距离阈值（MinMax核心约束）
# #         threshold = torch.cdist(benign_linear, benign_linear, p=2).max()

# #         # 二分查找最优mid值（MinMax核心逻辑）
# #         low = self.low_init
# #         high = self.high_init
# #         mid = (low + high) / 2
# #         while abs(high - low) > self.threshold_diff:
# #             mid = (low + high) / 2
# #             # 构造仅linear层的恶意更新
# #             mal_update_linear = torch.stack([mean_linear - mid * std_linear])
# #             # 计算恶意更新与benign更新的最大距离
# #             loss = torch.cdist(mal_update_linear, benign_linear, p=2).max()
# #             # 调整二分查找边界
# #             if loss < threshold:
# #                 low = mid
# #             else:
# #                 high = mid
# #         return mid

# #     def on_local_round_end(self, trainer: Algorithm):
# #         self.trainer = trainer  # 保存trainer引用供内部方法使用
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)  # 全量梯度均值

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 2. 获取linear层的整体切片范围
# #         linear_slice = self._get_linear_layer_slice()

# #         # 3. 仅基于linear层参数计算MinMax的mid值（替代原z_max）
# #         mid = self._compute_linear_layer_mid(benign_updates, linear_slice)

# #         # 4. 初始化最终更新向量（默认所有层用全量均值）
# #         final_update = mean.clone()

# #         # 5. 仅对linear层参数应用扰动（mean + std * mid，复用ALIE原公式结构）
# #         mean_linear = mean[linear_slice]
# #         std_linear = benign_updates[:, linear_slice].std(dim=0)
# #         # 应用ALIE公式（mean + std * 扰动系数），此处用mid替代原z_max
# #         final_update[linear_slice] = mean_linear - std_linear * mid

# #         # 6. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update




# # 70%
# # import random
# # import torch
# # import scipy.stats as stats  # 导入scipy统计库
# # from blades.constants import CLIENT_UPDATE, CLIENT_ID
# # from blades.trainers import Trainer as Algorithm
# # from blades.trainers import Trainer
# # from blades.aggregators import Signguard
# # from typing import Dict, Tuple  # 核心缺失的导入
# # from .adversary import Adversary


# # class SPEARAdversary(Adversary):
# #     def on_trainer_init(self, trainer: Trainer):
# #         super().on_trainer_init(trainer)
        
# #         self.num_clients = trainer.config.num_clients
# #         num_byzantine = len(self.clients)

# #         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
# #         cdf_value = (self.num_clients - num_byzantine - s) / (
# #             self.num_clients - num_byzantine
# #         )
# #         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
# #         self.z_max = dist.icdf(cdf_value)

# #         self.negative_indices = None
# #         # 预存模型参数的层信息（名称、形状、切片范围）
# #         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
# #         # 标记最后一层全连接层的前缀（适配你的linear层）
# #         self.target_fc_prefix = "linear"
# #         # 修正：改为字典，key=参数名，value=该参数的符号张量
# #         self.fc_signs: Dict[str, torch.Tensor] = {}
# #         # 新增：标记是否是第一次攻击（全局）
# #         self.is_first_attack = True

# #     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
# #         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
# #         param_info = {}
# #         start_idx = 0
# #         # 遍历模型所有可训练参数，记录每个参数的切片范围
# #         for name, param in model.named_parameters():
# #             param_numel = param.numel()  # 参数元素总数
# #             end_idx = start_idx + param_numel
# #             param_info[name] = (param.size(), slice(start_idx, end_idx))
# #             start_idx = end_idx
# #         return param_info

# #     def on_local_round_end(self, trainer: Algorithm):
# #         benign_updates = self.get_benign_updates(trainer)
# #         mean = benign_updates.mean(dim=0)  # 展平的均值向量
# #         std = benign_updates.std(dim=0)    # 展平的标准差向量

# #         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
# #         if not self.model_param_info:
# #             global_model = trainer.server.get_global_model()
# #             self.model_param_info = self._get_model_param_slices(global_model)

# #         # 2. 初始化最终更新向量（默认所有层用mean）
# #         final_update = mean.clone()

# #         # 3. 仅对linear层（最后一层全连接层）应用ALIE攻击公式
# #         for param_name, (_, param_slice) in self.model_param_info.items():
# #             # 匹配linear层的所有参数（linear.weight / linear.bias）
# #             if param_name.startswith(self.target_fc_prefix):
# #                 # 提取当前参数对应的切片数据
# #                 mean_slice = mean[param_slice]
# #                 std_slice = std[param_slice]
# #                 # 计算当前轮次的攻击值（仅针对当前参数）
# #                 current_attack = mean_slice - std_slice * self.z_max
                
# #                 if self.is_first_attack:
# #                     # 首次攻击：按参数名保存符号
# #                     sign = torch.sign(current_attack)
# #                     sign[sign == 0] = 1  # 处理符号为0的情况
# #                     self.fc_signs[param_name] = sign  # 字典key为参数名
# #                     # 首次攻击直接使用当前计算值
# #                     final_update[param_slice] = current_attack
# #                 else:
# #                     # 非首次攻击：读取当前参数对应的符号
# #                     if param_name not in self.fc_signs:
# #                         raise ValueError(f"未找到参数 {param_name} 的首次攻击符号！")
# #                     # 复用符号，仅更新数值大小
# #                     current_abs = torch.abs(current_attack)
# #                     final_attack = current_abs * self.fc_signs[param_name]
# #                     final_update[param_slice] = final_attack

# #         # 4. 标记首次攻击完成（仅执行一次）
# #         if self.is_first_attack:
# #             self.is_first_attack = False

# #         # 5. 将恶意更新赋值给恶意客户端
# #         for result in trainer.local_results:
# #             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
# #             if client.is_malicious:
# #                 result[CLIENT_UPDATE] = final_update



# import random
# import torch
# from typing import Dict, Tuple

# from blades.constants import CLIENT_UPDATE, CLIENT_ID
# from blades.trainers import Trainer as Algorithm
# from blades.trainers import Trainer
# from blades.aggregators import Signguard
# from .adversary import Adversary


# class SPEARAdversary(Adversary):
#     def on_trainer_init(self, trainer: Trainer):
#         super().on_trainer_init(trainer)
        
#         self.num_clients = trainer.config.num_clients
#         num_byzantine = len(self.clients)

#         # 初始化参数
#         self.negative_indices = None
#         # 预存模型参数的层信息（名称、形状、切片范围）
#         self.model_param_info: Dict[str, Tuple[torch.Size, slice]] = {}
#         # 标记最后一层全连接层的前缀（适配你的linear层）
#         self.target_fc_prefix = "linear"
#         # MinMax二分查找参数
#         self.threshold_diff = 1e-4  # 二分查找精度
#         self.high_init = 5.0         # 二分查找上界
#         self.low_init = 0.0          # 二分查找下界

#     def _get_model_param_slices(self, model: torch.nn.Module) -> Dict[str, Tuple[torch.Size, slice]]:
#         """解析模型参数，返回各层参数的形状和展平后的切片范围"""
#         param_info = {}
#         start_idx = 0
#         for name, param in model.named_parameters():
#             param_numel = param.numel()
#             end_idx = start_idx + param_numel
#             param_info[name] = (param.size(), slice(start_idx, end_idx))
#             start_idx = end_idx
#         return param_info

#     def _get_linear_layer_slice(self) -> slice:
#         """合并所有linear层参数的切片范围，返回整体slice"""
#         linear_slices = []
#         for param_name, (_, param_slice) in self.model_param_info.items():
#             if param_name.startswith(self.target_fc_prefix):
#                 linear_slices.append(param_slice)
        
#         if not linear_slices:
#             raise ValueError(f"未找到前缀为 {self.target_fc_prefix} 的层参数")
        
#         # 合并linear层所有参数的切片范围
#         start = min(s.start for s in linear_slices)
#         end = max(s.stop for s in linear_slices)
#         return slice(start, end)

#     def _get_min_l2_update(self, benign_updates: torch.Tensor) -> torch.Tensor:
#         """计算良性梯度中L2范数最小的更新向量（替换原中位数逻辑）"""
#         # 计算每个良性客户端更新的L2范数
#         l2_norms = torch.norm(benign_updates, p=2, dim=1)
#         # 找到L2范数的最小值对应的索引
#         min_idx = torch.argmin(l2_norms)
#         # 获取L2范数最小的更新向量
#         min_l2_update = benign_updates[min_idx].clone()
        
#         return min_l2_update

#     def _attack_by_binary_search(self, benign_updates: torch.Tensor, min_l2_update: torch.Tensor) -> Tuple[torch.Tensor, float]:
#         """
#         严格按你的逻辑实现MinMax二分查找：
#         1. mal_update = min_l2_update（非全连接层） + mean - mid*std（全连接层）
#         2. 用完整mal_update计算loss来更新mid
#         """
#         # 1. 基础计算
#         mean_grads = benign_updates.mean(dim=0)
#         deviation = benign_updates.std(dim=0)
#         threshold = torch.cdist(benign_updates, benign_updates, p=2).max()
#         linear_slice = self._get_linear_layer_slice()

#         # 2. 二分查找最优mid值
#         low = self.low_init
#         high = self.high_init
#         mid = (low + high) / 2
#         best_mid = mid  # 记录最优mid值

#         while abs(high - low) > self.threshold_diff:
#             mid = (low + high) / 2
#             # 构造当前mid对应的mal_update（核心：非全连接层用min_l2，全连接层用扰动值）
#             mal_update = min_l2_update.clone()
#             mal_update[linear_slice] = mean_grads[linear_slice] - mid * deviation[linear_slice]
            
#             # 用完整的mal_update计算loss（和良性更新的最大L2距离）
#             loss = torch.cdist(mal_update.unsqueeze(0), benign_updates, p=2).max()
            
#             # 调整二分查找边界
#             if loss < threshold:
#                 low = mid
#                 best_mid = mid  # 更新最优mid
#             else:
#                 high = mid

#         # 3. 计算最终的全连接层扰动值（用最优mid）
#         final_perturbed_linear = mean_grads[linear_slice] - best_mid * deviation[linear_slice]
#         return final_perturbed_linear, best_mid

#     def on_local_round_end(self, trainer: Algorithm):
#         self.trainer = trainer
#         benign_updates = self.get_benign_updates(trainer)

#         # 1. 首次运行时解析模型参数的层信息（仅解析一次）
#         if not self.model_param_info:
#             global_model = trainer.server.get_global_model()
#             self.model_param_info = self._get_model_param_slices(global_model)

#         # 2. 步骤1：计算良性客户端L2范数最小的更新向量（基准，替换原中位数）
#         min_l2_update = self._get_min_l2_update(benign_updates)

#         # 3. 步骤2：执行MinMax二分查找（严格按你的逻辑构造mal_update）
#         perturbed_linear_grad, mid = self._attack_by_binary_search(benign_updates, min_l2_update)

#         # 4. 步骤3：构造最终恶意更新（和二分查找中的mal_update完全一致）
#         final_update = min_l2_update.clone()
#         linear_slice = self._get_linear_layer_slice()
#         final_update[linear_slice] = perturbed_linear_grad

#         # 5. 将恶意更新赋值给恶意客户端
#         for result in trainer.local_results:
#             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 result[CLIENT_UPDATE] = final_update
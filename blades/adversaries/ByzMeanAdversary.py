
# import torch

# from blades.constants import CLIENT_UPDATE, CLIENT_ID
# from blades.trainers import Trainer
# from .adversary import Adversary
# from blades.aggregators import Signguard

# class ByzMeanAdversary(Adversary):
#     """实现byzMean攻击策略的对抗者类"""
#     def __init__(self, z: float = 0.5):
#         super().__init__()
#         self.z = z  # 攻击参数，控制与均值的偏离程度

#     def on_local_round_end(self, trainer: Trainer):
#         """在本地训练回合结束时注入攻击更新"""
#         # 获取良性客户端的模型更新
#         benign_updates = self.get_benign_updates(trainer)
#         num_benign = len(benign_updates)
        
#         # 统计恶意客户端数量
#         malicious_clients = []
#         for result in trainer.local_results:
#             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 malicious_clients.append(result[CLIENT_ID])
#         num_byzs = len(malicious_clients)
        
#         if num_byzs == 0:
#             return  # 无恶意客户端时不执行攻击
        
#         # 计算良性更新的均值和标准差（核心攻击参数）
#         device = benign_updates.device
#         mu = benign_updates.mean(dim=0)  # 均值
#         sigma = benign_updates.std(dim=0)  # 标准差
        
#         # 生成两类恶意更新
#         m1 = int(0.5 * num_byzs)  # 第一类恶意客户端数量
#         m2 = num_byzs - m1  # 第二类恶意客户端数量
        
#         # 第一类攻击梯度：mu - z*sigma
#         byz_grad1 = mu - self.z * sigma
#         byz_grads1 = [byz_grad1] * m1
        
#         # 第二类攻击梯度：((num_benign - m1)*byz_grad1 - sum(benign_updates)) / m2
#         sum_benign = benign_updates.sum(dim=0)
#         byz_grad2 = ((num_benign - m1) * byz_grad1 - sum_benign) / m2
#         byz_grads2 = [byz_grad2] * m2
        
#         # 合并所有恶意更新并分配给恶意客户端
#         all_byz_grads = byz_grads1 + byz_grads2
#         grad_idx = 0
#         for result in trainer.local_results:
#             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 result[CLIENT_UPDATE] = all_byz_grads[grad_idx].to(device)
#                 grad_idx += 1



import random
import torch

from blades.constants import CLIENT_UPDATE, CLIENT_ID
from blades.trainers import Trainer
from .adversary import Adversary
from blades.aggregators import Signguard

class ByzMeanAdversary(Adversary):
    """实现byzMean攻击策略的对抗者类，对Signguard聚合器添加标准差取反处理"""
    def __init__(self, z: float = 0.5):
        super().__init__()
        self.z = z  # 攻击参数，控制与均值的偏离程度
        self.z_max = None  # 基于客户端数量计算的z值
        self.negative_indices = None  # 要取反的标准差元素索引

    def on_trainer_init(self, trainer: Trainer):
        """初始化时计算z_max（参考ALIEAdversary的计算逻辑）"""
        super().on_trainer_init(trainer)
        
        self.num_clients = trainer.config.num_clients
        num_byzantine = len(self.clients)  # 恶意客户端数量

        # 计算z_max：基于良性客户端数量和容错阈值
        s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
        # 确保分母不为0（避免除以0错误）
        denominator = self.num_clients - num_byzantine
        if denominator <= 0:
            cdf_value = 0.0
        else:
            cdf_value = (self.num_clients - num_byzantine - s) / denominator
        
        # 限制cdf_value在合理范围内（0到1之间）
        cdf_value = max(0.001, min(0.999, cdf_value))
        
        # 计算标准正态分布的分位数
        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.z_max = dist.icdf(torch.tensor(cdf_value)).item()  # 转为标量

    def on_local_round_end(self, trainer: Trainer):
        """在本地训练回合结束时注入攻击更新"""
        # 获取良性客户端的模型更新
        benign_updates = self.get_benign_updates(trainer)
        num_benign = len(benign_updates)
        
        if num_benign == 0:
            return  # 无良性客户端时不执行攻击
        
        # 统计恶意客户端数量
        malicious_clients = []
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                malicious_clients.append(result[CLIENT_ID])
        num_byzs = len(malicious_clients)
        
        if num_byzs == 0:
            return  # 无恶意客户端时不执行攻击
        
        # 计算良性更新的均值和标准差
        device = benign_updates.device
        mu = benign_updates.mean(dim=0)  # 均值
        sigma = benign_updates.std(dim=0)  # 标准差
        
        # 对于Signguard聚合器，对标准差的部分元素进行取反（参考ALIEAdversary逻辑）
        if isinstance(trainer.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = sigma.numel()  # 获取总元素个数
                num_negate = num_elements // 2  # 取一半元素进行取反
                # 生成要取反的元素索引（展平为一维索引）
                self.negative_indices = random.sample(range(num_elements), num_negate)
                # 将一维索引转换为多维索引（适应不同形状的参数）
                self.negative_indices = torch.unravel_index(
                    torch.tensor(self.negative_indices), sigma.shape
                )
            
            # 对选中的索引对应的标准差元素取反
            sigma[self.negative_indices] *= -1
        
        # 生成两类恶意更新
        m1 = int(0.5 * num_byzs)  # 第一类恶意客户端数量
        m2 = num_byzs - m1  # 第二类恶意客户端数量
        
        # 第一类攻击梯度：采用ALIE的形式（mu + sigma * z_max），并结合原有的z参数控制强度
        byz_grad1 = mu + sigma * self.z_max * self.z
        
        # 第二类攻击梯度：保持原有计算逻辑不变
        sum_benign = benign_updates.sum(dim=0)
        byz_grad2 = ((num_benign - m1) * byz_grad1 - sum_benign) / m2
        
        # 生成对应数量的恶意梯度
        byz_grads1 = [byz_grad1] * m1
        byz_grads2 = [byz_grad2] * m2
        
        # 合并所有恶意更新并分配给恶意客户端
        all_byz_grads = byz_grads1 + byz_grads2
        grad_idx = 0
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = all_byz_grads[grad_idx].to(device)
                grad_idx += 1
import torch
from blades.trainers import Trainer as Algorithm
from blades.constants import CLIENT_UPDATE, CLIENT_ID
from .adversary import Adversary

# 14 80 50
class ScaleSignAdversary(Adversary):
    def __init__(self, gamma_init=10.0, omega=0.001):
        super().__init__()
        self.gamma_init = gamma_init
        self.omega = omega

    def on_local_round_end(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)

        mean_grads = benign_updates.mean(dim=0)

        g_scaling = self._scaling_attack(mean_grads, benign_updates)

        g_final = self._sign_modification(g_scaling, benign_updates)

        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = g_final

        return g_final

    def _scaling_attack(self, mean_grads, benign_updates):
        d = mean_grads.shape[0]

        theta1 = torch.where(mean_grads > 0)[0]
        theta2 = torch.where(mean_grads < 0)[0]

        gamma = self.gamma_init
        step = (gamma - 1)  / 2

        threshold = torch.cdist(benign_updates, benign_updates, p=2).max()

        while step > self.omega:
            phi = self._generate_scaling_matrix(d, theta1, theta2, gamma,mean_grads.device)
            g_star = phi * mean_grads

            mal_update = g_star.unsqueeze(0)
            loss = torch.cdist(mal_update, benign_updates, p=2).max()

            if loss < threshold:
                gamma += step/2
            else:
                gamma -= step/2

            step = step / 2

        phi = self._generate_scaling_matrix(d, theta1, theta2, gamma,mean_grads.device)
        g_star = phi * mean_grads
        return g_star

    def _generate_scaling_matrix(self, d, theta1, theta2, gamma,device):
        phi = torch.ones(d,device = device)
        phi[theta1] = gamma
        phi[theta2] = 1.0 / gamma
        return phi

    def _sign_modification(self, g_star, benign_updates):
        g_final = g_star.clone()

        sign_stats_list = []
        for update in benign_updates:
            p = torch.sum(update > 0).item()
            z = torch.sum(update == 0).item()
            n = torch.sum(update < 0).item()
            sign_stats_list.append((p, z, n))

        ref_p, _ref_z, ref_n = self._select_reference_sign_stat(sign_stats_list)

        curr_p = torch.sum(g_final > 0).item()
        curr_n = torch.sum(g_final < 0).item()

        g_final = self._adjust_positive_gradients(g_final, curr_p, ref_p)
        g_final = self._adjust_negative_gradients(g_final, curr_n, ref_n)

        return g_final

    def _select_reference_sign_stat(self, sign_stats_list):
        m = len(sign_stats_list)
        if m == 0:
            return (0, 0, 0)

        min_dist = float('inf')
        ref_idx = 0

        for i in range(m):
            total_dist = 0.0
            for j in range(m):
                if i != j:
                    dist = (
                        (sign_stats_list[i][0] - sign_stats_list[j][0]) ** 2 +
                        (sign_stats_list[i][1] - sign_stats_list[j][1]) ** 2 +
                        (sign_stats_list[i][2] - sign_stats_list[j][2]) ** 2
                    )
                    total_dist += dist

            if total_dist < min_dist:
                min_dist = total_dist
                ref_idx = i

        return sign_stats_list[ref_idx]

    def _adjust_positive_gradients(self, g, curr_p, target_p):
        if curr_p > target_p:
            positive_indices = torch.where(g > 0)[0]
            sorted_indices = positive_indices[torch.argsort(g[positive_indices])]
            num_to_zero = curr_p - target_p
            g[sorted_indices[:num_to_zero]] = 0.0
        elif curr_p < target_p:
            non_positive_indices = torch.where(g <= 0)[0]
            sorted_indices = non_positive_indices[torch.argsort(-g[non_positive_indices])]
            num_to_positive = target_p - curr_p
            positive_values = g[g > 0]
            if len(positive_values) > 0:
                min_positive = positive_values.min().item()
                g[sorted_indices[:num_to_positive]] = min_positive
        return g

    def _adjust_negative_gradients(self, g, curr_n, target_n):
        if curr_n > target_n:
            negative_indices = torch.where(g < 0)[0]
            sorted_indices = negative_indices[torch.argsort(-g[negative_indices])]
            num_to_zero = curr_n - target_n
            g[sorted_indices[:num_to_zero]] = 0.0
        elif curr_n < target_n:
            non_negative_indices = torch.where(g >= 0)[0]
            sorted_indices = non_negative_indices[torch.argsort(g[non_negative_indices])]
            num_to_negative = target_n - curr_n
            negative_values = g[g < 0]
            if len(negative_values) > 0:
                max_negative = negative_values.max().item()
                g[sorted_indices[:num_to_negative]] = max_negative
        return g

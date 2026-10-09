# import torch

# from blades.trainers import Trainer
# from blades.clients import ClientCallback
# from .adversary import Adversary


# class LabelFlipAdversary(Adversary):
#     def on_trainer_init(self, trainer: Trainer):
#         num_class = self._get_num_model_outputs(trainer)

#         class LabelFlipCallback(ClientCallback):
#             def on_train_batch_begin(self, data, target):
#                 return data, num_class - 1 - target

#         for client in self.clients:
#             client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)

#     def _get_num_model_outputs(self, trainer: Trainer):
#         dataset = trainer._dataset
#         test_client_0 = dataset.test_client_ids[0]
#         client_dataset = dataset.get_client_dataset(test_client_0)
#         test_loader = client_dataset.get_test_loader()
#         model = trainer.server.get_global_model()
#         with torch.no_grad():
#             for data, _ in test_loader:
#                 output = model(data)
#                 break
#         return output.shape[1]


import torch

from blades.trainers import Trainer
from blades.clients import ClientCallback
from .adversary import Adversary


class LabelFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        num_class = self._get_num_model_outputs(trainer)

        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                return data, num_class - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)

    def _get_num_model_outputs(self, trainer: Trainer):
        dataset = trainer._dataset
        test_client_0 = dataset.test_client_ids[0]
        client_dataset = dataset.get_client_dataset(test_client_0)
        test_loader = client_dataset.get_test_loader()
        model = trainer.server.get_global_model()
        
        # 1. 获取模型所在的设备（通过模型参数推断）
        device = next(model.parameters()).device  # 关键：获取模型设备（cuda:0 或 cpu）
        
        with torch.no_grad():
            for data, _ in test_loader:
                # 2. 将数据移动到模型所在的设备
                data = data.to(device)  # 关键：数据设备与模型设备同步
                output = model(data)
                break
        return output.shape[1]

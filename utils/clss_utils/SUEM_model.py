import torch
import numpy as np
import torch.nn.functional as F
import random
import torch.nn as nn
# randon seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # using GPU
torch.cuda.manual_seed_all(seed)  # Mutli-gpus
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SUEM_model(torch.nn.Module):
    def __init__(self, clip_adapter_model, model, loss_func, device):
        super(SUEM_model, self).__init__()
        self.clip_adapter_model = clip_adapter_model.to(device)
        self.model = model.to(device)
        self.loss_func = loss_func
        self.device = device

    def forward(self, xyz, points, labels):
        loss_cur, pred = self.model(xyz.to(self.device), points.to(self.device),
                                                                  labels.to(self.device), self.clip_adapter_model, self.loss_func)

        loss = self.loss_func(pred, labels)

        loss = loss + loss_cur

        return pred, loss
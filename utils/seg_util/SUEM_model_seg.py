import torch
import numpy as np
import torch.nn.functional as F
import random
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def cosine_loss(A, B, t=1):
    return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()


class SUEM_model_seg(torch.nn.Module):
    def __init__(self, clip_adapter_model_seg, model, loss_func, device):
        super(SUEM_model_seg, self).__init__()
        self.clip_adapter_model_seg = clip_adapter_model_seg.to(device)
        self.model = model.to(device)
        self.loss_func = loss_func
        self.device = device

    def forward(self, xyz, points, labels):
        loss_cur, pred = self.model(xyz.to(self.device), points.to(self.device),
                                                                  labels.to(self.device), self.clip_adapter_model_seg, self.loss_func, self.device)
        loss = self.loss_func(pred, labels)

        loss = loss + loss_cur

        return pred, loss
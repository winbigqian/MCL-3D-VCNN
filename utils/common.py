import numpy as np
import torch


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C) B：表示批次大小 N：表示每个样本中的点数 C：表示每个点的特征数
    :param inds: shape=(B, M) or shape=(B, M, K)  M：表示要收集的点数 K：表示每个点的最近邻数量
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)  # 获取 inds 的形状信息并转换为列表
    inds_shape[1:] = [1] * len(inds_shape[1:])   # 将除了第一维之外的维度都设为 1
    repeat_shape = list(inds.shape)  # 获取 inds 的形状信息并转换为列表
    repeat_shape[0] = 1   # 将第一维设为 1
    # 构造一个与 inds 形状相同的张量，用于索引 points 中的 batch
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape) # 生成一个与批次大小相同的一维张量，用于构造批次索引
    return points[batchlists, inds, :]   #根据给定的索引从原始点云数据中获取采样的点


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
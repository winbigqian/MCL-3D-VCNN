import torch
from utils.common import gather_points, get_dists


def ball_query(xyz, new_xyz, radius, K):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]  ## 获取新的质心点数量
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)  ## 创建一个张量，包含原始点云中所有点的索引
    dists = get_dists(new_xyz, xyz)  #计算新质心点与原始点云中所有点之间的距离
    grouped_inds[dists > radius] = N   ## 将距离超过半径的点的索引设置为N（即不在球形邻域内）
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]  # 对每个质心点的邻域内的点按照距离进行排序，并取前K个点
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)   ## 为邻域内没有点的质心点指定一个最近的点的索引
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds
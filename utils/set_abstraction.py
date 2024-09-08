import torch
import torch.nn as nn
from utils.sampling import fps
from utils.grouping import ball_query
from utils.common import gather_points


def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)  N是原始点云中每个样本的点数
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);

             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
             new_xyz: 经过sampling后，得到的512个中心点的坐标
             idx：是每个区域内点的索引
             grouped_xyz：分组后的点集，是一个四维向量（batch_size, 512个区域，每个区域的32个点，每个点3个坐标）
             new_points：也是就是分组后的点集，不过里面存的是特征，如果是第一次，就等于grouped_xyz，可以选择在卷积的时候把坐标和特征进行concat后卷积。

    '''
    new_xyz = gather_points(xyz, fps(xyz, M))  # 根据给定的索引从原始点云数据中获取采样点的质心坐标
    grouped_inds = ball_query(xyz, new_xyz, radius, K)   #在原始点云中查询每个新质心点的邻域内的点，存储了每个新质心点的邻域内的点的索引
    grouped_xyz = gather_points(xyz, grouped_inds)  ## 每个质心点的邻域内点的绝对坐标（在点云中的原始坐标位置），每个邻域内的点相对于质心点的相对坐标
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)  # 将质心点坐标减去对应的新质心点坐标，得到相对坐标
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)  # 根据邻域内的点索引，获取对应的点特征，从原始点云特征中收集邻域内的点的特征 [btachsize, M, k, 3]
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)  # 将局部点坐标和点特征连接起来 [btachsize, M, k, 6]
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz  #新的采样点的质心、这些点内的局部特征、每个新质心点的邻域内的点的索引、每个邻域内的点相对于质心点的相对坐标


def sample_and_group_all(xyz, points, use_xyz=True):
    '''

    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':  # 进行池化操作，可以选择最大池化或平均池化
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_MSG, self).__init__()
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbones = nn.ModuleList()
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                         nn.Conv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                             nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        new_xyz = gather_points(xyz, fps(xyz, self.M))
        new_points_all = []
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            grouped_xyz = gather_points(xyz, grouped_inds)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
            if points is not None:
                grouped_points = gather_points(points, grouped_inds)
                if self.use_xyz:
                    new_points = torch.cat(
                        (grouped_xyz.float(), grouped_points.float()),
                        dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]
            new_points = new_points.permute(0, 2, 1).contiguous()
            new_points_all.append(new_points)
        return new_xyz, torch.cat(new_points_all, dim=-1)


if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    setup_seed(2)
    xyz = torch.randn(4, 1024, 3)
    points = torch.randn(4, 1024, 3)

    M, radius, K = 5, 5, 6
    new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz, points, M, radius, K)
    print(new_xyz[0])
    print(new_points[0])
    '''
    print('='*20, 'backbone', '='*20)
    M, radius, K, in_channels, mlp = 2, 0.2, 3, 6, [32, 64, 128]
    new_xyz, new_points, grouped_inds = pointnet_sa_module(xyz, points, M, radius, K, in_channels, mlp)
    print('new_xyz: ', new_xyz.shape)
    print('new_points: ', new_points.shape)
    print('grouped_inds: ', grouped_inds.shape)

    print('='*20, 'backbone msg', '='*20)
    M, radius_list, K_list, in_channels, mlp_list = 2, [0.2, 0.4], [3, 4], 6, [[32, 64, 128], [64, 64]]
    new_xyz, new_points_cat = pointnet_sa_module_msg(xyz, points, M, radius_list, K_list, in_channels, mlp_list)
    print('new_xyz: ', new_xyz.shape)
    print(new_points_cat.shape)
    '''
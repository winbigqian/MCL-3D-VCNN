import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from clip.adapter import adapter
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG
from utils.mv_utils_zs import Realistic_Projection

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 如果使用GPU
torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True  # 确保每次结果一致
torch.backends.cudnn.benchmark = False  # 设置为 False 以确保每次结果一致


def cosine_loss(A, B, t=1):
    return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()


class pointnet2_cls_ssg(nn.Module):  #MRG
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        # M：最大采样点数 radius：质心点的邻域半径， K：质心点邻域中的点数， in_channels=in_channels：输入特征的通道数， mlp：MLP的隐藏层大小， group_all：是否对所有点进行操作
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

        self.adapter = adapter.AdapterModel(input_size=512, output_size=512, alpha=0.5)  #微调

        self.my_fc1_a = nn.Linear(128, 256, bias=False)
        self.my_bn1_a = nn.BatchNorm1d(256)
        self.my_fc1_b = nn.Linear(256, 512, bias=False)
        self.my_bn1_b = nn.BatchNorm1d(512)
        self.my_dropout1 = nn.Dropout(0.5)
        
        self.my_fc2 = nn.Linear(256, 512, bias=False)
        self.my_bn2 = nn.BatchNorm1d(512)
        self.my_dropout2 = nn.Dropout(0.5)

    def forward(self, xyz, points, labels, clip_adapter_model, loss_func):        #---------------------------------- 原本的pints：[32, 1024, 3]
        batchsize = xyz.shape[0]

        ####################第一阶段，原始点云投影+与PointNET++第一次抽象结果做对比
        features_2D_1, logits_image_text_1 = clip_adapter_model(xyz)

        new_xyz, new_points = self.pt_sa1(xyz, points)   #new_xyz:[32, 512, 3],   new_points:[32, 512, 128]

        features_3D_1 = new_points
        features_3D_1 = torch.mean(features_3D_1, dim=1)   #将中心点进行平均池化  [32, 128]

        features_3D_1 = self.my_dropout1(F.relu(self.my_bn1_a(self.my_fc1_a(features_3D_1))))  # [4,256]
        features_3D_1 = self.my_dropout1(F.relu(self.my_bn1_b(self.my_fc1_b(features_3D_1))))  #[4,512]
        features_3D_1 = self.adapter(features_3D_1)

        loss_sim_1 = cosine_loss(features_2D_1, features_3D_1, 0.8)

        ####################第二阶段，第一次抽象之后投影结果+与PointNET++第二次抽象结果做对比
        features_2D_2, logits_image_text_2 = clip_adapter_model(new_xyz)   #上一轮的new_xyz

        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)  #new_xyz:[32, 128, 3],  new_points: [32, 128, 256]

        features_3D_2 = new_points
        features_3D_2 = torch.mean(features_3D_2, dim=1)  # 将中心点进行平均池化[32,256]
        features_3D_2 = self.my_dropout2(F.relu(self.my_bn2(self.my_fc2(features_3D_2))))  #[32,512]
        features_3D_2 = self.adapter(features_3D_2)


        loss_sim_2 = cosine_loss(features_2D_2, features_3D_2, 0.8)

        ####################第三阶段，第二次抽象之后投影结果+与PointNET++第三次抽象结果做对比
        features_2D_3, logits_image_text_3 = clip_adapter_model(new_xyz)   #上一轮的new_xyz

        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)  #new_xyz:[32, 1, 3],  new_points: [32, 1, 1024]
        features_3D_3 = new_points
        features_3D_3 = features_3D_3.view(batchsize, -1)    #[4,1024]

        features_3D_3 = self.dropout1(F.relu(self.bn1(self.fc1(features_3D_3))))  #[4,512]
        features_3D_3 = self.adapter(features_3D_3)

        loss_sim_3 = cosine_loss(features_2D_3, features_3D_3, 0.8)

        net = new_points.view(batchsize, -1)    #[4,1024]
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))  # [batchsize,512]

        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))  #[4,256]
        net = self.cls(net)

        ######################################################

        clip_loss_1 = loss_func(logits_image_text_1, labels)
        clip_loss_2 = loss_func(logits_image_text_2, labels)
        clip_loss_3 = loss_func(logits_image_text_3, labels)

        loss_cur = loss_sim_1 + loss_sim_2 + loss_sim_3 + clip_loss_1 + clip_loss_2 + clip_loss_3

        return loss_cur, net


class pointnet2_cls_msg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_msg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module_MSG(M=512,
                                             radiuses=[0.1, 0.2, 0.4],
                                             Ks=[16, 32, 128],
                                             in_channels=in_channels,
                                             mlps=[[32, 32, 64],
                                                   [64, 64, 128],
                                                   [64, 96, 128]])
        self.pt_sa2 = PointNet_SA_Module_MSG(M=128,
                                             radiuses=[0.2, 0.4, 0.8],
                                             Ks=[32, 64, 128],
                                             in_channels=323,
                                             mlps=[[64, 64, 128],
                                                   [128, 128, 256],
                                                   [128, 128, 256]])
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=643, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, lable):
        '''

        :param pred: shape=(B, nclass)
        :param lable: shape=(B, )
        :return: loss
        '''
        loss = self.loss(pred, lable)
        return loss


if __name__ == '__main__':
    xyz = torch.randn(16, 2048, 3)
    points = torch.randn(16, 2048, 3)
    label = torch.randint(0, 40, size=(16, ))
    ssg_model = pointnet2_cls_ssg(6, 40)

    print(ssg_model)
    #net = ssg_model(xyz, points)
    #print(net.shape)
    #print(label.shape)
    #loss = cls_loss()
    #loss = loss(net, label)
    #print(loss)
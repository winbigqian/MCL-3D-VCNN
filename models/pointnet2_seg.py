import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG
from utils.feature_propagation import PointNet_FP_Module

from clip.adapter import adapter

def cosine_loss(A, B, t=1):
    return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()

def convert_labels_to_objects(label):
    # 定义seg_classes列表
    seg_classes = [
        [0, 1, 2, 3],  # 0
        [4, 5],         # 1
        [6, 7],          #2
        [8, 9, 10, 11],  #3
        [12, 13, 14, 15], # 4
        [16, 17, 18],      #5
        [19, 20, 21],     #6
        [22, 23],          #7
        [24, 25, 26, 27],   # 8
        [28, 29],           #9
        [30, 31, 32, 33, 34, 35],   # 10
        [36, 37],                 # 11
        [38, 39, 40],          # 12
        [41, 42, 43],         # 13
        [44, 45, 46],  # 14
        [47, 48, 49]         # 15
    ]

    # 创建一个新的张量，形状为 (32)
    new_label = torch.zeros((label.shape[0]), dtype=torch.long)

    # 遍历每一行，将部件标签转换为索引
    for i in range(label.shape[0]):
        # 获取当前行的第一个标签
        current_label = label[i][0].item()  # 将张量转换为 Python 数字

        # 在seg_classes中查找当前标签，并获取其索引
        found = False
        for idx, parts in enumerate(seg_classes):
            if current_label in parts:
                new_label[i] = idx
                found = True
                break

        if not found:
            raise ValueError(f"Label {current_label} not found in seg_classes")

    return new_label


class pointnet2_seg_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_seg_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)

        self.pt_fp1 = PointNet_FP_Module(in_channels=1024+256, mlp=[256, 256], bn=True)
        self.pt_fp2 = PointNet_FP_Module(in_channels=256 + 128, mlp=[256, 128], bn=True)
        self.pt_fp3 = PointNet_FP_Module(in_channels=128 + 6, mlp=[128, 128, 128], bn=True)

        self.conv1 = nn.Conv1d(128, 128, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.cls = nn.Conv1d(128, nclasses, 1, stride=1)

        self.adapter = adapter.AdapterModel(input_size=512, output_size=512, alpha=0.5)  #微调

        self.cls_fc1 = nn.Linear(1024, 512, bias=False)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_dropout1 = nn.Dropout(0.5)


        self.my_fc1_a = nn.Linear(128, 256, bias=False)
        self.my_bn1_a = nn.BatchNorm1d(256)
        self.my_fc1_b = nn.Linear(256, 512, bias=False)
        self.my_bn1_b = nn.BatchNorm1d(512)
        self.my_dropout1 = nn.Dropout(0.5)

        self.my_fc2 = nn.Linear(256, 512, bias=False)
        self.my_bn2 = nn.BatchNorm1d(512)
        self.my_dropout2 = nn.Dropout(0.5)


    def forward(self, l0_xyz, l0_points, label, clip_adapter_model_seg, loss_func, device):
        new_label = convert_labels_to_objects(label)
        new_label = new_label.long().to(device)     # 用于

        batchsize = l0_xyz.shape[0]
        ################ 第一次抽象
        image_feat_1, logits_image_text_1 = clip_adapter_model_seg(l0_xyz)
        l1_xyz, l1_points = self.pt_sa1(l0_xyz, l0_points)  #l1_xyz: torch.Size([32, 512, 3]), l1_points: torch.Size([32, 512, 128])

        point_feat_1 = l1_points
        point_feat_1 = torch.mean(point_feat_1, dim=1)   #将中心点进行平均池化  [32, 128]

        point_feat_1 = self.my_dropout1(F.relu(self.my_bn1_a(self.my_fc1_a(point_feat_1))))  # [4,256]
        point_feat_1 = self.my_dropout1(F.relu(self.my_bn1_b(self.my_fc1_b(point_feat_1))))  #[4,512]
        point_feat_1 = self.adapter(point_feat_1)

        loss_sim_1 = cosine_loss(image_feat_1, point_feat_1, 0.8)

        ############### 第二次抽象
        image_feat_2, logits_image_text_2 = clip_adapter_model_seg(l1_xyz)
        l2_xyz, l2_points = self.pt_sa2(l1_xyz, l1_points)  # l2_xyz: torch.Size([32, 128, 3]) , l2_points: torch.Size([32, 128, 256])

        point_feat_2 = l2_points
        point_feat_2 = torch.mean(point_feat_2, dim=1)  # 将中心点进行平均池化[32,256]

        point_feat_2 = self.my_dropout2(F.relu(self.my_bn2(self.my_fc2(point_feat_2))))  # [32,512]
        point_feat_2 = self.adapter(point_feat_2)

        loss_sim_2 = cosine_loss(image_feat_2, point_feat_2, 0.8)

        ############### 第三次抽象
        image_feat_3, logits_image_text_3 = clip_adapter_model_seg(l2_xyz)  # l3_xyz: torch.Size([32, 1, 3]) , l3_points: torch.Size([32, 1, 1024])
        l3_xyz, l3_points = self.pt_sa3(l2_xyz, l2_points)

        point_feat_3 = l3_points
        point_feat_3 = point_feat_3.view(batchsize, -1)  # [4,1024]
        point_feat_3 = self.cls_dropout1(F.relu(self.cls_bn1(self.cls_fc1(point_feat_3))))  # [4,512]
        point_feat_3 = self.adapter(point_feat_3)

        loss_sim_3 = cosine_loss(image_feat_3, point_feat_3, 0.8)

        ####################
        l2_points = self.pt_fp1(l2_xyz, l3_xyz, l2_points, l3_points)  # torch.Size([32, 128, 256])
        l1_points = self.pt_fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # torch.Size([32, 512, 128])
        l0_points = self.pt_fp3(l0_xyz, l1_xyz, torch.cat([l0_points, l0_xyz], dim=-1), l1_points)  # torch.Size([32, 2500, 128])

        # 最后一次给他再求一下相似度
        point_feat_4 = l0_points
        point_feat_4 = torch.mean(point_feat_4, dim=1)   #将中心点进行平均池化  [32, 128]
        point_feat_4 = self.my_dropout1(F.relu(self.my_bn1_a(self.my_fc1_a(point_feat_4))))  # [4,256]
        point_feat_4 = self.my_dropout1(F.relu(self.my_bn1_b(self.my_fc1_b(point_feat_4))))  #[4,512]
        point_feat_4 = self.adapter(point_feat_4)


        loss_sim_4 = cosine_loss(image_feat_1, point_feat_4, 0.8)


        net = l0_points.permute(0, 2, 1).contiguous()
        net = self.dropout1(F.relu(self.bn1(self.conv1(net))))
        net = self.cls(net)   # torch.Size([32, 50, 2500])

        ######################################################

        clip_loss_1 = loss_func(logits_image_text_1, new_label)
        clip_loss_2 = loss_func(logits_image_text_2, new_label)
        clip_loss_3 = loss_func(logits_image_text_3, new_label)

        loss_cur = loss_sim_1 + loss_sim_2 + loss_sim_3 + loss_sim_4+ clip_loss_1 + clip_loss_2 + clip_loss_3

        return loss_cur, net


class seg_loss(nn.Module):
    def __init__(self):
        super(seg_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, label):
        '''

        :param pred: shape=(B, N, C)
        :param label: shape=(B, N)
        :return:
        '''
        loss = self.loss(pred, label)
        return loss


if __name__ == '__main__':
    in_channels = 6
    n_classes = 50
    l0_xyz = torch.randn(4, 1024, 3)
    l0_points = torch.randn(4, 1024, 3)
    model = pointnet2_seg_ssg(in_channels, n_classes)
    net = model(l0_xyz, l0_points)
    print(net.shape)
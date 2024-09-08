import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, random_point_dropout

from clip import clip
from utils.mv_utils_zs import Realistic_Projection
# import matplotlib.pyplot as plt
import os
from torchvision.transforms import functional as F


class ModelNet40(Dataset):

    def __init__(self, data_root, split, npoints, augment=False, dp=False, normalize=True):
    #def __init__(self, data_root, split, npoints, clip_model, preprocess, augment=False, dp=False, normalize=True):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        # self.clip_model = clip_model
        # self.preprocess = preprocess
        # self.conv_layer = nn.Conv2d(30, 3, kernel_size=1)  #----------用于改变注意力之后的维度给clip

        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'modelnet40_shape_names.txt'))
        train_list_path = os.path.join(data_root, 'modelnet40_train.txt')
        train_files_list = self.read_list_file(train_list_path, name2cls)
        test_list_path = os.path.join(data_root, 'modelnet40_test.txt')
        test_files_list = self.read_list_file(test_list_path, name2cls)
        self.files_list = train_files_list if split == 'train' else test_files_list
        self.caches = {}

    def read_list_file(self, file_path, name2cls):
        base = os.path.dirname(file_path)
        files_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = '_'.join(line.strip().split('_')[:-1])
                cur = os.path.join(base, name, '{}.txt'.format(line.strip()))
                files_list.append([cur, name2cls[name]])
        return files_list

    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path, 'r') as f:
            for i, name in enumerate(f.readlines()):
                cls2name[i] = name.strip()
                name2cls[name.strip()] = i
        return cls2name, name2cls

    def augment_pc(self, pc_normal):
        rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
        rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
        jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
        jittered_pc = shift_point_cloud(jittered_pc)
        jittered_pc = jitter_point_cloud(jittered_pc)
        rotated_pc_normal[:, :3] = jittered_pc
        return rotated_pc_normal

    def get_label_feature(self,classnames):                  #------------------------------获取文本特征
        clip_model = self.clip_model
        #clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
        # get norm clip weight
        str_classnames ="a photo of a" +classnames
        prompt = clip.tokenize([str_classnames]).cuda()

        with torch.no_grad():
            text_features = clip_model.encode_text(prompt)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

    def preprocess_image(self, image):
        # 将图像转换为大小为 224x224 的张量
        image = F.resize(image, (224, 224), antialias=True)
        # 将图像转换为 PyTorch 张量，并进行标准化
        # image =torch.tensor(image)
        image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image.unsqueeze(0)  # 添加批次维度



    def get_feature(self, label, xyz):
        # ---------------------------My code-----------
        label_names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone"
            , "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp"
            , "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink"
            , "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]
        label_name = label_names[label].strip()  # 找出对应下标的label名称
        label_feature = self.get_label_feature(label_name)  # 大小是 [1,512]
        print("label---------------", label_name)
        ###############################################################

        point_cloud_data = torch.tensor(xyz)  # 将点云数据转化为向量

        # 在加载点云数据后，将其添加一个批处理维度
        point_cloud_data = point_cloud_data.unsqueeze(0)
        point_cloud_data = point_cloud_data.to('cuda:0')

        # 初始化投影器
        projector = Realistic_Projection()

        # 获取多视角图像
        images = projector.get_img(point_cloud_data.float())

        images = images.view(30, 110, 110)  #将(10,3,110,110)改为(30, 110, 110)

        # print("images.shape1----------------     ", images.shape)
        selfAttention = CBAMLayer(30)  #自注意力机制
        selfAttention.to(device='cuda')
        images = selfAttention.forward(images.float())
        # print("images.shape2----------------     ", images.shape)

        conv_layer = self.conv_layer
        conv_layer = conv_layer.to(device='cuda')
        images = conv_layer(images)
        images = self.preprocess_image(images)

        # print("images.shape3----------------     ", images.shape)
        # 处理每个图像并生成特征
        clip_model = self.clip_model

        with torch.no_grad():
                features = clip_model.encode_image(images)

        # print("features.shape----------------     ", features.shape)        #可以使用clip，明天再看看能不能用来求loss

        loss_fun = nn.CrossEntropyLoss()

        cos_sim = torch.nn.functional.cosine_similarity(features, label_feature, dim=-1)
        loss = 1 - cos_sim  # 1 - 余弦相似度用作损失

        print("lossmdoel40-------------   ", loss)


        # features_list = []
        # for img in images:
        #     img_tensor = self.preprocess_image(img)  # 标准化
        #     with torch.no_grad():
        #         print("img_tensor.shape--------------",img_tensor.shape)
        #         features = clip_model.encode_image(img_tensor)  # 用全连接层？？？##太慢了！！！！
        #     features_list.append(features)




        # 将特征连接在一起
        # all_features = torch.cat(features_list, dim=0)
        # # 添加批次维度
        # final_feature_vector = all_features.mean(dim=0, keepdim=True)  # 取平均作为最终特征向量
        # # print("final_feature_vector-----------  ", final_feature_vector.shape)
        # loss_fun = nn.CrossEntropyLoss()
        # loss = loss_fun(F1.softmax(final_feature_vector, dim=1), F1.softmax(label_feature, dim=1))
        # print("lossmdoel40-------------   ", loss)

        ## 可视化多视图图像（这里只展示第一个批次的图像）
        batch_size = point_cloud_data.shape[0]
        # print(projector.num_views)

        num_views = 10  # 可以这么重新设置值视角，结果是对的

        cmap = 'gray'

        ## 可视化
        # 设置子图显示的行列数
        # num_rows = 2
        # num_columns = num_views // num_rows
        # # plt.figure(figsize=(4 * num_views, 4))
        # fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 10))  # 子图存储在axs变量中
        #
        # for i in range(num_views):
        #     ax = axs[i // num_columns, i % num_columns]  # 选择要在其中显示图像的特定子图,5代表列数
        #     # plt.subplot(2, 5, i+1)
        #     plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
        #     ax.imshow(images[i][0].detach().cpu().numpy(), cmap=cmap)  # + .detach()
        #     # plt.title(f'View {i+1}')
        #     ax.set_title(f'View {i + 1}')
        #     # ax.axis('off')  # 关闭坐标轴
        # plt.tight_layout()
        # plt.show()
        return loss

    def __getitem__(self, index):
        if index in self.caches:            # 如果数据项已经存在于缓存中，则直接返回缓存中的数据
            return self.caches[index]
        file, label = self.files_list[index]        # 获取文件路径和标签
        xyz_points = np.loadtxt(file, delimiter=',')        # 从文件中加载点云数据
        #if self.npoints > 0:
        #    inds = np.random.randint(0, len(xyz_points), size=(self.npoints, ))
        #    xyz_points = xyz_points[inds, :]
        xyz_points = xyz_points[:self.npoints, :]           # 获取前self.npoints个点
        if self.normalize:
            xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])         # 对前三列进行归一化
        if self.augment:
            xyz_points = self.augment_pc(xyz_points)            # 对点云数据进行增强
        if self.dp:
            xyz_points = random_point_dropout(xyz_points)           # 对点云数据进行随机丢弃

        # xyz, points = xyz_points[:, :3], xyz_points[:, 3:]
        # loss = self.get_feature(label,xyz)  #------------  获取文本特征和多视图特征
        #
        # self.caches[index] = [xyz_points, label, loss]            # 将处理后的点云数据和标签存入缓存
        #
        # return xyz_points, label, loss           # 返回处理后的点云数据和标签，语义特征
        #---------------------------------------------

        self.caches[index] = [xyz_points, label]            # 将处理后的点云数据和标签存入缓存
        return xyz_points, label            # 返回处理后的点云数据和标签，语义特征

    def __len__(self):
        return len(self.files_list)



if __name__ == '__main__':
    modelnet40 = ModelNet40(data_root='/root/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled', split='test')
    test_loader = DataLoader(dataset=modelnet40,
                              batch_size=16,
                              shuffle=True)
    for point, label in test_loader:
        print(point.shape)
        print(label.shape)
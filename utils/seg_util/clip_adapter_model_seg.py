import torch
import random
import numpy as np
from torch import nn

from clip import clip
from utils.mv_utils_zs import Realistic_Projection
from clip.adapter import adapter

import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2



seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class clip_adapter_model_seg(torch.nn.Module):
    def __init__(self, clip_model, device):
        super(clip_adapter_model_seg, self).__init__()
        self.clip_model = clip_model
        self.adapter = adapter.AdapterModel(input_size=512, output_size=512, alpha=0.5)

        self.visual_encoder = self.clip_model.visual
        self.device = device

        # Realistic projection
        self.num_views = 10  # Mutli-views
        self.channel = 512
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img
        self.w = nn.Parameter(torch.ones(self.num_views))


    def get_label_feature(self):
        clip_model = self.clip_model
        # Defines a class and its corresponding class code
        '''
        seg_classes = {
            'Earphone': [16, 17, 18],
            'Motorbike': [30, 31, 32, 33, 34, 35],
            'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11],
            'Laptop': [28, 29], 'Cap': [6, 7],
            'Skateboard': [44, 45, 46], 'Mug': [36, 37],
            'Guitar': [19, 20, 21], 'Bag': [4, 5],
            'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
        }
        '''
        label_names = ["Airplane", "Bag", "Cap", "Car",
                    "Chair", "Earphone", "Guitar", "Knife",
                    "Lamp", "Laptop", "Motorbike", "Mug",
                    "Pistol", "Rocket", "Skateboard", "Table"]


        prompt = torch.cat([clip.tokenize(c) for c in label_names]).to(self.device)

        # with torch.no_grad():
        #     text_features_pc = clip_model.encode_text(prompt)  #  [40,512]
        with torch.no_grad():
            text_features = clip_model.encode_text(prompt).repeat(1, self.num_views)  #[40,512*num_views]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()


    def preprocess_image(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)     # HXW: 224x224
        return img

    def draw_image(self, images):
        ################################

        batch_size = 32
        files_name = 'my_image'

        num_views = 10  # Set value view

        # save image
        output_dir = "../data/results_seg_2"
        os.makedirs(output_dir, exist_ok=True)

        cmap = 'gray'


        for i in range(batch_size):
            for j in range(num_views):
                img = images[i * num_views + j].detach().cpu().numpy()


                # save image
                img = (img * 255).astype(np.uint8)

                img = img.transpose(1, 2, 0)  #  (3, 224, 224) -> (224, 224, 3)
                image_filename = f'{output_dir}/MultView_{cmap}_NumViews_{num_views}/{files_name}_view{j + 1}_batch_{i}.png'
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                cv2.imwrite(image_filename, img)
            print("***********保存完成************")


        num_rows = 2
        num_columns = num_views // num_rows

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 10))

        for i in range(num_views):
            ax = axs[i // num_columns, i % num_columns]
            # plt.subplot(2, 5, i+1)
            plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
            ax.imshow(images[i][0].detach().cpu().numpy(), cmap=cmap)  # + .detach()
            # plt.title(f'View {i+1}')
            ax.set_title(f'View {i + 1}')
            # ax.axis('off')
        plt.tight_layout()
        plt.show()

        ###############################




    def forward(self, points):

        prompt = self.get_label_feature()
        prompt = prompt.float()

        points = points.to(device=self.device)


        # get image
        images = self.preprocess_image(points.float())  #[batchszie * view_nums, 3, 224, 224]

        # self.draw_image(images)

        with torch.no_grad():
            clip_model = self.clip_model
            images_features = clip_model.encode_image(images)  #  #torch.Size([num_view * batch_size, 512])


        images_features = self.adapter(images_features)  #
        images_features = images_features / images_features.norm(dim=-1, keepdim=True)

        # Normalized weights
        view_weights = torch.softmax(self.w, dim=0).to(self.device)
        images_features = images_features.reshape(-1, self.num_views, self.channel) * view_weights.reshape(1, -1, 1)  # torch.Size([batch_size, view_num, 512])

        features_2D = images_features.mean(dim=1)


        # Change dimension
        images_features = images_features.reshape(-1, self.num_views * self.channel).type(torch.float)  # torch.Size([batch_size, 512*view_nums])

        logits = 100. * images_features @ prompt.t()



        return features_2D, logits #logits: torch.Size([32, 40])
import torch
import random
import numpy as np
from torch import nn

from clip import clip
from utils.mv_utils_zs import Realistic_Projection
from clip.adapter import adapter


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class clip_adapter_model(torch.nn.Module):
    def __init__(self, clip_model, preprocess, device):
        super(clip_adapter_model, self).__init__()
        self.clip_model = clip_model
        self.adapter = adapter.AdapterModel(input_size=512, output_size=512, alpha=0.5)

        self.visual_encoder = self.clip_model.visual
        self.preprocess = preprocess
        self.device = device

        # Realistic projection
        self.num_views = 15  # Mutli-views
        self.channel = 512
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img
        self.w = nn.Parameter(torch.ones(self.num_views))

        #模型维度变化
        self.conv_layer = nn.Conv2d(30, 3, kernel_size=1)
        self.text_projection = nn.Linear(512, 40)

        #把 (batch_size, view_num * channel)变成 (batch_size, channel)
        self.fc1 = nn.Linear(self.num_views * 512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)


    def get_label_feature(self):
        clip_model = self.clip_model
        label_names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
                       "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
                       "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand",
                       "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs",
                       "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

        prompt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_names]).to(device=self.device)  # generate prompt
        # with torch.no_grad():
        #     text_features_pc = clip_model.encode_text(prompt)  # [40,512]
        with torch.no_grad():
            text_features = clip_model.encode_text(prompt).repeat(1, self.num_views)  #[40,512*num_views]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()


    def preprocess_image(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)     # 224x224
        return img

    def forward(self, points):

        batch_size = points.shape[0]

        # print("labels--------", labels)
        prompt = self.get_label_feature()
        prompt = prompt.float()

        points = points.to(device=self.device)

        # mutli-images
        images = self.preprocess_image(points.float())  #[40, 3, 224, 224]

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


        return features_2D, logits
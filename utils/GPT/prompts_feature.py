import torch
import json
from clip import clip

class prompts_feature(torch.nn.Module):
    def __init__(self, clip_model, device):
        super(prompts_feature, self).__init__()
        self.clip_model = clip_model
        self.device = device

    def get_label_feature(self):
        clip_model = self.clip_model
        label_names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
                       "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
                       "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand",
                       "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs",
                       "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

        # GPT prompt
        with open("", "r") as file:
            descriptions = json.load(file)

        all_features = []

        for c in label_names:
            if c in descriptions:

                text_features_for_c = []
                for desc in descriptions[c]:
                    prompt = clip.tokenize(desc).to(device=self.device)
                    with torch.no_grad():
                        feature = clip_model.encode_text(prompt)
                    text_features_for_c.append(feature)


                text_features_for_c = torch.cat(text_features_for_c, dim=-1)  # [1, 512*num_views]
            else:
                # no descriptions in object
                prompt = clip.tokenize(f"a photo of a {c}").to(device=self.device)
                with torch.no_grad():
                    text_features_for_c = clip_model.encode_text(prompt)  # [1, 512]

            all_features.append(text_features_for_c)

        # [40, 512*num_views]
        final_features = torch.cat(all_features, dim=0)
        # print('final_features---------',final_features.shape)

        final_features = final_features / final_features.norm(dim=-1, keepdim=True)

        return final_features.float()

    def forward(self):
        prompt = self.get_label_feature()
        prompt = prompt.float()
        return prompt

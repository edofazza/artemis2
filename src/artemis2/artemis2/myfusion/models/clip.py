import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel
from torchmetrics.classification import MultilabelAveragePrecision


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=x)
        image_features = image_features.view(b, t, -1).mean(dim=1)
        image_features = F.normalize(image_features, dim=-1)
        return image_features


class CLIPExecutor(object):
    def __init__(self, test_loader, class_list, device):
        super(CLIPExecutor, self).__init__()
        self.test_loader = test_loader
        self.device = device
        self.image_model = CLIP().to(self.device)
        self.image_model.eval()
        self.class_embeds = self._get_text_features(class_list).to(self.device)
        self.num_classes = len(class_list)

    def _get_text_features(self, cl_names):
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        prompts = self._get_prompt(cl_names)
        text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    @staticmethod
    def _get_prompt(cl_names):
        return [c for c in cl_names]

    def evaluate(self):
        metric = MultilabelAveragePrecision(num_labels=self.num_classes, average="micro").to(self.device)

        for images, targets in self.test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device).int()

            with torch.no_grad():
                img_feats = self.image_model(images)  # [B, D]
                img_feats = F.normalize(img_feats, dim=-1)
                sims = img_feats @ self.class_embeds.T  # [B, num_classes]

            metric.update(sims, targets)

        mAP = metric.compute().item()
        print(f"Micro mAP (torchmetrics): {mAP:.4f}")
        return mAP

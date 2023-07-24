import argparse

import torch
import torch.nn as nn
import torchvision

from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor
import re


class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
    
class CLAPZeroShotClassifier(nn.Module):
    def __init__(self, model_path, labels, multi_label=False) -> None:
        super().__init__()
        print("model!", model_path)
        self.clap = ClapModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.loss_func = nn.CrossEntropyLoss()
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        self.labels = labels
        self.multi_label = multi_label
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
        out = self.clap(**inputs).logits_per_audio
        loss = self.loss_func(out, y)
        return loss, out
    
class CLAPClassifier(nn.Module):
    def __init__(self, model_path, num_classes, multi_label = False) -> None:
        super().__init__()
        self.clap = ClapAudioModelWithProjection.from_pretrained(model_path, projection_dim=num_classes,
                                                                ignore_mismatched_sizes=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.multi_label = multi_label

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
        out = self.clap(**inputs).audio_embeds
        loss = self.loss_func(out, y)
        return loss, out

class CLAPLanguageAudioClassifier(nn.Module):
    def __init__(self, model_path, labels, multi_label = False) -> None:
        super().__init__()
        self.clap = ClapModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
        clap_output = self.clap(**inputs, return_loss=True)
        out = clap_output.logits_per_audio
        loss = self.multimodal_loss(audio_embeds=clap_output.audio_embeds,
                                    text_embeds=clap_output.text_embeds, logit_scale_audio=self.clap.logit_scale_a.exp(), label_indices=y)
        return loss, out
    
    def multimodal_loss(self, audio_embeds, text_embeds, logit_scale_audio, label_indices):
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio
        return self.loss_func(logits_per_audio, label_indices)
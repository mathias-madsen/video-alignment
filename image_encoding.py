import os
import torch
import torchvision
import numpy as np
from PIL import Image


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])


def gridlike(image):
    h, w = image.shape[-2:]
    yspan = torch.linspace(-1, +1, h)
    xspan = torch.linspace(-1, +1, w)
    return torch.meshgrid(yspan, xspan, indexing="ij")


def softmax2d(logits):
    maxima = torch.max(logits, -1, keepdims=True).values
    maximaxima = torch.max(maxima, -2, keepdims=True).values
    scaledprobs = torch.exp(logits - maximaxima)
    norms = torch.sum(scaledprobs, dim=(-2, -1), keepdims=True)
    return scaledprobs / norms


def soft_argmax2d(logits):
    probs = softmax2d(logits)
    yspan, xspan = gridlike(logits)
    ymean = torch.sum(probs * yspan, dim=(-2, -1))
    xmean = torch.sum(probs * xspan, dim=(-2, -1))
    return torch.stack([ymean, xmean], axis=-1)


class ImageEncoder:

    def __init__(self):
        # weights = torchvision.models.resnet.ResNet18_Weights.DEFAULT
        weights = torchvision.models.resnet.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.eval()
        def _forward_impl(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            return soft_argmax2d(x)
            # x = model.avgpool(x)
            # x = torch.flatten(x, 1)
            # return torch.log(1e-5 + x)
        model._forward_impl = _forward_impl
        self.model = model

    def encode(self, uint8_rgb_image):
        """ Compute a latent vector for a raw RGB image. """
        # the preproceesor expects `PIL.Image`s:
        pil_rgb_image = Image.fromarray(uint8_rgb_image)
        input_tensor = preprocess(pil_rgb_image)
        with torch.no_grad():
            encoding = self.model(input_tensor.unsqueeze(0))
            return encoding.squeeze(0).numpy()



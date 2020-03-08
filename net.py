import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, weight_path=None, pretrained=True):
        super(Encoder, self).__init__()
        if weight_path:
            vgg = models.vgg19()
            vgg.load_state_dict(torch.load(weight_path))

        elif pretrained:
            vgg = models.vgg19(pretrained=True)

        else:
            raise OSError("VGG Model initialization error!")

        features = vgg.features

        self.enc_1 = features[0:2]
        self.enc_2 = features[2:7]
        self.enc_3 = features[7:12]
        self.enc_4 = features[12:21]
        self.name = ["enc_1", "enc_2", "enc_3", "enc_4"]

    def fix_param(self):
        for name in self.name:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, input_tensor):
        feature_1 = self.enc_1(input_tensor)
        feature_2 = self.enc_2(feature_1)
        feature_3 = self.enc_3(feature_2)
        feature_4 = self.enc_4(feature_3)
        return feature_1, feature_2, feature_3, feature_4


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class AdaINNetwork(nn.Module):
    def __init__(self, enc, dec):
        super(AdaINNetwork, self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.loss = nn.MSELoss()

        self.encoder.fix_param()

    def encode(self, input_tensor):
        # returns a tuple containing (ReLU1_1, ReLU2_1, ReLU3_1, ReLU4_1)
        return self.encoder(input_tensor)

    def content_loss(self, content, target):
        assert (content.size() == target.size())
        assert (target.requires_grad is False)
        return self.loss(content, target)

    def style_loss(self, content, target):
        assert (content.size() == target.size())
        assert (target.requires_grad is False)
        return content

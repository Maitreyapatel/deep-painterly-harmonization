from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--style", help="style image path",
                    type=str, required=False, default="images/16_target.jpg")
parser.add_argument("--input", help="input image path",
                    type=str, required=False, default="images/16_naive.jpg")
parser.add_argument("--mask", help="tight mask path",
                    type=str, required=False, default="images/16_c_mask.jpg")
parser.add_argument("--loose_mask", help="loose mask path",
                    type=str, required=False, default="images/16_c_mask_dilated.jpg")
args = parser.parse_args()


style_img = Image.open(args.style)
input_img = Image.open(args.input)
mask_img = Image.open(args.mask)
l_mask_image = Image.open(args.loose_mask)

# Define vgg model


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
vgg = VGG().cuda()
vgg.load_state_dict(torch.load("model_dir/" + 'vgg_conv.pth'))

# Define ImageNet Normalization
img_size = 256
prep = transforms.Compose([transforms.ToTensor(),
                           transforms.Lambda(
                               lambda x: x[torch.LongTensor([2, 1, 0])]),
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])

# normalize input and style image
input_norm = prep(input_img)
style_norm = prep(style_img)

# Now start the main algorithm

# First-Pass Algorithms(2,3)

# Take out conv layer features for both input and style image
output_input = vgg(Variable(input_norm.unsqueeze(0).cuda()),
                   out_keys=style_layers)
style_input = vgg(Variable(style_norm.unsqueeze(0).cuda()),
                  out_keys=style_layers)


#Another model for converting mask image input to the same size like output of conv3_1, conv4_1, conv5_1
class Convert(nn.Module):
    def __init__(self):
        super(Convert, self).__init__()
        pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x, out_keys):
        out = {}

        m = pool(Variable(x[None][None]))
        m = pool(m)
        m = m.data.squeeze().numpy()

        h,w = m.shape
        m = cv2.resize(m, (w//2,h//2))

        m = pool(Variable(m[None][None]))
        m = pool(m)
        m = m.data.squeeze().numpy()

        h,w = m.shape
        m = cv2.resize(m, (w//2,h//2))

        m = pool(Variable(m[None][None]))
        t1 = m.data.squeeze().numpy()

        out.append(t1)

        m = pool(Variable(t1[None][None]))
        m = pool(m)
        m = m.data.squeeze().numpy()

        h,w = m.shape
        m = cv2.resize(m, (w//2,h//2))

        m = pool(Variable(m[None][None]))
        t2 = m.data.squeeze().numpy()

        out.append(t2)

        m = pool(Variable(t2[None][None]))
        m = pool(m)
        m = m.data.squeeze().numpy()

        h,w = m.shape
        m = cv2.resize(m, (w//2,h//2))

        m = pool(Variable(m[None][None]))
        t3 = m.data.squeeze().numpy()

        out.append(t3)

        return out

# 1)Mapping


# 2)Reconstruction

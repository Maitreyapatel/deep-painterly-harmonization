from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from functools import partial
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--style", help="style image path",
                    type=str, required=False, default="images/6_target.jpg")
parser.add_argument("--input", help="input image path",
                    type=str, required=False, default="images/6_naive.jpg")
parser.add_argument("--mask", help="tight mask path",
                    type=str, required=False, default="images/6_c_mask.jpg")
parser.add_argument("--loose_mask", help="loose mask path",
                    type=str, required=False, default="images/6_c_mask_dilated.jpg")
args = parser.parse_args()


style_img = Image.open(args.style)
input_img = Image.open(args.input)
mask_img = Image.open(args.mask)#.convert('RGB')
l_mask_image = Image.open(args.loose_mask)#.convert('RGB')

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
vgg = VGG().cuda().eval()

for param in vgg.parameters():
    param.requires_grad = False

vgg.load_state_dict(torch.load("model_dir/" + 'vgg_conv.pth'))
print("Pretrained model loaded...!!")


# Define ImageNet Normalization
img_size = 256
prep = transforms.Compose([transforms.ToTensor(),
                           transforms.Lambda(
                               lambda x: x[torch.LongTensor([2, 1, 0])]),
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])


def halve_size(mask):
    h, w = np.array(mask).shape[:2]
    return cv2.resize(np.array(mask), (w // 2, h // 2))


input_img = Image.fromarray(halve_size(input_img))
style_img = Image.fromarray(halve_size(style_img))
mask_img = Image.fromarray(halve_size(mask_img))
l_mask_image = Image.fromarray(halve_size(l_mask_image))


# normalize input and style image
input_norm = prep(input_img)
style_norm = prep(style_img)


# Now start the main algorithm

# First-Pass Algorithms(2,3)

# Take out conv layer features for both input and style image
input_ftr = vgg(Variable(input_norm.unsqueeze(0)).cuda(),
                out_keys=style_layers)
style_ftr = vgg(Variable(style_norm.unsqueeze(0)).cuda(),
                out_keys=style_layers)


# Another model for converting mask image input to the same size like output of conv3_1, conv4_1, conv5_1
class Convert(nn.Module):
    def __init__(self):
        super(Convert, self).__init__()
        self.pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x):
        out = []

        m = self.pool(Variable(torch.from_numpy(
            x[None][None]).type(torch.DoubleTensor)))
        m = self.pool(m)

        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        m = self.pool(m)
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t1 = m.data.squeeze().numpy()

        out.append(t1)

        m = self.pool(Variable(torch.from_numpy(
            t1[None][None]).type(torch.DoubleTensor)))
        m = self.pool(m)
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t2 = m.data.squeeze().numpy()

        out.append(t2)

        m = self.pool(Variable(torch.from_numpy(
            t2[None][None]).type(torch.DoubleTensor)))
        m = self.pool(m)
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t3 = m.data.squeeze().numpy()

        out.append(t3)

        return out


cot = Convert()


temp = np.array(l_mask_image)[:,:,0]
mask_ftrs = cot(temp)


# 1)Mapping

def patches(x, ks=3, stride=1, padding=1):     # Time complexity is too much must be removed
    main = []
    ch, n1, n2 = x.shape
    y = np.zeros((ch, n1 + 2, n2 + 2))
    y[:, 1:n1 + 1, 1:n2 + 1] = x
    for i in range(n1):
        for j in range(n2):
            temp = []
            for c in range(ch):
                for m in range(3):
                    for n in range(3):
                        temp.append(y[c, i + m, j + n])
            main.append(temp)
    return torch.from_numpy(np.array(main))

def match_ftrs():
    res = []
    for i in range(3):
        l_inp = Variable(patches(
            input_ftr[2 + i].cpu().data.numpy()[0])).cuda()
        s_inp = Variable(patches(
            style_ftr[2 + i].cpu().data.numpy()[0])).cuda()
        scals = torch.mm(l_inp, s_inp.t())
        norms_in = torch.sqrt((l_inp ** 2).sum(1))
        norms_st = torch.sqrt((s_inp ** 2).sum(1))
        l_inp.cpu()
        del l_inp
        s_inp.cpu()
        del s_inp
        cosine_sim = scals / (1e-15 + norms_in.unsqueeze(1)
                              * norms_st.unsqueeze(0))
        _, idx_max = cosine_sim.max(1)
        res.append(idx_max.cpu().data.numpy())
    return res


map_ftrs = match_ftrs()


def map_style():
    res = []
    for sf, mapf in zip(style_ftr[2:], map_ftrs):
        sf = sf.cpu().data.numpy().reshape(sf.size(1), -1)
        sf = sf[:, mapf]
        res.append(Variable(torch.from_numpy(sf)).cuda())
    return res


sty_ftrs = map_style()



# 2)Reconstruction
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                  std=[1, 1, 1]),
                             transforms.Lambda(
                                 lambda x: x[torch.LongTensor([2, 1, 0])]),
                             ])

postpb = transforms.Compose([transforms.ToPILImage()])


def postp(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


opt_img = input_norm.clone().cuda()

opt_img_v = Variable(opt_img[None], requires_grad=True)
print(opt_img_v.shape)

max_iter = 700
show_iter = 100
optimizer = optim.LBFGS([opt_img_v], lr=1)


def gram(input):
    b, c, h, w = input.size()
    F = input.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G


def gram_org(input):
    x = input
    return torch.mm(x, x.t())


def gram_mse_loss(input, target): return F.mse_loss(
    gram_org(input), gram_org(target))

n_iter = 0
while n_iter < max_iter:


    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img_v, style_layers)
        out_ftrs = out[2:]

        msk_of = out_ftrs[1].cuda() * Variable(torch.from_numpy(
            mask_ftrs[1][None, None]).type(torch.FloatTensor), requires_grad=False).cuda()
        msk_if = input_ftr[3].cuda() * Variable(torch.from_numpy(mask_ftrs[1]
                                                                 [None, None]).type(torch.FloatTensor), requires_grad=False).cuda()
        c_loss = F.mse_loss(msk_of, msk_if, size_average=False) / float(out_ftrs[1].size(1) * torch.from_numpy(mask_ftrs[1]).sum())


        loss = 0
        i = 0
        for of, sf, mf in zip(out_ftrs, sty_ftrs, mask_ftrs):
            to_pass = of.cuda() * Variable(torch.from_numpy(mf[None, None]).type(
                torch.FloatTensor), requires_grad=False).cuda()
            to_pass = to_pass.view(to_pass.size(1), -1)
            sf = sf.cuda() * Variable(torch.from_numpy(mf).type(torch.FloatTensor),
                                      requires_grad=False).view(1, -1).cuda()
            i += 1
            loss += gram_mse_loss(to_pass, sf)

        s_loss = loss / 3

        t_loss = 1 * c_loss + 10 * s_loss
        t_loss.backward()

        global n_iter
        n_iter += 1
        if n_iter % show_iter == 0:
            print('Iteration: {}, loss: {}'.format(n_iter, loss.data[0]))

        return t_loss


    optimizer.step(closure)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
out_img = np.array(postp(opt_img_v.cpu().data.squeeze()))/255
print("asj:",out_img.shape)


out_img = out_img * (np.array(mask_img)/255) + (np.array(style_img)/255) * (1 - (np.array(mask_img))/255)
ax.imshow(out_img)
ax.axis('off')
plt.show()
fig.savefig('out.png', bbox_inches='tight', pad_inches=0)

print("Stage 1 is completed...!")

np.save('Results/numpy/6_stage1.npy', out_img)      #Just for the shake of saving

## Phase 2 - Algorithms 3 and 4(Main)

# Load stage1 result numpy file
stage1_img = np.load('Results/numpy/5_stage1.npy')
stage1_norm = prep(stage1_img)

stage1_ftr = vgg(Variable(stage1_norm.unsqueeze(0)).type(torch.FloatTensor).cuda(),
                out_keys=style_layers)

# Another convertng model for mask images
class Convert(nn.Module):
    def __init__(self):
        super(Convert, self).__init__()
        self.pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x):
        out = []

        m = self.pool(Variable(torch.from_numpy(
            x[None][None]).type(torch.DoubleTensor)))
        t1 = m.data.squeeze().numpy()

        out.append(t1)


        m = self.pool(Variable(torch.from_numpy(
            t1[None][None]).type(torch.DoubleTensor)))
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t2 = m.data.squeeze().numpy()

        out.append(t2)



        m = self.pool(Variable(torch.from_numpy(
            t2[None][None]).type(torch.DoubleTensor)))
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t3 = m.data.squeeze().numpy()

        out.append(t3)

        m = self.pool(Variable(torch.from_numpy(
            t3[None][None]).type(torch.DoubleTensor)))
        m = self.pool(m)
        m = m.data.squeeze().numpy()

        h, w = m.shape
        m = cv2.resize(m, (w // 2, h // 2))

        m = self.pool(Variable(torch.from_numpy(
            m[None][None]).type(torch.DoubleTensor)))
        t4 = m.data.squeeze().numpy()

        out.append(t4)

        return out


cot = Convert()


temp = np.array(l_mask_image)[:,:,0]
mask_ftrs = cot(temp)

def match_ftrs():
    res = []
    l_inp = Variable(patches(
        stage1_ftr[3].cpu().data.numpy()[0])).cuda()
    s_inp = Variable(patches(
        style_ftr[3].cpu().data.numpy()[0])).cuda()
    scals = torch.mm(l_inp, s_inp.t())
    norms_in = torch.sqrt((l_inp ** 2).sum(1))
    norms_st = torch.sqrt((s_inp ** 2).sum(1))
    l_inp.cpu()
    del l_inp
    s_inp.cpu()
    del s_inp
    cosine_sim = scals / (1e-15 + norms_in.unsqueeze(1)
                          * norms_st.unsqueeze(0))
    _, idx_max = cosine_sim.max(1)
    res.append(idx_max.cpu().data.numpy())
    return res

stage_style_map_ftrs = match_ftrs()

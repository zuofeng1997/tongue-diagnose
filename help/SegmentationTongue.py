from PIL import Image

import numpy as np

from torchvision import transforms, utils,models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt




def Segmentation(good_pics):
    device = torch.device('cuda:0')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model = model.to(device)
    model.load_state_dict(torch.load('weights/SegmetationTongue.pkl'))
    model.eval()

    def showImg(img, label):
        img = img.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        x = [label[0] * 224, label[2] * 224, label[4] * 224, label[6] * 224]
        y = [label[1] * 224, label[3] * 224, label[5] * 224, label[7] * 224]
        plt.cla()
        plt.scatter(x, y, s=100, marker='.', c='r')
        plt.imshow(img)
        plt.pause(0.1)
    n = len(good_pics)
    for i in range(n):
        ax = plt.figure(n)

        plt.subplot(1,n,i+1)
        origin_img = Image.open(good_pics[i])
        img = data_transform(origin_img)
        prediction = model(img.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        im_size = origin_img.size
        # print(im_size)
        p1 = [prediction[0]*im_size[0],prediction[1]*im_size[1]]
        p3 = [prediction[4]*im_size[0],prediction[5]*im_size[1]]
        # print(p1,p3)
        pic_name = good_pics[i][-(good_pics[i])[::-1].index("/"):]
        print("保存完整舌头图片到/home/zf/segmentation/"+pic_name)
        origin_img.crop((p1[0],p1[1],p3[0],p3[1])).save("/home/zf/segmentation/"+pic_name)
        showImg(img,prediction)
    plt.show()
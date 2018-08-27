import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from PIL import Image
def classifyTongue(data_dir):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load('weights/classify.pkl'))

    data_transforms =transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    list_img = os.listdir(data_dir)

    data = torch.stack([data_transforms(Image.open(data_dir+imgurl)) for imgurl in list_img])
    with torch.no_grad():
        out = model_ft(data.to(device))

    result = out.cpu().numpy()
    good_img = []
    for i,image in enumerate(list_img):
        if result[i][0]<result[i][1]:
            good_img.append(image)
    n = len(good_img)
    plt.figure(n)

    fullGoogPath = [data_dir+img_url for img_url in good_img]
    for i in range(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(Image.open(fullGoogPath[i]))
        ax.axis('off')
    plt.show()
    print("筛选出正常舌头图片",good_img)
    return fullGoogPath








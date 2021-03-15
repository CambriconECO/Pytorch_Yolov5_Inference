import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
import argparse
from collections import OrderedDict
from PIL import Image
import cv2
import os
import yolo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml',help='model.yaml')
    parser.add_argument('--device', default='cpu',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    # 获取yolov5网络文件
    net = yolo.get_model(opt)
    # 配置量化参数
    qconfig={'iteration': 1, 'use_avg':False, 'data_scale':1.0, 'firstconv':False, 'per_channel': False}
    # 调用量化接口
    quantized_net = mlu_quantize.quantize_dynamic_mlu(net.float(),qconfig_spec=qconfig, dtype='int8', gen_quant=True)
    # 设置为推理模式
    quantized_net = quantized_net.eval().float()
    # 对图片作预处理
    img_mat = Image.open("./images/image.jpg")
    if img_mat.mode != 'RGB':
        img_mat = img_mat.convert('RGB')
    crop = 640
    resize = 640 
    transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
        ])
    img = transform(img_mat)
    im_tensor = torch.unsqueeze(img, 0)
    im_tensor = im_tensor.float()
    print(im_tensor.shape)
    # 执行推理生成量化值 
    quantized_net(im_tensor)
    # 保存量化模型
    torch.save(quantized_net.state_dict(), './yolov5s_int8.pt')

import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torchvision.transforms as transforms
import argparse
from collections import OrderedDict
from PIL import Image
from data_loader import LoadImages
from post_process import plot_one_box,scale_coords
import cv2
import os
import yolo
import random

def get_boxes(prediction, batch_size=1):
    """
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_pred)
    """
    reshape_value = torch.reshape(prediction, (-1, 1))

    num_boxes_final = reshape_value[0].item()
    print('num_boxes_final: ',num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64 + i * 7 + 0].item())
        if batch_idx >= 0 and batch_idx < batch_size:
            bl = reshape_value[64 + i * 7 + 3].item()
            br = reshape_value[64 + i * 7 + 4].item()
            bt = reshape_value[64 + i * 7 + 5].item()
            bb = reshape_value[64 + i * 7 + 6].item()

            if bt - bl > 0 and bb -br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 1].item())

    outputs = [torch.FloatTensor(all_list[i]).reshape(-1, 6) for i in range(batch_size)]
    return outputs

def letter_box(img, new_shape= (640, 640), color=(114, 114, 114)):
    img = img[:, :, (2, 1, 0)]
    
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.float()
    return img

def draw_boxes(box_result):
    names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(box_result):  # detections per image
                im0 = cv2.imread('images/image.jpg')
                save_path = './results/yolov5_image.jpg'
                s = ''
                #save_path=Path(p).name
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    print(s)
                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_path, im0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml',help='model.yaml')
    parser.add_argument('--device', default='cpu',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--jit',type=bool,help='fusion',default=False)
    parser.add_argument('--save',type=bool,default=False,help='selection of save *.cambrcion')
    opt = parser.parse_args()
    # 获取yolov5网络文件
    net = yolo.get_empty_model(opt)
    quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(net)
    state_dict = torch.load('yolov5s_int8.pt')
    quantized_net.load_state_dict(state_dict, strict=False)
    # 设置为推理模式
    quantized_net = quantized_net.eval().float()
    device = ct.mlu_device()
    quantized_net.to(ct.mlu_device())
    # 读取图片
    img_mat = cv2.imread('images/image.jpg')
    # 预处理
    img = letter_box(img_mat)
    print(img.shape)
    # 设置在线融合模式
    if opt.jit:
        if opt.save:
            ct.save_as_cambricon('yolov5s')
        torch.set_grad_enabled(False)
        ct.set_core_number(4)
        trace_input = torch.randn(1, 3, 640, 640, dtype=torch.float)
        trace_input = trace_input.to(ct.mlu_device())
        quantized_net = torch.jit.trace(quantized_net, trace_input, check_trace = False)
    # 推理
    detect_out = quantized_net(img.to(ct.mlu_device()))

    if opt.jit:
        if opt.save:
            ct.save_as_cambricon('')
    # 后处理
    #anchors = [10, 13, 16, 30, 33, 23,30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    detect_out=detect_out.to(torch.device('cpu'))
    print(detect_out.shape)
    #print('==============================')
    #det = detect_out.clone()
    #det = det.numpy().tolist()
    #print(det)
    #print('==============================')
    # 画框，标出类别和置信度并保存图片
    box_result = get_boxes(detect_out)
    print(box_result)
    draw_boxes(box_result)

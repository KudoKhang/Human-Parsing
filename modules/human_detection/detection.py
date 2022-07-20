import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def crop(self, bbox, image):
        bbox = self.remove_bbox_noise(bbox, image)
        for bb in bbox:
            return image[bb[1]:bb[3], bb[0]:bb[2]]

    def remove_bbox_noise(self, bbox, image, thresh=0.3):
        area_image = image.shape[0] * image.shape[1]
        new_bb = []
        for bb in bbox:
            x1, y1, x2, y2 = bb
            area_bbox = (bb[2] - bb[0]) * (bb[3] - bb[1])
            ratio = area_bbox / area_image
            if ratio > thresh:
                new_bb.append(bb)
        return new_bb

    def run(self, image):
        result = self.model(image)
        output = result.pandas().xyxy[0]
        bbox = np.int32(np.array(output)[:,:4][np.where(np.array(output)[:,6] == 'person')])
        image = self.crop(bbox, image)
        return image

# path = 'g1.jpg'
# detection = Detection()
# image = cv2.imread('g1.jpg')
# i = detection.run(image)
# cv2.imshow('result', i)
# cv2.waitKey(0)
# print(bbox)

#
# import torch
# from PIL import Image
# import torchvision
#
# #other lib
# import sys
# import numpy as np
# import os
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt
#
# #Yolov5
# sys.path.insert(0, "yolov5")
#
# # from yolov5.utilss.general import non_max_suppression, scale_coords, check_img_size
# from yolov5.models.experimental import attempt_load
# # from yolov5.utilss.datasets import letterbox
# from yolov5.yolo_utils import non_max_suppression, scale_coords, resize_image
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = attempt_load("yolov5/yolov5s.pt", map_location=device)
# size_convert = 640  # setup size de day qua model
# conf_thres = 0.4
# iou_thres = 0.5
#
# path_query = "g1.jpg"
# orgimg = cv2.imread(path_query)  # BGR
# img = resize_image(orgimg.copy(), size_convert).to(device)
#
# with torch.no_grad():
#     pred = model(img[None, :])[0]
#     det = non_max_suppression(pred, conf_thres, iou_thres)[0] # x1,y1,x2,y2, score, class
#     det = det[det[:, -1] == 0] # body: 0, face:1
#
#     bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], orgimg.shape[:-1]).round().cpu().numpy())
#
# print(bboxs)
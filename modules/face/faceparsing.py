import os
import cv2
from PIL import Image
import time
import numpy as np
import torch
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap

class Ibug_Parsing():
    def __init__(self, threshold=0.8, encoder='rtnet50', decoder='fcn',
                        num_classes=11, max_num_faces=50,
                        weights='./ibug/face_parsing/rtnet/weights/rtnet50-fcn-11.torch'):
        self.threshold = threshold
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.max_num_faces = max_num_faces
        self.weights = weights
        self.alphas = np.linspace(0.75, 0.25, num=self.max_num_faces)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.face_detector = RetinaFacePredictor(threshold=self.threshold, device=self.device,
                                            model=(RetinaFacePredictor.get_model('mobilenet0.25')))

        self.face_parser = RTNetPredictor(device=self.device,
                                     ckpt=self.weights,
                                     encoder=self.encoder,
                                     decoder=self.decoder,
                                     num_classes=self.num_classes)

        self.colormap = label_colormap(self.num_classes)
        # print('Face detector created using RetinaFace.')

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def get_face_mask(self, mask):
        for i in range(1, 10):
            mask[np.where(mask == i)] = 255
        mask[np.where(mask != 255)] = 0
        return mask

    def run(self, frame):
        # Detect faces
        frame = self.check_type(frame)
        faces = self.face_detector(frame, rgb=False)
        if len(faces) > 0:
            # Parse faces
            masks = self.face_parser.predict_img(frame, faces, rgb=False)
            masks = self.get_face_mask(masks[0])
            return masks
        print('No face detected!!!')
        return frame # Mask da len mau

#--------------------------------------------------------------------------------------------------
def image(path_img='../human_detection/g1.jpg'):
    masks = Face_parsing_predictor.run(path_img)
    for i, mask in enumerate(masks):
        m = mask.astype(np.uint8) * 100
        cv2.imshow(f'Result{i}', m)
    cv2.waitKey(0)
#--------------------------------------------------------------------------------------------------
Face_parsing_predictor = Ibug_Parsing()
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # image('../human_detection/g1.jpg')
    masks = Face_parsing_predictor.run('../human_detection/g1.jpg')

    # cv2.imshow('mask', masks)
    # cv2.waitKey(0)
    print(np.unique(masks))

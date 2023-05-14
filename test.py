import os
from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt

path = "D:\project\FIDS30\FIDS30 customized\mix\\"

for photo in os.listdir(path):
    if photo[-4:] == '.txt':
        continue
    img = cv2.imread(path + photo)#[:,:,::-1] #OpenCV uses BGR channels
    bboxes = pkl.load(open(f'{path[:-1]} pkl\\{photo[:-4]}.pkl', "rb"))
    # plotted_img = draw_rect(img, bboxes)
    # plt.imshow(plotted_img)
    # plt.show()
    for i in range(5):
        seq = Sequence([RandomHSV(100, 100, 100),
                        RandomHorizontalFlip(),
                        RandomScale(0.1), #
                        RandomTranslate(0.2),
                        RandomRotate(10),
                        RandomShear(0.1)])
        # seq = Sequence([RandomHorizontalFlip()])
        img_, bboxes_ = seq(img, bboxes)
        cv2.imwrite(f'D:\project\codes\DataAugmentationForObjectDetection\outputs\\{photo[:-4]}_{i}.jpg',img_)
        #save bbox
        plt.imshow(draw_rect(img_, bboxes_))


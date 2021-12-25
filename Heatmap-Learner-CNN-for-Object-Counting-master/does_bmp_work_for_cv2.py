# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:42:30 2021

@author: aaa
"""

import cv2
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

image = cv2.imread('results/heatmap-20160331_NTU_00001.bmp', cv2.IMREAD_COLOR)
# cv2.imshow('a', image)
# cv2.waitKey()

# 归一化
image = (image - image.min()) / (image.max() - image.min())
# 阈值选0.1
image[image < 0.1] = 0
# 取出背景
background = (image == 0)
# 对背景做形态学腐蚀
eroded_background = binary_erosion(background, border_value=1)
cv2.imwrite('bg-20160331_NTU_00001.png', eroded_background * 255)
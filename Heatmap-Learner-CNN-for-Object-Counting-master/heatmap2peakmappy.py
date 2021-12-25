# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:32:15 2021

@author: aaa
"""

import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):
    # 归一化
    image = (image - image.min()) / (image.max() - image.min())
    # 阈值选0.1
    image[image < 0.1] = 0
    neighborhood = generate_binary_structure(2, 2)
    # 局部极大
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # 取出背景
    background = (image == 0)
    # 对背景做形态学腐蚀
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    # 局部极大跟背景做异或
    detected_peaks = local_max ^ eroded_background
    return background


if __name__ == "__main__":
    image = cv2.imread("results/heatmap-456.png", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    PMimage = detect_peaks(image)
    # PMimage = np.array(PMimage * 255, dtype="uint8")
    # cv2.imwrite("peaks.png", PMimage)
    # 一样的，用cv2和PIL.Image存都行，PIL不需要*255，convert("L")的意思是转换为灰度图
    peakI = Image.fromarray(PMimage).convert("L")
    peakI.save("peakmap.bmp")

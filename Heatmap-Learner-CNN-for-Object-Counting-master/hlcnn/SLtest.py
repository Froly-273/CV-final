# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:45:36 2021

@author: Tianhong Wang @ AI department, SEIEE, SJTU
rinkitsune@sjtu.edu.cn
"""

import cv2
import numpy as np
import sys
from numpy import *
import math

def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.zeros(gray_image.shape)
  for i in range(gray_image.shape[0]):
      for j in range(gray_image.shape[1]):
          if gray_image[i][j] >= thresh_val:
              binary_image[i][j] = 255
  return mat(binary_image, dtype='uint8')

def find(i, j, parent):
  # find operation in union find set, using path compression
  tmp = parent[i][j]
  if [i, j] != tmp:
      parent[i][j] = find(tmp[0], tmp[1], parent)
  return parent[i][j]

labels = 0
def label(binary_image):
  # TODO
  BG = 0
  FG = 1
  global labels         # 对全局变量的修改必须在函数里声明global
  
  # implement this by union find set
  # this array memorizes the parent node of each vertex
  # parent = [[-1,-1]] * binary_image.shape[0] * binary_image.shape[1]
  parent = [[[-1, -1] for col in range(binary_image.shape[1])] for row in range(binary_image.shape[0])]
  roots = {}
  label = 0
  
  # first pass : only find the connected component, do no labeling
  # initialize : first row & first column
  for j in range(binary_image.shape[1]):
      if (binary_image[0, j] == FG):
          # the first pixel
          if (j == 0):
              parent[0][j] = [0, j]
              # roots[(0, j)] = ++label     # python没有++操作
              label = label + 1
              roots[(0, j)] = label
          # other pixels
          elif (binary_image[0, j-1] == FG):
              parent[0][j] = find(0, j-1, parent)
          else:
              parent[0][j] = [0, j]
              label = label + 1
              roots[(0, j)] = label
  
  for i in range(1, binary_image.shape[0]):
      if (binary_image[i, 0] == FG):
          if (binary_image[i-1, 0] == FG):
              parent[i][0] = find(i-1, 0, parent)
          else:
              parent[i][0] = [i, 0]
              label = label + 1
              roots[(i, 0)] = label
  
  # sequential labeling via union find set
  for i in range(1, binary_image.shape[0]):
      for j in range(1, binary_image.shape[1]):
          # X X
          # X 0 : BG
          if (binary_image[i, j] == BG):
              continue
          # 0 0
          # 0 1 : new label
          elif (binary_image[i-1, j-1] == BG and binary_image[i-1, j] == BG 
                and binary_image[i, j-1] == BG and binary_image[i, j] == FG):
              parent[i][j] = [i, j]
              label = label + 1
              roots[(i, j)] = label
          # D X
          # X 1 : label = D
          elif (binary_image[i-1, j-1] == FG and binary_image[i, j] == FG):
              parent[i][j] = find(i-1, j-1, parent)
          # 0 0
          # C 1 : label = C
          elif (binary_image[i-1, j-1] == BG and binary_image[i-1, j] == BG 
                and binary_image[i, j-1] == FG and binary_image[i, j] == FG):
              parent[i][j] = find(i, j-1, parent)
          # 0 B
          # 0 1 : label = B
          elif (binary_image[i-1, j-1] == BG and binary_image[i-1, j] == FG 
                and binary_image[i, j-1] == BG and binary_image[i, j] == FG):
              parent[i][j] = find(i-1, j, parent)
          # 0 B
          # C 1 : label = B = C
          else:
              rootB = find(i-1, j, parent)
              rootC = find(i, j-1, parent)
              if (rootB != rootC):
                  # parent[i][j-1] = rootB      # 你为什么只C->rB，而不是rC->rB
                  parent[rootC[0]][rootC[1]] = rootB
                  del roots[(rootC[0], rootC[1])]
              parent[i][j] = rootB
  
  # second pass : labeling
  #labeled_image = [[[0] for col in range(binary_image.shape[1])] for row in range(binary_image.shape[0])]
  labeled_image = np.zeros(binary_image.shape)  # cv2不认list，得用numpy创建
  labels = len(roots)
  # print(labels)
  
  # rearrange labels
  label = 0
  for key in roots:
      label = label + 1
      roots[key] = label
  # print(roots)
  # print(labels)
  
  for i in range(binary_image.shape[0]):
      for j in range(binary_image.shape[1]):
          if (binary_image[i, j] == FG):
              tmp = find(i, j, parent)
              if (tmp[0], tmp[1]) in roots:
                  #labeled_image[i][j] = int(FG / labels) * roots[(tmp[0], tmp[1])]  # eg: 6 of 7 components with grayscale 255*6/7
                  labeled_image[i][j] = int(roots[(tmp[0], tmp[1])] * 255.0 / labels)   # 上面那种写法labels特别多的时候整个矩阵就变成0了
                  
  return mat(labeled_image, dtype='uint8')


if __name__ == "__main__":
    img_name = '20161225_TPZ_00004'
    img = cv2.imread('res1/heatmap-'+img_name+'.bmp', cv2.IMREAD_COLOR)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = (gimg - gimg.min()) / (gimg.max() - gimg.min())
    gimg[gimg < 0.5] = 0
    gimg[gimg >= 0.5] = 1
    lbimg = label(gimg)
    cv2.imwrite(img_name+'_SLvisualize.png',lbimg)

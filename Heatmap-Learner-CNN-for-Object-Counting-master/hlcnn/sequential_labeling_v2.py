# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:45:01 2021

@author: aaa
"""

import numpy as np
import cv2
import os
from scipy.ndimage.morphology import binary_erosion

# 跟v1的sequential labeling有一点不同
# 返回的不再是len(roots)，因为sequential labeling只是基于connected component的，无法对多个粘连在一起的东西分开计数
# 所以这次roots字典统计的是每一个connected component里的像素数
# 一遍sequential labeling跑完之后，先求出每个connected component像素数的均值
# 差不多等于这个均值的2倍的，就分成2个计数；差不多等于3倍的，就分成3个计数
# 这个idea来自于一个直觉：经过thresholding之后，两个东西就算粘连在一起了，由于粘连部分不会很粗，不影响它们总体加起来是2倍
# 不建议使用在bg上，因为bg更粗，会有一大段一大段的粘连


# 使用路径压缩的并查集find操作
def find(i, j, parent):
    tmp = parent[i][j]
    if [i, j] != tmp:
        parent[i][j] = find(tmp[0], tmp[1], parent)
    return parent[i][j]


def sequential_labeling(binary_image, clip_thr=0.5, aggregate='median'):
    BG = 0
    FG = 1

    # implement this by union find set
    # this array memorizes the parent node of each vertex
    parent = [[[-1, -1] for col in range(binary_image.shape[1])] for row in range(binary_image.shape[0])]
    roots = {}

    # first pass : only find the connected component, do no labeling
    # initialize : first row & first column
    for j in range(binary_image.shape[1]):
        if (binary_image[0, j] == FG):
            # the first pixel
            if (j == 0):
                parent[0][j] = [0, j]
                roots[(0, j)] = 1
            # other pixels
            elif (binary_image[0, j - 1] == FG):
                parent[0][j] = find(0, j - 1, parent)
                roots[(parent[0][j][0], parent[0][j][1])] += 1
            else:
                parent[0][j] = [0, j]
                roots[(0, j)] = 1

    for i in range(1, binary_image.shape[0]):
        if (binary_image[i, 0] == FG):
            if (binary_image[i - 1, 0] == FG):
                parent[i][0] = find(i - 1, 0, parent)
                roots[(parent[i][0][0], parent[i][0][1])] += 1
            else:
                parent[i][0] = [i, 0]
                roots[(i, 0)] = 1

    # sequential labeling via union find set
    for i in range(1, binary_image.shape[0]):
        for j in range(1, binary_image.shape[1]):
            # X X
            # X 0 : BG
            if (binary_image[i, j] == BG):
                continue
            # 0 0
            # 0 1 : new label
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == BG
                  and binary_image[i, j - 1] == BG and binary_image[i, j] == FG):
                parent[i][j] = [i, j]
                roots[(i, j)] = 1
            # D X
            # X 1 : label = D
            elif (binary_image[i - 1, j - 1] == FG and binary_image[i, j] == FG):
                parent[i][j] = find(i - 1, j - 1, parent)
                roots[(parent[i][j][0], parent[i][j][1])] += 1
            # 0 0
            # C 1 : label = C
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == BG
                  and binary_image[i, j - 1] == FG and binary_image[i, j] == FG):
                parent[i][j] = find(i, j - 1, parent)
                roots[(parent[i][j][0], parent[i][j][1])] += 1
            # 0 B
            # 0 1 : label = B
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == FG
                  and binary_image[i, j - 1] == BG and binary_image[i, j] == FG):
                parent[i][j] = find(i - 1, j, parent)
                roots[(parent[i][j][0], parent[i][j][1])] += 1
            # 0 B
            # C 1 : label = B = C
            else:
                rootB = find(i - 1, j, parent)
                rootC = find(i, j - 1, parent)
                if (rootB != rootC):
                    parent[rootC[0]][rootC[1]] = rootB
                    if (rootC[0], rootC[1]) in roots:
                        roots[(rootB[0], rootB[1])] += roots[(rootC[0], rootC[1])]
                        del roots[(rootC[0], rootC[1])]
                parent[i][j] = rootB
    
    # print(roots)
    # return len(roots)       # 已弃用
    
    # 后处理，求每个component的mean pixels -> 根据每个component的pixel相对于baseline的比值，clip到整数的counts
    key_list = list(roots.values())
    
    # 求均值
    if aggregate == 'mean':
        baseline = np.mean(key_list)
    # median of medians。先分组求中位数，再求它们的中位数
    # 默认每组5个元素
    elif aggregate == 'medianofmedians':
        medians = []
        for i in range(int(len(key_list) / 5)):
            sublist = key_list[5*i:5*i+4]
            medians.append(np.median(sublist))
        if i < len(key_list) - 1:
            sublist = key_list[5*i:]
            medians.append(np.median(sublist))
        baseline = np.median(medians)
    # 中位数
    else:
        baseline = np.median(key_list)
    
    counts = 0
    # 在clip threshold=0.5的情况下，0~1.5会被算作1，1.5~2.5会被算作2，2.5~3.5会被算作3，以此类推
    # 如果clip threshold不是0.5，那么向下的误差限会更大些
    # 比如0.3，那么0~1.3会被算作1，1.3~2.3会被算作2，以此类推
    for key in roots.keys():
        ratio = float(roots[key] / baseline)
        if ratio > 0 and ratio <= 1 + clip_thr:
            counts += 1
        elif ratio > 0:
            counts += int(ratio - clip_thr) + 1
    
    return counts

# 只是把detect_peaks的最后一部分删掉了，这次我要获取背景
def get_bg(image, thr=0.8, bg=False):
    # 归一化
    image = (image - image.min()) / (image.max() - image.min())
    # 阈值选0.1
    image[image < thr] = 0
    # （只在需要用前景的时候）前景变成1
    image[image >= thr] = 1
    if not bg:
        return image
    else:
        # 取出背景
        background = (image == 0)
        # 对背景做形态学腐蚀
        eroded_background = binary_erosion(background, border_value=1)
        return eroded_background



# 在heatmap上使用sequential labeling进行counting
def counting_sl(thr=0.8, clip_thr=0.5, is_bg=False, aggregate='median'):
    assert aggregate in ['mean', 'median', 'medianofmedians']
    
    with open('CARPK/ImageSets/test.txt', 'r') as f:
        lines = f.readlines()
    
    AE = 0
    SE = 0
    RAE = 0
    for line in lines:
        image_name = line.strip('\n')
        
        # 获取ground truth counts
        with open('CARPK/Annotations/' + image_name + '.txt', 'r') as anno:
            anno_lines = anno.readlines()
        gt_counts = len(anno_lines)
        
        # heatmap -> eroded background -> sequential labeling -> predicted counts
        image = cv2.imread('res1/heatmap-' + image_name + '.bmp', cv2.IMREAD_COLOR)
        
        # save_path = 'problematic images/'+ str(thr)
        # cv2.imwrite(save_path + '/' + image_name + '_heatmap.jpg', image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg = get_bg(image, thr, bg=is_bg)
        counts = sequential_labeling(bg, clip_thr, aggregate)
        
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        
        # cv2.imwrite(save_path + '/' + image_name + '_bg.jpg', bg * 255)
        
        # 更新统计量
        err = abs(counts - gt_counts)
        AE += err
        SE += err ** 2
        RAE = max(RAE, err)
        print(image_name, '\t', counts, ' ', gt_counts, '\t', 'AE: ', err)
    
    print('MAE:', AE / len(lines))
    print('RMSE:', np.sqrt(SE / len(lines)))
    print('RAE:', RAE)
    

if __name__ == "__main__":
    counting_sl(thr=0.9, clip_thr=0.4, is_bg=False, aggregate='median')
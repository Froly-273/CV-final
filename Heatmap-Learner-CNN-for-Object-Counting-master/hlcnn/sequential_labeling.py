import numpy as np
import cv2
import os
from scipy.ndimage.morphology import binary_erosion


# 跟课上的sequential labeling有几点不同
# 第一，不需要给出每一类的区别，直接len(roots)就是基于senquantial labeling的counting


# 使用路径压缩的并查集find操作
def find(i, j, parent):
    tmp = parent[i][j]
    if [i, j] != tmp:
        parent[i][j] = find(tmp[0], tmp[1], parent)
    return parent[i][j]


def sequential_labeling(binary_image):
    BG = 0
    FG = 1

    # implement this by union find set
    # this array memorizes the parent node of each vertex
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
                label = label + 1
                roots[(0, j)] = label
            # other pixels
            elif (binary_image[0, j - 1] == FG):
                parent[0][j] = find(0, j - 1, parent)
            else:
                parent[0][j] = [0, j]
                label = label + 1
                roots[(0, j)] = label

    for i in range(1, binary_image.shape[0]):
        if (binary_image[i, 0] == FG):
            if (binary_image[i - 1, 0] == FG):
                parent[i][0] = find(i - 1, 0, parent)
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
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == BG
                  and binary_image[i, j - 1] == BG and binary_image[i, j] == FG):
                parent[i][j] = [i, j]
                label = label + 1
                roots[(i, j)] = label
            # D X
            # X 1 : label = D
            elif (binary_image[i - 1, j - 1] == FG and binary_image[i, j] == FG):
                parent[i][j] = find(i - 1, j - 1, parent)
            # 0 0
            # C 1 : label = C
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == BG
                  and binary_image[i, j - 1] == FG and binary_image[i, j] == FG):
                parent[i][j] = find(i, j - 1, parent)
            # 0 B
            # 0 1 : label = B
            elif (binary_image[i - 1, j - 1] == BG and binary_image[i - 1, j] == FG
                  and binary_image[i, j - 1] == BG and binary_image[i, j] == FG):
                parent[i][j] = find(i - 1, j, parent)
            # 0 B
            # C 1 : label = B = C
            else:
                rootB = find(i - 1, j, parent)
                rootC = find(i, j - 1, parent)
                if (rootB != rootC):
                    parent[rootC[0]][rootC[1]] = rootB
                    if (rootC[0], rootC[1]) in roots:
                        del roots[(rootC[0], rootC[1])]
                parent[i][j] = rootB
    
    return len(roots)       # 这个值就是在黑白图上面数出来的counts了，对于测试例来说是112个




# 只是把detect_peaks的最后一部分删掉了，这次我要获取背景
def get_bg(image, thr=0.1, bg=False):
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
def counting_sl(thr=0.1, is_bg=False):
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
        save_path = 'problematic images/'+ str(thr)
        cv2.imwrite(save_path + '/' + image_name + '_heatmap.jpg', image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg = get_bg(image, thr, bg=is_bg)
        counts = sequential_labeling(bg)
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        cv2.imwrite(save_path + '/' + image_name + '_bg.jpg', bg * 255)
        
        
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
    counting_sl(0.95, is_bg=False)
import cv2
import os
from PIL import Image
import random
import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

# ?????????detect_peaks??????????????????????????????????????????????????????
def get_bg(image):
    # ?????????
    image = (image - image.min()) / (image.max() - image.min())
    # ?????????0.1
    image[image < 0.1] = 0
    # ????????????
    background = (image == 0)
    # ???????????????????????????
    eroded_background = binary_erosion(background, border_value=1)

    return eroded_background


# ????????????????????????
def drop_boxes(ratio=1):
    # ?????????????????????????????????
    with open('CARPK/ImageSets/train.txt', 'r') as f:
        fnames = f.readlines()

    # ?????????annotation????????????
    dirpath = 'CARPK/Annotations_{%.2f}'%ratio
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    # ??????????????????
    for imagename in fnames:
        imagename = imagename.strip('\n')
        boxes = []
        with open('CARPK/Annotations/'+imagename+'.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if ratio < 1:
                r = random.random()
                if r > ratio:
                    continue
            boxes.append(line)

        # ??????drop???box?????????
        img = cv2.imread('CARPK/Images/'+imagename+'.png', cv2.IMREAD_COLOR)
        vis_img = vis(img, boxes)
        cv2.imwrite(dirpath+'/'+imagename+'.jpg', vis_img)

        # ??????????????????heatmap
        hmap = cv2.imread('res1/heatmap-'+imagename+'.bmp', cv2.IMREAD_COLOR)
        print(imagename)
        bg = get_bg(hmap)
        cv2.imwrite(dirpath+'/'+imagename+'.jpg', bg)

        with open(dirpath+'/'+imagename+'.txt', 'w') as f:
            f.writelines(boxes)


def vis(img, boxes):
    box_counts = 0  # ????????????????????????
    cls_id = 0
    for i in range(len(boxes)):
        box = boxes[i].strip('\n').split()

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        box_counts += 1
    return img


if __name__ == "__main__":
    drop_boxes(0.2)



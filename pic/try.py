
import skimage.io as io
# import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v

import torch

# dog.jpg    width = 1599, height=1066, channel=3

# 使用skimage读取图像
img_skimage = io.imread('pic.png')        # skimage.io imread()-----np.ndarray,  (H x W x C), [0, 255],RGB
# print(img_skimage)

# 使用opencv读取图像
img_cv = cv2.imread('pic.png')            # cv2.imread()------np.array, (H x W xC), [0, 255], BGR
# print(img_cv.shape)

# 使用PIL读取
img_pil = Image.open('pic.png')         # PIL.Image.Image对象
img_pil_1 = np.array(img_pil)           # (H x W x C), [0, 255], RGB
# print(img_pil_1)

import numpy as np

from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v




# 提取边缘特征
#reading the image
image = imread('pic.png',as_gray=True)
#calculating horizontal edges using prewitt kernel
edges_prewitt_horizontal = prewitt_h(image)
#calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)
imshow(edges_prewitt_vertical, cmap='gray')

# 提取纹理
# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

plt.imshow(image, plt.cm.gray)
lbp = local_binary_pattern(image, n_points, radius)

plt.imshow(lbp, plt.cm.gray)


plt.figure()
for i, im in enumerate([image,lbp]):
    ax = plt.subplot(1, 3, i + 1)
    ax.imshow(im)
    plt.pause(0.01)
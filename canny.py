import numpy as np
from matplotlib import pyplot as plt
import math

img = plt.imread('./RGB人脸.tif')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 第一步：灰度化
img = rgb2gray(img)
plt.subplot(231)
plt.imshow(img, cmap='gray')

# 第二步：高斯滤波，图像平滑（去噪声）
# a、生成二维高斯分布
gaussian = np.zeros([5, 5])
sigma = 0.6  # 方差
gaussianSum = 0
for i in range(5):
    for j in range(5):
        gaussian[i, j] = math.exp(-(np.square(i-2)+np.square(j-2))/(2 * np.square(sigma))) / (2*math.pi*np.square(sigma))
        gaussianSum = gaussianSum + gaussian[i, j]
gaussian = gaussian / gaussianSum

# b、与灰度图像进行卷积，实现滤波
H, W = img.shape
new_gray = np.zeros([W-4, H-4])
for i in range(W-4):
    for j in range(H-4):
        new_gray[i, j] = np.sum(img[i:i+5, j:j+5] * gaussian)

plt.subplot(232)
plt.imshow(new_gray, cmap='gray')

# 第三步：计算梯度值和方向
H1, W1 = new_gray.shape
G = np.zeros([W1-2, H1-2])
beta = np.zeros([W1-2, H1-2])

K_GX = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
K_GY = np.array([[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]])

for i in range(W1-2):
    for j in range(H1-2):
        GX = np.sum(new_gray[i:i+3, j:j+3] * K_GX)
        GY = np.sum(new_gray[i:i+3, j:j+3] * K_GY)
        G[i, j] = np.sqrt(np.square(GX) + np.square(GY))
        beta[i, j] = math.atan2(GY, GX)

plt.subplot(233)
plt.imshow(G, cmap='gray')

# 第四步：非极大值抑制 NMS
H2, W2 = G.shape
nms = np.copy(G)

# a、把图像中不可能为边缘的像素点设为0
nms[0, :] = nms[H2-1, :] = nms[:, 0] = nms[:, W2-1] = 0
# b、根据梯度方向进行 非极大值抑制
beta1 = 0.125 * math.pi
beta2 = 0.375 * math.pi
beta3 = 0.625 * math.pi
beta4 = 0.875 * math.pi

for i in range(1, H2-1):
    for j in range(1, W2-1):

        # 如果当前像素点的梯度值为0，则一定不是边缘点
        if G[i, j] == 0:
            nms[i, j] = 0
        elif beta1 < beta[i, j] <= beta2 or -beta4 < beta[i, j] <= -beta3:
            if nms[i, j] != max(nms[i, j], G[i+1, j+1], G[i-1, j-1]):
                nms[i, j] = 0
        elif beta2 < beta[i, j] <= beta3 or -beta3 < beta[i, j] <= -beta2:
            if nms[i, j] != max(nms[i, j], G[i + 1, j + 1], G[i - 1, j - 1]):
                nms[i, j] = 0
        elif beta3 < beta[i, j] <= beta4 or -beta2 < beta[i, j] <= -beta1:
            if nms[i, j] != max(nms[i, j], G[i - 1, j + 1], G[i + 1, j - 1]):
                nms[i, j] = 0
        else:
            if nms[i, j] != max(nms[i, j], G[i-1, j], G[i+1, j]):

                nms[i, j] = 0
plt.subplot(234)
plt.imshow(nms, cmap='gray')

# 第五步：双阈值检测，并利用滞后的边界跟踪
H3, W3 = nms.shape
DT = np.zeros([H3, W3])
# 定义高低阈值
TL = 0.1 * np.max(nms)
TH = 0.3 * np.max(nms)

for i in range(1, H3-1):
    for j in range(1, W3-1):
        if nms[i, j] < TL:
            DT[i, j] = 0
        elif nms[i, j] > TH:
            DT[i, j] = 1
        elif (nms[i-1:i+2, j-1:j+2] > TH).any:
            DT[i, j] = 1
        else:
            DT[i, j] = 0
plt.subplot(235)
plt.imshow(DT, cmap='gray')

print()



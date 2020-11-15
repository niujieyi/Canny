import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('round.tif', 0)
edge = cv2.Canny(img, 200, 230)
plt.imshow(edge, cmap='gray')
print()



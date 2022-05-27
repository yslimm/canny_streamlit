import cv2
import numpy as np

def draw_heatmap(img1, img2):
    #input images are in bgr format
    #difference
    #img_diff = np.abs(img1 - img2)
    img_diff = cv2.subtract(img1, img2)

    img_diff_norm = (img_diff - np.min(img_diff)) / (np.max(img_diff) - np.min(img_diff)) #Min-Max Normalization

    img_map = np.uint8(255 * img_diff_norm)  # 255階調　tensor

    img_jet = cv2.applyColorMap(img_map, cv2.COLORMAP_JET)  # colormap image


    #合成
    img_gousei = cv2.addWeighted(src1=img_jet, alpha=0.4, src2=img1, beta=0.6, gamma=0)


    return img_gousei

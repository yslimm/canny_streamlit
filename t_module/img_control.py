import cv2
import numpy as np


def draw_Centerline(img, draw=False):
    if draw:
        height, width, channel = img.shape
        img = cv2.line(img, (int(0.5 * width), 0), (int(0.5 * width), height),
                       (255, 255, 255), 1)  # vertical
        img = cv2.line(img, (0, int(0.5 * height)), (width, int(0.5 * height)),
                       (255, 255, 255), 1)  # #horizontal

    return img


def insert_Text(img, msg, draw=False, ):
    if draw:
        cv2.putText(img, msg, (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2, cv2.LINE_AA)
    return img


def get_Canny(img, kernel_size, threshold1, threshold2):

    # 00. Make a Copy
    img_copy = img.copy()
    img_black = np.zeros_like(img)

    # 1. Color -> GrayScale
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    kernel = kernel_size
    kernel = (kernel * 2) + 1
    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), None)

    # 3. Canny Edge Detection

    # 3.1. Automatic Thresholds Finding
    med_val = np.median(img_blur)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    img_edge1 = cv2.Canny(img_blur, threshold1 = min_val, threshold2 = max_val)
    cv2.putText(img_edge1, 'th 1: {}'.format(str(min_val)), (0, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_edge1, 'th 2: {}'.format(str(max_val)), (0, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_edge1, 'k size: {}'.format('(' + str(kernel) + ',' + str(kernel) + ')'), (0, 210),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                (255, 255, 255), 2, cv2.LINE_AA)

    # 3.2 Quiita Thresholds Finding :: parameter -> threshold1, threshold2
    thres1_val = threshold1
    thres2_val = threshold2

    img_edge2 = cv2.Canny(img_blur, threshold1 = thres1_val, threshold2 = thres2_val)

    return min_val, max_val, img_edge1, img_edge2, img_gray, img_black


def get_Threshold(img_edge, kernel_size, iter_d, iter_e):
    img_edge2 = img_edge
    kernel2 = kernel_size
    kernel2 = (kernel2 * 2) + 1
    k_mat = np.ones((kernel2, kernel2), np.uint8)

    # 01. Dilate
    iter_dilate = iter_d
    img_dilate = cv2.dilate(img_edge2, kernel = k_mat, iterations = iter_dilate)

    # 02. Erode
    iter_erode = iter_e
    img_erode = cv2.erode(img_dilate, kernel = k_mat, iterations = iter_erode)

    # 03. Threshold
    _, img_thresh = cv2.threshold(img_erode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_dilate, img_erode, img_thresh


def get_Contour(img_org, img_th, minArea):
    img_thresh = img_th
    img_copy = img_org.copy()

    height, width, channel = img_copy.shape
    img_size = height * width

    # ----------------------------------------------------------------------
    # Find Contour
    # -----------------------------------------------------------------------------
    # 01. 輪郭抽出
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours:', len(contours))
    final_contours = []

    for i, cnt in enumerate(contours):
        # print('contours[{}].shape{}: '.format(i, cnt.shape))
        area = cv2.contourArea(cnt)

        if area > (0.95 * img_size):
            continue  # この次は実行しない

        if area > minArea:
            # polygon近似
            # arclen = cv2.arcLength(cnt, closed = True)
            # approx_cnt = cv2.approxPolyDP(cnt, 0.001 * arclen, closed = True) #ratio 0.02 暫定
            # final_contours.append(approx_cnt)
            # print('contours[{}].shape{}: '.format(i, approx_cnt.shape))
            final_contours.append(cnt)
            # print('contours[{}].shape{}: '.format(i, cnt.shape))

    # print('final contours:', len(final_contours))

    cv2.drawContours(image = img_copy, contours = final_contours, contourIdx = -1, color = (0, 0, 255), thickness = 3)

    return img_copy


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

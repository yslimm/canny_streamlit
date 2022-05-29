import streamlit as st
from PIL import Image
import cv2
import numpy as np
from t_module import img_array_module

#variables


st.title('Canny Edge Detection Demo')

def canny_edge_detection(img_array):
    st.header('Header')
    st.subheader('Subheader')

    #00.make a copy
    img_copy = img_array.copy()
    img_black = np.zeros_like(img_array)

    #01. Color -> Gray Scale
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    st.image(img_gray,
             caption='Gray Scale',
             use_column_width=True)
    #02. Gaussian Blur
    kernel = st.sidebar.slider('Blur kernel size', 0, 10)
    kernel = (kernel * 2) + 1
    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), None)
    st.image(img_blur,
             caption='Blur Image',
             use_column_width=True)
    #03 Canny Edge Detection
    thres1_val = st.sidebar.slider('Canny Threshold 1', 1, 500)
    thres2_val = st.sidebar.slider('Canny Threshold 2', 1, 500)

    img_edge = cv2.Canny(img_blur, threshold1= thres1_val, threshold2=thres2_val)
    st.image(img_edge,
             caption='Canny Image',
             use_column_width=True)
    #04. dilate , erode
    kernel2 = st.sidebar.slider('dialate, erode k_mat_size', 0, 10)
    kernel2 = (kernel2 * 2) + 1
    k_mat = np.ones((kernel2, kernel2), np.uint8)

    #04.01. dilate
    iter_dialate = st.sidebar.slider('iter_d', 0,20)
    img_dialate = cv2.dilate(img_edge, kernel=k_mat, iterations=iter_dialate)
    st.image(img_dialate,
             caption='Iteration Image',
             use_column_width=True)
    #04.02. Erode
    iter_erode = st.sidebar.slider('iter_e', 0, 20)
    img_erode = cv2.erode(img_dialate, kernel=k_mat, iterations=iter_erode)
    st.image(img_dialate,
             caption='Iteration Image',
             use_column_width=True)


    #04.03. Binary Threshold + Otsu Threshold
    _, img_thresh = cv2.threshold(img_erode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.image(img_thresh,
             caption='Binary Threshold + Otsu Threshold',
             use_column_width=True)

    img_stack = img_array_module.stackImages(0.3, ([[img_array, img_blur, img_edge],
                                                    [img_dialate,img_erode,img_thresh]]))
    st.subheader('1st Preprocessing')
    st.image(img_stack,
             caption='[img_array, img_blur, img_edge], \n '
                     '[img_dialate,img_erode,img_thresh]',
             use_column_width=True)
    return img_thresh

def find_contour(img_copy, img_thresh, img_size):
    minArea = st.sidebar.slider('minArea', min_value=1, max_value=10000, step=10)
    # 01. 輪郭抽出
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for i, cnt in enumerate(contours):
        # print('contours[{}].shape{}: '.format(i, cnt.shape))
        area = cv2.contourArea(cnt)

        if area > (0.9 * img_size):
            continue  # この次は実行しない

        if area > minArea:
            # polygon近似
            # arclen = cv2.arcLength(cnt, closed = True)
            # approx_cnt = cv2.approxPolyDP(cnt, 0.001 * arclen, closed = True) #ratio 0.02 暫定
            # final_contours.append(approx_cnt)
            # print('contours[{}].shape{}: '.format(i, approx_cnt.shape))
            final_contours.append(cnt)

    img_cont = cv2.drawContours(image = img_copy, contours = final_contours, contourIdx = -1, color = (255, 0, 255), thickness = 3)
    st.image(img_cont,
             caption='Contour Images',
             use_column_width=True
             )




#Side Bar
st.sidebar.image('./image/Logo_Small_new.png')
#File upload
upload_file = st.sidebar.file_uploader('1. Choose a image file',
                                       type=['png','jpg','BMP','jpeg'],
                                       accept_multiple_files=False)
print('upload_file',upload_file)
if upload_file is not None:
    img = Image.open(upload_file)
    img_array = np.array(img)
    height, width, channel = img_array.shape
    img_size = height * width
    st.image(img_array,
             caption= 'Original Image',
             use_column_width=True)

    #canny
    img_result = canny_edge_detection(img_array=img_array)

    #Find Contour
    find_contour(img_copy=img_array,img_thresh=img_result,img_size=img_size)








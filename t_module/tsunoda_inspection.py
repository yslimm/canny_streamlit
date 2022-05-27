#Tsunoad Inspection module

import cv2

def inspect_stopper(img_input, img_template):
    #Image read
    img_source = img_input.copy()
    img_input_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    img_temp = cv2.cvtColor(img_template,cv2.COLOR_BGR2GRAY)

    # get the template image height, width
    h, w = img_temp.shape

    # template matching
    result = cv2.matchTemplate(img_input_gray, img_temp, cv2.TM_CCOEFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(result)


    pass
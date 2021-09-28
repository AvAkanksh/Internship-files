import pyautogui
import sys
import time
import cv2
import numpy as np

# for i in range(1000):
#     sys.stdout.write("\r{}".format(pyautogui.position()))
#     sys.stdout.flush()
factor = 3/4
image_number = 5
left_img = cv2.imread('../Stereo-Images/{}l.png'.format(image_number))[48:]
disparity = cv2.imread('../Stereo-Images/disp_test/{}ldisp.png'.format(image_number))
while True:
    x,y = pyautogui.position()
    y = y-60
    Baseline = 9 #cm
    focal_length = np.load('../Stereo-Images/camera_params/camera_paramsL.npy')[0][0]
    print("Baseline : ",Baseline)
    print("Focal Length : ",focal_length)
    print('x : ',x , 'y : ',y)
    print("Disparity value : ",disparity[y][x][0])
    Z = factor*Baseline*focal_length/disparity[y][x][0]
    print("Z = ",Z)

    k = cv2.waitKey(5)
    if k == 27:
        break
    final_left = left_img.copy()
    final_disparity = disparity.copy()
    # cv2.putText(final_left, "X : {:.1f}*x_length/focal_length cms".format(Z),(0,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)
    # cv2.putText(final_left, "Y : {:.1f}*y_length/focal_length cms".format(Z),(0,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 3)
    cv2.putText(final_left, "Depth : {:.1f} cms".format(Z),(0,150), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)
    # cv2.putText(final_disparity, "X : {:.1f}*x_length/focal_length cms".format(Z),(0,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)
    # cv2.putText(final_disparity, "Y : {:.1f}*y_length/focal_length cms".format(Z),(0,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 3)
    cv2.putText(final_disparity, "Depth : {:.1f} cms".format(Z),(0,150), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)
    cv2.imshow('Color Depth Map',final_left)
    cv2.imshow('Grayscale Depth Map',final_disparity)

# cv2.waitKey(0)
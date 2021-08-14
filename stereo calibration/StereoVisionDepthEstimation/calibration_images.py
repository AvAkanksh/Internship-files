import cv2
import time

cam1_number = 2
cam2_number = 4
cap = cv2.VideoCapture(cam1_number)
cap2 = cv2.VideoCapture(cam2_number)

num = 0
c=30
x = 1
delay = 3
t1 = time.time()

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    final = cv2.hconcat([img,img2])
    cv2.imshow('final', final)

    k = cv2.waitKey(5)
    if ((time.time()-t1)>delay):
        if k == 27:
            break
        if x>=c*20:
            break
        if x%c==0: # wait for 's' key to save and exit
            # cv2.imwrite('/home/akanksheos/Documents/Academics/My github repos/Internship/stereo calibration/StereoVisionDepthEstimation/images/stereoLeft/imageL' + str(num) + '.png', img)
            # cv2.imwrite('/home/akanksheos/Documents/Academics/My github repos/Internship/stereo calibration/StereoVisionDepthEstimation/images/stereoRight/imageR' + str(num) + '.png', img2)
            cv2.imwrite('./stereo calibration/StereoVisionDepthEstimation/images/stereoLeft/imageL' + str(num) + '.png', img)
            cv2.imwrite('./stereo calibration/StereoVisionDepthEstimation/images/stereoRight/imageR' + str(num) + '.png', img2)
            print("images saved!")
            num += 1
        x+=1


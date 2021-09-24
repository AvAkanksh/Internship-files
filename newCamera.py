import cv2
import numpy as np
import time
import os

# VIDEO MODE	OUTPUT RESOLUTION (SIDE BY SIDE)	FRAME RATE (FPS)	FIELD OF VIEW
# 2.2K	            4416x1242	                          15	            Wide
# 1080p	            3840x1080	                        30, 15	            Wide
# 720p	            2560x720	                       60, 30, 15	      Extra Wide
# WVGA	            1344x376	                     100, 60, 30, 15	  Extra Wide

def heightWidth(resolution = '720p'):
    if(resolution == 'wvga' or resolution == 0):
        width = 1344
        height = 376
        factor = 1.5
    elif(resolution == '720p' or resolution == 1):
        width = 2560
        height = 720
        factor = 0.75
    elif(resolution == '1080p' or resolution == 2):
        width = 3840
        height = 1080
        factor = 0.5
    elif(resolution == '2.2k' or resolution == 3):
        width = 4416
        height = 1242
        factor = 0.435
    return height , width , factor


def epipolarlines(frame,spacing):
    for y in range(frame.shape[0]):
        if(y%spacing==0):
            cv2.line(frame_resized,(0,y),(frame_resized.shape[1],y),(0,255,0),1)
    return frame



cam = cv2.VideoCapture(0)
height , width , factor = heightWidth(3)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
num = 0
while True:
    res, frame = cam.read()
    frame_resized = cv2.resize(frame, (int(frame.shape[1]*factor),int(frame.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    # frame_resized = epipolarlines(frame_resized,20)
    mid = int(frame.shape[1]/2)

    cv2.imshow('Resized',frame_resized)

    # cv2.imshow('Left View',frame[:,:mid])
    # cv2.imshow('Right View',frame[:,mid:])
    k = cv2.waitKey(1)
    if(k == ord('q') or k == 27):
        break
    if(k == 32):
        cv2.imwrite('./my_stereo_images/newCamImages/'+str(num)+'l.png',frame[:,:mid])
        cv2.imwrite('./my_stereo_images/newCamImages/'+str(num)+'r.png',frame[:,mid:])
        print('Image ',num,' taken!')
        num+=1


cam.release()
cv2.destroyAllWindows()
os.system('git add . ; git commit -m "updating the files" ; git push')



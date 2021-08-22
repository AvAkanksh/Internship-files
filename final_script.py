import cv2
import time

n1 = 2
n2 = 4
count = 15
interval = 1.75
cap = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(n2)#,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
start_time = time.time()
num = 0
x = 0

factor = 0.25



while cap.isOpened():

    succes1, frame1 = cap.read()
    succes2, frame2 = cap2.read()
    frame1=cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2=cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    frame1_resized = cv2.resize(frame1, (int(frame1.shape[1]*factor),int(frame1.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    frame2_resized = cv2.resize(frame2, (int(frame2.shape[1]*factor),int(frame2.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    final = cv2.hconcat([frame1_resized,frame2_resized])
    k = cv2.waitKey(5)

    if k == 27 or num==count:
        break
    # elif k == ord('s'): # wait for 's' key to save and exit
    if(x%40*interval==0 and time.time()-start_time>5):
        cv2.imwrite('./my_stereo_images/stereoLeft/' + str(num) + 'l.png', frame1)
        cv2.imwrite('./my_stereo_images/stereoRight/' + str(num) + 'r.png', frame2)
        print("images saved!")
        num += 1
    cv2.imshow('final',final)
    x+=1


# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()
import numpy as np
import cv2 as cv
import glob
import time

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10,7)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 24
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


num_images_l = len(glob.glob('./my_stereo_images/stereoLeft/*.png'))
num_images_r = len(glob.glob('./my_stereo_images/stereoRight/*.png'))
both_true = 0

# for imgLeft, imgRight in zip(imagesLeft, imagesRight):
for i in range(num_images_l):
    l_img_path='./my_stereo_images/stereoLeft/'+str(i)+'l.png'
    r_img_path='./my_stereo_images/stereoRight/'+str(i)+'r.png'

    imgL = cv.imread(l_img_path)
    imgR = cv.imread(r_img_path)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)


    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    print(str(i)+'*'*100)
    print(retL)
    print(retR)
    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        print('this is the number which has both the chess patterene detected: ',i)
        both_true+=1
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1)


cv.destroyAllWindows()

print((both_true))


############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)




########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()


import numpy as np
import cv2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# print(stereoMapL_y)
# Open both cameras
# cap_left =  cv2.VideoCapture(4)
# cap_right = cv2.VideoCapture(2)

n1 = 4
n2 = 2
num = 0

factor = 0.25

cam1 = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(n2)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while(cam1.isOpened() and cam2.isOpened()):

    succes_right, frame_right = cam1.read()
    succes_left, frame_left = cam2.read()
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    frame1_resized = cv2.resize(frame_right, (int(frame_right.shape[1]*factor),int(frame_right.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    frame2_resized = cv2.resize(frame_left, (int(frame_left.shape[1]*factor),int(frame_left.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    final = cv2.hconcat([frame1_resized,frame2_resized])

    # Show the frames
    # cv2.imshow("frame right", frame_right)
    # cv2.imshow("frame left", frame_left)
    cv2.imshow("final", final)


    k = cv2.waitKey(5)

    if k == ord('s'):
        cv2.imwrite('./my_stereo_images/test_dataset/'+str(num)+'l.png',frame_left)
        cv2.imwrite('./my_stereo_images/test_dataset/'+str(num)+'r.png',frame_right)
        print('Image ',num,' taken!')
        num+=1

    if k == 27:
        break

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Release and destroy all windows before termination
cam1.release()
cam2.release()

cv2.destroyAllWindows()

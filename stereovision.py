import numpy as np
import cv2
import os

lines = True
line_width = 1



# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap_best2.xml', cv2.FileStorage_READ)

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

factor = 0.5

cam1 = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(n2)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while(cam1.isOpened() and cam2.isOpened()):

    succes_right, frame_right_original = cam1.read()
    succes_left, frame_left_original = cam2.read()
    frame_left_original = cv2.cvtColor(frame_left_original, cv2.COLOR_RGB2BGR)
    frame_right_original = cv2.cvtColor(frame_right_original, cv2.COLOR_RGB2BGR)

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right_original, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left_original, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    frame1_resized = cv2.resize(frame_right, (int(frame_right.shape[1]*factor),int(frame_right.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    frame2_resized = cv2.resize(frame_left, (int(frame_left.shape[1]*factor),int(frame_left.shape[0]*factor)), interpolation=cv2.INTER_AREA)

    frame1_resized_original = cv2.resize(frame_right_original, (int(frame_right.shape[1]*factor),int(frame_right.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    frame2_resized_original = cv2.resize(frame_left_original, (int(frame_left.shape[1]*factor),int(frame_left.shape[0]*factor)), interpolation=cv2.INTER_AREA)


    final_original = cv2.hconcat([frame1_resized_original,frame2_resized_original])
    final = cv2.hconcat([frame1_resized,frame2_resized])


    if(lines):
        for i in range(int(final.shape[0]/15)):
            # r , g , b = np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)
            if i%2 == 0:
                r , g , b = 255,0,0
            else:
                r , g , b = 0,0,255


            final = cv2.line(final, (0,i*15), (final.shape[1],i*15), (b,g,r),line_width)
            final_original = cv2.line(final_original, (0,i*15), (final.shape[1],i*15), (b,g,r),line_width)

    original_plus_rectified = cv2.vconcat([final,final_original])

    cv2.imshow("final", final)
    cv2.imshow("final_original", final_original)


    k = cv2.waitKey(5)

    if ((k == 27) | ( k==ord('q'))):
        os.system("cd ../Stereo-Images ;git add . ; git status ; git commit -m 'updating the files' ; git push")
        break

    if k == ord('s'):
        cv2.imwrite('./my_stereo_images/test_dataset/'+str(num)+'l.png',frame_left)
        cv2.imwrite('../Stereo-Images/'+str(num)+'l.png',frame_left)
        cv2.imwrite('./my_stereo_images/test_dataset/'+str(num)+'r.png',frame_right)
        cv2.imwrite('../Stereo-Images/'+str(num)+'r.png',frame_right)
        print('Image ',num,' taken!')
        num+=1

    # Hit "q" to close the window
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break




# Release and destroy all windows before termination
cam1.release()
cam2.release()

cv2.destroyAllWindows()

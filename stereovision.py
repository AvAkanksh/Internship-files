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
cap_right = cv2.VideoCapture(4)
cap_left =  cv2.VideoCapture(2)

while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    final = cv2.hconcat([frame_left,frame_right])

    # Show the frames
    # cv2.imshow("frame right", frame_right)
    # cv2.imshow("frame left", frame_left)
    cv2.imshow("final", final)

    k = cv2.waitKey(5)

    if k == 27:
        break

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()

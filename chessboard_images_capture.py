import cv2
import time


cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(4)
start_time = time.time()
num = 0
x = 0
while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2=cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    final = cv2.hconcat([img,img2])
    k = cv2.waitKey(5)

    if k == 27 or num==25:
        break
    # elif k == ord('s'): # wait for 's' key to save and exit
    if(x%100==0 and time.time()-start_time>5):
        cv2.imwrite('./my_stereo_images/stereoLeft/' + str(num) + 'l.png', img)
        cv2.imwrite('./my_stereo_images/stereoRight/' + str(num) + 'r.png', img2)
        print("images saved!")
        num += 1
    cv2.imshow('final',final)
    x+=1


# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()

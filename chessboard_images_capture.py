import cv2
import time

n1 = 2
n2 = 4
count = 10
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

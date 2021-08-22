import cv2


n1 = 2
n2 = 4

factor = 0.5

cam1 = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(n2)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1:
        print("failed to grab frame")
        break

    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    frame1_resized = cv2.resize(frame1, (int(frame1.shape[1]*factor),int(frame1.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    frame2_resized = cv2.resize(frame2, (int(frame2.shape[1]*factor),int(frame2.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    final = cv2.hconcat([frame1_resized,frame2_resized])

    cv2.imshow("final",final)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name1 = "opencv_frame1_{}.png".format(img_counter)

        cv2.imwrite(img_name1, frame1)

        print("{} written!".format(img_name1))



cam1.release()


cv2.destroyAllWindows()
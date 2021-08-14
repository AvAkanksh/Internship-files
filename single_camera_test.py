import cv2
n = 0
cam1 = cv2.VideoCapture(n)

while True:
    ret1, frame1 = cam1.read()

    if not ret1:
        print("failed to grab frame")
        break
    if(not(n == 0)):
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)

    cv2.imshow("cam1",frame1)


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
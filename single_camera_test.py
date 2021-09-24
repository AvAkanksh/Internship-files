import cv2

n = 4
img_counter = 0
cam1 = cv2.VideoCapture(n)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cam1.set(cv2.CAP_PROP_FRAME_COUNT,3)



while True:
    ret1, frame1 = cam1.read()

    if not ret1:
        print("failed to grab frame")
        break

    if(not(n == 0)):
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    # print(frame1.shape)
    cv2.imshow("cam1",frame1)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name1 = "test{}.png".format(img_counter)
        img_counter+=1
        cv2.imwrite(img_name1, frame1)

        print("{} written!".format(img_name1))



cam1.release()


cv2.destroyAllWindows()
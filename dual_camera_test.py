import cv2
import numpy as np
import time

def final_output(frame):
    gaussian = cv2.GaussianBlur(frame, (31,31), 0)
    hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([157,91,177],dtype='uint8')
    # upper_green = np.array([179,168,255],dtype='uint8')
    # Bright lighting Red
    # lower_green = np.array([0,29,80],dtype='uint8')
    # upper_green = np.array([34,193,255],dtype='uint8')
    # dim lighting Red
    lower_green = np.array([0,124,0],dtype='uint8')
    upper_green = np.array([179,255,255],dtype='uint8')

    mask = cv2.inRange(hsv, lower_green, upper_green)
    # cv2.imshow('mask',mask)
    output = frame.copy()
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('result',res)
    mask = cv2.Canny(mask,threshold1 = 100, threshold2 = 200)
    gray_blurred = cv2.blur(mask, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 1)
    x = None
    max_r = None
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        # print(len(detected_circles[0]))
        max_r = detected_circles[0][0][2]
        index = 0
        for i in range(len(detected_circles[0])):
            if(max_r<detected_circles[0][i][2]):
                max_r = detected_circles[0][i][2]
                index = i

        x,y,r = detected_circles[0][index]
        cv2.circle(output,(x,y),r, (0, 255, 0), 2)
        cv2.circle(output,(x,y),1, (0, 0, 255), 2)
        cv2.line(output,(x-r,y),(x+r,y), (255,0,0),1)
        cv2.line(output,(x,y-r),(x,y+r), (255,0,0),1)
        return output ,res, x , max_r
    return output ,res, x , max_r

n1 = 2
n2 = 4

factor = 0.5
f_size_actual = float(input('Enter the actual size of the fruit : '))
cam1 = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(n2)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

image_counter = 0

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

    output1 ,filtered_1, u_l ,r1= final_output(frame1_resized)
    output2 ,filtered_2, u_r ,r2= final_output(frame2_resized)
    original = cv2.hconcat([filtered_1,filtered_2])
    final = cv2.hconcat([output1,output2])
    baselength = 9

    if u_l != None and u_r!=None:
        r_mean = (r1+r2)/2
        disparity = u_l - u_r
        print(f"U_l : {u_l} | U_r : {u_r}")
        print(f"Disparity : {(u_r - u_l)}")
        f_size = (2*r_mean*baselength)/(u_r - u_l)
        z = baselength* focal_length
        b = 7/f_size
        error = abs((f_size_actual - f_size)*100/f_size_actual)
        cv2.putText(final, f"Disparity :{(u_r - u_l)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
        cv2.putText(final, f"size of the fruit : {f_size:.2f}", (30,80), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
        cv2.putText(final, f"Error : {error:.1f} %", (30,120), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
        # cv2.putText(final, f"Error : {b:.3f} %", (30,150), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)


        # cv2.putText(final, f"baselength: {7/(r_mean*baselength/(u_r - u_l))}", (30,120), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)
    # print(f"U_l : {u_l} | U_r : {u_r}")

    # original = cv2.hconcat([frame1_resized,frame2_resized])
    # cv2.putText(orig, "{:.1f}in".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)


    cv2.imshow("original",original)
    cv2.imshow("final",final)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == ord('s'):
        # Press s to save the image
        img_name1 = "opencv_frame1_{}.png".format(image_counter)
        image_counter+=1

        cv2.imwrite(img_name1, frame1_resized)

        print("{} written!".format(img_name1))
    elif k%256 == 32:
        # SPACE pressed to pause
        time.sleep(0.5)


cam1.release()


cv2.destroyAllWindows()


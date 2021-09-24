from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA,ptB):
    return((ptA[0]+ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)

delay = 500
factor = 0.5
n1 = 0
width = 1
factor = 0.5

cam1 = cv2.VideoCapture(n1)#,cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while True:
    ret1, image = cam1.read()

    if not ret1:
        print("failed to grab frame")
        break
    # cv2.imshow('Original Image',image)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray Image',gray)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()

    gray = cv2.GaussianBlur(gray, (7,7), 0)
    # cv2.imshow('Gray Gaussian Image',gray)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()


    edged = cv2.Canny(gray, 50, 10)
    # cv2.imshow('Canny Edge Image',edged)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()

    edged = cv2.dilate(edged,None,iterations=1)
    # cv2.imshow('Canny Edge Image dilated',edged)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()

    edged = cv2.erode(edged, None,iterations=1)
    edged_resized = cv2.resize(edged, (int(edged.shape[1]*factor),int(edged.shape[0]*factor)), interpolation=cv2.INTER_AREA)
    # cv2.imshow('Canny Edge Image dilated and eroded',edged_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(cnts)

    (cnts,x) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    i= 0
    orig = image.copy()
    for c in cnts :
        if (cv2.contourArea(c) <1000):
            continue
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box,dtype='int')
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype('int')], -1, (255,0,0),2)
        for (x,y) in box :
            cv2.circle(orig,(int(x),int(y)),5,(0,0,255),-1)



        (tl, tr, br, bl) = box

        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(155, 100, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(155, 100, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        # draw the object sizes on the image

        cv2.putText(orig, "{:.1f}in".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)
        cv2.putText(orig, "{:.1f}in".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        orig_resized = cv2.resize(orig, (int(orig.shape[1]*factor),int(orig.shape[0]*factor)), interpolation=cv2.INTER_AREA)
        # final = cv2.hconcat([orig_resized , edged_resized])
        k = cv2.waitKey(1)
        if k == 27 | k == ord('q'):
            break
    cv2.imshow('Image',orig_resized)

    print("Area :" , cv2.contourArea(c))
    print('Box :\n',box,'\n'+"*"*100+'\n',)
    k = cv2.waitKey(1)
    if k == 27 :
        break



cam1.release()
cv2.destroyAllWindows()


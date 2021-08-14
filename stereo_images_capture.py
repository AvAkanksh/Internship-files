import cv2

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(4)

num = 0

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2=cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    final = cv2.hconcat([img,img2])
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./my_stereo_images/' + str(num) + 'l.png', img)
        cv2.imwrite('./my_stereo_images/' + str(num) + 'r.png', img2)
        print("images saved!")
        num += 1
    cv2.imshow('final',final)


# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()

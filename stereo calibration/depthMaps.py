# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# n1 = 4
# n2 = 2
# cam1 = cv.VideoCapture(n1)
# cam2 = cv.VideoCapture(n2)

# while True:
#     ret1, frame1 = cam1.read()
#     ret2, frame2 = cam2.read()

#     if not ret1:
#         print("failed to grab frame")
#         break

#     left_image = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
#     right_image = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
# #left_image = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
# #right_image = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)

# # left_image = cv.imread('items_l.png', cv.IMREAD_GRAYSCALE)
# # right_image = cv.imread('items_r.png', cv.IMREAD_GRAYSCALE)

# #left_image = cv.imread('items2_l.png', cv.IMREAD_GRAYSCALE)
# #right_image = cv.imread('items2_r.png', cv.IMREAD_GRAYSCALE)         
#     stereo = cv.StereoBM_create(numDisparities=32, blockSize=21)
#     # For each pixel algorithm will find the best disparity from 0
#     # Larger block size implies smoother, though less accurate disparity map
#     depth = stereo.compute(left_image, right_image)

#     print(depth)

#     cv.imshow("Left", left_image)
#     cv.imshow("right", right_image)
#     break
#     # cv.imshow('Depth',depth)
#     cv.waitKey(1)
# plt.imshow(depth)
# plt.axis('off')
# plt.show()



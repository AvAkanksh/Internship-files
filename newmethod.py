import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the left and right images
# in gray scale
imgLeft = cv2.imread('/home/akanksheos/Documents/Academics/My github repos/Internship/my_stereo_images/stereoLeft/1l.png',0)
imgRight = cv2.imread('/home/akanksheos/Documents/Academics/My github repos/Internship/my_stereo_images/stereoRight/1r.png',0)

# Detect the SIFT key points and
# compute the descriptors for the
# two images
sift = cv2.xfeatures2d.SIFT_create()
keyPointsLeft, descriptorsLeft = sift.detectAndCompute(imgLeft,None)
keyPointsRight, descriptorsRight = sift.detectAndCompute(imgRight,None)

# Create FLANN matcher object
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams,searchParams)
matches = flann.knnMatch(descriptorsLeft,descriptorsRight,k=2)


# Apply ratio test
goodMatches = []
ptsLeft = []
ptsRight = []

for m, n in matches:

	if m.distance <  n.distance:

		goodMatches.append([m])
		ptsLeft.append(keyPointsLeft[m.trainIdx].pt)
		ptsRight.append(keyPointsRight[n.trainIdx].pt)


#**********************************************************************************************************************************************


ptsLeft = np.int32(ptsLeft)
ptsRight = np.int32(ptsRight)
F, mask = cv2.findFundamentalMat(ptsLeft,ptsRight,cv2.FM_LMEDS)

# We select only inlier points
ptsLeft = ptsLeft[mask.ravel() == 1]
ptsRight = ptsRight[mask.ravel() == 1]

#**********************************************************************************************************************************************

def drawlines(img1, img2, lines, pts1, pts2):

	r, c = img1.shape
	img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

	for r, pt1, pt2 in zip(lines, pts1, pts2):

		color = tuple(np.random.randint(0, 255,3).tolist())

		x0, y0 = map(int, [0, -r[2] / r[1] ])
		x1, y1 = map(int,[c, -(r[2] + r[0] * c) / r[1] ])

		img1 = cv2.line(img1,(x0, y0), (x1, y1), color, 1)
		img1 = cv2.circle(img1,tuple(pt1), 5, color, -1)
		img2 = cv2.circle(img2,tuple(pt2), 5, color, -1)
	return img1, img2

#**********************************************************************************************************************************************


# Find epilines corresponding to points
# in right image (second image) and
# drawing its lines on left image
linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1,1,2),2, F)
linesLeft = linesLeft.reshape(-1, 3)
img5, img6 = drawlines(imgLeft, imgRight,linesLeft, ptsLeft,ptsRight)

# Find epilines corresponding to
# points in left image (first image) and
# drawing its lines on right image
linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2),1, F)
linesRight = linesRight.reshape(-1, 3)

img3, img4 = drawlines(imgRight, imgLeft,linesRight, ptsRight,ptsLeft)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()


# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt


# img1 = cv.imread('/home/akanksheos/Documents/Academics/My github repos/Internship/my_stereo_images/stereoLeft/0l.png',0)  #queryimage # left image
# img2 = cv.imread('/home/akanksheos/Documents/Academics/My github repos/Internship/my_stereo_images/stereoRight/0r.png',0) #trainimage # right image
# sift = cv.SIFT()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# good = []
# pts1 = []
# pts2 = []

# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.8*n.distance:
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)

# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]



# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2


# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)


# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()
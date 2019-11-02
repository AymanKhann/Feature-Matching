#this code implements BF based matching between trained image and query image

import cv2
import numpy as np

#reads the static frames
img1 = cv2.imread('model/the-sun-is-also-a-star.jpg',1)
img2 = cv2.imread('model/query_image.jpg',1)

orb = cv2.ORB_create()

#compute the keypoints and descriptors for both the images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching Algorithm
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imshow('feature matching',matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

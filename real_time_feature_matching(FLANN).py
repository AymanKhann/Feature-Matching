#this code implements FLANN based matching to match the object in the real time frame with the computed descriptors of the trained image

import cv2
import numpy as np

# load image
img = cv2.imread('model/the-sun-is-also-a-star.jpg', 1)
# capture real time video
cap = cv2.VideoCapture(0)

# feature detection of model
orb = cv2.ORB_create()
kp_img, des_img = orb.detectAndCompute(img, None)  # argument none=no mask

# featurematching (can use ORB match detector)
# loading the object
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)  # passing setting of the algorithm

while True:
    # read the camera
    _, frame = cap.read()

    # feature detection
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    # findingmatches
    matches = flann.knnMatch(np.asarray(des_img, np.float32), np.asarray(des_frame, np.float32), 2)  # 2

    # neglecting the false matches
    best_matches = []
    for m, n in matches:  # 2 arrays in matches where, m=model and n=object in frame
        # comparing distances
        if m.distance < 0.7 * n.distance:  # ratio test ( the lower the distance between the descriptors the better the match)
            # considering descriptors with shorter distance between them
            best_matches.append(m)

    # drawingmatches
    img3 = cv2.drawMatches(img, kp_img, frame, kp_frame, best_matches, frame)

    # showing image and camera in real time
    cv2.imshow('matching', img3)

    # key event to get out of the loop
    key = cv2.waitKey(1)
    if key == 27:  # escapekey
        break

# release camera
cap.release()
cv2.destroyAllWindows()


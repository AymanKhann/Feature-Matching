# Feature-Matching
## using numpy and opencv

> **USAGE**

* add the trained image to be matched with in `model` folder.
* replace the `query_image` with the image you want to match.
* open jupyter notebook in the folder.
* change the name of the image in the codes for static matching and real-time matching i.e `img = cv2.imread('model\abc.jpg',1)` *1 in argument represents the scale image i.e BGR that can be altered as opted*
* press `esc` key to clear output window 

### feature matching: 
* the features founded of both the trained image and the real-time frame were the object is to be found are matched with thevcomputed descriptors.

> A descriptor provides a representation of the information given by a feature and its surroundings. Which is abstracted to a feature vector *a vector that contains the descriptors of the keypoints found in the image with the reference object.*

Feature matching can be implemented using Brute-Force based matching, FLANN based matching and many a more effective algorithm. However, I find FLANN based matching somewhat accurate giving a ratio of 500 best matches with approximately false matches in range of 1 to 5 which is quite amazing!!!

So, I have implemented BF based matching in static frames and FLANN in real-time-frame matching. There you go with the code ...

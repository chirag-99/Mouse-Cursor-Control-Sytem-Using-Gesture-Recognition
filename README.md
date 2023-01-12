Considering the useful aspects of virtual mouse operations, our model will contactlessly take inputs from the user and perform all the required operations.
For doing so, this model performs the following steps:

1) Model will accept the input from the user in the form of hand gestures through webcam in ROI. I have indicated a green box in the output window termed as Region of Interest(ROI) to focus on gestures.
2) The received image is processed using OpenCV and the backgroun subtraction is performed on ROI, so that the foreground gesture can be accurately processed. 
3) Noise removal pre-processing operations will be performed on it like morphological transformation, applying contour on hand and drawing the convex hull.
4) While drawing the hand contour, the centroid of input is detected to locate the coordinates of the cursor and then the contour is extracted.
5) Counting of the blobs is done. 
5.1 If only cursor needs to move across the screen then only one blob needs to be fed as input.
5.2 However, two fingers are provided as input for performing left-click. 
5.3 Three or four fingers for performing right click accordingly.


TECHNOLOGY USED:

Python 3.9

OpenCV 4.5.1

NumPy

Mac as OS

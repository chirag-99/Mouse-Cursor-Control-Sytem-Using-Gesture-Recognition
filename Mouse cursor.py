import cv2 
import numpy as np
import mouse
import math
from sklearn.metrics import pairwise
import wx
x = 200
y = 180
cap = cv2.VideoCapture(0) 
skin_calibration = False
back_calibration = False
_,backGround = cap.read()
width=np.shape(backGround)[1]
height=np.shape(backGround)[0]

backGround=backGround[1:height-199,250:width].copy()
app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
def take_out_base(image):
    back = image
    return back

def remove_background(image):
    frame_1 = np.copy(image)
    mask=cv2.absdiff(backGround,frame_1,0)
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask=cv2.threshold(mask,10,255,0)[1]
    mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ERODE,(2,2)),iterations=2)
    mask1=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,\
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    mask1=cv2.erode(mask1,cv2.getStructuringElement(cv2.MORPH_ERODE,(2,2)),iterations=2)
    foreg_frame=cv2.bitwise_and(frame_1,frame_1,mask=mask1)
    
    gr_frame=cv2.cvtColor(foreg_frame,cv2.COLOR_BGR2GRAY)
    gr_frame=cv2.blur(gr_frame,(10,10))
    bw_frame=cv2.threshold(gr_frame,50,255,0)[1]
    cv2.imshow("bw_frame",bw_frame)
    
    return image,bw_frame


def centroid(max_contour):
    M = cv2.moments(max_contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx ,cy

    else:
        return None


def draw_con(img,thresh):
    if skin_calibration == False or back_calibration == False:
        return 
    contour =  cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    try:
        max_contour_list =  max(contour, key=cv2.contourArea)
    except:
        max_contour_list=np.array([[[1,0],[1,2],[2,3]]],dtype=np.int32)
    centroid_val = centroid(max_contour_list)
    cv2.circle(img, centroid_val, 5, (0,255,0), -1)

    cnt = max_contour_list
    conv_hull = cv2.convexHull(cnt)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    thresholded = thresh
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    (contours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0

    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        if  out_of_wrist and limit_points:
            count += 1
    print(count)
    m_x=sx-((top[0]-60)*sx/(width-340))
    m_y=(top[1]*sy/(height-281))

    if(count == 1):
        mouse.move(sx-m_x,m_y,absolute=True, duration=.1)

    if(count == 2):
        mouse.click(button='left')
        cv2.putText(roi,str('Left click'),top,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)

    if(count == 3 or count == 4):
        mouse.click(button='right')
        cv2.putText(roi,str('Right click'),top,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1,cv2.LINE_AA)
    
while(True):
    pressed_key = cv2.waitKey(1)
    ret, frame=cap.read()
    frame= cv2.flip(frame,1)
    roi=frame[1:height-199,250:width].copy()
    temp_roi=roi.copy()
    frame_background,mask = remove_background(roi)
    draw_con(frame_background,mask)
    if pressed_key & 0xFF == ord("b"):
        backGround = take_out_base(roi)
        back_calibration = True
    if pressed_key & 0xFF == ord("s"):
        skin_calibration = True
    frame[1:height-199,250:width]=roi
    cv2.rectangle(frame,(250,1),(width-1,height-200),(0,255,0),1)
    cv2.imshow('frame',frame)
    cv2.imshow("frame_background",frame_background)
    if pressed_key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

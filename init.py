import numpy as np
import cv2
import math

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and cv2.contourArea(cnt) < 100000:
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

img = cv2.VideoCapture(0)
while True:
    """
    Little bit pre-processing of webcam feed
    """
    ret,frame = img.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.equalizeHist(gray,gray)
    """
    detects the different squares in the webcam feed
    """
    """
    This is a dirty hack to compute the rectangle to select the square detected and save the subsquent image
    TODO - implement it properly (hint - Eucledian Distance(squares array) )
    """
    try:
        squares = find_squares(gray)
        cv2.drawContours(frame, squares, -1, (0,255,0), 3)
        cv2.imshow('Webcam Feed',frame)
        cropped_width = math.sqrt( ((squares[0][3][0]-squares[0][0][0])**2) + ((squares[0][3][1]-squares[0][0][1])**2) )
        cropped_height = math.sqrt( ((squares[0][1][0]-squares[0][0][0])**2) + ((squares[0][1][1]-squares[0][0][1])**2) )
        print cropped_width, cropped_height
        rect = gray[squares[0][0][1]:squares[0][0][1]+cropped_width,squares[0][0][0]:squares[0][0][0]+cropped_height]
        cv2.imwrite("./img/image.png",rect)
    except:
        continue

    cv2.imshow("Selected Image",rect)

    """
    press q to exit
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()



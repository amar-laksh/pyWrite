import numpy as np
import cv2
import tensorflow as tf
from scipy import ndimage
import sys
import os
import math

def getBestShift(img):
    """
    params - image to get shifts of
    returns - finds the best shifts to do on the image and returns the x and y coordinates of shifts
    """
    cy,cx = ndimage.measurements.center_of_mass(img)
    print cy,cx

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    """
    params - image , xshift and yshift
    returns - shifts the image by x and y shift
    """
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted




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

def learn_image(rect):
        """
        This is the tensorflow mnist neural network placeholders
        """
        checkpoint_dir = "cps/"
        learnt = False

        x = tf.placeholder("float", [None, 784])
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        y_ = tf.placeholder("float", [None,10])

        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'No checkpoint found'
            exit(1)


        """
        This is the original colored image and gray pre-processed image.
        """
        color_complete = rect

        gray_complete = rect

        _, gray_complete = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        """
        This is the digitized_image array filled with -1's
        """

        digit_image = -np.ones(gray_complete.shape)

        height, width = gray_complete.shape

        """
        crop into several images
        """
        for cropped_width in range(100, 300, 20):
            for cropped_height in range(100, 300, 20):
                for shift_x in range(0, width-cropped_width, cropped_width/4):
                    for shift_y in range(0, height-cropped_height, cropped_height/4):
                        gray = gray_complete[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]

                        """
                        This checks if the image is almost empty: which means it contains less than 20 non-zero values
                        """
                        if np.count_nonzero(gray) <= 20:
                             continue

                        """
                        This checks if we are cutting a digit somwhere .i.e. it checks if there is white border or not
                        """
                        if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,-1]) != 0):
                            continue

                        """
                        Saving the top-left and bottom-right positions of each image to draw the rectangles later
                        """
                        top_left = np.array([shift_y, shift_x])
                        bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

                        """
                        This removes rows and columns from image which are completely black. This helps to crop the images
                        which contain the digits.
                        """
                        while np.sum(gray[0]) == 0:
                            top_left[0] += 1
                            gray = gray[1:]

                        while np.sum(gray[:,0]) == 0:
                            top_left[1] += 1
                            gray = np.delete(gray,0,1)

                        while np.sum(gray[-1]) == 0:
                            bottom_right[0] -= 1
                            gray = gray[:-1]

                        while np.sum(gray[:,-1]) == 0:
                            bottom_right[1] -= 1
                            gray = np.delete(gray,-1,1)

                        actual_w_h = bottom_right-top_left

                        """
                        This checks if the rectangle that we have currently selected contain more than 20% of the
                        actual image then we can say we have already found that digit
                        """
                        rectangle = digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
                        if (np.count_nonzero(rectangle+1) >
                                    0.2*actual_w_h[0]*actual_w_h[1]):
                            continue

                        """
                        Converts the image to 28x28 array to feed into our neural network by applying padding
                        """
                        rows,cols = gray.shape
                        compl_dif = abs(rows-cols)
                        half_Sm = compl_dif/2
                        half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
                        if rows > cols:
                            gray = np.lib.pad(gray,((0,0),(half_Sm,half_Big)),'constant')
                        else:
                            gray = np.lib.pad(gray,((half_Sm,half_Big),(0,0)),'constant')
                        gray = cv2.resize(gray, (20, 20))
                        gray = np.lib.pad(gray,((4,4),(4,4)),'constant')

                        """
                        This gets the best shifting x and y and shifts the each image in such a way that
                        the digit is in the center of the image
                        """
                        shiftx,shifty = getBestShift(gray)
                        shifted = shift(gray,shiftx,shifty)
                        gray = shifted

                        """
                        This flatten our image array to values between 0 and 1 for the neural network
                        and makes a prediction.
                        """
                        flatten = gray.flatten() / 255.0
                        prediction = [tf.reduce_max(y),tf.argmax(y,1)[0]]
                        pred = sess.run(prediction, feed_dict={x: [flatten]})
                        print pred
                        if pred:
                            learnt = True

                        """
                        This draws a rectangle on each digit and writes the prediction probability and the prediciton
                        value
                        """
                        cv2.rectangle(color_complete,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),
                                      thickness=5)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(color_complete,str(pred[1]),(top_left[1],bottom_right[0]+50),font,fontScale=1.4,
                                    color=(0,255,0),thickness=4)
                        cv2.putText(color_complete,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+60),
                                    font,fontScale=0.8,color=(0,255,0),thickness=2)


        """
        Finaly, we save the digitized image( TODO - combine this image with webcam feed by using cv2.overlay methods)
        """
        sess.close()
        tf.reset_default_graph()
        return color_complete,learnt



learnt = False
img = cv2.VideoCapture(0)
while True:
    """
    Little bit pre-processing of webcam feed
    """
    ret,frame = img.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.equalizeHist(gray,gray)
    """
    detects the different squares in the webcam feed
    """
    """
    This is a dirty hack to compute the rectangle to select the square detected and save the subsquent image
    TODO - implement it properly (hint - Eucledian Distance(squares array) )
    """
    try:
        squares = find_squares(grey)
        cv2.drawContours(frame, squares, -1, (0,255,0), 3)
        cropped_width = math.sqrt( ((squares[0][3][0]-squares[0][0][0])**2) + ((squares[0][3][1]-squares[0][0][1])**2) )
        cropped_height = math.sqrt( ((squares[0][1][0]-squares[0][0][0])**2) + ((squares[0][1][1]-squares[0][0][1])**2) )
        rect = grey[squares[0][0][1]:squares[0][0][1]+cropped_width,squares[0][0][0]:squares[0][0][0]+cropped_height]
        grey[squares[0][0][1]:squares[0][0][1]+cropped_width,squares[0][0][0]:squares[0][0][0]+cropped_height] = rect
        cv2.imshow("Pre-feed",frame)
        digitized_image = rect
        if not learnt:
            digitized_image,learnt = learn_image(rect)
        #frame[squares[0][0][1]:squares[0][0][1]+cropped_width,squares[0][0][0]:squares[0][0][0]+cropped_height] = digitized_image
    except:
        e = sys.exc_info()
        print e
    cv2.imshow("Webcam Feed",digitized_image)
    """
    press q to exit
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()



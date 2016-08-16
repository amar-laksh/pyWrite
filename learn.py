import tensorflow as tf
import cv2
import numpy as np
from scipy import ndimage
import sys
import os

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


"""
This is the tensorflow mnist neural network placeholders
"""

image = sys.argv[1] #Takes the detected image of the paper from webcam. (TODO - remove the need for it.)
checkpoint_dir = "cps/"


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


if not os.path.exists("img/" + image + ".png"):
    print "File img/" + image + ".png doesn't exist"
    exit(1)


"""
This is the original colored image and gray pre-processed image.
"""
color_complete = cv2.imread("img/" + image + ".png")

gray_complete = cv2.cvtColor(color_complete,cv2.COLOR_BGR2GRAY)

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
cv2.imwrite("pro-img/"+image+"_digitized_image.png", color_complete)




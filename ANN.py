############################################################################################################################################

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

import tensorflow as tf
import sys, os, time
import numpy as np
import math, cv2

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

############################################################################################################################################

class Inpainter():

    inputImage = None
    mask = None
    workImage = None
    sourceRegion = None
    targetRegion = None
    confidence = None
    LAPLACIAN_KERNEL =  None
    patchHeight = patchWidth = 0
    fillFront = []
    train_region = []
    pred_region = []
    halfPatchWidth = None

    def __init__(self, inputImage, mask, halfPatchWidth):

        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.halfPatchWidth = halfPatchWidth

############################################################################################################################################

    def Inpaint(self):

        height, width = self.inputImage.shape[:2]
        out = cv2.VideoWriter('output.avi',cv2.cv.CV_FOURCC('M','J','P','G'), 20.0, (width, height))
        out.write(self.inputImage)
        
        self.initializeMats()
        self.computeFillFront()
        self.compute_training_area(out)
        
        out.release()

##########################################################################################################################################

    def initializeMats(self):

        _, self.confidence = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        self.confidence = np.float32(self.confidence)

        self.sourceRegion = np.copy(self.confidence)
        self.sourceRegion = np.uint8(self.sourceRegion) 
        
        _, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)        
                
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8

############################################################################################################################################

    def computeFillFront(self):
 
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)
        del self.fillFront[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y))

##############################################################################################################################################
    
    def getPatch(self, point):

        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = centerX - self.halfPatchWidth
        maxX = centerX + self.halfPatchWidth
        minY = centerY - self.halfPatchWidth
        maxY = centerY + self.halfPatchWidth
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight

#############################################################################################################################################

    def compute_training_area(self,out):

        height, width = self.workImage.shape[:2]      
        self.patchHeight, self.patchWidth = self.halfPatchWidth*2+1, self.halfPatchWidth*2+1
        area = self.patchHeight * self.patchWidth

        SUM_KERNEL = np.ones((self.patchHeight, self.patchWidth), dtype = np.uint8)
        convolvedMat = cv2.filter2D(self.sourceRegion, cv2.CV_8U, SUM_KERNEL, anchor = (0, 0))

# ----------- <<< listing of pixels information for training purpose >>> ---------------------#

        self.train_region = []
        self.pred_region = []

        for y in range(self.halfPatchWidth,height - self.halfPatchWidth):
            for x in range(self.halfPatchWidth,width - self.halfPatchWidth):
                if convolvedMat[y, x] == area:
                    self.train_region.append((y, x))
                else:
                    self.pred_region.append((y,x))

# ------------- <<< end marker >>>> ---------------- #
# ------------- <<< extraction of training data pixel-by-pixel >>> ---------------------#

        data_points = []
        input_for_model = []

        for point in self.train_region:
            (mX,mY),(MX,MY) = self.getPatch(point)
            del data_points[:]
            for i in range(mX,MX+1):
                for j in range(mY,MY+1):
                    data_points.append(self.workImage[i][j])
            for i in data_points:
                for j in i:
                    input_for_model.append(float(j))

# ----------- <<< pixel data extracted and stored in input_for_model list >>> ---------------------#
# ----------- <<< extraction of data to be generated from the image pixel-by-pixel >>> ---------------------#
 
        data_points = []          
        pred_by_model = []

        for point in self.pred_region:
            (mX,mY),(MX,MY) = self.getPatch(point)
            del data_points[:]
            for i in range(mX,MX+1):
                for j in range(mY,MY+1):
                    data_points.append(self.workImage[i][j])
            for i in data_points:
                for j in i:
                    pred_by_model.append(float(j))

# ----------- <<< pixel data extracted and stored in input_for_model list >>> ---------------------#

        self.trainModel(input_for_model,pred_by_model,out)

#############################################################################################################################################

    def huber_loss(self,labels, predictions, delta=1.0):
        residual = tf.abs(predictions - labels)
        def f1(): return 0.5 * tf.square(residual)
        def f2(): return delta * residual - 0.5 * tf.square(delta)
        return tf.cond(residual < delta, f1, f2)

#############################################################################################################################################
    
    def trainModel(self,input_for_model,pred_by_model,out):

        input_size = (((self.halfPatchWidth*2+1)**2)*3)
        no_of_epoches = 10
        iter_size = input_size
        learning_rate = 0.001
        minput = []
        res = []
        val = []
        temp = [] 
        count = 0 

        X = tf.placeholder(tf.float32,[1,input_size])
        Y = tf.placeholder(tf.float32,[1,input_size])

        w1 = tf.Variable(tf.zeros(shape=[input_size,500]))
        w2 = tf.Variable(tf.zeros(shape=[500,500]))
        w3 = tf.Variable(tf.zeros(shape=[500,500]))
        w4 = tf.Variable(tf.zeros(shape=[500,input_size]))
        
        b1 = tf.Variable(tf.zeros(shape=[1,500]))
        b2 = tf.Variable(tf.zeros(shape=[1,500]))
        b3 = tf.Variable(tf.zeros(shape=[1,500]))
        b4 = tf.Variable(tf.zeros(shape=[1,input_size]))

        inter1 = tf.add(tf.matmul(X,w1),b1)
        inter2 = tf.add(tf.matmul(inter1,w2),b2)
        inter3 = tf.add(tf.matmul(inter2,w3),b3)

        Y_predicted = tf.add(tf.matmul(inter3,w4),b4)
        loss = self.huber_loss(tf.reduce_sum(Y_predicted),tf.reduce_sum(Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iter_cnt = 1
            for _ in range(no_of_epoches):
                iter_size,cnt = input_size,0
                for item in input_for_model:
                    del minput[:]
                    while cnt<iter_size and cnt<len(input_for_model):
                        minput.append(item)
                        cnt += 1
                    if(len(minput)>0):
                        iter_size += input_size
                        sess.run(optimizer,feed_dict = {X:[minput],Y:[minput]})
                    else:
                        break
                print("epoch",iter_cnt,"completed")
                iter_cnt += 1

            iter_size,cnt = input_size,0
            for item in pred_by_model:
                del minput[:]
                temp = []
                while cnt<iter_size and cnt<len(pred_by_model):
                    minput.append(item)
                    cnt += 1
                if(len(minput)>0):
                    iter_size += input_size
                    temp = sess.run([Y_predicted],feed_dict = {X:[minput]})
                    val = (temp[0].size/2)
                    res.append([temp[0][0][val-1],temp[0][0][val],temp[0][0][val+1]])
                else:
                    break

            print(res)

# ----------- <<< Extracted the required values from my tensor-flow model >>> ---------------------#
# ----------- <<< Feeding the values obtained by model into workImage >>> ---------------------#

        for i in range(len(self.pred_region)):
            x_coord = self.pred_region[i][0]
            y_coord = self.pred_region[i][1]
            value = np.asarray(res[i],dtype = np.uint8)
            print(value)
            print(self.workImage[x_coord][y_coord])
            self.workImage[x_coord][y_coord] = value


        cv2.imwrite('result.jpg',self.workImage)


#############################################################################################################################################



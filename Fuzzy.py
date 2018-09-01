#!/usr/bin/python

############################################################################################################################################

import sys, os, time
import math, cv2
import numpy as np

############################################################################################################################################

class Inpainter():

    inputImage = None
    mask = updatedMask = None
    result = None
    workImage = None
    sourceRegion = None
    targetRegion = None
    originalSourceRegion = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None
    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None
    bestMatchUpperLeft = bestMatchLowerRight = None
    patchHeight = patchWidth = 0
    fillFront = []
    normals = []
    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    mode = None
    halfPatchWidth = None
    targetIndex = None
    fuzzy_mat = None

#############################################################################################################################################

    def __init__(self, inputImage, mask, halfPatchWidth):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.updatedMask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.result = np.ndarray(shape = inputImage.shape, dtype = inputImage.dtype)
        self.halfPatchWidth = halfPatchWidth

#############################################################################################################################################

    def inpaint(self):
        self.initializeMats()
        self.calculateGradients()
		height, width = self.inputImage.shape[:2]
		out = cv2.VideoWriter('output.avi',cv2.cv.CV_FOURCC('M','J','P','G'), 20.0, (width, height))
		out.write(self.inputImage)
        stay = True
        
        while stay:
            self.computeFillFront()
            self.computeConfidence()
            self.computeData()
            self.computeTarget()
			self.computeRemainingPixels()
            self.computeBestPatch()
            self.updateMats()
            stay = self.checkEnd()

			out.write(self.workImage)
            cv2.imwrite("updatedMask.jpg", self.updatedMask)
            cv2.imwrite("UpdatedImage.jpg", self.workImage)
        
        self.result = np.copy(self.workImage)
		out.release()
        cv2.imshow("Confidence", self.confidence)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#############################################################################################################################################
    
    def initializeMats(self):

        retval, self.confidence = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        retval, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        
        self.sourceRegion = np.copy(self.confidence)
        self.sourceRegion = np.uint8(self.sourceRegion) 
        self.originalSourceRegion = np.copy(self.sourceRegion)
        
        self.confidence = np.float32(self.confidence)
 
        retval, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        retval, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = np.uint8(self.targetRegion)
        self.data = np.ndarray(shape = self.inputImage.shape[:2],  dtype = np.float32)
        
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype = np.float32)
        self.NORMAL_KERNELX[1, 0] = -1
        self.NORMAL_KERNELX[1, 2] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

        self.fuzzy_mat = np.zeros((9, 9), dtype = np.float32)
		for i in xrange(1,8):
			self.fuzzy_mat[i,1] = 0.3
		for i in xrange(1,8):
			self.fuzzy_mat[i,7] = 0.3
		for i in xrange(1,8):
			self.fuzzy_mat[1,i] = 0.3
		for i in xrange(1,8):
			self.fuzzy_mat[7,i] = 0.3
		for i in xrange(2,7):
			self.fuzzy_mat[i,2] = 0.6
		for i in xrange(2,7):
			self.fuzzy_mat[i,6] = 0.6
		for i in xrange(2,7):
			self.fuzzy_mat[2,i] = 0.6
		for i in xrange(2,7):
			self.fuzzy_mat[6,i] = 0.6
		for i in xrange(3,6):
			for j in xrange(3,6):
				self.fuzzy_mat[i,j] = 1.0


#############################################################################################################################################

    def calculateGradients(self):

        srcGray = cv2.cvtColor(self.workImage, cv2.COLOR_RGB2GRAY)        
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0) 
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)
    
        height, width = self.sourceRegion.shape
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    self.gradientX[y, x] = 0
                    self.gradientY[y, x] = 0
        
        self.gradientX /= 255
        self.gradientY /= 255

#############################################################################################################################################
    
    def computeFillFront(self):
 
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)
        sourceGradientX = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)
        
        del self.fillFront[:]
        del self.normals[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y))
                    dx = sourceGradientX[y, x]
                    dy = sourceGradientY[y, x]
                    
                    normalX, normalY = dy, - dx 
                    tempF = math.sqrt(pow(normalX, 2) + pow(normalY, 2))
                    if not tempF == 0:
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))
    
##############################################################################################################################################
    
    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight

#############################################################################################################################################
    
    def computeConfidence(self):
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)
            total = 0
			i=0
			j=0
            for y in range(aY, bY + 1):
                for x in range(aX, bX + 1):
                    if self.targetRegion[y, x] == 0:
                        total += self.confidence[y, x] * self.fuzzy_mat[i,j]
					j = j+1
				i = i+1
				j = 0
            self.confidence[pY, pX] = total / ((bX-aX+1) * (bY-aY+1))
    
#############################################################################################################################################

    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[y, x] = math.fabs(self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY)
    
#############################################################################################################################################

    def computeTarget(self):
        self.targetIndex = 0
        maxPriority, priority = 0, 0
        omega, alpha, beta = 0.7, 0.2, 0.8
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            priority = self.data[y, x] * self.confidence[y, x]
            if priority > maxPriority:
                maxPriority = priority
                self.targetIndex = i
    
#############################################################################################################################################

    def computeBestPatch(self):
        minError = bestPatchVariance = 9999999999999999
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        height, width = self.workImage.shape[:2]
        workImage = self.workImage
        
        self.patchHeight, self.patchWidth = pHeight, pWidth
        area = pHeight * pWidth
        SUM_KERNEL = np.ones((pHeight, pWidth), dtype = np.uint8)
        convolvedMat = cv2.filter2D(self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor = (0, 0))
        self.sourcePatchULList = []

        for y in range(height - pHeight):
	    for x in range(width - pWidth):
		if convolvedMat[y, x] == area:
			self.sourcePatchULList.append((y, x))
	
		countedNum = 0
        self.targetPatchSList = []
        self.targetPatchTList = []      

        for i in range(pHeight):
            for j in range(pWidth):
                if self.sourceRegion[aY+i, aX+j] == 1:
                    countedNum += 1
                    self.targetPatchSList.append((i, j)) ## case to be considered
                else:
                    self.targetPatchTList.append((i, j))
                    
        
				for (y, x) in self.sourcePatchULList:
				patchError = 0 
                meanR = meanG = meanB = 0

                for (i, j) in self.targetPatchSList:
                        sourcePixel = workImage[y+i][x+j]
                        targetPixel = workImage[aY+i][aX+j]
                        
                        for c in range(3):
                            difference = float(sourcePixel[c]) - float(targetPixel[c])
                            patchError += math.pow(difference, 2)
                        meanR += sourcePixel[0]
                        meanG += sourcePixel[1]
                        meanB += sourcePixel[2]
                
                countedNum = float(countedNum)
                patchError /= countedNum
                meanR /= countedNum
                meanG /= countedNum
                meanB /= countedNum

                if patchError <= minError:
                    patchVariance = 0
                    
                    for (i, j) in self.targetPatchTList:
                                sourcePixel = workImage[y+i][x+j]
                                difference = sourcePixel[0] - meanR
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[1] - meanG
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[2] - meanB
                                patchVariance += math.pow(difference, 2)

					patchError = patchError * self.fuzzy_mat[i,j]

                    if patchError < minError or patchVariance < bestPatchVariance:
                        bestPatchVariance = patchVariance
                        minError = patchError
                        self.bestMatchUpperLeft = (x, y)
                        self.bestMatchLowerRight = (x+pWidth-1, y+pHeight-1)
                    
#############################################################################################################################################
    
    def updateMats(self):
        targetPoint = self.fillFront[self.targetIndex]
        tX, tY = targetPoint
        (aX, aY), (bX, bY) = self.getPatch(targetPoint)
        bulX, bulY = self.bestMatchUpperLeft
        pHeight, pWidth = bY-aY+1, bX-aX+1
        
        for (i, j) in self.targetPatchTList:
			if(self.fuzzy_mat[i,j]==1.0):
				self.workImage[aY+i, aX+j] = self.workImage[bulY+i, bulX+j]
				self.gradientX[aY+i, aX+j] = self.gradientX[bulY+i, bulX+j]
				self.gradientY[aY+i, aX+j] = self.gradientY[bulY+i, bulX+j]
				self.confidence[aY+i, aX+j] = self.confidence[tY, tX]
				self.sourceRegion[aY+i, aX+j] = 1
				self.targetRegion[aY+i, aX+j] = 0
				self.updatedMask[aY+i, aX+j] = 0
    
#############################################################################################################################################

    def checkEnd(self):
        height, width = self.sourceRegion.shape[:2]
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    return True
        return False

#############################################################################################################################################

    def computeRemainingPixels(self):
        height, width = self.sourceRegion.shape[:2]
		cnt=0
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    cnt = cnt + 1
        print cnt

#############################################################################################################################################

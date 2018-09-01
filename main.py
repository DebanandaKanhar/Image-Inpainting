###################################################################################################################################

import cv2
import numpy as np
#import tkFileDialog

from tkinter import *
#from PIL import ImageTk,
from PIL import Image
from Inpainter import Inpainter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

##################################################################################################################################

refpt = []

##################################################################################################################################

def crop_it():

    global refpt

    if len(refpt)>0:

        root.destroy()

        img = cv2.imread('image.jpg')
        height,width,layers = img.shape

        clone = img.copy()

        img = img-clone
        polygon = Polygon(refpt)

        mask = np.zeros((height,width))

        for i in xrange(height):
            for j in xrange(width):
                point = Point(j,i)
                if(polygon.contains(point)):
                    mask[i][j] = 255

        img = clone.copy()

        for i in xrange(height):
            for j in xrange(width):
                point = Point(j,i)
                if(polygon.contains(point)):
                    img[i][j] = [0,255,0]


        cv2.imwrite('Masked_image.jpg',img)
        cv2.imwrite('Mask.jpg',mask)

        mask = cv2.imread('Mask.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

        obj = Inpainter(img,mask, 4)
        obj.inpaint()
        cv2.imwrite("result.jpg", obj.result)
        
#########################################################################################################################

def KeyPressed(event):
    if(event.char)=='c':
        crop_it()
    elif (event.char)=='r':
        global refpt
        refpt = []
    elif(event.char)=='q':
        root.destroy()      
        
#########################################################################################################################

def OnLeftDrag(event):
    global refpt
    refpt.append((event.x,event.y))

#########################################################################################################################

root = Tk()
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_copy = Image.fromarray(img)
img_copy = ImageTk.PhotoImage(img_copy)

Window = Label(root,image = img_copy)
Window.pack(side = "bottom", fill = "both", expand = "yes")

Window.bind('<Key>',KeyPressed)
Window.bind('<B1-Motion>',OnLeftDrag)

Window.focus()
root.title('Image inpainting using Crimnisi Algorithm')

root.mainloop()

#########################################################################################################################

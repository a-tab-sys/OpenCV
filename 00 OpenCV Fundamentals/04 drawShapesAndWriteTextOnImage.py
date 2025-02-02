import cv2 as cv

# 04 DRAW SHAPES AND WRITE TEXT ON AN IMAGE
import numpy as np

#2 ways to draw on images
    #1. draw on a standalone image
    #2. draw on a dummy/blank image

blank=np.zeros((500, 500, 3), dtype='uint8')   #creating a blank image, uint8 is the datatype of an image. (500,500,3) is (height,width,number of color channels)
#cv.imshow('Blank', blank)                   #view the blank image

# 1. Paint the image a certain color
#blank[:] = 0,255,0            #[:] references all the pixels. 0,255,0 gives green. 0,0,255 paints red. Color is (Blue, Green, Red) order
#cv.imshow('Colored Green', blank)

# 2. Paint a range of pixels, to color portions of the image
#blank[200:300, 300:400] = 0,0,255            # dmensions are in pixels
#cv.imshow('Colored Red', blank)

#3. Draw a rectangle
#cv.rectangle(blank, (0, 0), (250,250), (0,255,0), thickness=2)             #parameters: (<image on which to draw the rect>, <bunch of others type .reactangle() to view>)
#cv.rectangle(blank, (0, 0), (250,500), (0,255,0), thickness=cv.FILLED)      #cv.FILLED or -1 = fills in rect with the color
#cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=cv.FILLED)         #scale the rectangle from the original images dimensions
#cv.imshow('Rectangle', blank)

#4. Draw a circle
#cv.circle(blank,(250,250),40, (0,0,255), thickness=3 )                                     #to fill in image give thickness=-1      
#cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1 )        #does the same thing as above line
#cv.imshow('CircleImage', blank)

#5. Draw a line
#cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
#cv.line(blank,(100,250),(300,400), (255,255,255), thickness=3)
#cv.imshow('Line', blank)

#6. Write text on an Image
#cv.putText(blank, 'Hello', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2 )  #cv.FONT_HERSHEY_TRIPLEX is a fontface. 1 is scale, we set s 1 bc we don want to scale.
cv.putText(blank, 'Keep longer text in screen', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2 )  #cv.FONT_HERSHEY_TRIPLEX is a fontface. 1 is scale, we set s 1 bc we don want to scale.
cv.imshow('Text', blank)
cv.waitKey(0)

"""
"""

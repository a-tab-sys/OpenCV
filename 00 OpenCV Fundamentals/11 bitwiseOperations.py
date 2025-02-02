import cv2 as cv

# 11 BITWISE Operations in opencv
#pixels are turned off when they have a value of 0, and turned on when they have a value of 1
import numpy as np

blank=np.zeros((400,400),dtype='uint8')  #create a blank window/variable. use to draw a rectangle and a circle

rectangle=cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)    #<blank image>,<starting point (30,30) means a margin of 30 pixels on either side>,<end point>, <color (if its not a color image, just a binary image you can just give it 1 parameter: 255 which is white)>,<thickness(-1 fills the image)>
circle=cv.circle(blank.copy(), (200, 200), 200, 255, -1)    #<blank img (the .copy makes a copy of the blank image so it doesnt affect the actual blank image) >, <center>, <radius>, <color>, <thickness>

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

#BITWISE AND --> intersecting regions are returned (only portions common to both images are returned)
#you get back took the 2 images, placed then atop another and returned the intersection
bitwise_and = cv.bitwise_and(rectangle, circle)      #pass in 2 source images
cv.imshow('Bitwise AND', bitwise_and)

#BITWISE OR --> non-intersecting and intersecting regions are returned
#superiposes the images
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('BITWISE OR', bitwise_or)

#BITWISE XOR --> non-intersecting regions are returned
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('BITWISE XOR', bitwise_xor)

#BITWISE NOT --> takes in 1 source image and inverts the binary color. turns black to white, turns white to black.
bitwise_not=cv.bitwise_not(rectangle)
cv.imshow('BITWISE NOT', bitwise_not)


cv.waitKey(0) 


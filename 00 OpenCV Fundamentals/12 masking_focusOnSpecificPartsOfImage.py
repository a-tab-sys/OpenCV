import cv2 as cv

# 12 Masking - allows us to focus on parts of an image that wed like to focus on
#Uses bitwise operations
#if you had an image with people and you only want to focus on their faces, you can apply masking
#and remove all unwanted parts of the image
#NOTE: size of mask has to be same dimensions as your img

import numpy as np

img=cv.imread('Photos/fourthcat.png')
#cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')       #[:2] is important bc dimensions of mask have to be the same size as that of the image, if it isnt it doesnt work
#cv.imshow('Blank Image', blank)

#we are gonna draw a circle over our blank image and call that our mask
mymask=cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//4), 100, 255, -1)   #<blank img>, <center>, <radius>, <color>, <thickness>
#cv.imshow('Mask', mymask)

masked = cv.bitwise_and(img, img, mask=mymask)       #<source image 1>, <source image 2>,<specify parameter>
#cv.imshow('Masked Image', masked)

#above we use a circle, if you wanted to use some weird shape, you could combine
#rectangles and circles using bitwise to make a wierd shape and use that for your mask

rectangle=cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)    #<blank image>,<starting point (30,30) means a margin of 30 pixels on either side>,<end point>, <color (if its not a color image, just a binary image you can just give it 1 parameter: 255 which is white)>,<thickness(-1 fills the image)>
circle=cv.circle(blank.copy(), (200, 200), 200, 255, -1)    #<blank img (the .copy makes a copy of the blank image so it doesnt affect the actual blank image) >, <center>, <radius>, <color>, <thickness>

weirdshape=cv.bitwise_and(circle, rectangle)
cv.imshow('Wierd Shape', weirdshape)

masked2 = cv.bitwise_and(img,img, mask=weirdshape)       #<source image 1>, <source image 2>,<specify parameter>
cv.imshow('Masked Image', masked2)

cv.waitKey(0) 


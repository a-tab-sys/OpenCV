import cv2 as cv


# 01 READ IN IMAGES

img = cv.imread('Photos/secondcat.png')                 #cv.imread method lets us read images, takes 
                                                        #a path to an image and returns that image as a matrix of pixels
                                                        #if you have very large image with huge dimensions larger than your screen, you can mitigate this issue by resizing and rescaling images

cv.imshow('Cat', img)                                   #displays image as new window. 2 parameters: (<name of windows>, <the matrix of pixels to dislay>)

cv.waitKey(0)               #keyboard binding function. waits for a specific delay (time in ms) for a key to be pressed
                            #0: waits infinite amount of time


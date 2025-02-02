import cv2 as cv

# 15 Gradients ad Edge Detection
# Gradients are edge like regions present in an image 
# from a mathematical POV Gradients and edges are completly different
# you can get away with thinking of them as the same from a programming perspective
# we knoww canny (advanced edge detection algorithm)- we will discuss 2 other ways: Laplacian Method, Sobel Method
import numpy as np

img=cv.imread('Photos/firstcat.png')
cv.imshow('Cats', img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# METHOD 1: Laplacian Method
# computes the gradients of a grayscale image. when you transition from white to black and black to white
# thats considered a pos and neg slope, images cant have neg pixel values so we compute the absolute value of all the pixel values in the image
# then convert it to uint8 (image specific datatype)
lap = cv.Laplacian(gray, cv.CV_64F)         #takes in <source img>, 
                                            #<ddepth (for now we use cv.CV_64F)>
lap=np.uint8(np.absolute(lap))              #computed absolute value and converted to uint8(img specific datatype) - dont worry about why

cv.imshow('Laplacian', lap)

# METHOD 2: Sobel Method
# sobel computes gradients in 2 directions (x nd y)

#Setting gradients coputed along the x axis = to something
sobelx=cv.Sobel(gray, cv.CV_64F, 1,0)             #<grayscale image>, <ddepth>, <x-dirn>, <y-dirn>
#Setting gradients coputed along the y axis = to something
sobely=cv.Sobel(gray, cv.CV_64F, 0, 1) 

#cv.imshow('Sobel x', sobelx)        #view the x axis gradiants. sobel x computed across the y axis. alot of vertical gradiants
#cv.imshow('Sobel y', sobely)        #view the y axis gradiants. sobel y computed across the x axis. alot of horizontal gradiants

combined_sobel=cv.bitwise_or(sobelx, sobely)     #combine sobel x and sobel y
cv.imshow('Combined Sobel', combined_sobel)        #view the y axis gradiants. sobel y computed across the x axis. alot of horizontal gradiants

# METHOD 3: Canny edge Detector - canny is very advanced, it actually uses sobel as one of its stages
canny=cv.Canny(gray, 150, 175 )
cv.imshow('Canny', canny)        #view the y axis gradiants. sobel y computed across the x axis. alot of horizontal gradiants

cv.waitKey(0) 


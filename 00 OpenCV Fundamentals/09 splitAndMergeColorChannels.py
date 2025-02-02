import cv2 as cv

# 09 Split and merge color channels in opencv
#color image consists of multiple channels(red,gren,blue)
#all images(bgr,rgb) are these 3 color channels merged together
#open cv allows you to split an image into its blue, green and red components
#it shows the intensity othat color on the pixels of the image. 
# lighter areas are more concentrated with that color, darker areas have less or none pixels of that color
import numpy as np

img=cv.imread('Photos/firstcat.png')
cv.imshow('Cats', img)

#we can create a blank image and set it = to the shape of the img
#blank img consists of the height and width not the number of color channels
blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r=cv.split(img)

blue=cv.merge([b,blank,blank])      #passing in a list, only displaying blue: setting green and red components to black
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

#prints the split of the colors in their respective color channels. shows where the respective color is in the image, if none of that color displays black
#cv.imshow('Blue', blue)
#cv.imshow('Green', green)
#cv.imshow('Red', red)

#prints the split of the colors in grayscale. white and black shows how much of a color is in the images pixels
#cv.imshow('Blue', b)
#cv.imshow('Green', g)
#cv.imshow('Red', r)

#these print statements let you visualise the shape of the image
print(img.shape)        #(#, #, 3) 3 is the number of color channels
print(b.shape)          # the b, g and r dont have the 3 in it because the shape of these is 1. gratscale images have a shape of 1
print(g.shape)
print(r.shape)

#merging color channels
merged=cv.merge([b,g,r])     #pass in a list of b, g, r
cv.imshow('Merged Img', merged)

cv.waitKey(0) 


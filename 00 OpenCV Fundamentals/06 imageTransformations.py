import cv2 as cv

# 06 Image Transformations
import numpy as np

img = cv.imread('Photos/firstcat.png')

cv.imshow('Image', img)

# Translation
def translatee(img, x, y):       #to translate an image you need to make a translation matrix, you can rename the function something else
    transMatrix=np.float32([[1,0,x], [0,1,y]])  #you need a transltion matrix. this takes in a list, with 2 lists inside it
    dimensions = (img.shape[1], img.shape[0])   #returns the dimension of the image. [1] is x. [0] is y. For (.shape) function, the height of the image is stored at the index 0. The width of the image is stored at index 1
    return cv.warpAffine(img, transMatrix, dimensions)         #return translated image

#-x --> left
#-y --> up
 
translated = translatee(img, -100, 100)        #moving it right 100 pixels and down 100 pixels
#cv.imshow('Translated', translated )

# Rotation
# open cv allow you to specify the point at which you would like to rotate the image, this could be the center or a corner
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]     #grab height and width of the image by setting it equal to img.shape[:2] of the first 2 values
                                        #[1] is x. [0] is y.
    if rotPoint is None:                #here we are saying that if the rotation point is none, we are assuming that we want to rotate around the center of the image
        rotPoint = (width//2, height//2)

    rotMat= cv.getRotationMatrix2D(rotPoint, angle,1)        #you ned a rotation matrix. this takes the (<rotation point>, <the angle to rotate around>, <scale value>) we dont want to scale the image so make <scale value>=1
    dimensions=(width, height)
    return cv.warpAffine(img, rotMat, dimensions)       #return rotated image (<needs the image>,<rotation matrix>,<destination size which is the dimensions>)

rotated=rotate(img, 45)         #-angle value will rotate CW
#cv.imshow('Rotated', rotated)

# Rotate an already rotated image - this can create missing parts of images corners because when you rotate once you get some black space where there was no image, and if you rotate that it also rotates the blakc space so you get some black triangle cutout on your image.
rotate_rotated=rotate(rotated, 45)
#cv.imshow('Rotate rotated',rotate_rotated )

#Resizing and Image
resized=cv.resize(img,(500, 500), interpolation=cv.INTER_CUBIC )     #pass in <image>, <destination size>, <interpolation>                                                               #cv.INTER_AREA is default(good for shrinking image). you can also put cv.INTER_LINEAR or cv.INTER_CUBIC (good for enlarging image, CUBIC is slow but best quality)
#cv.imshow('Resized', resized)

#Flipping an Image
#dont need to define a function, just need to create a variable and set = to cv.flip
flip=cv.flip(img, 1)      #takes in <img>, <flip code (0:flip vertically/over x axis, 1:flip horizontally/over y axis, -1:flip vertically&horizontally)>
#cv.imshow('Flip', flip)

#Cropping
cropped=img[200:400, 300:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)



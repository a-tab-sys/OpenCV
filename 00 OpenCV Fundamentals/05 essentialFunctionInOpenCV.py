import cv2 as cv

# 05 Essential Functions in OpenCV

img = cv.imread('Photos/fourthcat.png')
cv.imshow('Cat', img)

# Converting to grayscale
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('GrayImg', gray)

# Blur an image - Many ways - we use Gaussian blur
blur = cv.GaussianBlur(img,(7, 7), cv.BORDER_DEFAULT )     #ksize = 3,3. it has to be an odd number, will explain why later. for more blur increase ksize 7,7
#cv.imshow('BlurImg', blur)

# Edge Cascade - find the edges present in an image - many ways - we use canny edge detector. we need two threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded.
#canny=cv.Canny(img, 125, 175)      #<img>, <threshold-minvalue>, <threshold-max value> 
canny=cv.Canny(blur, 125, 175)      #pass in blurred image to reduce how many edges shown
#cv.imshow ('Canny Edges', canny)

# Dilating an image using a structuring element. We will use canny edges as structuring element
# bsically the lines you get from canny edges, appear sharper/brighter after dilating
dilated = cv.dilate(canny, (3,3), iterations=2)         #kernel could also be 7,7. keep it odd values. 
#cv.imshow ('Dilated', dilated)

# Eroding Dilated Image to get back struckturing element. the sharp/brighter edges are eroded so photo is closer to what you got for canny edges
eroded = cv.erode(dilated,(3,3), iterations=1 )
#cv.imshow ('Eroded', eroded)

# Resize Image
resized=cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)     #interpolation=cv.INTER_AREA - good for when you are shrinking the image
resized=cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)   #interpolation=cv.INTER_LINEAR - good for when you enlarge the image
resized=cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)    #interpolation=cv.INTER_CUBIC - good for when you enlarge the image - slowest but produces highest quality 
#cv.imshow('Resized', resized)

# Cropping Image
# Images are arrays, we can use array slicing, select a portion of image based on pixel values
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)



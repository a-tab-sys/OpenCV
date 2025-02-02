import cv2 as cv

# 14 Threshholding
#  Threshholding is binarizaion of an image. it is taing an image and convering it to a binary image
# that is an image where pixels are either 0(black) or 255(white)
# we will take an image and some value called the threshhold value and compare 
# each pixel of the image to the threshold value. if the pixels intensity is less that the 
# threshold value, we set pixel intensity to 0(black), other wise we set the pixel intensity to 255(white)
# so in the end, we can create a binary image from a regular img

# 2 types of thresholding: simple and adaptive

img=cv.imread('Photos/firstcat.png')
#cv.imshow('Cats', img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
# basically what this does is it looks at the image, compares each pixel value to the threshold value,
# if its above this value it sets it to the max value(255, white), otherwise it sets it to 0(black)
# the lower threshold value is the more white you will get in image
threshold, thresh = cv.threshold(gray,150, 255, cv.THRESH_BINARY)     #takes in <grayscale img>, <thresholded value>, <max value>, <thresholding type>
                                                                            #the max value means that if your pixel value is greater than threshold value (150), what do you want to set it to? we want to set it to white so we put 255
#cv.imshow('Simple Thresholded', thresh)

# Simple Thresholding Inverse
# instead of setting pixel intensities greater that 150, to 255
#it setes values less that 150 to 255
threshold, thresh_inverse = cv.threshold(gray,150, 255, cv.THRESH_BINARY_INV)     #takes in <grayscale img>, <thresholded value>, <max value>, <thresholding type>
                                                                            #the max value means that if your pixel value is greater than threshold value (150), what do you want to set it to? we want to set it to white so we put 255
#cv.imshow('Simple Thresholded Inverse', thresh_inverse)

# Adaptive Thresholding
# for simple thresholding we have to manually select a threshold value
# we can let the computer find the optimal thresholding value itself
# this is what adaptive thresholding does

adaptive_thresh_mean=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,3)      #<source img>, 
                                                                                                        #<max value>, 
                                                                                                        #<adaptive method (can be cv.THRESH_BINARY - the mean of some neighborhood of pixels or cv.ADAPTIVE_THRESH_GAUSSIAN_C)>, 
                                                                                                        #<threshold type>, 
                                                                                                        #<blocksize-neighborhood size/kernel size used to compute the mean(for now lets set to 11)>,
                                                                                                        #<c value - an integer that is subtracted from the mean, allows us to fine tune our thresholder (for now lets set to 0 or 3)>
cv.imshow('Adaptive Thresh Mean', adaptive_thresh_mean)                                                           #you can play around with the last 2 values, also you can set cv.THRESH_BINARY_INV to invert the white and black
#basically open cv is computing a mean over the neighborhood or pixels (window) and finds the optimal threshold value for that
#specific part, then the window slides to the right and slides down till it slides over every part
#of the image and computes the mean there

adaptive_thresh_gauss=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,3)                                                   
cv.imshow('Adaptive Thresh Gaussian', adaptive_thresh_gauss)    
#difference beween gaussian and mean: gaussian adds a weight to each pixel value and computes a mean accross those pixels
#thats why we got a better image that with the mean adaptive method
#is some cases mean works well, in some cases adaptive works well

cv.waitKey(0) 


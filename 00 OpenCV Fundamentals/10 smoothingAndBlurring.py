import cv2 as cv

# 10 Smoothing and Blurring in opencv

# first we need to define  kernel, which is a window ou draw over a specific portion of an image 
# this window has a size called a kernel size. kernel size depends on the number of rows and columns
# you have on your window. 3 rows and 3 columns has kernel size 3 by 3
# there are many methods to apply a blur
# blur is applied to the middle pixel as a result of the pixels around it, the surrounding pixels

img=cv.imread('Photos/firstcat.png')
cv.imshow('Cats', img)

# METHOD 1 BLURRING: Averaging
#Averaging defines a kernel window over a portion of an image
#this window computes the pixel intensity of the middle pixel (the true center)
#as the average of the surrounding pixel intensities. so it you have 3 by 3 kernel
#the middle pixel is the average of the blur intensity of the surrounding pixels which you define
#this window will then slide to the right and down to compute for all the pixels in the image

average = cv.blur(img, (3,3))     #<source image>,<kernel size> Higher kernel size results in more blur. kernal size must be odd because of how averaging works, read above
cv.imshow('Average blur', average)

#METHOD 2 BLURRING: Gaussian blur
#similar to averaging but instead of computing the average of pixel intensity
#each surrounding pixel is given a weight, the avg of the products of those weights
#give you the value for the true center
#with this method you get less blurring compared to averaging mehtod however looks more natural

gauss=cv.GaussianBlur(img, (3, 3),0)  #<source image>, <kernel size>, <sigma x: standard deviation in x direction (lets use 0)>
cv.imshow('Gaussian Blur', gauss)

#METHOD 3 BLURRING: Median blur
#same thing as averaging except as instead of finding the average of surrounding pixels
#it finds the median
#median blurring is more effective in reducing noise in images compared to averaging and gussian blur
#used in advaned cv projects that depend on the reduction of substancial amount of noise
#median blurring is not meant for ihh kernel sizes like 7 or even 5
median = cv.medianBlur(img, 3)        #<source img>, <kernel size(instead of a tuple, its just an integer bc open cv know its # by # with just a #)>, 
cv.imshow('Median Blur', median)

#METHOD 4 BLURRING: Bilateral blur
#most efective, used in alot of advanced cv projects bc of how it blurs
#traditional blurring blurs images without considering whether edges are being reduced or not
#bilateral blurring applies blurring but retains the edges in the image
#so you have a blurred image but get to retain edges

bilateral = cv.bilateralFilter(img,5,15,15 ) #<source image>, <diameter of pixel neighborhood - not a kernel size (lets use 5)>, <sigma color- larger value means there are more colors in the neighborhood that will be considered when blur is computed (lets use 15), <sigma space- larger values mean that pixels further out from the central pixel will influence blurring calculation more(lets use 15)>
cv.imshow('Bilateral', bilateral)

#NOTE: higher values sed in bilateral and median blur tend to make the image look more smudged than blurred

cv.waitKey(0) 




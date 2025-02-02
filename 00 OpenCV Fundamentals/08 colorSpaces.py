import cv2 as cv

# 08 Color Spaces
import matplotlib.pyplot as plt
#BGR is opencv default way of reading in images
#how to switch between color spaces in open cv
#rgb, grayscale, hsv, lab and all those are color spaces
#opencv reads in images in a BGR format (blue, green, red) thats not the system
#used outside of opencv. outside opencv RBG is used
#if you wee o disply the img image in  pyhton librayry thats not open cv
#you will see an inversion of colors

img=cv.imread('Photos/secondcat.png')
cv.imshow('Cats', img)

#BGR image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#BBR to HSV (hue saturation value) hsv is based on how humans think an can see with color
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
#cv.imshow('HSV', hsv)

#BGR to LAB(L*A*B) - kind of looks like a washed down HSV image
lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)
#cv.imshow('LAB', lab)

#Grayscale to HSV & Grayscale to LAB - cant be done directly
    #Grayscale-->BGR--->HSV
    #Grayscale--BGR-->LAB
#HSV to BGR
hsv_bgr=cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR',hsv_bgr )

#LAB to BGR
lab_bgr=cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR',lab_bgr )

#KEEP IN MIND WHEN WORKING WITH LIBRARIES OUTSIDE OPENCV, INVERSION OF COLORS MAY OCCUR
#display this image varible, you will notice that the image displayed with plt
#is very different from the original image, this is because opencv is displaying a BGR
#as it normally would. but if you display this in matplotlib, matplotlib does not know this
#is a BGR image and displays the image as if it is an RGB image, thats why matplotlib
#shows the colors as inverted. Shows Red as blue. blue as red.
#plt.imshow(img)
#plt.show()

#BGR to RGB (BGR and RGB are inverse of one another)
#rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)       #pass in <image>, <color code>
#cv.imshow('RGB', rgb)

#so here we provide open cv an rbg image, it assumes its a BGR image and thats why you get a color inversion.
#we pass in Rgb to matplotlib and matplot libs default is rgb so it display the correct coloring
#plt.imshow(rgb)    
#plt.show()

cv.waitKey(0) 




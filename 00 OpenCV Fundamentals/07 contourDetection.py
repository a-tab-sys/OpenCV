import cv2 as cv

# 07 Contour Detection - Detecting the contours of an image
        #contours are not the same as edges
import numpy as np

img=cv.imread('Photos/secondcat.png')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      #converting image to gray scale
#cv.imshow("Gray", gray)

blur=cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)        #if we blur image before finding contours, we get fewer countours
#cv.imshow("Blur", blur)

canny_edges= cv.Canny(img, 125, 175)         #find present edges using canny
cv.imshow('Canny', canny_edges)

canny_edges_blurred= cv.Canny(blur, 125, 175)         #find present edges using canny
cv.imshow('Canny', canny_edges)


#Method 1 for finding contours - uses canny edges - this method returns 2 things: contours and heirarchies
contours, hierarchies = cv.findContours(canny_edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)      #this takes in <canny edges image>, <mode in which to find the contours>, <contour aproximation method>
                                                                        #cv.RETR_TREE: if you want want all heirarhal contours
                                                                        #cv.RETR_EXTERNAL: if you want only external contours
                                                                        #cv.RETR_LIST: if you want all the contours in the image
                                                                        #CHAIN_APPROX_NONE: does nothing, it just returns all the contours
                                                                        #CHAIN_APPROX_SIMPLE: compresses all the contours returned into simple ones that make sense
                                                                        #ex if you had a line: NONE will give you all the contours/points of that line, SIMPLE will takes all the points of the line and compress them into the 2 endpoints only

    #contours is a list, so we can find the number of contours found by printing the length of the list
#print(len(contours))            #prints how many contours there are
#print(f'{len(contours)} countour(s) found')#prints how many contours there are



#Method 2 for finding contours - uses thresholding - thresholding looks at the image and tries to binarize the image (convert it to white(255) or black(0))
                                                    #if a pixels intensity is below 125:it will be set to 0 or black
                                                    #if a pixels intensity is above 125:it is set to white or 255


ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) #This will take <gray image>, <threshold value(125)>, <max value (255)>, <cv.THRESH_BINARY>
cv.imshow('Thresh',thresh )

contours2, hierarchies2 = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)      #this takes in <canny edges image>, <mode in which to find the contours>, <contour aproximation method>
#print(f'{len(contours)} countour(s) found')#prints how many contours there are



#Cool feature - You can visualize the contours found on the image by drawing over the image, basically you are drawing the contours found in thresh and canny and drawing it on the blank image
    #we are going to draw these contours on the blank kimage so we know what type of contours opencv found
    #we are going to do this with the cv.drawContours() method
blankforcanny = np.zeros(img.shape, dtype='uint8')     #create a blank image
cv.imshow('Blank', blankforcanny)

blankforthresh = np.zeros(img.shape, dtype='uint8')     #create a blank image
cv.imshow('Blank', blankforthresh)

cv.drawContours(blankforcanny, contours, -1, (0,0,255),1)      #takes in an <image to draw over>, <the contours list>, <contour index: how many contours do you want? (we want all of them so we put -1)>, <color (we can use red so 0,0,255)>,<thickness>
cv.drawContours(blankforthresh, contours2, -1, (0,0,255),1)

cv.imshow('Contours Drawn from canny', blankforcanny)
cv.imshow('Contours Drawn from thresh', blankforthresh)

cv.waitKey(0)       #all methods need this


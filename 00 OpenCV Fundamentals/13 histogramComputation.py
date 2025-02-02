import cv2 as cv

# 13 Histogram Comptation
# histograms allow you to visualize the distribution of pixel intensity in an image
#whether its a color image or grayscale, you can visualize pixel intensity distributions
#with help help of histogram which is a sort of graph/plot
#also you cn create a mask, so that youre only computing the histogram for a selected portion of an image
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('Photos/fourthcat.png')
cv.imshow('Cats', img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#creating mask
blank = np.zeros((img.shape[:2]), dtype='uint8')
circle=cv.circle(blank, (img.shape[1]//2, (img.shape[0]//3)), 100, 255, -1)

#mask_gray=cv.bitwise_and(gray, gray, mask=circle)
#cv.imshow('Mask', mask_gray)

#Computing histogram for grayscale images
#gray_hist=cv.calcHist([gray], [0], None, [256], [0,256])        #<image as a list (we are only doing for 1 image so pass in image)>, 
                                                                #<number of channels as a list-specifies the index of the channel we want to compute a histogram for(we are computing histogram for grayscale img so pass in 0)>, 
                                                                #<provide a mask - if you want to compleate a histogram for a specific portion of an image (in this case no so pass None)>,
                                                                #<histSize as a list- number of bins we want to use to compute histogram (for now pass [256]))>,
                                                                #<specify the range of all possible pixel values as a list (for our case [0, 256])>,

#plt.figure()                         #instanciate
#plt.title('Grayscale Histogam')       #give it a title
#plt.xlabel('Bins')                    #give it a label across x-axis
#plt.ylabel('# of Pixels')             #give it a label across y-axis
#plt.plot(gray_hist)                   #plot it
#plt.xlim([0, 256])                    #give it a limit across x-axis as list
#plt.show()                            #display the image

#gray_hist_masked=cv.calcHist([gray], [0], mask_gray, [256], [0,256]) 

#plt.figure()                         #instanciate
#plt.title('Grayscale Histogam With Mask')       #give it a title
#plt.xlabel('Bins')                    #give it a label across x-axis
#plt.ylabel('# of Pixels')             #give it a label across y-axis
#plt.plot(gray_hist_masked)            #plot it
#plt.xlim([0, 256])                    #give it a limit across x-axis as list
#plt.show()  

#so what you see is that the number of bins across the x axis represent the intervals of pixel intensity
#so we have a peak at about 65 on the x, and 2100 on the y. this means we have about 2100 pixels
#with an intensity of 65



#Computing histogam for a color rgb image
#plt.figure()                         #instanciate
#plt.title('Color Histogam')       #give it a title
#plt.xlabel('Bins')                    #give it a label across x-axis
#plt.ylabel('# of Pixels')             #give it a label across y-axis

#colors= ('b', 'g','r')    #define a tuple of colors
#for i,col in enumerate(colors):
#    hist=cv.calcHist([img],[i],None, [256],[0,256])     #plot the histogram <img>,<channels>,<mask>, <histsize [256]>,<ranges [0,256]>
#    plt.plot(hist, color=col)
#    plt.xlim([0, 256])

#plt.show()

#cv.waitKey(0) 


#Computing histogam for a color rgb image with a mask -GOT AN ERROR HERE IDK WHYYYYYYYYYYYYYYY
mask_clr=cv.bitwise_and(img, img, mask=circle)
cv.imshow('Mask or color', mask_clr)


plt.figure()                         #instanciate
plt.title('Color Histogam With Mask')       #give it a title
plt.xlabel('Bins')                    #give it a label across x-axis
plt.ylabel('# of Pixels')             #give it a label across y-axis
colors= ('b', 'g','r')    #define a tuple of colors

for i,col in enumerate(colors):
    hist=cv.calcHist([img],[i],mask_clr, [256],[0,256])     #plot the histogram <img>,<channels>,<mask>, <histsize [256]>,<ranges [0,256]>
    plt.plot(hist, color=col)
    plt.xlim([0, 256])


plt.show()

cv.waitKey(0) 



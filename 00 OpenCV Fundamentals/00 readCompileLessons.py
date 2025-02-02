import cv2 as cv

"""
# 01 READ IN IMAGES

img = cv.imread('Photos/secondcat.png')                 #cv.imread method lets us read images, takes 
                                                        #a path to an image and returns that image as a matrix of pixels
                                                        #if you have very large image with huge dimensions larger than your screen, you can mitigate this issue by resizing and rescaling images

cv.imshow('Cat', img)                                   #displays image as new window. 2 parameters: (<name of windows>, <the matrix of pixels to dislay>)

cv.waitKey(0)               #keyboard binding function. waits for a specific delay (time in ms) for a key to be pressed
                            #0: waits infinite amount of time

"""
"""

# 02 READ IN VIDEOS
                                                        #capture is an instance of the VideoCapture class
capture = cv.VideoCapture('Videos/firstvideo.mp4')      #read in videos with Int or string
                                                        #takes in int value 0,1,2,3 if you are using a webcam connected to your computer
                                                        #0: refrences webcam
                                                        #if you have multiple webcams use the correct intger argument. 
                                                        #1:references first camea connected to your computer
                                                        #2:references second camera connected to your computer



                                                        #inside the while loop we read the video frame by frame using the capture.read method
                                                        #displa each frame of the video using cv.imshow method
                                                        #then to break out of the while loop
while True:                                             #for reading in videos you use a while loop and read the video frame by frame
    isTrue, frame = capture.read()                      #capture.read reads in the video frame by frame                 
    cv.imshow('Video', frame)                           #to display an individual frame. displays image as new window. 2 parameters: (<name of windows>, <the matrix of pixels to dislay>)                                              #it returns the frame (freame) and a boolean (isTrue) that says whether the frame was successfully read in or not                    

    #to stop the video from playing indefinitely
    if cv.waitKey(20) & 0xFF==ord('d'):                 #this says the if d is pressed, break out of this loop and stop displaying the video
        break

capture.release()              #release the capture pointer/device
cv.destroyAllWindows           #destroy all windows

#cv.waitKey(0)      #you can comment this line out or have it in, no big difference
                    #when its in d pauses the video and any other button pressed closes the window
                    #when its out, pressing d just closes the video

                    #ERROR: -215 assertion failed : this error means that opencv could not find a media file at the specified location
                    #once your video reaches the end an stops playing, opencv cant find anymore frames so it gives this error
                    #so it unexpetedly breaks out of the while loop by raising an error

"""
"""

# 03 RESIZING AND RESCALING
# we usually resize/rescale videos to prevent computational strain
# media files tend to store a lot of information, displaying it takes alot of processing needs
# resizing/rescaling gets rid of some of that information
# rescaling is modifying height ad width
# your camera dont support going higher than is max capability. a camera that shoots in 720P cant shoot in 1080P or higher.


#METHOD 1: def rescaleFrame function works for video, image, live video
def rescaleFrame(frame, scale=0.75):         #to rescale we crete a function. 2 parameters (<frame>, <scale value>)
    width = int(frame.shape[1]*scale)            #you can cast your floating point values to an int       
    height= int(frame.shape[1]*scale)
    dimensions = (width, height)                   #put dimensions in a tuple

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)        #resizes frame (frame) to a particular dimension (dimensions)
cv.waitKey(0)

#METHOD 2: def changeRes function works for Live Video only
def changeRes(width, height):   #3 and 4 represent properties of this capture class
    capture.set(3, width)       #3 represents the width
    capture.set(4, height)      #4 represents the height
    #capture.set(10, <>)        #10 lets you chage brightness?     

#METHOD 1: EXAMPLE ON IMAGE
#Used the read in image code from above
img = cv.imread('Photos/secondcat.png')

cv.imshow('Cat', img)

image_resized=rescaleFrame(img)                 #added this line to read in image code
cv.imshow('Image Resized', image_resized)           #added this line to read in image code

cv.waitKey(0)

#METHOD 1: EXAMPLE ON VIDEO
#Used the read in video code from above
capture = cv.VideoCapture('Videos/firstvideo.mp4')     
                                                        
while True:                                             
    isTrue, frame = capture.read()                                      
    
    frame_resized=rescaleFrame(frame)                 #added this line to read in video code
    
    cv.imshow('Video', frame)                      

    cv.imshow('Video Resized', frame_resized)         #added this line to read in video code

    if cv.waitKey(20) & 0xFF==ord('d'):  
        break

capture.release()        
cv.destroyAllWindows     

"""
"""

# 04 DRAW SHAPES AND WRITE TEXT ON AN IMAGE
import numpy as np

#2 ways to draw on images
    #1. draw on a standalone image
    #2. draw on a dummy/blank image

blank=np.zeros((500, 500, 3), dtype='uint8')   #creating a blank image, uint8 is the datatype of an image. (500,500,3) is (height,width,number of color channels)
#cv.imshow('Blank', blank)                   #view the blank image

# 1. Paint the image a certain color
#blank[:] = 0,255,0            #[:] references all the pixels. 0,255,0 gives green. 0,0,255 paints red. Color is (Blue, Green, Red) order
#cv.imshow('Colored Green', blank)

# 2. Paint a range of pixels, to color portions of the image
#blank[200:300, 300:400] = 0,0,255            # dmensions are in pixels
#cv.imshow('Colored Red', blank)

#3. Draw a rectangle
#cv.rectangle(blank, (0, 0), (250,250), (0,255,0), thickness=2)             #parameters: (<image on which to draw the rect>, <bunch of others type .reactangle() to view>)
#cv.rectangle(blank, (0, 0), (250,500), (0,255,0), thickness=cv.FILLED)      #cv.FILLED or -1 = fills in rect with the color
#cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=cv.FILLED)         #scale the rectangle from the original images dimensions
#cv.imshow('Rectangle', blank)

#4. Draw a circle
#cv.circle(blank,(250,250),40, (0,0,255), thickness=3 )                                     #to fill in image give thickness=-1      
#cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1 )        #does the same thing as above line
#cv.imshow('CircleImage', blank)

#5. Draw a line
#cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
#cv.line(blank,(100,250),(300,400), (255,255,255), thickness=3)
#cv.imshow('Line', blank)

#6. Write text on an Image
#cv.putText(blank, 'Hello', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2 )  #cv.FONT_HERSHEY_TRIPLEX is a fontface. 1 is scale, we set s 1 bc we don want to scale.
cv.putText(blank, 'Keep longer text in screen', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2 )  #cv.FONT_HERSHEY_TRIPLEX is a fontface. 1 is scale, we set s 1 bc we don want to scale.
cv.imshow('Text', blank)
cv.waitKey(0)

"""
"""

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

"""
"""

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

"""
"""

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

"""
"""

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


"""
"""

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


"""
"""

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

"""
"""

# 11 BITWISE Operations in opencv
#pixels are turned off when they have a value of 0, and turned on when they have a value of 1
import numpy as np

blank=np.zeros((400,400),dtype='uint8')  #create a blank window/variable. use to draw a rectangle and a circle

rectangle=cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)    #<blank image>,<starting point (30,30) means a margin of 30 pixels on either side>,<end point>, <color (if its not a color image, just a binary image you can just give it 1 parameter: 255 which is white)>,<thickness(-1 fills the image)>
circle=cv.circle(blank.copy(), (200, 200), 200, 255, -1)    #<blank img (the .copy makes a copy of the blank image so it doesnt affect the actual blank image) >, <center>, <radius>, <color>, <thickness>

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

#BITWISE AND --> intersecting regions are returned (only portions common to both images are returned)
#you get back took the 2 images, placed then atop another and returned the intersection
bitwise_and = cv.bitwise_and(rectangle, circle)      #pass in 2 source images
cv.imshow('Bitwise AND', bitwise_and)

#BITWISE OR --> non-intersecting and intersecting regions are returned
#superiposes the images
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('BITWISE OR', bitwise_or)

#BITWISE XOR --> non-intersecting regions are returned
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('BITWISE XOR', bitwise_xor)

#BITWISE NOT --> takes in 1 source image and inverts the binary color. turns black to white, turns white to black.
bitwise_not=cv.bitwise_not(rectangle)
cv.imshow('BITWISE NOT', bitwise_not)


cv.waitKey(0) 

"""
"""

# 12 Masking - allows us to focus on parts of an image that wed like to focus on
#Uses bitwise operations
#if you had an image with people and you only want to focus on their faces, you can apply masking
#and remove all unwanted parts of the image
#NOTE: size of mask has to be same dimensions as your img

import numpy as np

img=cv.imread('Photos/fourthcat.png')
#cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')       #[:2] is important bc dimensions of mask have to be the same size as that of the image, if it isnt it doesnt work
#cv.imshow('Blank Image', blank)

#we are gonna draw a circle over our blank image and call that our mask
mymask=cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//4), 100, 255, -1)   #<blank img>, <center>, <radius>, <color>, <thickness>
#cv.imshow('Mask', mymask)

masked = cv.bitwise_and(img, img, mask=mymask)       #<source image 1>, <source image 2>,<specify parameter>
#cv.imshow('Masked Image', masked)

#above we use a circle, if you wanted to use some weird shape, you could combine
#rectangles and circles using bitwise to make a wierd shape and use that for your mask

rectangle=cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)    #<blank image>,<starting point (30,30) means a margin of 30 pixels on either side>,<end point>, <color (if its not a color image, just a binary image you can just give it 1 parameter: 255 which is white)>,<thickness(-1 fills the image)>
circle=cv.circle(blank.copy(), (200, 200), 200, 255, -1)    #<blank img (the .copy makes a copy of the blank image so it doesnt affect the actual blank image) >, <center>, <radius>, <color>, <thickness>

weirdshape=cv.bitwise_and(circle, rectangle)
cv.imshow('Wierd Shape', weirdshape)

masked2 = cv.bitwise_and(img,img, mask=weirdshape)       #<source image 1>, <source image 2>,<specify parameter>
cv.imshow('Masked Image', masked2)

cv.waitKey(0) 

"""
"""

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

"""
"""

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

"""
"""

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

"""
"""

# 16 Face Detection / Recognition with Haar Cascades
# face detection merely detects the presence of a face in an image
# face recognition can identify the face

# face detection is performed using classifiers
# a classifier is an algorithm that decides whether a given image is positive or negative (whether a face is present or not present)
# a classifier needs to be trained on several thousand images with and without faces
# open cv comes with alot of pretrained classifiers that we can use
# 2 main classifiers in use today: haar cascades and more advanced classifiers called local binary patterns(these are more advanced than haar cascades, and not as prone to noise in an image)
#Continued in another file


"""
"""



# 17 Build Face Recognition Model using Open CV's built in Face Recognizer
# Continued in another file


"""


# Note/QUESTIONS:  

#when to use .copy() blank.copy(), img.copy()
#diff between module, method, function, class, instance of a class










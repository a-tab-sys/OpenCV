import cv2 as cv

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


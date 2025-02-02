import cv2 as cv

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


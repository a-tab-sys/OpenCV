import cv2 as cv

# face Detection / r ecognition
# face detection merely detects the presence of a face in an image, face recognition can identify the face

# face detection is performed using classifiers
# a classifier is an algorithm that decides whether a given image is positive or negative (whether a face is present or not present)
# a classifier needs to be trained on several thousand images with and without faces
# open cv comes with alot of pretrained classifiers that we can use
# 2 main classifiers in use today: haar cascades and more advanced classifiers called local binary patterns(these are more advanced than haar cascades, and not as prone to noise in an image)

# See more haar cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

# we are using a haar cascade
# face detection does not involve skin tone or colors that are present in an image.
# haar cascades look at an object in an image and using the edges tries to
# determine whether its a face or not, so you can use grayscale images
# for videos, you would detect on each individual frame of the video

img = cv.imread('photos/group_1.png')
# cv.imshow('Original Image', img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Person', gray)

# read in the haar xml file by createing a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_cascade/haar_frontalface.xml')

# detect the face
# detectMultiScale is an instance of CascadeClassifier class
# it will take in the image and use variables scaleFactor and minNeighbors to detect a face
# and return the rectangular coordinates of that face as a list to the variable faces_rect
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)       # <pass in img you want to detect>, 
                                                                                        # <scaleFactor= (for now lets use 1.1)>, 
                                                                                        # <minNeighbors= (this parameter specifies the number of neighbors a rectangle should have to be called a face, for now use 3)>
                                                                                        # minimizing minNeighbors can provide you a more robust result, more detection but it also makes the haar cascades more prone to noise
# cv: module
# CascadeClassifier: class 
# detectMultiScale: instance of CascadeClassifier class

# prints how many faces it found in the image                                            
print(f'Number of faces found = {len(faces_rect)}')

# since faces_rect has the rectangular coordinates for the faces that are present in the image
# we can loop over this list and grab the coordinates of those images and draw
# a rectangle over the detected faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)         # draw rectangle over the original image. draw rect futakes in <img>, <point 1>, <point 2>, <color>, <thickness>

cv.imshow('Detected Faces', img)


cv.waitKey(0)


# we found that it often detested an incorrect number of faces
# this is because haar cascades are really sensitive to noise in an image
# we can try to minimize sensitivity to noise by modifying scaleFactor and minNeighbors
# minimizing minNeighbors can give you more face detection, a more robust result but it also makes opencv haar cascades more prone to noise, so it might be more likely to detect something thats not a face as a face
# haar cascades are popular and easy to use but are not the most effective in detecting faces
# delibs face recognizer is more effective and less sensitive to noise


